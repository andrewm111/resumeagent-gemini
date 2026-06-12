import os
import io
import re
import asyncio
import logging
from functools import partial

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from agent import tailor_resume, convert_to_corporate
from database import list_specialists_summary, load_specialist, save_specialist
from docx_utils import build_output_docx, extract_resume_text, extract_text_from_pdf


def _make_filename(data: dict, fallback: str) -> str:
    try:
        name = data.get("name", fallback).strip()
        role = data.get("role", "").strip()
        combined = f"{name} {role}" if role else name
        combined = re.sub(r"[^\w\s-]", "", combined)
        combined = re.sub(r"\s+", "_", combined.strip())
        return f"{combined}.docx"
    except Exception:
        return f"{fallback}.docx"


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WAITING_SPECIALIST = 0
WAITING_DOC_NAME = 1
CONFIRM_TAILOR = 2
CONFIRM_CONVERT = 3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TEMPLATE_PATH = "template.docx" if os.path.exists("template.docx") else None

ALLOWED_IDS = set(
    int(x.strip())
    for x in os.getenv("ALLOWED_IDS", "").split(",")
    if x.strip().isdigit()
)


def _allowed(user_id: int) -> bool:
    return not ALLOWED_IDS or user_id in ALLOWED_IDS


HELP_TEXT = (
    "Привет! Вот что я умею:\n\n"
    "*Адаптация резюме под бриф:*\n"
    "1. Отправь бриф клиента текстом\n"
    "2. Выбери специалиста из списка\n"
    "3. Получи готовый DOCX\n\n"
    "*Конвертация нового резюме:*\n"
    "1. Прикрепи PDF или DOCX в чат\n"
    "2. Укажи имя — резюме сохранится в базу и получишь DOCX\n\n"
    "/list — список специалистов в базе\n"
    "/start — показать эту инструкцию\n"
    "/cancel — отменить текущее действие"
)

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [["📋 База специалистов"]],
    resize_keyboard=True,
    is_persistent=True,
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=MAIN_KEYBOARD)


async def handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return ConversationHandler.END

    brief = update.message.text.strip()

    if brief == "📋 База специалистов":
        await cmd_list(update, context)
        return ConversationHandler.END

    if len(brief) < 30:
        await update.message.reply_text(
            "Сообщение слишком короткое. Отправь полный текст требований клиента."
        )
        return ConversationHandler.END

    context.user_data["brief"] = brief

    specialists = list_specialists_summary()
    if not specialists:
        await update.message.reply_text(
            "База специалистов пуста. Сначала загрузи PDF/DOCX резюме в чат."
        )
        return ConversationHandler.END

    context.user_data["specialists"] = specialists

    keyboard = [
        [InlineKeyboardButton(s["label"][:60], callback_data=f"spec:{i}")]
        for i, s in enumerate(specialists)
    ]
    await update.message.reply_text(
        "Бриф получен. Выбери специалиста или напиши имя для поиска:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return WAITING_SPECIALIST


async def handle_specialist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not _allowed(query.from_user.id):
        return ConversationHandler.END

    idx = int(query.data[len("spec:"):])
    specialists = context.user_data.get("specialists", [])
    key = specialists[idx]["key"]
    context.user_data["selected_key"] = key

    resume_data = load_specialist(key)
    specialist_name = resume_data.get("name", key)
    role = resume_data.get("role", "")

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, адаптировать", callback_data="tailor:yes"),
        InlineKeyboardButton("❌ Отмена", callback_data="tailor:no"),
    ]])
    await query.edit_message_text(
        f"Адаптировать *{specialist_name}*{f' ({role})' if role else ''} под этот бриф?",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    return CONFIRM_TAILOR


async def handle_confirm_tailor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not _allowed(query.from_user.id):
        return ConversationHandler.END

    if query.data == "tailor:no":
        specialists = context.user_data.get("specialists", [])
        keyboard = [
            [InlineKeyboardButton(s["label"][:60], callback_data=f"spec:{i}")]
            for i, s in enumerate(specialists)
        ]
        await query.edit_message_text(
            "Выбери другого специалиста или напиши имя для поиска:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return WAITING_SPECIALIST

    key = context.user_data.get("selected_key", "")
    brief = context.user_data.get("brief", "")

    await query.edit_message_text("Адаптирую резюме... (~30 сек)")
    try:
        resume_data = load_specialist(key)
        loop = asyncio.get_running_loop()
        tailored = await loop.run_in_executor(
            None, partial(tailor_resume, brief, resume_data, GEMINI_API_KEY)
        )
        docx_bytes = build_output_docx(tailored, key, TEMPLATE_PATH)
        specialist_name = resume_data.get("name", key)
        filename = _make_filename(tailored, specialist_name)
        await query.message.reply_document(
            document=io.BytesIO(docx_bytes),
            filename=filename,
            caption=f"Готово! Резюме {specialist_name} адаптировано под бриф.",
        )
        await query.edit_message_text("✅ Готово")
    except Exception as e:
        logger.exception("Tailor error")
        await query.edit_message_text(f"Ошибка при адаптации: {e}")

    context.user_data.clear()
    return ConversationHandler.END


async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return ConversationHandler.END

    query = update.message.text.strip().lower()
    specialists = context.user_data.get("specialists", [])

    matches = [
        (i, s) for i, s in enumerate(specialists)
        if query in s["label"].lower() or query in s["key"].lower()
    ]

    if not matches:
        await update.message.reply_text(
            f"Никого не нашёл по запросу «{query}». Попробуй другое имя."
        )
        return WAITING_SPECIALIST

    keyboard = [
        [InlineKeyboardButton(s["label"][:60], callback_data=f"spec:{i}")]
        for i, s in matches
    ]
    await update.message.reply_text(
        f"Найдено {len(matches)}: выбери специалиста:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return WAITING_SPECIALIST


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return ConversationHandler.END

    doc = update.message.document
    allowed_mimes = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if doc.mime_type not in allowed_mimes:
        await update.message.reply_text("Поддерживаются только PDF и DOCX файлы.")
        return ConversationHandler.END

    tg_file = await doc.get_file()
    raw = await tg_file.download_as_bytearray()
    context.user_data["doc_bytes"] = bytes(raw)
    context.user_data["doc_mime"] = doc.mime_type
    context.user_data["doc_filename"] = doc.file_name or "resume"

    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Пропустить — взять из резюме", callback_data="doc:skip")]])
    await update.message.reply_text(
        "Файл получен. Напиши имя специалиста для сохранения в базу\n"
        "_(например: Иванов Антон)_\n\n"
        "Или нажми кнопку — имя возьмётся автоматически из резюме.",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    return WAITING_DOC_NAME


async def _do_convert(context, name_override: str | None, reply_fn, send_doc_fn):
    doc_bytes = context.user_data.get("doc_bytes", b"")
    mime = context.user_data.get("doc_mime", "")

    try:
        file_io = io.BytesIO(doc_bytes)
        file_io.name = context.user_data.get("doc_filename", "resume")

        if mime == "application/pdf":
            raw_text = extract_text_from_pdf(file_io)
        else:
            raw_text = extract_resume_text(file_io)

        loop = asyncio.get_running_loop()
        structured = await loop.run_in_executor(
            None, partial(convert_to_corporate, raw_text, GEMINI_API_KEY)
        )
        name = name_override or structured.get("name", "Специалист")
        save_specialist(name, structured)

        docx_bytes = build_output_docx(structured, name, TEMPLATE_PATH)
        filename = _make_filename(structured, name)
        await send_doc_fn(docx_bytes, filename, name)
        await reply_fn(f"✅ Сохранено в базу как *{name}*.")
    except Exception as e:
        logger.exception("Convert error")
        await reply_fn(f"Ошибка при конвертации: {e}")

    context.user_data.clear()


def _confirm_convert_keyboard():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, конвертировать", callback_data="conv:yes"),
        InlineKeyboardButton("❌ Отмена", callback_data="conv:no"),
    ]])


async def handle_doc_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return ConversationHandler.END

    name = update.message.text.strip()
    context.user_data["convert_name"] = name
    filename = context.user_data.get("doc_filename", "файл")
    await update.message.reply_text(
        f"Конвертировать *{filename}* и сохранить как *{name}*?",
        parse_mode="Markdown",
        reply_markup=_confirm_convert_keyboard(),
    )
    return CONFIRM_CONVERT


async def handle_doc_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not _allowed(query.from_user.id):
        return ConversationHandler.END

    filename = context.user_data.get("doc_filename", "файл")
    context.user_data["convert_name"] = None
    await query.edit_message_text(
        f"Конвертировать *{filename}*? Имя возьмётся из резюме автоматически.",
        parse_mode="Markdown",
        reply_markup=_confirm_convert_keyboard(),
    )
    return CONFIRM_CONVERT


async def handle_confirm_convert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not _allowed(query.from_user.id):
        return ConversationHandler.END

    if query.data == "conv:no":
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Пропустить — взять из резюме", callback_data="doc:skip")]])
        await query.edit_message_text(
            "Напиши имя специалиста для сохранения в базу\n_(например: Иванов Антон)_\n\nИли нажми кнопку.",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        return WAITING_DOC_NAME

    name_override = context.user_data.get("convert_name")
    await query.edit_message_text("Конвертирую резюме... (~30 сек)")

    async def send_doc(docx_bytes, filename, specialist_name):
        await query.message.reply_document(
            document=io.BytesIO(docx_bytes),
            filename=filename,
            caption=f"Корпоративный формат: {specialist_name}",
        )

    await _do_convert(context, name_override, lambda t: query.edit_message_text(t, parse_mode="Markdown"), send_doc)
    return ConversationHandler.END


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update.effective_user.id):
        return
    specialists = list_specialists_summary()
    if not specialists:
        await update.message.reply_text("База пуста.")
        return
    keyboard = [
        [InlineKeyboardButton(s["label"][:60], callback_data=f"dl:{i}")]
        for i, s in enumerate(specialists)
    ]
    context.user_data["dl_specialists"] = specialists
    await update.message.reply_text(
        f"📋 *Специалисты в базе ({len(specialists)}):*\nНажми — получишь DOCX.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if not _allowed(query.from_user.id):
        return

    idx = int(query.data[len("dl:"):])
    specialists = context.user_data.get("dl_specialists", [])
    if not specialists:
        specialists = list_specialists_summary()
    key = specialists[idx]["key"]

    await query.edit_message_text("Готовлю файл...")
    try:
        resume_data = load_specialist(key)
        docx_bytes = build_output_docx(resume_data, key, TEMPLATE_PATH)
        filename = _make_filename(resume_data, key)
        specialist_name = resume_data.get("name", key)
        await query.message.reply_document(
            document=io.BytesIO(docx_bytes),
            filename=filename,
            caption=f"{specialist_name} — корпоративный формат",
        )
        await query.edit_message_text(f"📋 Специалисты в базе ({len(specialists)}):")
    except Exception as e:
        logger.exception("Download error")
        await query.edit_message_text(f"Ошибка: {e}")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Отменено.")
    return ConversationHandler.END


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_brief),
            MessageHandler(filters.Document.ALL, handle_document),
        ],
        states={
            WAITING_SPECIALIST: [
                CallbackQueryHandler(handle_specialist, pattern=r"^spec:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_search),
            ],
            CONFIRM_TAILOR: [
                CallbackQueryHandler(handle_confirm_tailor, pattern=r"^tailor:"),
            ],
            WAITING_DOC_NAME: [
                CallbackQueryHandler(handle_doc_skip, pattern=r"^doc:skip$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_doc_name),
            ],
            CONFIRM_CONVERT: [
                CallbackQueryHandler(handle_confirm_convert, pattern=r"^conv:"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_user=True,
        per_chat=True,
        per_message=False,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CallbackQueryHandler(handle_download, pattern=r"^dl:"))
    app.add_handler(conv)

    logger.info("Gemini бот запущен")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
