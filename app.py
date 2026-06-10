import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from agent import convert_to_corporate, tailor_resume, tailor_resume_from_text, humanize_resume
from docx_utils import extract_resume_text, extract_text_from_pdf, build_output_docx
from database import list_specialists, list_specialists_summary, load_specialist, save_specialist, delete_specialist
from agent import get_last_usage
import re


def _make_filename(data: dict, fallback: str) -> str:
    name = data.get("name", fallback).strip()
    role = data.get("role", "").strip()
    combined = f"{name} {role}" if role else name
    combined = re.sub(r"[^\w\s-]", "", combined)
    combined = re.sub(r"\s+", "_", combined.strip())
    return f"{combined}.docx"

load_dotenv()

TEMPLATE_PATH = "template.docx" if Path("template.docx").exists() else None

st.set_page_config(
    page_title="Resume Agent — Outkod",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Resume Agent")
st.caption("Конвертация и адаптация резюме")

# --- Password protection ---
try:
    app_password = st.secrets["APP_PASSWORD"]
except Exception:
    app_password = os.getenv("APP_PASSWORD", "")

if app_password:
    if not st.session_state.get("authenticated"):
        pwd = st.text_input("Введите пароль для входа", type="password")
        if st.button("Войти"):
            if pwd == app_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Неверный пароль")
        st.stop()

# --- API Key ---
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = ""
if not api_key:
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Вставь ключ или добавь его в файл .env",
    )

st.divider()

tab_convert, tab_tailor, tab_specialists = st.tabs([
    "📥 Конвертация PDF",
    "⚡ Адаптация под клиента",
    "👥 Специалисты",
])

# =====================================================================
# TAB 1: PDF → Corporate format
# =====================================================================
with tab_convert:
    st.subheader("PDF резюме → корпоративный формат")
    st.caption("Загрузи резюме от специалиста в любом виде — агент переведёт в наш формат и дополнит недостающее")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        pdf_file = st.file_uploader(
            "Резюме специалиста (.pdf или .docx)",
            type=["pdf", "docx"],
            key="convert_upload",
        )
        save_to_specialists = st.checkbox("Сохранить в базу специалистов после конвертации", value=True)
        sp_name_convert = st.text_input(
            "Имя для базы (латиницей)",
            placeholder="Ivan_Petrov",
            key="sp_name_convert",
        )

    with col2:
        if pdf_file:
            st.success(f"Файл загружен: {pdf_file.name}")
            with st.expander("Что будет сделано"):
                st.markdown("""
- Извлечь текст из файла
- Разобрать на разделы: имя, роль, проекты, стек, образование
- **Дополнить недостающее**: нет достижений → агент придумает; нет стека → выведет из контекста
- Выдать готовый `.docx` в корпоративном формате
- (Опционально) сохранить в базу специалистов
                """)

    convert_btn = st.button("Конвертировать", type="primary", use_container_width=True, key="convert_btn")

    if convert_btn:
        if not api_key:
            st.error("Нужен API ключ")
        elif not pdf_file:
            st.error("Загрузи файл резюме")
        else:
            with st.spinner("Читаю файл и конвертирую..."):
                try:
                    if pdf_file.name.lower().endswith(".pdf"):
                        raw_text = extract_text_from_pdf(pdf_file)
                    else:
                        raw_text = extract_resume_text(pdf_file)

                    if not raw_text.strip():
                        st.error("Не удалось извлечь текст из файла")
                        st.stop()

                    st.session_state["conv_raw_text"] = raw_text

                    converted = convert_to_corporate(raw_text, api_key)

                    name = converted.get("name", sp_name_convert or pdf_file.name)
                    docx_bytes = build_output_docx(converted, name, TEMPLATE_PATH)
                    filename = _make_filename(converted, name)

                    st.session_state["conv_docx"] = docx_bytes
                    st.session_state["conv_filename"] = filename
                    st.session_state["conv_preview"] = converted

                    u = get_last_usage()
                    st.session_state["conv_usage"] = u

                    if save_to_specialists:
                        save_name = (
                            sp_name_convert.strip()
                            or converted.get("name", "").replace(" ", "_")
                            or pdf_file.name.rsplit(".", 1)[0]
                        )
                        if save_name:
                            try:
                                save_specialist(save_name, converted)
                                st.success(f"Сохранено в базу: {save_name}")
                            except Exception as save_err:
                                st.warning(f"Конвертация прошла успешно, но сохранить в базу не удалось: {save_err}")

                except json.JSONDecodeError as e:
                    st.error(f"Ошибка разбора ответа Claude: {e}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    if st.session_state.get("conv_raw_text"):
        with st.expander("🔍 Debug: что агент получил на вход (первые 3000 символов)"):
            st.text(st.session_state["conv_raw_text"][:3000])

    if st.session_state.get("conv_usage"):
        u = st.session_state["conv_usage"]
        st.caption(f"Токены: {u['prompt_tokens']} вход / {u['completion_tokens']} выход | ~${u['cost_usd']:.4f}")

    if st.session_state.get("conv_docx"):
        st.success("Готово!")
        st.download_button(
            label="📥 Скачать .docx",
            data=st.session_state["conv_docx"],
            file_name=st.session_state["conv_filename"],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_convert_result",
        )
        with st.expander("Предпросмотр"):
            c = st.session_state["conv_preview"]
            st.markdown(f"**{c.get('name', '')}** — {c.get('role', '')}")
            st.markdown(f"*{c.get('summary', '')}*")
            projects = c.get("projects", [])
            st.markdown(f"**Проектов:** {len(projects)}")
            for p in projects:
                st.markdown(f"- {p.get('role_company', '')} ({p.get('dates', '')})")

# =====================================================================
# TAB 2: Tailor for client
# =====================================================================
with tab_tailor:
    st.subheader("Адаптация резюме под запрос клиента")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Бриф клиента**")
        brief = st.text_area(
            "Бриф",
            height=280,
            placeholder="Нужен React-разработчик, 3+ года, финтех, английский B2...",
            label_visibility="collapsed",
            key="tailor_brief",
        )

    with col2:
        st.markdown("**Специалист**")

        summaries = list_specialists_summary()

        search = st.text_input("Поиск по имени, роли или стеку", placeholder="DevOps, React, Иванов...", key="specialist_search")

        filtered = [s for s in summaries if search.lower() in s["label"].lower()] if search else summaries
        labels = ["Загрузить файл"] + [s["label"] for s in filtered]
        keys = [None] + [s["key"] for s in filtered]

        selected_idx = st.selectbox(
            "Источник", range(len(labels)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
            key="tailor_choice",
        )
        choice_key = keys[selected_idx]

        resume_data = None
        resume_text_fallback = ""
        specialist_name = ""

        if choice_key is None:
            uploaded = st.file_uploader(
                "Резюме (.docx или .pdf)",
                type=["docx", "pdf"],
                key="tailor_upload",
            )
            save_to_db = st.checkbox("Сохранить в базу специалистов", value=True, key="tailor_save_db")
            sp_name_tailor = st.text_input("Имя для базы (латиницей)", placeholder="Ivan_Petrov", key="tailor_sp_name")
            if uploaded:
                if uploaded.name.lower().endswith(".pdf"):
                    resume_text_fallback = extract_text_from_pdf(uploaded)
                else:
                    resume_text_fallback = extract_resume_text(uploaded)
                specialist_name = uploaded.name.rsplit(".", 1)[0]
                st.session_state["_tailor_cached_key"] = None
                st.session_state["_tailor_cached_data"] = None
                st.session_state["_tailor_cached_text"] = resume_text_fallback
                st.session_state["_tailor_cached_name"] = specialist_name
                st.success(f"Загружено: {uploaded.name}")
            else:
                if st.session_state.get("_tailor_cached_key") is None:
                    resume_text_fallback = st.session_state.get("_tailor_cached_text", "")
                    specialist_name = st.session_state.get("_tailor_cached_name", "")
        else:
            if choice_key != st.session_state.get("_tailor_cached_key"):
                data = load_specialist(choice_key)
                st.session_state["_tailor_cached_key"] = choice_key
                if "projects" in data:
                    st.session_state["_tailor_cached_data"] = data
                    st.session_state["_tailor_cached_text"] = ""
                else:
                    st.session_state["_tailor_cached_data"] = None
                    st.session_state["_tailor_cached_text"] = data.get("resume_text", "")
                st.session_state["_tailor_cached_name"] = choice_key
            resume_data = st.session_state.get("_tailor_cached_data")
            resume_text_fallback = st.session_state.get("_tailor_cached_text", "")
            specialist_name = st.session_state.get("_tailor_cached_name", choice_key)
            st.info(f"Специалист: **{labels[selected_idx]}**")

    out_name = st.text_input(
        "Имя для файла (необязательно)",
        value=specialist_name,
        placeholder="Ivan_Petrov",
        key="tailor_out_name",
    )

    humanize_check = st.checkbox(
        "Улучшить стиль (звучит живее, менее роботизированно)",
        value=False,
        key="humanize_check",
        help="Дополнительный проход редактора — переформулирует summary и задачи, сохраняя все факты",
    )

    tailor_btn = st.button("⚡ Адаптировать резюме", type="primary", use_container_width=True, key="tailor_btn")

    if tailor_btn:
        if not api_key:
            st.error("Нужен API ключ")
        elif not brief.strip():
            st.error("Вставь бриф клиента")
        elif resume_data is None and not resume_text_fallback.strip():
            st.error("Загрузи или выбери резюме специалиста")
        else:
            spinner_msg = "Агент адаптирует резюме... (это займёт ~30 сек)" if humanize_check else "Агент адаптирует резюме..."
            with st.spinner(spinner_msg):
                try:
                    if resume_data is not None:
                        tailored = tailor_resume(brief, resume_data, api_key)
                    else:
                        tailored = tailor_resume_from_text(brief, resume_text_fallback, api_key)

                    if humanize_check:
                        tailored = humanize_resume(tailored, api_key)

                    st.session_state["tailor_usage"] = get_last_usage()

                    notes = tailored.pop("match_notes", "")

                    if choice_key is None and save_to_db:
                        save_name = (sp_name_tailor.strip() or specialist_name.strip()).replace(" ", "_")
                        if save_name:
                            save_specialist(save_name, tailored)
                            st.success(f"Сохранено в базу: {save_name}")

                    docx_bytes = build_output_docx(tailored, out_name or specialist_name, TEMPLATE_PATH)
                    filename = _make_filename(tailored, out_name or specialist_name)

                    # Сохраняем в session_state чтобы кнопка не гасла при перезапуске
                    st.session_state["last_docx"] = docx_bytes
                    st.session_state["last_filename"] = filename
                    st.session_state["last_notes"] = notes
                    st.session_state["last_summary"] = tailored.get("summary", "")
                    st.session_state["last_skills"] = tailored.get("skills", "")

                except json.JSONDecodeError as e:
                    st.error(f"Ошибка разбора ответа Claude: {e}")
                    st.caption("Попробуй нажать ещё раз — иногда модель возвращает некорректный JSON")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    # Показываем результат из session_state — не пропадёт при перезапуске
    if st.session_state.get("tailor_usage"):
        u = st.session_state["tailor_usage"]
        st.caption(f"Токены: {u['prompt_tokens']} вход / {u['completion_tokens']} выход | ~${u['cost_usd']:.4f}")

    if st.session_state.get("last_docx"):
        if st.session_state.get("last_notes"):
            st.info(f"**Почему подходит:** {st.session_state['last_notes']}")
        st.success("Готово!")
        st.download_button(
            label="📥 Скачать .docx",
            data=st.session_state["last_docx"],
            file_name=st.session_state["last_filename"],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="dl_tailor_result",
        )
        with st.expander("Что изменилось"):
            st.markdown("**Summary:**")
            st.write(st.session_state.get("last_summary", ""))
            st.markdown("**Skills:**")
            st.write(st.session_state.get("last_skills", ""))

# =====================================================================
# TAB 3: Manage specialists
# =====================================================================
with tab_specialists:
    st.subheader("База специалистов")
    st.caption("Сохранённые резюме в корпоративном формате. Появляются в выпадающем списке при адаптации.")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("**Добавить вручную (docx)**")
        sp_name = st.text_input("Имя (латиницей, без пробелов)", placeholder="Ivan_Petrov", key="sp_name_manual")
        sp_file = st.file_uploader("Резюме .docx", type=["docx"], key="sp_upload")
        if st.button("Сохранить", use_container_width=True, key="sp_save"):
            if not sp_name.strip():
                st.error("Введи имя")
            elif not sp_file:
                st.error("Загрузи файл")
            else:
                text = extract_resume_text(sp_file)
                save_specialist(sp_name.strip(), {"name": sp_name, "resume_text": text})
                st.success(f"Сохранено: {sp_name}")
                st.rerun()

    st.markdown("---")
    all_summaries = list_specialists_summary()
    if not all_summaries:
        st.caption("Пока пусто")
    else:
        for s in all_summaries:
            name = s["key"]
            data = load_specialist(name)
            fmt = "structured" if "projects" in data else "text"

            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    display_name = data.get("name", name)
                    role = data.get("role", "")
                    skills_raw = data.get("skills", "")
                    skills_short = ", ".join(
                        sk.strip() for sk in skills_raw.replace("•", ",").split(",")[:4] if sk.strip()
                    )
                    st.markdown(f"**{display_name}**")
                    if role:
                        st.caption(role)
                    if skills_short:
                        st.caption(f"🛠 {skills_short}")
                with right:
                    if fmt == "structured":
                        docx_bytes = build_output_docx(data, name, TEMPLATE_PATH)
                        st.download_button(
                            label="📥",
                            data=docx_bytes,
                            file_name=_make_filename(data, name),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"dl_{name}",
                        )
                    if st.button("❌", key=f"del_{name}"):
                        st.session_state[f"confirm_del_{name}"] = True

            if st.session_state.get(f"confirm_del_{name}"):
                st.warning(f"Удалить **{display_name}** из базы?")
                col_yes, col_no = st.columns([1, 1])
                if col_yes.button("Да, удалить", key=f"yes_{name}", type="primary"):
                    delete_specialist(name)
                    st.session_state.pop(f"confirm_del_{name}", None)
                    st.rerun()
                if col_no.button("Отмена", key=f"no_{name}"):
                    st.session_state.pop(f"confirm_del_{name}", None)
                    st.rerun()
