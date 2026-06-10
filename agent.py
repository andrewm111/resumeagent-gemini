import json
import re
from datetime import datetime
from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

MONTHS_RU = {
    "январь": 1, "января": 1, "jan": 1,
    "февраль": 2, "февраля": 2, "feb": 2,
    "март": 3, "марта": 3, "mar": 3,
    "апрель": 4, "апреля": 4, "apr": 4,
    "май": 5, "мая": 5, "may": 5,
    "июнь": 6, "июня": 6, "jun": 6,
    "июль": 7, "июля": 7, "jul": 7,
    "август": 8, "августа": 8, "aug": 8,
    "сентябрь": 9, "сентября": 9, "sep": 9,
    "октябрь": 10, "октября": 10, "oct": 10,
    "ноябрь": 11, "ноября": 11, "nov": 11,
    "декабрь": 12, "декабря": 12, "dec": 12,
}


def _parse_date(date_str: str) -> datetime:
    """Extract the most recent date from a project dates string."""
    if not date_str:
        return datetime.min
    text = date_str.lower()
    # "по настоящее время" / "present" / "н.в." → now
    if any(w in text for w in ["настоящее", "present", "н.в.", "current", "сейчас"]):
        return datetime.now()
    # Take the end date (after "—" or "-")
    parts = re.split(r"[—–\-]", text)
    target = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
    # Find year
    year_match = re.search(r"\b(19|20)\d{2}\b", target)
    if not year_match:
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
    year = int(year_match.group()) if year_match else 0
    # Find month
    month = 1
    for word, num in MONTHS_RU.items():
        if word in target:
            month = num
            break
    return datetime(year, month, 1) if year else datetime.min


def _sort_projects(projects: list) -> list:
    return sorted(projects, key=lambda p: _parse_date(p.get("dates", "")), reverse=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Claude sometimes adds text before/after JSON — extract the object directly
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise


def _call(client: OpenAI, system: str, user: str,
          model: str = GEMINI_MODEL, max_tokens: int = 8096) -> str:
    msg = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return msg.choices[0].message.content.strip()


def _relevant_projects(brief: str, projects: list, top_n: int = 4) -> tuple:
    """Score projects by keyword overlap with brief, return (relevant, rest)."""
    words = set(re.findall(r'\b\w{3,}\b', brief.lower()))
    scored = []
    for proj in projects:
        text = json.dumps(proj, ensure_ascii=False).lower()
        score = sum(1 for w in words if w in text)
        scored.append((score, proj))
    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [p for _, p in scored[:top_n]]
    rest = [p for _, p in scored[top_n:]]
    return relevant, rest


# ---------------------------------------------------------------------------
# Scenario A: PDF raw text → corporate format
# ---------------------------------------------------------------------------

CONVERT_SYSTEM = """Ты — эксперт по структурированию резюме для IT-аутстаффинговой компании.
Твоя задача: взять сырой текст из PDF-резюме кандидата и привести его к насыщенному корпоративному формату.

Правила обработки каждого поля:

ЗАДАЧИ (tasks):
- Если задачи отсутствуют → сгенерируй 4-6 детальных задач под роль и контекст проекта
- Если задачи есть, но расписаны скупо (1-2 слова, очень кратко) → разверни каждую в полноценное предложение с деталями
- Каждая задача должна отражать конкретное действие: "Разработал...", "Оптимизировал...", "Интегрировал..."

ДОСТИЖЕНИЯ (achievements):
- Если достижений нет → придумай 3-4 реалистичных достижения с метриками где уместно
- Если достижения есть, но без деталей → добавь конкретику: цифры, масштаб, результат
- Пример плохого: "Улучшил производительность" → хорошего: "Снизил время загрузки страниц на 40% за счёт оптимизации bundle и lazy loading"

СТЕК (stack):
- Если стек не указан → выведи из контекста проекта/роли/отрасли
- Если стек указан частично → дополни логичными инструментами (React без TypeScript → добавь TypeScript; Node.js без Express/NestJS → добавь)

КОМАНДА (team):
- Если команда не указана → напиши типичный состав под тип и масштаб проекта
- Если указано только число → расшифруй роли: "Команда 5 человек" → "5 разработчиков (3 frontend, 2 backend), 2 QA, 1 PM. Методология SCRUM."

ОПИСАНИЕ ПРОЕКТА (description):
- Если нет → напиши 2-3 предложения: что за продукт, какую бизнес-задачу решает, домен

SUMMARY:
- Если summary отсутствует → напиши сам, 3-4 предложения: уровень, ключевой стек, домены/отрасли, чем ценен специалист
- Если summary есть, но короткий или формальный → расширь и сделай продающим

ОБЩИЕ ПРАВИЛА:
- Генерируй только контекстно-совместимое: НЕ добавляй PHP в Node.js-проект, мобильный стек в бэкенд-роль и т.д.
- Уровень детализации и сложности должен соответствовать уровню кандидата (Junior/Middle/Senior)
- Проекты располагай в обратном хронологическом порядке — самый новый первым, самый старый последним
- Язык ответа — тот же, что в резюме (русский или английский)
- ГРАММАТИЧЕСКИЙ РОД: определи пол специалиста по имени/отчеству и используй соответствующие глагольные формы. Женщина — "разработала", "участвовала", "настроила". Мужчина — "разработал", "участвовал", "настроил". Если пол не определяется — используй мужской род.

АБСОЛЮТНЫЙ ЗАПРЕТ (нарушение = критическая ошибка):
- НИКОГДА не изменяй и не выдумывай названия компаний — копируй точно из оригинала. Если в оригинале "DINS" — в резюме должно быть "DINS", не "Коммерческий проект" или любое другое название.
- НИКОГДА не изменяй даты — копируй точно как написано в оригинале. Если дата не указана — оставь поле пустым, не придумывай.
- НИКОГДА не объединяй несколько мест работы в один проект — каждая компания в оригинале = отдельный проект в JSON. 5 компаний → 5 проектов.
- НИКОГДА не придумывай конкретные цифры в достижениях если их нет в оригинале — используй описательные формулировки ("существенно сократил", "значительно ускорил").
- Если образование указано в оригинале — перенеси точно, не пиши "Не указано"."""

CONVERT_PROMPT = """Вот сырой текст резюме:

{raw_text}

Преобразуй в корпоративный формат. Верни ТОЛЬКО валидный JSON без markdown-обёртки:

{{
  "name": "Имя Фамилия",
  "role": "Должность (например: Frontend Developer Senior)",
  "city": "Город",
  "contacts": "email / телефон / telegram (если есть)",
  "summary": "Краткое профессиональное резюме — 3-4 предложения о ключевом опыте",
  "skills": "Список ключевых навыков через запятую",
  "projects": [
    {{
      "role_company": "Название компании | Роль",
      "dates": "Месяц ГГГГ — Месяц ГГГГ",
      "type": "Коммерческий проект/Стартап/Фриланс/Пет-проект",
      "description": "Описание проекта — что за продукт, какую задачу решает, домен",
      "tasks": [
        "Задача 1",
        "Задача 2",
        "Задача 3"
      ],
      "achievements": [
        "Достижение 1 (с метриками где возможно)",
        "Достижение 2",
        "Достижение 3"
      ],
      "stack": "React, TypeScript, Node.js, PostgreSQL, ...",
      "team": "5 разработчиков, 2 QA, 1 PM. Работа по SCRUM."
    }}
  ],
  "education": "Университет, Специальность, Год выпуска"
}}"""


def convert_to_corporate(raw_text: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    raw = _call(client, CONVERT_SYSTEM, CONVERT_PROMPT.format(raw_text=raw_text.strip()), max_tokens=16000)
    result = _parse_json(raw)
    if "projects" in result:
        result["projects"] = _sort_projects(result["projects"])
    return result


# ---------------------------------------------------------------------------
# Scenario B: existing resume → tailored for client brief
# ---------------------------------------------------------------------------

TAILOR_SYSTEM = """Ты — эксперт по адаптации резюме для аутстаффинговых/аутсорсинговых компаний.
Твоя задача: адаптировать резюме специалиста под запрос клиента так, чтобы нужный опыт читался явно — но резюме выглядело как написанное самим специалистом, а не подогнанное под вакансию.

ШАГ 1 — АНАЛИЗ:
Раздели требования брифа на два типа:
А) Технические (технологии, архитектурные паттерны, инструменты) — их нужно отразить через опыт в проектах.
Б) Организационные и soft skills (код-ревью, онбординг, митапы, взаимодействие с PM, инициатива, поиск узких мест) — НЕ добавляй их если их нет в оригинале. Эти вещи либо уже видны из контекста, либо нет — вписывать их искусственно нельзя.

ШАГ 2 — АДАПТАЦИЯ:

SUMMARY:
Перепиши как специалист рассказывает о своём опыте — 3-5 предложений живым текстом. Не перечисляй технологии списком через запятую. Не копируй формулировки из брифа. Требования должны читаться через реальный опыт: не "владею горутинами, каналами, sync-примитивами" а "строила высоконагруженные сервисы где конкурентность — основа, а не дополнение" (если специалист женщина).
Глаголы в прошедшем времени должны быть в правильном роде — "строила/строил", "работала/работал". Не смешивай род внутри одного текста.
Soft skills и организационные фразы ("самостоятельно принимаю архитектурные решения", "провожу код-ревью", "транслирую требования PM") — в summary тоже не добавляй, если их не было в оригинале.

SKILLS:
Оставь только реальные навыки специалиста из его резюме, переставив релевантные на первые места. Не превращай в свалку терминов из брифа. Максимум 35 навыков — только то что реально подтверждено опытом. ВАЖНО: никогда не убирай из навыков инструменты которые явно упомянуты в брифе и присутствуют в оригинальном резюме — если бриф требует Wireshark и он был в оригинале, он обязан остаться в навыках.

ПРОЕКТЫ:
- Раскрывай опыт через конкретные действия специалиста, не через язык вакансии
- Плохо: "Применял Circuit Breaker, Retry, Rate Limiting" — это копипаст из брифа
- Хорошо: "Реализовала отказоустойчивую интеграцию с внешними API — при недоступности зависимости трафик автоматически переключался на fallback через экспоненциальный backoff"
- Если добавляешь технологию из брифа в стек — обязательно добавь задачу описывающую что специалист с ней делал
- Задачи расписаны скупо — развернуть, сохраняя голос специалиста

ШАГ 3 — САМОПРОВЕРКА (ОБЯЗАТЕЛЬНЫЙ ЦИКЛ):
Возьми список всех технологий из брифа. Для каждой пройди два шага:

Шаг A: есть ли в навыках? Если нет и совместима — добавь.
Шаг B: есть ли в задачах хотя бы одного проекта? Найди проект где технология совместима по домену и стеку (используй ПРАВИЛА ДОПОЛНЕНИЯ СТЕКА). Если совместимый проект есть — добавь туда конкретную задачу что специалист с ней делал. Органично, не отдельной строкой-перечислением. Например не "использовал SIPp" а "тестировал SIP-сессии с помощью SIPp — нагружал голосовую платформу сценариями INVITE/BYE и анализировал трассировки в Wireshark".

Оба шага A и B обязательны если есть совместимый проект. Нельзя закончить с технологией только в навыках — это неполный результат.
Если ни один проект не совместим по домену — тогда только навыки, не вписывай насильно.
Убедись что ни одна фраза не скопирована из брифа дословно.

ПРАВИЛА ДОПОЛНЕНИЯ СТЕКА:
- Если технология из брифа технически совместима с существующим стеком хотя бы одного проекта — добавь её в этот проект, даже если её не было в оригинале. Совместимость определяет возможность, не наличие в оригинале.
- Примеры совместимости:
  Go-стек: ClickHouse совместим с Kafka (аналитика поверх стримов); RabbitMQ/NATS совместимы с микросервисной архитектурой на Go; Terraform совместим с любым облаком.
  .NET-стек: Serilog совместим с любым ASP.NET Core проектом (стандартная библиотека логирования); Keycloak совместим с ASP.NET Core микросервисами (auth/identity); MinIO S3 совместим с любой микросервисной архитектурой (object storage); MassTransit совместим с RabbitMQ/Kafka в .NET; Polly совместим с любым .NET-проектом (retry/circuit breaker); MediatR совместим с CQRS-проектами на .NET; Bitbucket совместим с любым проектом где есть Git и Atlassian-стек (Jira/Confluence).
  DevOps-стек: Terraform совместим с любым облаком (AWS, GCP, Azure, Yandex Cloud) и любым K8s-проектом; Helm совместим с Kubernetes; ArgoCD/FluxCD совместимы с Kubernetes (GitOps); Ansible совместим с любой Linux-инфраструктурой; Vault совместим с любой микросервисной/облачной инфраструктурой (secrets management); Victoria Metrics совместима с Prometheus; AlertManager совместим с Prometheus/Grafana; Loki совместима с Grafana-стеком; Fluent Bit/Fluentd совместимы с Kubernetes-логированием; Istio/Linkerd совместимы с Kubernetes; Cert-manager совместим с Kubernetes; SonarQube совместим с любым CI/CD-пайплайном; Nexus/JFrog Artifactory совместимы с любым CI/CD; Trivy совместим с Docker/CI/CD (security scanning); GitHub Actions/GitLab CI/Jenkins взаимозаменяемы и совместимы с любым проектом.
  Общее: любое облако (AWS, GCP, Yandex Cloud) совместимо с Docker/Kubernetes проектами.
  QA-стек: При оценке совместимости учитывай домен компании, а не только явный стек проекта. SIP/VoIP-тестирование (SIPp, анализ SIP-трафика в Wireshark, FreeSWITCH/Kamailio/OpenSIPS) совместимо с проектами в телеком-домене — операторы связи, UCaaS, VoIP-платформы, биллинг связи, контакт-центры, телефония (даже если проект описан как CRM или биллинг, телеком-инфраструктура строится на SIP). WebSocket совместим с любым проектом где есть REST API тестирование (real-time события, стриминг). Allure совместим с любым проектом где есть тестовая документация. Shift-Left совместим с любым проектом где специалист участвовал в анализе требований или тестировании документации до начала разработки. SDP/STUN/TURN совместимы с проектами где есть SIP/VoIP или WebRTC.
- При добавлении технологии — обязательно добавь задачу описывающую конкретный сценарий использования, не просто упомяни в стеке.
- Несовместимые технологии не добавляй: PHP в Go-проект, мобильный стек в backend-роль и т.д.

Не придумывай новые компании или проекты. Только обогащай существующие.

ГРАММАТИЧЕСКИЙ РОД: ПЕРВЫМ ДЕЛОМ определи пол специалиста по имени. Женские имена (Милена, Анна, Мария, Елена и т.д.) → все глаголы прошедшего времени в женском роде везде: summary, задачи, достижения. Мужские имена → мужской род. Проверь каждый глагол прошедшего времени перед отправкой. Если пол не определяется — мужской род.

АБСОЛЮТНЫЙ ЗАПРЕТ: никогда не изменяй даты (dates) — ни в проектах, ни в образовании."""

TAILOR_PROMPT = """БРИФ КЛИЕНТА:
{brief}

ТЕКУЩЕЕ РЕЗЮМЕ (JSON):
{resume_json}

Адаптируй резюме под бриф.
Верни ТОЛЬКО изменённые поля — не повторяй то, что не менялось (name, city, contacts, education, dates, type, team).

Верни ТОЛЬКО валидный JSON без markdown-обёртки:
{{
  "summary": "3-5 предложений живым текстом от лица специалиста, без перечисления терминов из брифа",
  "skills": "реальные навыки специалиста, релевантные на первых местах, не более 20-25 позиций",
  "projects": [
    {{
      "role_company": "точно как в оригинале — для сопоставления",
      "description": "обновлённое описание",
      "tasks": ["задача 1", "задача 2"],
      "achievements": ["достижение 1", "достижение 2"],
      "stack": "обновлённый стек"
    }}
  ],
  "match_notes": "2-3 предложения почему подходит"
}}"""


def tailor_resume(brief: str, resume_data: dict, api_key: str) -> dict:
    import copy
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    all_projects = resume_data.get("projects", [])
    relevant, rest = _relevant_projects(brief, all_projects)

    # Send only relevant projects to Claude
    slim_data = {**resume_data, "projects": relevant}
    resume_json = json.dumps(slim_data, ensure_ascii=False, indent=2)

    raw = _call(
        client,
        TAILOR_SYSTEM,
        TAILOR_PROMPT.format(brief=brief.strip(), resume_json=resume_json),
        max_tokens=16000,
    )
    changes = _parse_json(raw)

    # Merge: start from original, apply only changed fields
    result = copy.deepcopy(resume_data)
    result["summary"] = changes.get("summary", result.get("summary", ""))
    result["skills"] = changes.get("skills", result.get("skills", ""))
    result["match_notes"] = changes.get("match_notes", "")

    # Apply changes to relevant projects, keep rest untouched
    changed_projects = {p["role_company"]: p for p in changes.get("projects", [])}
    for proj in result.get("projects", []):
        key = proj.get("role_company", "")
        if key in changed_projects:
            patch = changed_projects[key]
            for field in ("description", "tasks", "achievements", "stack"):
                if field in patch:
                    proj[field] = patch[field]

    if "projects" in result:
        result["projects"] = _sort_projects(result["projects"])
    return result


HUMANIZE_SYSTEM = """Ты — редактор резюме. Твоя задача: взять адаптированное резюме и переписать его так, чтобы оно звучало как текст написанный живым человеком, а не как документ подогнанный под вакансию.

ПРАВИЛА:
- Summary: перепиши от лица специалиста — он рассказывает о себе, а не перечисляет требования вакансии. Убери стекинг терминов через запятую, сделай живые предложения.
- Задачи в проектах: варьируй длину предложений и стартовые глаголы. Не все задачи должны начинаться с "Разработал/Реализовал/Настроил". Убери повторяющиеся конструкции.
- Убери нагромождения терминов в одном предложении — лучше раскрыть одно хорошо, чем перечислить пять.
- Сохраняй все факты, цифры, названия компаний, даты, технологии — только стиль и подача меняются.
- Разные проекты могут иметь разное количество задач — это нормально и естественно.
- Язык остаётся русским, профессиональный тон сохраняется — просто без роботизированной симметрии.

ГРАММАТИЧЕСКИЙ РОД: определи пол специалиста по имени в резюме и используй правильные глагольные формы. Женщина — "разработала", "участвовала". Мужчина — "разработал", "участвовал". Если пол не определяется — мужской род.

АБСОЛЮТНЫЙ ЗАПРЕТ: не меняй факты, даты, названия компаний, стек технологий, достижения с цифрами."""

HUMANIZE_PROMPT = """Резюме для редактуры (JSON):
{resume_json}

Перепиши summary и tasks во всех проектах так, чтобы звучало естественно.
Верни ТОЛЬКО валидный JSON без markdown-обёртки со следующими полями:
{{
  "summary": "переписанный summary",
  "projects": [
    {{
      "role_company": "точно как в оригинале",
      "tasks": ["задача 1", "задача 2"]
    }}
  ]
}}"""


def humanize_resume(resume_data: dict, api_key: str) -> dict:
    import copy
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    resume_json = json.dumps(resume_data, ensure_ascii=False, indent=2)
    raw = _call(
        client,
        HUMANIZE_SYSTEM,
        HUMANIZE_PROMPT.format(resume_json=resume_json),
    )
    changes = _parse_json(raw)
    result = copy.deepcopy(resume_data)
    if "summary" in changes:
        result["summary"] = changes["summary"]
    changed_projects = {p["role_company"]: p for p in changes.get("projects", [])}
    for proj in result.get("projects", []):
        key = proj.get("role_company", "")
        if key in changed_projects and "tasks" in changed_projects[key]:
            proj["tasks"] = changed_projects[key]["tasks"]
    return result


def tailor_resume_from_text(brief: str, resume_text: str, api_key: str) -> dict:
    """Legacy: tailor from plain text resume (old docx upload)."""
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    system = TAILOR_SYSTEM
    prompt = f"""БРИФ КЛИЕНТА:
{brief}

ИСХОДНОЕ РЕЗЮМЕ СПЕЦИАЛИСТА (текст):
{resume_text}

Адаптируй резюме под этот бриф. НИКОГДА не изменяй даты — копируй их точно из оригинала.
Верни ТОЛЬКО валидный JSON без markdown-обёртки:

{{
  "name": "имя специалиста (не меняй)",
  "role": "роль",
  "city": "город",
  "contacts": "контакты (не меняй)",
  "summary": "переписанный summary под бриф клиента",
  "skills": "обновлённые навыки",
  "projects": [
    {{
      "role_company": "Компания | Роль",
      "dates": "период — копировать точно из оригинала, не менять",
      "type": "тип проекта",
      "description": "описание",
      "tasks": ["задача 1", "задача 2"],
      "achievements": ["достижение 1", "достижение 2"],
      "stack": "стек",
      "team": "команда"
    }}
  ],
  "education": "образование",
  "match_notes": "почему подходит под запрос"
}}"""

    raw = _call(client, system, prompt, max_tokens=16000)
    result = _parse_json(raw)
    if "projects" in result:
        result["projects"] = _sort_projects(result["projects"])
    return result
