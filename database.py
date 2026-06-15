import json
import os
import httpx
from pathlib import Path

# ---------------------------------------------------------------------------
# Storage backend: Supabase (cloud) or local files (local)
# ---------------------------------------------------------------------------

def _supabase_config():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        try:
            import streamlit as st
            url = st.secrets.get("SUPABASE_URL", "")
            key = st.secrets.get("SUPABASE_KEY", "")
        except Exception:
            pass
    return url.rstrip("/"), key


def _table() -> str:
    name = os.getenv("SUPABASE_TABLE", "")
    if not name:
        try:
            import streamlit as st
            name = st.secrets.get("SUPABASE_TABLE", "")
        except Exception:
            pass
    return name or "specialists"


def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def use_supabase() -> bool:
    url, key = _supabase_config()
    return bool(url and key)


# ---------------------------------------------------------------------------
# Public API — used by app.py
# ---------------------------------------------------------------------------

LOCAL_DIR = Path("specialists")


TIMEOUT = 10


def list_specialists() -> list[str]:
    if use_supabase():
        url, key = _supabase_config()
        r = httpx.get(f"{url}/rest/v1/{_table()}?select=name&order=name.asc", headers=_headers(key), timeout=TIMEOUT)
        r.raise_for_status()
        return [row["name"] for row in r.json()]
    else:
        LOCAL_DIR.mkdir(exist_ok=True)
        return sorted(f.stem for f in LOCAL_DIR.glob("*.json"))


def list_specialists_summary() -> list[dict]:
    """Return list of {key, label} for display in selectbox."""
    if use_supabase():
        url, key = _supabase_config()
        r = httpx.get(
            f"{url}/rest/v1/{_table()}?select=name,data&order=name.asc",
            headers=_headers(key),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
    else:
        LOCAL_DIR.mkdir(exist_ok=True)
        rows = []
        for f in sorted(LOCAL_DIR.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"name": f.stem, "data": data})

    result = []
    for row in rows:
        key_name = row["name"]
        data = row["data"]
        display_name = data.get("name", key_name)
        role = data.get("role", "")
        skills = data.get("skills", "")
        if isinstance(skills, list):
            skills = ", ".join(str(s) for s in skills)
        skills_short = ", ".join(s.strip() for s in skills.replace("•", ",").split(",")[:3] if s.strip())
        parts = [p for p in [display_name, role, skills_short] if p]
        label = " | ".join(parts)
        result.append({"key": key_name, "label": label})
    return result


def load_specialist(name: str) -> dict:
    if use_supabase():
        url, key = _supabase_config()
        r = httpx.get(
            f"{url}/rest/v1/{_table()}?name=eq.{name}&select=data",
            headers=_headers(key),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            raise KeyError(f"Specialist '{name}' not found")
        return rows[0]["data"]
    else:
        path = LOCAL_DIR / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))


def save_specialist(name: str, data: dict) -> None:
    if use_supabase():
        url, key = _supabase_config()
        payload = {"name": name, "data": data}
        headers = {**_headers(key), "Prefer": "resolution=merge-duplicates"}
        r = httpx.post(f"{url}/rest/v1/{_table()}", json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
    else:
        LOCAL_DIR.mkdir(exist_ok=True)
        (LOCAL_DIR / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def delete_specialist(name: str) -> None:
    if use_supabase():
        url, key = _supabase_config()
        r = httpx.delete(
            f"{url}/rest/v1/{_table()}?name=eq.{name}",
            headers=_headers(key),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
    else:
        path = LOCAL_DIR / f"{name}.json"
        if path.exists():
            path.unlink()
