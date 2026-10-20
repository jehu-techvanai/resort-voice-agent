"""Persistent settings, admin login tokens and the unanswered-questions log. Everything lives in DATA_DIR."""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

import config

DATA_DIR: Path = config.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
UNANSWERED_FILE = DATA_DIR / "unanswered.jsonl"
SECRET_FILE = DATA_DIR / ".secret"

_lock = threading.Lock()

# Kokoro English voices. Prefix a = American, b = British; f = female, m = male.
VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_aoede", "af_kore", "af_nova", "af_sky",
    "am_michael", "am_adam", "am_fenrir", "am_puck", "am_echo",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
]

DEFAULTS = {
    "business_name": config.RESORT_NAME,
    "agent_name": config.AGENT_NAME,
    "business_type": "resort",
    "greeting": "Hi, this is {agent} from {business}. How can I help you?",
    "fallback_text": "I'm sorry, I cannot answer that. But I can log that for you and have my boss call you back.",
    "extra_instructions": "",
    "voice": config.TTS_VOICE,
    "speed": config.TTS_SPEED,
    "wake_required_web": True,
    "followup_seconds": 20,
    "min_retrieval_score": config.MIN_RETRIEVAL_SCORE,
    "active_kb": None,
    "stt_engine": config.STT_ENGINE,
}
STT_ENGINES = ["whisper-small", "whisper-medium", "whisper-large-v3-turbo", "sarvam"]


def load_settings() -> dict:
    with _lock:
        data = {}
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
            except json.JSONDecodeError:
                data = {}
        return {**DEFAULTS, **data}


def save_settings(update: dict) -> dict:
    current = load_settings()
    allowed = {k: v for k, v in update.items() if k in DEFAULTS}
    if "voice" in allowed and allowed["voice"] not in VOICES:
        raise ValueError(f"unknown voice {allowed['voice']}")
    if "stt_engine" in allowed and allowed["stt_engine"] not in STT_ENGINES:
        raise ValueError(f"unknown speech-to-text engine {allowed['stt_engine']}")
    if "speed" in allowed:
        allowed["speed"] = max(0.6, min(1.6, float(allowed["speed"])))
    if "followup_seconds" in allowed:
        allowed["followup_seconds"] = max(0, min(120, int(allowed["followup_seconds"])))
    if "min_retrieval_score" in allowed:
        allowed["min_retrieval_score"] = max(0.0, min(1.0, float(allowed["min_retrieval_score"])))
    for k in ("agent_name", "business_name"):
        if k in allowed:
            allowed[k] = str(allowed[k]).strip()[:60] or DEFAULTS[k]
    merged = {**current, **allowed}
    with _lock:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(SETTINGS_FILE)
    return merged


# ---------------------------------------------------------------- admin auth

def _secret() -> bytes:
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(secrets.token_hex(32))
        os.chmod(SECRET_FILE, 0o600)
    return SECRET_FILE.read_text().strip().encode()


def admin_password() -> str:
    """ADMIN_PASSWORD env wins; otherwise a random one is generated once and stored in data/admin_password.txt."""
    if config.ADMIN_PASSWORD:
        return config.ADMIN_PASSWORD
    f = DATA_DIR / "admin_password.txt"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(9))
        os.chmod(f, 0o600)
    return f.read_text().strip()


def make_token(hours: int = 12) -> str:
    exp = str(int(time.time()) + hours * 3600)
    sig = hmac.new(_secret(), f"admin:{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def check_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    good = hmac.new(_secret(), f"admin:{exp}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()


def check_password(pw: str) -> bool:
    return hmac.compare_digest((pw or "").encode(), admin_password().encode())


# ---------------------------------------------------------------- API keys (phase 3 uses these)

def api_keys() -> set[str]:
    return {k.strip() for k in config.API_KEYS.split(",") if k.strip()}


# ---------------------------------------------------------------- unanswered log

def log_unanswered(question: str, session_id: str, mode: str, name: str = "", phone: str = "", note: str = "") -> dict:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "name": name,
        "phone": phone,
        "note": note,
        "mode": mode,
        "session": session_id,
        "resolved": False,
    }
    with _lock, UNANSWERED_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def update_unanswered(entry_id: str, **fields) -> bool:
    with _lock:
        if not UNANSWERED_FILE.exists():
            return False
        rows = [json.loads(l) for l in UNANSWERED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        hit = False
        for r in rows:
            if r["id"] == entry_id:
                r.update({k: v for k, v in fields.items() if k in ("name", "phone", "note", "resolved")})
                hit = True
        if hit:
            UNANSWERED_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return hit


def list_unanswered() -> list[dict]:
    if not UNANSWERED_FILE.exists():
        return []
    rows = [json.loads(l) for l in UNANSWERED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list(reversed(rows))
