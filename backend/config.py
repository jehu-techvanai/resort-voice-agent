"""All settings come from environment variables so the same code runs on a GPU box, a CPU box or in mock mode."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default):
    return os.environ.get(name, default)


# MOCK=1 swaps every model for a stand-in so the pipeline and UI can be tested without downloads or a GPU.
MOCK = _env("MOCK", "0") == "1"


def detect_device():
    forced = _env("DEVICE", "")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = "cpu" if MOCK else detect_device()

# Speech to text (faster-whisper). Use a multilingual model, not *.en, so non-English speech can be detected.
STT_MODEL = _env("STT_MODEL", "small")
STT_COMPUTE = _env("STT_COMPUTE", "int8_float16" if DEVICE == "cuda" else "int8")
# If Whisper says the language is not English with at least this confidence, the caller is asked to speak English.
NON_ENGLISH_THRESHOLD = float(_env("NON_ENGLISH_THRESHOLD", "0.8"))

# Retrieval (BERT-class embedding model)
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
KNOWLEDGE_FILE = Path(_env("KNOWLEDGE_FILE", str(BASE_DIR / "knowledge" / "resort.md")))
TOP_K = int(_env("TOP_K", "3"))

# LLM: any OpenAI-compatible server. Defaults to Ollama on the same box.
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = _env("LLM_MODEL", "llama3.2:3b")
LLM_API_KEY = _env("LLM_API_KEY", "ollama")
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "180"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "120"))

# Text to speech (Kokoro-82M)
# Kokoro: lang "a" = American English (voices af_*, am_*), "b" = British English (bf_*, bm_*)
TTS_LANG = _env("TTS_LANG", "a")
TTS_VOICE = _env("TTS_VOICE", "af_heart")
TTS_SPEED = float(_env("TTS_SPEED", "1.0"))

AGENT_NAME = _env("AGENT_NAME", "Maya")
RESORT_NAME = _env("RESORT_NAME", "Palm Lagoon Resort")
HISTORY_TURNS = int(_env("HISTORY_TURNS", "4"))

ENGLISH_ONLY_REPLY = (
    "Sorry, I can only help in English at the moment. Could you please say that again in English?"
)
NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?"

# ---- added in v0.2: hands-free sessions, knowledge base admin, API
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR.parent / "data")))
ADMIN_PASSWORD = _env("ADMIN_PASSWORD", "")
API_KEYS = _env("API_KEYS", "")           # comma separated; empty = API open (local testing only)
# Retrieval gate: if the best knowledge chunk scores below this, the agent gives the fallback line
# instead of asking the LLM. bge-small cosine scores; tune in Settings using the Test tab.
MIN_RETRIEVAL_SCORE = float(_env("MIN_RETRIEVAL_SCORE", "0.40"))
CHUNK_WORDS = int(_env("CHUNK_WORDS", "160"))
# Noise filters for always-listening mode
MIN_UTTERANCE_SEC = float(_env("MIN_UTTERANCE_SEC", "0.4"))
MIN_RMS = float(_env("MIN_RMS", "0.006"))
NO_SPEECH_PROB = float(_env("NO_SPEECH_PROB", "0.6"))

# ---- v0.3: speech-to-text choice
STT_ENGINE = _env("STT_ENGINE", "whisper-small")      # whisper-small | whisper-medium | whisper-large-v3-turbo | sarvam
SARVAM_API_KEY = _env("SARVAM_API_KEY", "")
SARVAM_MODEL = _env("SARVAM_MODEL", "saaras:v4")
SARVAM_LANGUAGE = _env("SARVAM_LANGUAGE", "unknown")   # unknown = detect (needed for the English check); or en-IN
