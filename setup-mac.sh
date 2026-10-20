#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac (M1 to M4). Needs Homebrew: https://brew.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v brew >/dev/null; then
  echo "Homebrew not found. Install it from https://brew.sh, then run this again."; exit 1
fi
brew install python@3.12 node espeak-ng ffmpeg
brew install --cask ollama-app 2>/dev/null || brew install ollama

# Python env (3.12, not the system Python)
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch            # macOS wheel; whisper, embeddings and TTS run on CPU
pip install -r backend/requirements.txt

# LLM server (uses the Apple GPU via Metal)
pgrep -x ollama >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &)
sleep 3
ollama pull "${LLM_MODEL:-llama3.2:3b}"

# Frontend
(cd frontend && npm install && npm run build)

# Pre-download speech and embedding models
python - <<'PY'
from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8")
from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")
from kokoro import KPipeline; list(KPipeline(lang_code="a")("Ready.", voice="af_heart"))
print("models cached")
PY

# Force CPU for the Python models (CTranslate2 has no Metal backend)
[ -f .env ] || echo "DEVICE=cpu" > .env
echo "Setup done. Start with ./run.sh"
