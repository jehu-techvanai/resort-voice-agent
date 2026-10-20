#!/usr/bin/env bash
# One-time setup on Ubuntu 22.04/24.04 with an NVIDIA GPU and driver installed (nvidia-smi works).
set -euo pipefail
cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip espeak-ng ffmpeg curl

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if command -v nvidia-smi >/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  # cuBLAS / cuDNN libraries used by faster-whisper (CTranslate2)
  pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r backend/requirements.txt

# LLM server
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh; fi
ollama pull "${LLM_MODEL:-llama3.2:3b}"

# Frontend (needs Node 20+)
if command -v npm >/dev/null; then
  (cd frontend && npm install && npm run build)
else
  echo "Node.js not found: install Node 20+ and run 'cd frontend && npm install && npm run build'"
fi

# Pre-download the speech and embedding models so the first call isn't slow
python - <<'PY'
from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8")
from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")
from kokoro import KPipeline; list(KPipeline(lang_code="a")("Ready.", voice="af_heart"))
print("models cached")
PY
echo "Setup done. Start with ./run.sh"
