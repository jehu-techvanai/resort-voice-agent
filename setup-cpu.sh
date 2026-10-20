#!/usr/bin/env bash
# One-time setup on Ubuntu 22.04 / 24.04, CPU only (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")"

echo "== System check"
echo "CPU cores: $(nproc)   RAM: $(free -g | awk '/Mem:/{print $2}') GB   Python: $(python3 --version)"
[ "$(free -g | awk '/Mem:/{print $2}')" -lt 8 ] && echo "WARNING: less than 8 GB RAM, use LLM_MODEL=llama3.2:1b"

echo "== System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential espeak-ng ffmpeg curl zstd

echo "== Python environment"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, avoids ~3 GB of CUDA libraries
pip install -r backend/requirements.txt

echo "== LLM server (Ollama)"
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh; fi
pgrep -x ollama >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &)
sleep 3
ollama pull llama3.2:3b
ollama pull llama3.2:1b   # faster fallback for slow CPUs

echo "== Settings"
[ -f .env ] || cat > .env <<'ENV'
DEVICE=cpu
STT_MODEL=small
LLM_MODEL=llama3.2:3b
ENV
cat .env

echo "== Pre-download speech and embedding models"
python - <<'PY'
from faster_whisper import WhisperModel; WhisperModel("small", device="cpu", compute_type="int8")
from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")
from kokoro import KPipeline; list(KPipeline(lang_code="a")("Ready.", voice="af_heart"))
print("models cached")
PY

echo "Setup done. Start with ./run.sh"
