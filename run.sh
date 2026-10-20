#!/usr/bin/env bash
# Starts the backend (which also serves the built UI) on port 8000.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
source .venv/bin/activate
# Let CTranslate2 find the pip-installed CUDA libraries
NV=$(python -c "import os,nvidia; print(':'.join(os.path.join(p,'lib') for d in nvidia.__path__ for p in [os.path.join(d,x) for x in os.listdir(d)] if os.path.isdir(os.path.join(p,'lib'))))" 2>/dev/null || true)
export LD_LIBRARY_PATH="${NV}:${LD_LIBRARY_PATH:-}"
pgrep -x ollama >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &)
cd backend
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
