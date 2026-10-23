#!/usr/bin/env bash
# Start everything needed for the Twilio voice agent:
#   1. Ollama (LLM backend, port 11434)
#   2. FastAPI backend on port 8000  → logs to log.txt
#   3. ngrok tunnel on port 8000     → logs to /tmp/ngrok.log
#
# Run from the project root:
#   cd backend && bash run.sh
#
# Stop everything with: pkill -f "uvicorn app:app" ; pkill -f "ollama serve" ; pkill -f "ngrok http 8000"

set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$BACKEND_DIR")"

# --- 1. Ollama ---
echo "=== Starting Ollama (if not running) ==="
if ! pgrep -f "ollama serve" >/dev/null; then
  ollama serve > /tmp/ollama.log 2>&1 &
  echo "Ollama starting (PID $!)…"
  # wait until port 11434 is accepting connections
  for i in $(seq 1 30); do
    if curl -s --max-time 2 "http://localhost:11434/api/tags" >/dev/null 2>&1; then
      echo "Ollama ready."
      break
    fi
    sleep 1
  done
else
  echo "Ollama already running."
fi

# --- 2. Backend ---
echo ""
echo "=== Starting backend → log.txt ==="
cd "$BACKEND_DIR"

# clear previous log so you see fresh output
> log.txt

PYTHONPATH="$BACKEND_DIR" \
  nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 > log.txt 2>&1 &
BACKEND_PID=$!
echo "Backend starting (PID $BACKEND_PID) → log.txt"

# wait for "Uvicorn running" in the log
for i in $(seq 1 60); do
  if grep -q "Uvicorn running on http://0.0.0.0:8000" log.txt 2>/dev/null; then
    echo "Backend ready (port 8000)."
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: backend exited early. Last log lines:"
    tail -20 log.txt
    exit 1
  fi
  sleep 1
done

# --- 3. ngrok ---
echo ""
echo "=== Starting ngrok → /tmp/ngrok.log ==="

# kill any stale ngrok tunnel on port 8000
pkill -f "ngrok http 8000" 2>/dev/null || true
sleep 1

NGROK_LOG="/tmp/ngrok.log"
> "$NGROK_LOG"

# use Python to launch ngrok detached (bash/nohup can't keep ngrok alive on this system)
python3 - <<'PY' &
import os, subprocess, time

def load_env(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k] = v

load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# start ngrok
proc = subprocess.Popen(
    ['ngrok', 'http', '8000'],
    stdout=open('/tmp/ngrok.log', 'a'),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
time.sleep(2)
print(f"ngrok PID: {proc.pid}", flush=True)
PY

NGROK_PID=$!
echo "ngrok starting (PID $NGROK_PID) → $NGROK_LOG"

# wait for the Forwarding line in the ngrok log
for i in $(seq 1 30); do
  if grep -q "Forwarding" "$NGROK_LOG" 2>/dev/null; then
    echo "ngrok tunnel ready."
    echo ""
    echo "=== ngrok URL ==="
    grep "Forwarding" "$NGROK_LOG" | tail -1
    echo ""
    echo "=== Health check ==="
    curl -s --max-time 5 "https://$(grep -o 'https://[^ ]*ngrok-free.dev' "$NGROK_LOG" | head -1)/api/health" || echo "(ngrok URL not yet reachable)"
    echo ""
    echo "=== Logs ==="
    echo "  backend: $BACKEND_DIR/log.txt"
    echo "  ngrok:   $NGROK_LOG"
    echo ""
    echo "=== Call your Twilio number now ==="
    exit 0
  fi
  if ! kill -0 "$NGROK_PID" 2>/dev/null; then
    echo "ERROR: ngrok exited early. Log:"
    tail -20 "$NGROK_LOG"
    exit 1
  fi
  sleep 1
done

echo "ERROR: ngrok did not start a tunnel within 30s. Log:"
tail -20 "$NGROK_LOG"
exit 1
