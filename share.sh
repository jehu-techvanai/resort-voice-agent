#!/usr/bin/env bash
# Gives the running agent a temporary https link (needed for the microphone on phones and other computers).
# Anyone with the link can use the agent while this runs, so set ADMIN_PASSWORD in .env first.
set -euo pipefail
PORT="${PORT:-8000}"
if ! command -v cloudflared >/dev/null; then
  if command -v brew >/dev/null; then brew install cloudflared
  else
    echo "Installing cloudflared..."
    curl -L -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i /tmp/cloudflared.deb
  fi
fi
echo "Starting tunnel to http://localhost:$PORT ... look for the https://...trycloudflare.com link below."
exec cloudflared tunnel --url "http://localhost:$PORT"
