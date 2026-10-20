#!/usr/bin/env bash
# Run after unzipping a new version over an existing install. Keeps your .venv, .env and data/.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
pip install -q -r backend/requirements.txt
echo "Upgrade done. Start with ./run.sh"
