#!/usr/bin/env bash
cd "$(dirname "$0")"

# Load .env into current shell
[ -f "../../.env" ] && set -a && source "../../.env" && set +a

export PYTHONPATH="$(pwd)"

# Write env file for the child process to source
ENV_FILE="$(pwd)/.env.v2"
declare -p | grep -E '^(declare --|[a-zA-Z_][a-zA-Z_0-9]*)=' | grep -v 'BASH\|DIR\|OLDPWD\|PWD\|SHLVL\|PATH\|HOME\|LOGNAME\|USER\|TERM\|LANG\|LC_\|COLORTERM\|DISPLAY\|XDG\|SSH\|GPG\|HOSTNAME\|HOSTTYPE\|MACHTYPE\|OSTYPE\|PPID\|SHELL\|IFS\|GROUPS\|EUID\|UID\|RANDOM\|LINENO\|SECONDS\|BASHOPTS\|BASH_ALIASES\|BASH_ARGC\|BASH_ARGV\|BASH_CMDS\|BASH_SUBSHELL\|BASH_VERSINFO\|BASH_VERSION\|OPTERR\|OPTIND\|FUNCNAME\|GROUPS' > "$ENV_FILE" 2>/dev/null || true

# Launch uvicorn in its own session, sourcing the env file
setsid bash -c "
  cd '$(pwd)'
  [ -f '.env.v2' ] && set -a && source '.env.v2' && set +a
  export PYTHONPATH='$(pwd)'
  exec nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info > backend.log 2>&1
" &
echo $! > uvicorn.pid
disown

echo "Started detached. PID: $(cat uvicorn.pid)"
