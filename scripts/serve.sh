#!/usr/bin/env bash
# Start the QA service on port 8000 (Linux / macOS / Git Bash).
# The token comes from the environment or from the untracked .env file -- never from code.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${QA_API_TOKEN:-}" ] && ! grep -qs '^QA_API_TOKEN=' .env; then
  echo "QA_API_TOKEN is not set (export it or create .env from .env.example)" >&2; exit 1
fi
# 127.0.0.1: only the tunnel (running on this laptop) can reach the service, not the classroom Wi-Fi.
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
