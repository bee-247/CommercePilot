#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_python="$project_dir/.venv/bin/python"

if [ ! -f "$project_dir/.env" ]; then
  echo "缺少 .env，请先执行: cp .env.example .env 并填写模型与认证配置" >&2
  exit 1
fi

if [ ! -x "$venv_python" ]; then
  echo "缺少 Python 虚拟环境，请先执行: python3 -m venv .venv" >&2
  exit 1
fi

backend_pid=""
frontend_pid=""

cleanup() {
  if [ -n "$backend_pid" ]; then
    kill "$backend_pid" 2>/dev/null || true
  fi
  if [ -n "$frontend_pid" ]; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "$project_dir/backend"
  exec "$venv_python" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
) &
backend_pid=$!

(
  cd "$project_dir/frontend"
  exec pnpm dev --host 0.0.0.0
) &
frontend_pid=$!

wait "$backend_pid" "$frontend_pid"
