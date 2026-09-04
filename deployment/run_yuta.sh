#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting YUTA — Unified Video Intelligence Platform"
echo " Gujarat Police Innovation Challenge 2026"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    echo " Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
fi

export PYTHONPATH="$ROOT_DIR"
echo " Running test suite..."
python -m pytest tests/ -q

echo " Launching YUTA Intelligence API & Dashboard at http://localhost:8000 ..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
