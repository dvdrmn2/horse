#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  echo ".venv already exists — skipping create."
  echo "  source .venv/bin/activate"
  echo "  python main.py"
  exit 0
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Core environment ready (one-time setup)."
echo "Each new terminal session, only run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
