#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  bash scripts/setup_venv.sh
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch mmengine openmim
python -m mim install "mmcv>=2.0.1,<2.2.0"
python -m pip install "mmpose>=1.3.0" --no-deps
python -m pip install json-tricks munkres matplotlib scipy pillow

# xtcocotools often fails on macOS; try prebuilt wheel first.
python -m pip install xtcocotools || echo "Warning: xtcocotools install failed; pose may still work."

echo
echo "Optional MMPose deps installed."
echo "  export POSE_BACKEND=horse10_mmpose"
echo "  python main.py"
