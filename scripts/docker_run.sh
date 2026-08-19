#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f yolov8x.pt ]]; then
  echo "Missing yolov8x.pt in project root."
  echo "Download YOLOv8x weights and place them at: $ROOT/yolov8x.pt"
  exit 1
fi

mkdir -p output

echo "Building Linux container (first run may take several minutes)..."
docker compose build

echo "Running horse detection + pose..."
docker compose run --rm horse

echo
echo "Done. Open output/annotated.mp4 to review results."
