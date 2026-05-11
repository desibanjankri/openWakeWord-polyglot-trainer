#!/usr/bin/env bash
# slice.sh — slice drop-zone recordings into wake-word training clips
# Usage: bash slice.sh [slicer.py options]
#   bash slice.sh --dry-run
#   bash slice.sh --silence-thresh -35 --min-silence 600
set -euo pipefail

ROOT="$HOME/kitty_training"
DROPZONE="$ROOT/data/dropzone"

if [ ! -d "$DROPZONE" ]; then
    mkdir -p "$DROPZONE"
    echo "Created drop zone: $DROPZONE"
    echo "Drop your long-format .wav recordings there and run this script again."
    exit 0
fi

source "$ROOT/venv_stable/bin/activate"
python "$ROOT/slicer.py" "$@"
