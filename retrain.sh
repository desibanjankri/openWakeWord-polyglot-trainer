#!/usr/bin/env bash
# retrain.sh — full pipeline: slice drop zone → optional inspect → train
# Usage: bash retrain.sh [slice.sh options]
#   bash retrain.sh --silence-thresh -35
set -euo pipefail

ROOT="$HOME/kitty_training"

echo "=== DJ Catnip Retrain Pipeline ==="
echo ""

bash "$ROOT/slice.sh" "$@"

echo ""
read -rp "Continue to training? [y/N] " CONFIRM
if [[ "${CONFIRM:-}" != "y" && "${CONFIRM:-}" != "Y" ]]; then
    echo ""
    echo "Stopped. Inspect clips in models/dj_catnip_v1/positive_train/ then run:"
    echo "  bash train.sh"
    exit 0
fi

echo ""
bash "$ROOT/train.sh"
