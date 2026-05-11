#!/usr/bin/env bash
# train.sh — train a new DJ Catnip model from clips in positive_train/
# Produces: models/dj_catnip_YYYYMMDD_HHMM.onnx + updates dj_catnip_latest.onnx symlink
# Usage: bash train.sh
set -euo pipefail

ROOT="$HOME/kitty_training"
MODELS="$ROOT/models"
CONFIG="$ROOT/training_config.yaml"
TRAIN_PY="$ROOT/openWakeWord/openwakeword/train.py"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$ROOT/training_run_${TIMESTAMP}.log"
V1_ONNX="$MODELS/dj_catnip_v1.onnx"
DATED_ONNX="$MODELS/dj_catnip_${TIMESTAMP}.onnx"
LATEST_LINK="$MODELS/dj_catnip_latest.onnx"
ORIGINAL_BACKUP="$MODELS/dj_catnip_v1_original.onnx"

# Sanity checks
TRAIN_COUNT=$(find "$MODELS/dj_catnip_v1/positive_train" -maxdepth 1 -iname '*.wav' 2>/dev/null | wc -l)
if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "ERROR: No clips found in positive_train/. Run slice.sh first."
    exit 1
fi
echo "positive_train/ has $TRAIN_COUNT clips."

# One-time backup of the original piper-only model before it gets overwritten
if [ -f "$V1_ONNX" ] && [ ! -f "$ORIGINAL_BACKUP" ]; then
    cp "$V1_ONNX" "$ORIGINAL_BACKUP"
    echo "Backed up original piper-only model → $ORIGINAL_BACKUP"
fi

source "$ROOT/venv_stable/bin/activate"

echo "Starting training at ${TIMESTAMP} — log: $LOG"
echo ""

# The TFLite conversion at the end of train.py always exits non-zero (onnx_tf not
# installed, by design). pipefail would abort here before cp/ln run. Capture the
# Python exit code separately and only fail on unexpected errors (exit > 1).
set +o pipefail
python "$TRAIN_PY" \
    --training_config "$CONFIG" \
    --augment_clips \
    --overwrite \
    --train_model \
    2>&1 | tee "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -o pipefail
if [ "$TRAIN_EXIT" -gt 1 ]; then
    echo ""
    echo "ERROR: Training exited with code $TRAIN_EXIT — check $LOG"
    exit "$TRAIN_EXIT"
fi

# train.py writes to models/dj_catnip_v1.onnx — copy to dated file and update symlink
if [ ! -f "$V1_ONNX" ]; then
    echo ""
    echo "ERROR: Expected model not found at $V1_ONNX"
    exit 1
fi

cp "$V1_ONNX" "$DATED_ONNX"
ln -sf "dj_catnip_${TIMESTAMP}.onnx" "$LATEST_LINK"

echo ""
echo "────────────────────────────────────────────"
echo "Model saved : $DATED_ONNX"
echo "Symlink     : $LATEST_LINK → dj_catnip_${TIMESTAMP}.onnx"
echo ""
echo "Final metrics:"
grep "Final Model" "$LOG" || echo "  (no metrics line found in log)"
echo "────────────────────────────────────────────"
