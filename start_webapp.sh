#!/usr/bin/env bash
# start_webapp.sh — launch the DJ Catnip web UI
# Usage: bash ~/kitty_training/start_webapp.sh
# Then open: http://localhost:5000
set -euo pipefail

ROOT="$HOME/kitty_training"
source "$ROOT/venv_stable/bin/activate"

echo "Starting DJ Catnip Trainer at http://localhost:5000"
echo "Press Ctrl+C to stop."
echo ""

python "$ROOT/webapp/app.py"
