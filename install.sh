#!/usr/bin/env bash
# install.sh — openWakeWord WSL Factory bootstrap
# Run once after cloning: bash install.sh

set -euo pipefail

# ── Configurable download URLs ────────────────────────────────────────────────
# These point to the official openWakeWord HuggingFace dataset.
# Verify at: https://huggingface.co/datasets/davidscripka/openwakeword_features
ACAV_URL="https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VAL_URL="https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

# Pinned openWakeWord commit — patches are applied against this exact revision
OWW_COMMIT="368c037"
OWW_REPO="https://github.com/dscripka/openWakeWord.git"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GRN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YLW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()  { echo -e "\n${GRN}══ $* ${NC}"; }

# ── 1. System prerequisites ───────────────────────────────────────────────────
step "Checking prerequisites"

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.12: sudo apt install python3.12 python3.12-venv"
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PY_VER"

if ! command -v ffmpeg &>/dev/null; then
    warn "ffmpeg not found — required for audio processing."
    echo "  Install with: sudo apt install ffmpeg"
    echo "  Then re-run install.sh"
    exit 1
fi

if ! command -v git &>/dev/null; then
    error "git not found. Install with: sudo apt install git"
fi

# ── 2. Virtual environment ────────────────────────────────────────────────────
step "Creating virtual environment (venv_stable)"

if [[ -d "$ROOT/venv_stable" ]]; then
    warn "venv_stable already exists — skipping creation."
else
    python3 -m venv "$ROOT/venv_stable"
    info "Virtual environment created."
fi

VENV_PY="$ROOT/venv_stable/bin/python"
VENV_PIP="$ROOT/venv_stable/bin/pip"

# ── 3. Python dependencies ────────────────────────────────────────────────────
step "Installing Python dependencies from requirements.txt"

"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r "$ROOT/requirements.txt"
info "Dependencies installed."

# ── 4. Clone openWakeWord ─────────────────────────────────────────────────────
step "Cloning openWakeWord @ $OWW_COMMIT"

if [[ -d "$ROOT/openWakeWord" ]]; then
    warn "openWakeWord/ already exists — skipping clone."
else
    git clone --depth 1 "$OWW_REPO" "$ROOT/openWakeWord"
    # Fetch and checkout the pinned commit so patches always hit the right lines
    (cd "$ROOT/openWakeWord" && git fetch --depth 1 origin "$OWW_COMMIT" && git checkout "$OWW_COMMIT")
    info "openWakeWord cloned and pinned to $OWW_COMMIT."
fi

# Install in editable mode so imports resolve without sys.path tricks
"$VENV_PIP" install -e "$ROOT/openWakeWord" --quiet
info "openWakeWord installed in editable mode."

# ── 5. WSL compatibility patches ──────────────────────────────────────────────
step "Applying WSL compatibility patches"

# ── Patch A: openWakeWord/openwakeword/data.py ────────────────────────────────
# torchaudio.load fails on WSL with torchaudio 2.11+; replace with soundfile.read.
DATA_PY="$ROOT/openWakeWord/openwakeword/data.py"

"$VENV_PY" - "$DATA_PY" <<'PYEOF'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

if 'import soundfile as sf' in text and 'sf.read(clip' in text:
    print("  data.py: already patched — skipping.")
    sys.exit(0)

if 'import soundfile as sf' not in text:
    text = text.replace('import torch\n', 'import torch\nimport soundfile as sf\n', 1)

old = (
    '            clip_data, clip_sr = torchaudio.load(clip)\n'
    '            clip_data = clip_data[0]'
)
new = (
    '            _data, clip_sr = sf.read(clip, dtype="float32", always_2d=True)\n'
    '            clip_data = torch.from_numpy(_data.T[0].copy())'
)
if old not in text:
    print("  data.py: target block not found — patch may already be applied or version mismatch.")
    sys.exit(0)

text = text.replace(old, new, 1)
path.write_text(text)
print("  data.py: patched OK.")
PYEOF

# ── Patch B: openWakeWord/openwakeword/train.py ───────────────────────────────
# os.scandir only reads one level — misses nested RIR dirs; replace with rglob.
TRAIN_PY="$ROOT/openWakeWord/openwakeword/train.py"

"$VENV_PY" - "$TRAIN_PY" <<'PYEOF'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

if 'rglob("*.wav")' in text:
    print("  train.py: already patched — skipping.")
    sys.exit(0)

old = '    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]'
new = (
    '    # Patched: recurse into RIR dataset which is nested (e.g. largeroom/RoomNNN/*.wav)\n'
    '    from pathlib import Path as _Path\n'
    '    rir_paths = [str(p) for j in config["rir_paths"] for p in _Path(j).rglob("*.wav")]'
)
if old not in text:
    print("  train.py: target line not found — patch may already be applied or version mismatch.")
    sys.exit(0)

text = text.replace(old, new, 1)
path.write_text(text)
print("  train.py: patched OK.")
PYEOF

# ── Patch C: torch_audiomentations/utils/io.py ───────────────────────────────
# torchaudio.info() was removed in torchaudio 2.11; replace with soundfile.info.
IO_PY=$(find "$ROOT/venv_stable" -path "*/torch_audiomentations/utils/io.py" 2>/dev/null | head -1)

if [[ -z "$IO_PY" ]]; then
    warn "torch_audiomentations io.py not found — skipping io.py patch."
else
    "$VENV_PY" - "$IO_PY" <<'PYEOF'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
text = path.read_text()

if 'sf.info(' in text:
    print("  io.py: already patched — skipping.")
    sys.exit(0)

if 'import soundfile as sf' not in text:
    text = text.replace('import torchaudio\n', 'import torchaudio\nimport soundfile as sf\n', 1)

old = (
    '        info = torchaudio.info(str(file_path))\n'
    '        # Deal with backwards-incompatible signature change.\n'
    '        # See https://github.com/pytorch/audio/issues/903 for more information.\n'
    '        if type(info) is tuple:\n'
    '            si, ei = info\n'
    '            num_samples = si.length\n'
    '            sample_rate = si.rate\n'
    '        else:\n'
    '            num_samples = info.num_frames\n'
    '            sample_rate = info.sample_rate\n'
    '        return num_samples, sample_rate'
)
new = (
    '        # Patched: torchaudio.info removed in torchaudio 2.11; using soundfile instead.\n'
    '        info = sf.info(str(file_path))\n'
    '        num_samples = info.frames\n'
    '        sample_rate = info.samplerate\n'
    '        return num_samples, sample_rate'
)
if old not in text:
    print("  io.py: target block not found — patch may already be applied or version mismatch.")
    sys.exit(0)

text = text.replace(old, new, 1)
path.write_text(text)
print("  io.py: patched OK.")
PYEOF
fi

# ── 6. Directory structure ────────────────────────────────────────────────────
step "Creating data directory structure"

mkdir -p "$ROOT/data/negatives"
mkdir -p "$ROOT/data/rir"
mkdir -p "$ROOT/data/dropzone"
mkdir -p "$ROOT/models"
info "Directories ready."

# ── 7. Silence seed file ──────────────────────────────────────────────────────
step "Generating silence.wav seed file"

SILENCE="$ROOT/data/negatives/silence.wav"
if [[ -f "$SILENCE" ]]; then
    warn "silence.wav already exists — skipping."
else
    ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono \
        -t 1 -acodec pcm_s16le -ar 16000 -ac 1 "$SILENCE" -loglevel quiet
    info "silence.wav generated at data/negatives/silence.wav"
fi

# ── 8. Training data download ─────────────────────────────────────────────────
step "Downloading openWakeWord training data"

echo ""
echo "  The training data is large (~17 GB ACAV100M features)."
echo "  Downloads will begin now. You can Ctrl-C and resume later — wget resumes."
echo ""

ACAV="$ROOT/data/negatives/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if [[ -f "$ACAV" ]]; then
    warn "ACAV100M feature array already present — skipping."
else
    info "Downloading ACAV100M features (~17 GB) ..."
    wget --continue --show-progress -O "$ACAV" "$ACAV_URL" || {
        warn "Download failed. Retry with:"
        echo "  wget --continue -O data/negatives/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \\"
        echo "    '$ACAV_URL'"
    }
fi

VAL="$ROOT/data/negatives/validation_set_features.npy"
if [[ -f "$VAL" ]]; then
    warn "Validation feature array already present — skipping."
else
    info "Downloading validation features ..."
    wget --continue --show-progress -O "$VAL" "$VAL_URL" || {
        warn "Download failed. Retry with:"
        echo "  wget --continue -O data/negatives/validation_set_features.npy \\"
        echo "    '$VAL_URL'"
    }
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}══ Setup complete! ══${NC}"
echo ""
echo "  Start the web UI:  bash start_webapp.sh"
echo "  Open in browser:   http://localhost:5000"
echo ""
echo "  If data/rir/ is empty, populate it manually — see README.md."
echo ""
