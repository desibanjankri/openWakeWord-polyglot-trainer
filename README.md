# openWakeWord WSL Factory

A local, polyglot, UI-driven openWakeWord model trainer for Home Assistant. Runs entirely on WSL/Ubuntu without cloud GPUs.

Train custom wake words in any language directly from your browser: record phrases, slice them into clips, train an ONNX model, and drop it straight into Home Assistant.

---

## Features

- Multi-persona: train multiple wake words simultaneously, each with their own model
- Polyglot: language profiles control variation generation and adversarial negative pipelines
- Drop-zone UI: drag-and-drop audio files, configure slicer params, train — no CLI needed
- Live training output streamed to the browser via SSE
- Model version history with metrics (FPR/FNR) per run
- One-click ONNX download, ready for Home Assistant

---

## Prerequisites

- **WSL2** with Ubuntu 22.04 or 24.04 (tested on Ubuntu 22.04)
- **Python 3.12** — `sudo apt install python3.12 python3.12-venv python3.12-dev`
- **ffmpeg** — `sudo apt install ffmpeg`
- **git** — `sudo apt install git`
- ~20 GB of free disk space for training data

---

> [!IMPORTANT]
> **Windows Users:** These commands must be run inside your **WSL/Ubuntu terminal**, NOT PowerShell or CMD.

## Installation

```bash
git clone https://github.com/desibanjankri/openWakeWord-polyglot-trainer.git
cd openWakeWord-polyglot-trainer
bash install.sh
```

`install.sh` will:
1. Create a `venv_stable/` virtual environment
2. Install all Python dependencies from `requirements.txt`
3. Clone `openWakeWord` from GitHub and apply three WSL compatibility patches
4. Generate the `silence.wav` seed file in `data/negatives/`
5. Download the ACAV100M feature arrays from HuggingFace (~17 GB — resumable)

> **Note on training data:** The ACAV100M negative features download is large.
> If you have them from a previous setup, copy the `.npy` files into `data/negatives/`
> before running `install.sh` and the download step will be skipped.

### RIR (Room Impulse Response) files

The RIR dataset is not auto-downloaded due to its structure. Populate `data/rir/` with the
[openWakeWord RIR dataset](https://github.com/dscripka/openWakeWord#training-data):

```bash
# Download from openWakeWord's documented source, then extract into:
data/rir/largeroom/
data/rir/mediumroom/
data/rir/smallroom/
```

The installer's train.py patch makes `rglob` recurse into these subdirectories automatically.

---

## Usage

### Start the web UI

```bash
bash start_webapp.sh
```

Open `http://localhost:5000` in your browser.

### Create a wake word persona

1. Click **+ New Wake Word**
2. Enter your wake phrase (e.g. `Hey Computer`)
3. Select a language profile
4. For languages with variations enabled (e.g. Punjabi), add phonetic/spelling variants
5. Click **Create**

### Record and slice positive clips

1. Record yourself saying the wake phrase many times (~50+ repetitions)
2. Drop the recording into the **Positive Audio** drop zone
3. Adjust silence detection sliders if needed
4. Click **Slice Positive**

### (Polyglot) Record and slice adversarial negatives

For languages that require custom negatives (e.g. Punjabi), a second **Adversarial Audio** drop zone appears:
1. Record words that sound similar but are NOT the wake phrase
2. Drop into the **Adversarial Audio** drop zone
3. Click **Slice Adversarial**

### Train

Click **Train Model** and watch the output stream in real time. Training takes 5–30 minutes on CPU depending on clip count and augmentation rounds.

### Export

Click **Download Model** to get the `.onnx` file (or `.zip` for split-weight models).
Drop it into your Home Assistant `custom_sentences/` wake-word directory.

---

## Project structure

```
.
├── install.sh              # Bootstrap script
├── start_webapp.sh         # Launch the web UI
├── requirements.txt        # Python dependencies
├── language_profiles.json  # Language configuration
├── template_config.yaml    # openWakeWord training config template
├── slicer.py               # Audio slicing pipeline (CLI + UI)
├── webapp/
│   ├── app.py              # Flask application
│   ├── persona.py          # Persona management logic
│   └── templates/
│       └── index.html      # Single-page UI
├── data/
│   ├── negatives/          # ACAV100M .npy arrays + silence.wav
│   ├── rir/                # Room impulse response .wav files
│   └── dropzone/           # Per-persona audio upload staging
└── models/                 # Trained ONNX models (gitignored)
```

---

## WSL patches applied by install.sh

| File | Problem | Fix |
|------|---------|-----|
| `openWakeWord/openwakeword/data.py` | `torchaudio.load` fails on WSL with torchaudio 2.11+ | Replaced with `soundfile.read` |
| `openWakeWord/openwakeword/train.py` | `os.scandir` only reads one level — misses nested RIR dirs | Replaced with `Path.rglob("*.wav")` |
| `torch_audiomentations/utils/io.py` | `torchaudio.info` removed in torchaudio 2.11 | Replaced with `soundfile.info` |

---

## Adding a new language

Edit `language_profiles.json`:

```json
"yoruba": {
    "display_name": "Yoruba",
    "supports_variations": true,
    "requires_custom_negatives": true,
    "hint": "Add tonal variants of the wake phrase."
}
```

The UI picks up the new profile on next page load — no server restart needed.

---

## License

MIT
