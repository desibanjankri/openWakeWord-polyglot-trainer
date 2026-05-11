"""
persona.py — backend logic for multi-persona wake-word management.

Responsibilities:
  - Parse a wake phrase into (persona_id, target_phrase)
  - Create the full directory structure for a new persona
  - Seed negative_train/ and negative_test/ from the canonical silence.wav
  - Scan models/ to auto-detect existing personas on startup
  - Inject template_config.yaml and write a per-persona <persona_id>_config.yaml
  - Store and retrieve per-persona metadata (language profile, target phrase list)
  - Return the current state (clip counts, last training, metrics) for any persona

Nothing here touches app.py, index.html, or slicer.py — those are Phase 2/3.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — all relative to this file so the project is portable
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent.parent          # kitty_training/
MODELS_DIR    = ROOT / "models"
DROPZONE_BASE = ROOT / "data" / "dropzone"
SILENCE_SRC   = ROOT / "data" / "negatives" / "silence.wav"
TEMPLATE_CFG  = ROOT / "template_config.yaml"
LANG_PROFILES = ROOT / "language_profiles.json"
TRAIN_PY      = ROOT / "openWakeWord" / "openwakeword" / "train.py"
VENV_PYTHON   = ROOT / "venv_stable" / "bin" / "python3"


# ---------------------------------------------------------------------------
# Language profiles
# ---------------------------------------------------------------------------

def load_language_profiles() -> dict:
    if LANG_PROFILES.exists():
        return json.loads(LANG_PROFILES.read_text())
    return {
        "english": {
            "display_name": "English",
            "supports_variations": False,
            "requires_custom_negatives": False,
            "description": "Standard single-phrase wake word.",
        }
    }


# ---------------------------------------------------------------------------
# Wake phrase parsing
# ---------------------------------------------------------------------------

def parse_wake_phrase(wake_phrase: str) -> tuple[str, str]:
    """
    Convert a user-supplied wake phrase into (persona_id, target_phrase).

    Input:   "Open Garage"
    Returns: ("open_garage", "open garage")
    """
    cleaned = wake_phrase.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        raise ValueError("Wake phrase must contain at least one alphanumeric character.")

    persona_id    = cleaned.replace(" ", "_")
    target_phrase = cleaned
    return persona_id, target_phrase


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _persona_model_dir(persona_id: str) -> Path:
    return MODELS_DIR / f"{persona_id}_v1"


def _persona_dropzone(persona_id: str) -> Path:
    """Positive-audio staging area."""
    return DROPZONE_BASE / persona_id


def _persona_neg_dropzone(persona_id: str) -> Path:
    """Adversarial-negative staging area."""
    return DROPZONE_BASE / f"{persona_id}_neg"


def _persona_meta_path(persona_id: str) -> Path:
    return _persona_model_dir(persona_id) / "meta.json"


# ---------------------------------------------------------------------------
# Persona metadata
# ---------------------------------------------------------------------------

def _read_meta(persona_id: str) -> dict:
    path = _persona_meta_path(persona_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    # Default: derive everything from persona_id
    return {
        "language": "english",
        "target_phrases": [persona_id.replace("_", " ")],
        "requires_custom_negatives": False,
    }


def _write_meta(persona_id: str, meta: dict) -> None:
    path = _persona_meta_path(persona_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Persona creation
# ---------------------------------------------------------------------------

def create_persona(wake_phrase: str,
                   language: str = "english",
                   variations: Optional[list[str]] = None) -> dict:
    """
    Initialise all on-disk resources for a new persona.

    Creates:
      models/<persona_id>_v1/positive_train/
      models/<persona_id>_v1/positive_test/
      models/<persona_id>_v1/negative_train/  (seeded with silence_1..5.wav)
      models/<persona_id>_v1/negative_test/   (seeded with silence_1..5.wav)
      models/<persona_id>_v1/meta.json
      data/dropzone/<persona_id>/
      data/dropzone/<persona_id>_neg/         (only when requires_custom_negatives)

    Returns a dict with keys:
      persona_id, target_phrase, target_phrases, dirs_created, error
    """
    try:
        persona_id, target_phrase = parse_wake_phrase(wake_phrase)
    except ValueError as exc:
        return {"persona_id": None, "target_phrase": None, "target_phrases": [],
                "dirs_created": [], "error": str(exc)}

    base = _persona_model_dir(persona_id)
    if base.exists():
        return {"persona_id": persona_id, "target_phrase": target_phrase,
                "target_phrases": [], "dirs_created": [],
                "error": f"Persona '{persona_id}' already exists."}

    if not SILENCE_SRC.exists():
        return {"persona_id": persona_id, "target_phrase": target_phrase,
                "target_phrases": [], "dirs_created": [],
                "error": f"Canonical silence clip not found at {SILENCE_SRC}"}

    # Build the full target phrases list
    target_phrases = [target_phrase]
    if variations:
        for v in variations:
            _, cleaned = parse_wake_phrase(v)
            if cleaned and cleaned not in target_phrases:
                target_phrases.append(cleaned)

    # Load language profile metadata
    profiles = load_language_profiles()
    profile  = profiles.get(language, profiles.get("english", {}))
    requires_neg = profile.get("requires_custom_negatives", False)

    dirs_created: list[str] = []

    # Create the four model subdirectories
    for subdir in ("positive_train", "positive_test", "negative_train", "negative_test"):
        d = base / subdir
        d.mkdir(parents=True, exist_ok=True)
        dirs_created.append(str(d))

    # Seed both negative dirs with 5 copies of the canonical silence clip
    for neg_dir in ("negative_train", "negative_test"):
        for i in range(1, 6):
            shutil.copy2(SILENCE_SRC, base / neg_dir / f"silence_{i}.wav")

    # Write persona metadata
    _write_meta(persona_id, {
        "language":                language,
        "target_phrases":          target_phrases,
        "requires_custom_negatives": requires_neg,
    })

    # Create positive staging drop zone
    dz = _persona_dropzone(persona_id)
    dz.mkdir(parents=True, exist_ok=True)
    dirs_created.append(str(dz))

    # Create negative staging drop zone if the profile requires it
    if requires_neg:
        neg_dz = _persona_neg_dropzone(persona_id)
        neg_dz.mkdir(parents=True, exist_ok=True)
        dirs_created.append(str(neg_dz))

    return {
        "persona_id":    persona_id,
        "target_phrase": target_phrase,
        "target_phrases": target_phrases,
        "dirs_created":  dirs_created,
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Persona discovery
# ---------------------------------------------------------------------------

def scan_personas() -> list[str]:
    """
    Scan models/ for directories named *_v1 and return their persona IDs.
    Returns an alphabetically sorted list.
    """
    if not MODELS_DIR.exists():
        return []

    personas = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.is_dir() and entry.name.endswith("_v1"):
            persona_id = entry.name[:-3]   # strip trailing "_v1"
            personas.append(persona_id)
    return personas


# ---------------------------------------------------------------------------
# Config injection
# ---------------------------------------------------------------------------

def inject_config(persona_id: str,
                  target_phrases) -> Path:
    """
    Read template_config.yaml, substitute all placeholders, write
    <persona_id>_config.yaml next to the template.

    target_phrases: str (single phrase) or list[str] (multi-phrase / polyglot)

    Substitutions:
      {{PERSONA_ID}}          -> persona_id
      {{TARGET_PHRASES_YAML}} -> YAML list block  (one '  - "phrase"' per line)
      {{ROOT}}                -> str(ROOT)
    """
    if not TEMPLATE_CFG.exists():
        raise FileNotFoundError(f"template_config.yaml not found at {TEMPLATE_CFG}")

    if isinstance(target_phrases, str):
        target_phrases = [target_phrases]

    phrases_yaml = "\n".join(f'  - "{p}"' for p in target_phrases)

    text = TEMPLATE_CFG.read_text()
    text = text.replace("{{PERSONA_ID}}",          persona_id)
    text = text.replace("{{TARGET_PHRASES_YAML}}", phrases_yaml)
    text = text.replace("{{ROOT}}",                str(ROOT))

    out_path = ROOT / f"{persona_id}_config.yaml"
    out_path.write_text(text)
    return out_path


# ---------------------------------------------------------------------------
# Persona state
# ---------------------------------------------------------------------------

def get_persona_state(persona_id: str) -> dict:
    """
    Return the current on-disk state for a persona.

    Keys:
      persona_id            str
      language              str
      target_phrases        list[str]
      requires_custom_negatives  bool
      train_clips           int
      test_clips            int
      neg_train_clips       int   — total clips in negative_train/
      neg_train_custom      int   — adversarial sliced clips (neg_*.wav)
      last_trained          str | None
      metrics               dict | None  — {recall, fp_hr, accuracy} floats
      onnx_path             str | None
    """
    base = _persona_model_dir(persona_id)
    meta = _read_meta(persona_id)

    def count_wavs(subdir: str) -> int:
        d = base / subdir
        return len(list(d.glob("*.wav"))) if d.exists() else 0

    def count_neg_custom(subdir: str) -> int:
        d = base / subdir
        return len(list(d.glob("neg_*.wav"))) if d.exists() else 0

    last_trained: Optional[str] = None
    metrics: Optional[dict]     = None

    log = _find_latest_log(persona_id)
    if log:
        last_trained = datetime.fromtimestamp(
            log.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
        metrics = _parse_metrics(log)

    onnx = MODELS_DIR / f"{persona_id}_v1.onnx"

    return {
        "persona_id":               persona_id,
        "language":                 meta.get("language", "english"),
        "target_phrases":           meta.get("target_phrases", [persona_id.replace("_", " ")]),
        "requires_custom_negatives": meta.get("requires_custom_negatives", False),
        "train_clips":              count_wavs("positive_train"),
        "test_clips":               count_wavs("positive_test"),
        "neg_train_clips":          count_wavs("negative_train"),
        "neg_train_custom":         count_neg_custom("negative_train"),
        "last_trained":             last_trained,
        "metrics":                  metrics,
        "onnx_path":                str(onnx) if onnx.exists() else None,
    }


def _find_latest_log(persona_id: str) -> Optional[Path]:
    """Return the most recent log that belongs to this persona, or None."""
    named = sorted(ROOT.glob(f"{persona_id}_training_run_*.log"), reverse=True)
    if named:
        return named[0]

    marker   = f"{persona_id}_v1"
    all_logs = sorted(ROOT.glob("training_run_*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    for log in all_logs:
        try:
            if marker in log.read_text():
                return log
        except OSError:
            continue
    return None


def _parse_metrics(log_path: Path) -> Optional[dict]:
    """Extract final Recall / FP-per-hour / Accuracy from a training log."""
    metrics: dict = {}
    try:
        with open(log_path) as f:
            for line in f:
                if "Final Model Recall:" in line:
                    try:
                        metrics["recall"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif "Final Model False Positives per Hour" in line:
                    try:
                        metrics["fp_hr"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif "Final Model Accuracy:" in line:
                    try:
                        metrics["accuracy"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
    except OSError:
        return None
    return metrics if metrics else None
