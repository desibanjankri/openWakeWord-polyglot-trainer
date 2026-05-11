"""
Wake-Word Factory — multi-persona web UI
Run:  bash ~/kitty_training/start_webapp.sh
Open: http://localhost:5000
"""

import datetime
import io
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from flask import (Flask, Response, jsonify, render_template,
                   request, send_file)

sys.path.insert(0, str(Path(__file__).parent))
from persona import (
    ROOT, MODELS_DIR, DROPZONE_BASE, VENV_PYTHON, TRAIN_PY,
    create_persona, scan_personas, inject_config, get_persona_state,
    load_language_profiles,
    _persona_model_dir, _persona_dropzone, _persona_neg_dropzone,
    _read_meta, _parse_metrics,
)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg"}

app = Flask(__name__)

_training_active: set[str] = set()
_global_lock = threading.Lock()

DROPZONE_BASE.mkdir(parents=True, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _list_dropzone(dz: Path) -> list[dict]:
    if not dz.exists():
        return []
    files = []
    for f in sorted(dz.iterdir()):
        if f.suffix.lower() in AUDIO_EXTS:
            st = f.stat()
            files.append({
                "name":  f.name,
                "size":  f"{st.st_size / 1024:.0f} KB",
                "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return files


def _preview_clips(persona_id: str, n: int = 3) -> list[str]:
    import random
    train_dir = _persona_model_dir(persona_id) / "positive_train"
    clips = sorted(f.name for f in train_dir.glob("*.wav")) if train_dir.exists() else []
    return random.sample(clips, min(n, len(clips)))


def _persona_model_history(persona_id: str) -> list[dict]:
    models = []
    latest_link   = MODELS_DIR / f"{persona_id}_latest.onnx"
    latest_target = latest_link.resolve().name if latest_link.is_symlink() else None

    orig = MODELS_DIR / f"{persona_id}_v1_original.onnx"
    if orig.exists():
        models.append({
            "name":    orig.name,
            "label":   "original baseline",
            "size":    f"{orig.stat().st_size / 1024:.0f} KB",
            "mtime":   datetime.datetime.fromtimestamp(orig.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "metrics": None,
            "latest":  False,
        })

    for onnx in sorted(MODELS_DIR.glob(f"{persona_id}_2*.onnx"), reverse=True):
        ts  = onnx.stem[len(persona_id) + 1:]
        log = ROOT / f"{persona_id}_training_run_{ts}.log"
        if not log.exists():
            log = ROOT / f"training_run_{ts}.log"
        raw = _parse_metrics(log) if log.exists() else None
        fmt = {k: f"{v:.3f}" for k, v in raw.items()} if raw else None
        models.append({
            "name":    onnx.name,
            "label":   ts.replace("_", " "),
            "size":    f"{onnx.stat().st_size / 1024:.0f} KB",
            "mtime":   datetime.datetime.fromtimestamp(onnx.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "metrics": fmt,
            "latest":  onnx.name == latest_target,
        })

    return models


def _full_state(persona_id: str) -> dict:
    state = get_persona_state(persona_id)
    if state.get("metrics"):
        state["metrics"] = {k: f"{v:.3f}" for k, v in state["metrics"].items()}
    state["dropzone_files"]     = _list_dropzone(_persona_dropzone(persona_id))
    state["neg_dropzone_files"] = _list_dropzone(_persona_neg_dropzone(persona_id))
    state["training_busy"]      = persona_id in _training_active
    state["models"]             = _persona_model_history(persona_id)
    state["display_name"]       = persona_id.replace("_", " ")
    return state


def _save_and_upload(persona_id: str, dz: Path, files) -> list[dict]:
    """Save uploaded files to dz, return metadata list for each saved file."""
    dz.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if f and Path(f.filename).suffix.lower() in AUDIO_EXTS:
            dest = dz / Path(f.filename).name
            f.save(str(dest))
            st = dest.stat()
            saved.append({
                "name":  dest.name,
                "size":  f"{st.st_size / 1024:.0f} KB",
                "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return saved


def _run_slicer(persona_id: str, mode: str, form: dict) -> dict:
    """
    Run slicer.py for the given persona and mode (positive | negative).
    Returns the JSON-serialisable result dict or raises ValueError on error.
    """
    persona_dir = _persona_model_dir(persona_id)
    if not persona_dir.exists():
        raise ValueError(f"Persona '{persona_id}' not found")

    dz = _persona_neg_dropzone(persona_id) if mode == "negative" else _persona_dropzone(persona_id)

    cmd = [
        str(VENV_PYTHON), str(ROOT / "slicer.py"),
        "--input_dropzone",     str(dz),
        "--output_persona_dir", str(persona_dir),
        "--mode",               mode,
        "--silence-thresh", form.get("silence_thresh", "-40"),
        "--min-silence",    form.get("min_silence",    "700"),
        "--min-clip",       form.get("min_clip",       "500"),
        "--max-clip",       form.get("max_clip",       "3000"),
        "--keep-silence",   form.get("keep_silence",   "200"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        raise ValueError(result.stdout + result.stderr)

    accepted = rejected_short = rejected_long = 0
    for line in result.stdout.splitlines():
        l = line.strip()
        if l.startswith("Accepted clips"):
            try: accepted = int(l.split(":")[1].strip())
            except ValueError: pass
        elif l.startswith("Rejected short"):
            try: rejected_short = int((l.split(")")[1].split(":")[1]).strip() if ")" in l else l.split(":")[1].strip())
            except (ValueError, IndexError): pass
        elif l.startswith("Rejected long"):
            try: rejected_long = int((l.split(")")[1].split(":")[1]).strip() if ")" in l else l.split(":")[1].strip())
            except (ValueError, IndexError): pass

    state = get_persona_state(persona_id)
    return {
        "accepted":       accepted,
        "rejected_short": rejected_short,
        "rejected_long":  rejected_long,
        "n_train":        state["neg_train_custom"] if mode == "negative" else state["train_clips"],
        "n_test":         state["test_clips"],
        "neg_train_custom": state["neg_train_custom"],
        "preview_clips":  _preview_clips(persona_id) if mode == "positive" else [],
    }


# ─── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    personas = [_full_state(pid) for pid in scan_personas()]
    profiles = load_language_profiles()
    return render_template("index.html", personas=personas,
                           language_profiles=profiles)


@app.route("/api/language_profiles")
def api_language_profiles():
    return jsonify(load_language_profiles())


@app.route("/api/personas", methods=["POST"])
def api_create_persona():
    data = request.get_json(silent=True) or {}
    wake_phrase = (data.get("wake_phrase") or request.form.get("wake_phrase", "")).strip()
    language    = (data.get("language")    or request.form.get("language",    "english")).strip()
    variations  = data.get("variations") or []

    if not wake_phrase:
        return jsonify({"error": "wake_phrase is required"}), 400

    result = create_persona(wake_phrase, language=language, variations=variations)
    if result["error"]:
        return jsonify({"error": result["error"]}), 400

    try:
        inject_config(result["persona_id"], result["target_phrases"])
    except FileNotFoundError as exc:
        pass   # template missing — training will re-inject

    return jsonify(_full_state(result["persona_id"])), 201


# ── positive upload / delete / slice ──────────────────────────────────────────

@app.route("/upload/<persona_id>", methods=["POST"])
def upload(persona_id: str):
    saved = _save_and_upload(persona_id, _persona_dropzone(persona_id),
                             request.files.getlist("files"))
    return jsonify({"ok": True, "saved": saved})


@app.route("/delete/<persona_id>", methods=["POST"])
def delete_file(persona_id: str):
    name   = request.form.get("filename", "")
    dz     = _persona_dropzone(persona_id)
    target = dz / Path(name).name
    if target.exists() and target.is_file() and target.parent.resolve() == dz.resolve():
        target.unlink()
    return jsonify({"ok": True})


@app.route("/slice/<persona_id>", methods=["POST"])
def slice_route(persona_id: str):
    with _global_lock:
        if persona_id in _training_active:
            return jsonify({"error": "Training in progress"}), 409
    try:
        result = _run_slicer(persona_id, "positive", request.form)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500


# ── negative upload / delete / slice ──────────────────────────────────────────

@app.route("/upload_neg/<persona_id>", methods=["POST"])
def upload_neg(persona_id: str):
    saved = _save_and_upload(persona_id, _persona_neg_dropzone(persona_id),
                             request.files.getlist("files"))
    return jsonify({"ok": True, "saved": saved})


@app.route("/delete_neg/<persona_id>", methods=["POST"])
def delete_neg_file(persona_id: str):
    name   = request.form.get("filename", "")
    dz     = _persona_neg_dropzone(persona_id)
    target = dz / Path(name).name
    if target.exists() and target.is_file() and target.parent.resolve() == dz.resolve():
        target.unlink()
    return jsonify({"ok": True})


@app.route("/slice_neg/<persona_id>", methods=["POST"])
def slice_neg_route(persona_id: str):
    with _global_lock:
        if persona_id in _training_active:
            return jsonify({"error": "Training in progress"}), 409
    try:
        result = _run_slicer(persona_id, "negative", request.form)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500


# ── training ──────────────────────────────────────────────────────────────────

@app.route("/train_stream/<persona_id>")
def train_stream(persona_id: str):
    with _global_lock:
        if persona_id in _training_active:
            def already():
                yield f"data: {json.dumps({'line': 'ERROR: training already running\n'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            return Response(already(), content_type="text/event-stream")
        _training_active.add(persona_id)

    # Always re-inject config from stored metadata so variations are current
    meta = _read_meta(persona_id)
    config_path = ROOT / f"{persona_id}_config.yaml"
    try:
        inject_config(persona_id, meta.get("target_phrases", [persona_id.replace("_", " ")]))
    except FileNotFoundError:
        pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path  = ROOT / f"{persona_id}_training_run_{timestamp}.log"

    def generate():
        try:
            cmd = [
                str(VENV_PYTHON), str(TRAIN_PY),
                "--training_config", str(config_path),
                "--augment_clips", "--train_model", "--overwrite",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(ROOT),
            )
            with open(log_path, "w") as log_f:
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
                    yield f"data: {json.dumps({'line': line})}\n\n"
            proc.wait()

            v1_onnx = MODELS_DIR / f"{persona_id}_v1.onnx"
            if v1_onnx.exists():
                orig = MODELS_DIR / f"{persona_id}_v1_original.onnx"
                if not orig.exists():
                    shutil.copy2(v1_onnx, orig)
                dated  = MODELS_DIR / f"{persona_id}_{timestamp}.onnx"
                latest = MODELS_DIR / f"{persona_id}_latest.onnx"
                shutil.copy2(v1_onnx, dated)
                if latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(dated.name)

            raw = _parse_metrics(log_path) if log_path.exists() else None
            fmt = {k: f"{v:.3f}" for k, v in raw.items()} if raw else None
            yield f"data: {json.dumps({'done': True, 'metrics': fmt})}\n\n"
        finally:
            with _global_lock:
                _training_active.discard(persona_id)

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── static helpers ────────────────────────────────────────────────────────────

@app.route("/audio/<persona_id>/<filename>")
def audio_clip(persona_id: str, filename: str):
    train_dir = _persona_model_dir(persona_id) / "positive_train"
    target    = train_dir / Path(filename).name
    if target.exists() and target.parent.resolve() == train_dir.resolve():
        return send_file(str(target), mimetype="audio/wav")
    return "Not found", 404


@app.route("/download/<persona_id>")
def download_model(persona_id: str):
    onnx      = MODELS_DIR / f"{persona_id}_v1.onnx"
    data_file = Path(str(onnx) + ".data")

    if not onnx.exists():
        return f"No trained model found for '{persona_id}'", 404

    if data_file.exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(onnx, onnx.name)
            zf.write(data_file, data_file.name)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip",
                         as_attachment=True,
                         download_name=f"{persona_id}_v1_model.zip")

    return send_file(str(onnx), mimetype="application/octet-stream",
                     as_attachment=True, download_name=onnx.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
