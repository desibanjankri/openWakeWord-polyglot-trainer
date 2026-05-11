#!/usr/bin/env python3
"""
Slice long-format wake-word recordings from a drop zone into training clips.

Wipes positive_train/ and positive_test/ at the start of each run, then
repopulates them from every audio file found in the drop zone.

Default paths (legacy single-persona mode):
  --input_dropzone    data/dropzone/
  --output_persona_dir  models/dj_catnip_v1/
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg")


def parse_args():
    p = argparse.ArgumentParser(
        description="Slice drop-zone recordings into wake-word training clips"
    )
    p.add_argument("--input_dropzone",     type=Path,
                   default=ROOT / "data" / "dropzone",
                   metavar="DIR",
                   help="Directory containing source recordings (default: data/dropzone/)")
    p.add_argument("--output_persona_dir", type=Path,
                   default=ROOT / "models" / "dj_catnip_v1",
                   metavar="DIR",
                   help="Persona model dir containing positive_train/ and positive_test/ "
                        "(default: models/dj_catnip_v1/)")
    p.add_argument("--mode", choices=["positive", "negative"], default="positive",
                   help="positive: wipe and populate positive_train/test (default). "
                        "negative: append adversarial clips into negative_train/test "
                        "without touching silence_*.wav files.")
    p.add_argument("--silence-thresh", type=int, default=-40,
                   metavar="DBFS",
                   help="Silence threshold in dBFS (default: -40)")
    p.add_argument("--min-silence",   type=int, default=700,
                   metavar="MS",
                   help="Minimum silence duration ms to split on (default: 700)")
    p.add_argument("--min-clip",      type=int, default=500,
                   metavar="MS",
                   help="Reject clips shorter than this ms (default: 500)")
    p.add_argument("--max-clip",      type=int, default=3000,
                   metavar="MS",
                   help="Reject clips longer than this ms (default: 3000)")
    p.add_argument("--keep-silence",  type=int, default=200,
                   metavar="MS",
                   help="ms of silence to keep at clip edges (default: 200)")
    p.add_argument("--split",         type=float, default=0.9,
                   metavar="FRAC",
                   help="Fraction of clips going to train (default: 0.9)")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print what would be generated without writing files")
    return p.parse_args()


def main():
    args = parse_args()

    DROPZONE = args.input_dropzone
    if args.mode == "negative":
        TRAIN_DIR = args.output_persona_dir / "negative_train"
        TEST_DIR  = args.output_persona_dir / "negative_test"
    else:
        TRAIN_DIR = args.output_persona_dir / "positive_train"
        TEST_DIR  = args.output_persona_dir / "positive_test"

    # Import here so the --help above works without pydub installed
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    if not DROPZONE.exists():
        print(f"Drop zone not found: {DROPZONE}")
        print("Create the directory or pass --input_dropzone <path>.")
        sys.exit(1)

    recordings = sorted(
        p for p in DROPZONE.iterdir()
        if p.suffix.lower() in AUDIO_EXTS
    )
    if not recordings:
        print(f"No audio files found in {DROPZONE}")
        print(f"Supported formats: {', '.join(AUDIO_EXTS)}")
        sys.exit(1)

    print(f"Found {len(recordings)} recording(s) in drop zone:")
    for r in recordings:
        print(f"  {r.name}")

    all_clips = []
    rejected_short = 0
    rejected_long  = 0

    for rec in recordings:
        print(f"\nProcessing: {rec.name} ...")
        audio = AudioSegment.from_file(str(rec))
        # Normalise to 16 kHz mono 16-bit before splitting
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        chunks = split_on_silence(
            audio,
            min_silence_len=args.min_silence,
            silence_thresh=args.silence_thresh,
            keep_silence=args.keep_silence,
        )
        print(f"  Raw chunks from silence split: {len(chunks)}")

        for chunk in chunks:
            dur = len(chunk)  # pydub len() returns ms
            if dur < args.min_clip:
                rejected_short += 1
            elif dur > args.max_clip:
                rejected_long += 1
            else:
                all_clips.append(chunk)

    print(f"\n{'─'*50}")
    print(f"Accepted clips : {len(all_clips)}")
    print(f"Rejected short (<{args.min_clip} ms)  : {rejected_short}")
    print(f"Rejected long  (>{args.max_clip} ms) : {rejected_long}")

    if args.dry_run:
        n_train = round(len(all_clips) * args.split)
        n_test  = len(all_clips) - n_train
        print(f"\n[dry-run] Would write {n_train} clips → {TRAIN_DIR}")
        print(f"[dry-run] Would write {n_test}  clips → {TEST_DIR}")
        print("[dry-run] No files written.")
        return

    if not all_clips:
        print("\nNo clips to write. Try loosening --silence-thresh or --min-clip.")
        sys.exit(1)

    # Wipe and recreate output directories.
    # In negative mode: only remove previously sliced neg_*.wav files;
    # silence_*.wav seeds must never be deleted.
    for d in (TRAIN_DIR, TEST_DIR):
        if args.mode == "negative":
            d.mkdir(parents=True, exist_ok=True)
            for stale in d.glob("neg_*.wav"):
                stale.unlink()
        else:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # Shuffle then split
    random.shuffle(all_clips)
    n_train     = round(len(all_clips) * args.split)
    train_clips = all_clips[:n_train]
    test_clips  = all_clips[n_train:]

    prefix = "neg_clip" if args.mode == "negative" else "clip"
    for i, clip in enumerate(train_clips, 1):
        clip.export(str(TRAIN_DIR / f"{prefix}_{i:04d}.wav"), format="wav")

    for i, clip in enumerate(test_clips, 1):
        clip.export(str(TEST_DIR / f"{prefix}_{i:04d}.wav"), format="wav")

    print(f"\nWrote {len(train_clips):3d} clips → {TRAIN_DIR}")
    print(f"Wrote {len(test_clips):3d}  clips → {TEST_DIR}")


if __name__ == "__main__":
    main()
