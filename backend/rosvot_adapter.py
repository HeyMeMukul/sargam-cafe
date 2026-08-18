#!/usr/bin/env python3
"""Optional ROSVOT singing-note backend for Sargam Cafe.

This adapter is intentionally separate from vocal_melody.py. It lets the app
use a dedicated note-boundary/pitch model when ROSVOT and its checkpoints are
installed, while preserving the existing CREPE pipeline as a fallback.

Required environment variables when ROSVOT is not vendored in the project:
  SARGAM_ROSVOT_DIR=/absolute/path/to/ROSVOT
  SARGAM_ROSVOT_CKPT_DIR=/absolute/path/to/ROSVOT/checkpoints  (optional)
  SARGAM_ROSVOT_THRESHOLD=0.85  (optional)
  SARGAM_ROSVOT_CACHE_DIR=/absolute/path/to/shared/stem/cache  (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROSVOT_DIR = Path(os.getenv("SARGAM_ROSVOT_DIR", str(ROOT / "third_party" / "ROSVOT"))).resolve()
CKPT_DIR = Path(os.getenv("SARGAM_ROSVOT_CKPT_DIR", str(ROSVOT_DIR / "checkpoints"))).resolve()
THRESHOLD = float(os.getenv("SARGAM_ROSVOT_THRESHOLD", "0.85"))


def _shared_stem(audio_path: str) -> Path:
    """Reuse the exact Demucs cache used by vocal_melody.py."""
    from vocal_melody import CACHE_DIR, separate_vocals

    cache_override = os.getenv("SARGAM_ROSVOT_CACHE_DIR")
    if cache_override:
        import vocal_melody
        vocal_melody.CACHE_DIR = cache_override
    return Path(separate_vocals(audio_path))


def _note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _tempo_and_beats(audio_path: str):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else None
    beats = librosa.frames_to_time(beat_frames, sr=sr).astype(float).tolist()
    return tempo_value, beats, float(len(y) / sr)


def _run_rosvot(stem: Path, output_dir: Path) -> tuple[Path, Path | None]:
    if not ROSVOT_DIR.exists():
        raise FileNotFoundError(
            f"ROSVOT directory not found: {ROSVOT_DIR}. Set SARGAM_ROSVOT_DIR or install it under third_party/ROSVOT."
        )
    model_ckpt = CKPT_DIR / "rosvot" / "model.pt"
    wbd_ckpt = CKPT_DIR / "rwbd" / "model.pt"
    if not model_ckpt.exists() or not wbd_ckpt.exists():
        raise FileNotFoundError(
            f"ROSVOT checkpoints missing under {CKPT_DIR}; expected rosvot/model.pt and rwbd/model.pt."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROSVOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # ROSVOT uses its own model-device logic. Keep CUDA visible when available;
    # on CPU-only hosts the local ROSVOT CPU-safe patch is required.
    cmd = [
        sys.executable, "-m", "inference.rosvot",
        "-o", str(output_dir), "-p", str(stem),
        "--ckpt", str(model_ckpt), "--wbd_ckpt", str(wbd_ckpt),
        "--thr", str(THRESHOLD), "--ds_workers", "0",
        # Keep ROSVOT's MIDI artifact: it contains the real note intervals.
        # The [note]output.npy file stores only durations, so reconstructing
        # time by summing durations destroys rests and phrase timing.
        "--no_save_final_npy",
    ]
    proc = subprocess.run(cmd, cwd=str(ROSVOT_DIR), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ROSVOT failed ({proc.returncode}): {proc.stderr[-2000:]}")
    npy = output_dir / "npy" / "[note]output.npy"
    if not npy.exists():
        raise RuntimeError(f"ROSVOT completed without output: {proc.stdout[-2000:]}")
    midi = output_dir / "midi" / "output.mid"
    return npy, (midi if midi.exists() else None)


def _read_midi_events(midi_path: Path) -> list[dict]:
    """Read ROSVOT's saved note intervals without losing rests or offsets."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            if note.end <= note.start or note.pitch <= 0:
                continue
            events.append({
                "start": round(float(note.start), 4),
                "end": round(float(note.end), 4),
                "midi": int(note.pitch),
                "velocity": round(float(note.velocity) / 127.0, 3),
            })
    events.sort(key=lambda x: (x["start"], x["midi"], x["end"]))
    for previous, current in zip(events, events[1:]):
        same_pitch = previous["midi"] == current["midi"]
        gap = current["start"] - previous["end"]
        if same_pitch and 0 <= gap <= 0.03:
            current["retrigger"] = True
            current["articulation"] = "retrigger"
    return events


def extract(audio_path: str, root: str = "C", thaat: str = "Bilawal") -> dict:
    del root, thaat  # ROSVOT predicts absolute MIDI pitches; labels are applied upstream.
    stem = _shared_stem(audio_path)
    with tempfile.TemporaryDirectory(prefix="sargam_rosvot_") as td:
        npy_path, midi_path = _run_rosvot(stem, Path(td))
        result = np.load(npy_path, allow_pickle=True).item()
        # Read the MIDI while the temporary ROSVOT output directory still exists.
        midi_events = _read_midi_events(midi_path) if midi_path else []

    pitches = result.get("pitches", [])
    durations = result.get("note_durs", [])
    melody = []
    if midi_events:
        # MIDI preserves ROSVOT's true note intervals and silent gaps.
        for event in midi_events:
            midi = event["midi"]
            note = _note_name(midi)
            melody.append({
                "start": event["start"],
                "end": event["end"],
                "note": note,
                "pitch_class": note[:-1],
                "octave": midi // 12 - 1,
                "midi": midi,
                "velocity": event["velocity"],
                "pitch_confidence": 1.0,
                "voicing_confidence": 1.0,
                "source_model": "rosvot",
                "timing_source": "rosvot_midi",
                "retrigger": event.get("retrigger"),
                "articulation": event.get("articulation"),
                "render_role": "attack",
            })
    else:
        # Compatibility fallback for installations that cannot write/read MIDI.
        # This is explicitly marked because duration-only output cannot preserve
        # ROSVOT rests or absolute onsets.
        cursor = 0.0
        for raw_midi, raw_dur in zip(pitches, durations):
            midi = int(raw_midi)
            dur = float(raw_dur)
            if midi <= 0 or dur <= 0:
                cursor += max(0.0, dur)
                continue
            note = _note_name(midi)
            melody.append({
                "start": round(cursor, 4),
                "end": round(cursor + dur, 4),
                "note": note,
                "pitch_class": note[:-1],
                "octave": midi // 12 - 1,
                "midi": midi,
                "velocity": 0.7,
                "pitch_confidence": 1.0,
                "voicing_confidence": 1.0,
                "source_model": "rosvot",
                "timing_source": "duration_fallback",
                "render_role": "attack",
            })
            cursor += dur

    tempo, beats, duration = _tempo_and_beats(audio_path)
    return {
        "root": None,
        "thaat": None,
        "duration": round(duration, 3),
        "tempo": round(tempo, 2) if tempo else None,
        "beats": beats,
        "melody": melody,
        "transcriber": "rosvot",
        "model_threshold": THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_filepath")
    parser.add_argument("root_note")
    parser.add_argument("--thaat", default="Bilawal")
    args = parser.parse_args()
    try:
        print(json.dumps(extract(args.audio_filepath, args.root_note, args.thaat)))
    except Exception as exc:
        print(json.dumps({"error": f"ROSVOT extraction failed: {exc}"}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
