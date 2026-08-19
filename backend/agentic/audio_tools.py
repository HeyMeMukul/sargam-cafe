"""Specialist audio tools exposed to the pianist agent.

The tools call existing audio models but return narrow, queryable artifacts
instead of dumping an entire extraction into an LLM prompt. Heavy extraction is
cached by audio hash so repeated agent queries are cheap and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
CACHE_ROOT = Path(os.getenv("SARGAM_AGENTIC_EVIDENCE_DIR", "/tmp/sargam_agentic_evidence"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_last_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        values.append(value)
    if not values:
        raise RuntimeError("audio tool produced no JSON result")
    # The extractor payload contains nested note dictionaries. Prefer the
    # outer full melody object over the last nested object in stdout.
    melody_values = [
        value for value in values
        if isinstance(value, dict) and isinstance(value.get("melody"), list)
    ]
    if melody_values:
        return melody_values[-1]
    return values[-1]


def get_track_manifest(audio_path: str) -> dict[str, Any]:
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    import soundfile as sf
    info = sf.info(str(path))
    return {
        "artifact_id": f"track-{_sha256(path)[:16]}",
        "audio_path": str(path),
        "audio_sha256": _sha256(path),
        "duration": round(float(info.frames / info.samplerate), 4),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "evidence_cache_dir": str(CACHE_ROOT / _sha256(path)[:16]),
    }


def _cache_dir(audio_path: str) -> Path:
    directory = CACHE_ROOT / _sha256(audio_path)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_evidence(audio_path: str, root: str = "C", thaat: str = "Bilawal") -> tuple[dict[str, Any], Path]:
    directory = _cache_dir(audio_path)
    evidence_path = directory / "production.evidence.json"
    if evidence_path.exists():
        return json.loads(evidence_path.read_text(encoding="utf-8")), evidence_path
    command = [
        sys.executable,
        str(BACKEND / "vocal_melody.py"),
        str(audio_path),
        root,
        "--thaat",
        thaat,
        "--evidence-out",
        str(evidence_path),
    ]
    proc = subprocess.run(command, cwd=str(BACKEND), capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"evidence extraction failed: {proc.stderr[-2000:]}")
    if not evidence_path.exists():
        raise RuntimeError("extractor completed without evidence artifact")
    return json.loads(evidence_path.read_text(encoding="utf-8")), evidence_path


def get_note_candidates(
    audio_path: str,
    start: float,
    end: float,
    transcribers: list[str] | None = None,
    min_confidence: float = 0.0,
    root: str = "C",
    thaat: str = "Bilawal",
) -> dict[str, Any]:
    if end <= start:
        raise ValueError("end must be greater than start")
    evidence, evidence_path = ensure_evidence(audio_path, root, thaat)
    melody = evidence.get("melody", [])
    # Some evidence artifacts contain only frames; derive coarse candidates from
    # the production extractor once and cache them beside the evidence.
    candidate_path = evidence_path.with_name("production.melody.json")
    result: dict[str, Any] | None = None
    if candidate_path.exists():
        try:
            cached = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        # A pitch-class histogram is a valid extractor output for another mode,
        # but it is not a candidate artifact. Never expose it as an empty melody
        # because that forces the agent to debug cache state instead of audio.
        if isinstance(cached, dict) and isinstance(cached.get("melody"), list):
            result = cached
        else:
            candidate_path.unlink(missing_ok=True)
    if result is None:
        command = [sys.executable, str(BACKEND / "vocal_melody.py"), audio_path, root, "--thaat", thaat]
        proc = subprocess.run(command, cwd=str(BACKEND), capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(f"candidate extraction failed: {proc.stderr[-2000:]}")
        result = _parse_last_json(proc.stdout)
        if not isinstance(result, dict) or not isinstance(result.get("melody"), list):
            raise RuntimeError("candidate extraction returned no full melody artifact")
        candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    allowed = set(transcribers or [])
    events = []
    for event in result.get("melody", []):
        if event.get("end", 0) <= start or event.get("start", 0) >= end:
            continue
        source = str(event.get("source_model", event.get("transcriber", "unknown")))
        confidence = float(event.get("pitch_confidence") or event.get("voicing_confidence") or 0.0)
        if allowed and source not in allowed:
            continue
        if confidence < min_confidence:
            continue
        events.append(event)
    return {
        "artifact_id": f"candidates-{_sha256(audio_path)[:16]}-{start:.3f}-{end:.3f}",
        "audio_sha256": _sha256(audio_path),
        "evidence_path": str(evidence_path),
        "window": {"start": start, "end": end},
        "events": events,
        "available_frame_count": sum(start <= float(frame.get("t", -1)) < end for frame in evidence.get("frames", [])),
    }


def inspect_pitch_window(
    audio_path: str,
    start: float,
    end: float,
    trackers: list[str] | None = None,
    resolution: str = "coarse",
    root: str = "C",
    thaat: str = "Bilawal",
) -> dict[str, Any]:
    evidence, evidence_path = ensure_evidence(audio_path, root, thaat)
    frames = [frame for frame in evidence.get("frames", []) if start <= float(frame.get("t", -1)) < end]
    if resolution == "coarse" and len(frames) > 200:
        stride = max(1, len(frames) // 200)
        frames = frames[::stride]
    if trackers:
        allowed = set(trackers)
        frames = [{key: value for key, value in frame.items() if key in {"t", "voiced", "rms", "onset_strength", "brightness"} or any(name in key for name in allowed)} for frame in frames]
    return {
        "artifact_id": f"pitch-window-{_sha256(audio_path)[:16]}-{start:.3f}-{end:.3f}",
        "evidence_path": str(evidence_path),
        "window": {"start": start, "end": end},
        "frame_hop_seconds": evidence.get("frame_hop_seconds"),
        "frames": frames,
        "summary": {
            "frame_count": len(frames),
            "voiced_fraction": round(sum(bool(frame.get("voiced")) for frame in frames) / len(frames), 4) if frames else 0.0,
            "median_midi_crepe": round(float(np.median([frame["midi_crepe"] for frame in frames if frame.get("midi_crepe") is not None])), 4) if any(frame.get("midi_crepe") is not None for frame in frames) else None,
        },
    }


def inspect_boundary_window(
    audio_path: str,
    start: float,
    end: float,
    resolution: str = "coarse",
    root: str = "C",
    thaat: str = "Bilawal",
) -> dict[str, Any]:
    evidence, evidence_path = ensure_evidence(audio_path, root, thaat)
    frames = [frame for frame in evidence.get("frames", []) if start <= float(frame.get("t", -1)) < end]
    onsets = [float(t) for t in evidence.get("onsets", []) if start <= float(t) < end]
    ranked = sorted(
        [
            {
                "t": frame.get("t"),
                "onset_strength": frame.get("onset_strength", 0.0),
                "rms": frame.get("rms", 0.0),
                "voiced": frame.get("voiced", False),
                "midi_crepe": frame.get("midi_crepe"),
            }
            for frame in frames
        ],
        key=lambda item: float(item.get("onset_strength", 0.0)),
        reverse=True,
    )[:20]
    return {
        "artifact_id": f"boundary-window-{_sha256(audio_path)[:16]}-{start:.3f}-{end:.3f}",
        "evidence_path": str(evidence_path),
        "window": {"start": start, "end": end},
        "candidate_onsets": onsets,
        "strongest_local_frames": ranked,
    }


def audition_phrase(audio_path: str, events: list[dict[str, Any]], start: float, end: float, render_profile: str = "faithful") -> dict[str, Any]:
    """Write an exact MIDI audition artifact; audio synthesis is optional.

    The agent receives a truthful `audio_rendered` flag. The artifact is still
    useful for browser playback and for later comparison against the source.
    """
    directory = _cache_dir(audio_path) / "auditions"
    directory.mkdir(parents=True, exist_ok=True)
    audition_id = f"audition-{int(time.time() * 1000)}"
    json_path = directory / f"{audition_id}.json"
    selected = [event for event in events if float(event.get("end", 0)) > start and float(event.get("start", 0)) < end]
    payload = {
        "artifact_id": audition_id,
        "audio_sha256": _sha256(audio_path),
        "window": {"start": start, "end": end},
        "render_profile": render_profile,
        "events": selected,
        "audio_rendered": False,
        "note": "MIDI/browser audition adapter is the next renderer integration step.",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "artifact_path": str(json_path)}


def score_hypothesis(audio_path: str, events: list[dict[str, Any]], reference_path: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact_id": f"hypothesis-score-{_sha256(audio_path)[:16]}",
        "audio_sha256": _sha256(audio_path),
        "event_count": len(events),
        "reference_available": bool(reference_path),
        "evidence_consistency": {
            "events_with_evidence": sum(bool(event.get("evidence_refs")) for event in events),
            "events_with_pitch_confidence": sum(event.get("pitch_confidence") is not None for event in events),
            "events_with_boundary_confidence": sum(event.get("onset_confidence") is not None and event.get("offset_confidence") is not None for event in events),
        },
    }
    if reference_path:
        from evaluation.strict_sequence import load_notes, evaluate
        estimated_path = Path(tempfile.mkstemp(suffix=".json")[1])
        estimated_path.write_text(json.dumps(events), encoding="utf-8")
        result["strict_metrics"] = evaluate(load_notes(reference_path), load_notes(estimated_path))
        estimated_path.unlink(missing_ok=True)
    return result
