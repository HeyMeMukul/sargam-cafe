#!/usr/bin/env python3
"""Kaggle batch stage runner: clips -> scores with strict eval gate.

Per-clip stages:
  1. validate   (ffprobe duration + file sanity)
  2. transcribe (STARS | ROSVOT on GPU kernels;
                 in --dry-run: load an existing melody JSON, no GPU)
  3. arrange    (pop2piano | rule;
                 in --dry-run: local backend/pianist_v2 arranger)
  4. perform    (virtuosonet | phrase_arch;
                 in --dry-run: local backend/pianist_v2 performance)
  5. gate       (structural contract check; strict_sequence scoring when a
                 reference is supplied via --reference-dir)

GPU modes need vendored checkpoints (see kaggle/README.md and
docs/PIANIST_V2_ROADMAP.md M5). --dry-run needs no GPU and no
KAGGLE_API_TOKEN: it exercises the identical local
ensemble -> arranger -> performance code path on existing melody JSONs.

Fails closed: any stage error marks the clip `failed` instead of emitting a
silent fallback score.

Secrets: KAGGLE_API_TOKEN is read from the environment ONLY to decide whether
a GPU upload/session may start. Its value is NEVER written to disk: the
manifest records only a boolean `kaggle_token_present`, and a post-write
assertion aborts if the token value ever appears in serialized output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".opus", ".wma"}

STAGE_ORDER = ["validate", "transcribe", "arrange", "perform", "gate"]


def sha256_file(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(limit)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe. Raises on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out on {path.name}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {proc.stderr.strip()[:200]}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned non-numeric duration for {path.name}") from exc


def discover_clips(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    clips = [p for p in sorted(input_path.iterdir())
             if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return clips


def find_melody_json(clip: Path, melody_dir: Path | None) -> Path | None:
    """Lookup order: --melody-dir/<stem>.json, sibling <clip>.json, sibling <stem>.json."""
    candidates: list[Path] = []
    if melody_dir is not None:
        candidates.append(melody_dir / f"{clip.stem}.json")
    candidates.append(clip.parent / f"{clip.name}.json")
    candidates.append(clip.parent / f"{clip.stem}.json")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_melody_json(path: Path) -> tuple[list[dict], float, list[dict]]:
    """Return (melody, tempo, chords). Accepts {melody,...} dicts or bare lists."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tempo = 120.0
    chords: list[dict] = []
    if isinstance(data, dict):
        melody = data.get("melody", [])
        if isinstance(melody, dict):
            melody = melody.get("melody", [])
        try:
            tempo = float(data.get("tempo", 120.0) or 120.0)
        except (TypeError, ValueError):
            tempo = 120.0
        raw_chords = data.get("chords") or []
        if isinstance(raw_chords, list):
            chords = [c for c in raw_chords if isinstance(c, dict)]
        melody_list = melody if isinstance(melody, list) else []
    elif isinstance(data, list):
        melody_list = data
    else:
        raise ValueError(f"unsupported melody JSON top-level type in {path.name}")
    melody = [dict(e) for e in melody_list if isinstance(e, dict)]
    return melody, tempo, chords


def run_clip(clip: Path, *, args, out_dir: Path) -> dict:
    """Run all stages for one clip. Never raises: failures are recorded."""
    from pianist_v2.ensemble import merge_rosvot_crepe
    from pianist_v2.arranger import arrange_for_piano
    from pianist_v2.performance import apply_performance
    from pianist_v2.contracts import validate_score

    stem = clip.stem
    record: dict = {
        "clip": clip.name,
        "clip_sha256": sha256_file(clip),
        "stages": {},
        "status": "ok",
    }

    def fail(stage: str, reason: str) -> dict:
        record["stages"][stage] = {"ok": False, "reason": reason}
        record["status"] = "failed"
        record["fail_stage"] = stage
        record["fail_reason"] = reason
        return record

    # ---- 1. validate ----
    try:
        size = clip.stat().st_size
        if size == 0:
            return fail("validate", "empty file (0 bytes)")
        duration = ffprobe_duration(clip)
        if not duration > 0:
            return fail("validate", f"non-positive duration: {duration}")
        if args.min_duration is not None and duration < args.min_duration:
            return fail("validate", f"duration {duration:.2f}s below --min-duration")
        if args.max_duration is not None and duration > args.max_duration:
            return fail("validate", f"duration {duration:.2f}s above --max-duration")
    except RuntimeError as exc:
        return fail("validate", str(exc))
    record["duration"] = round(duration, 3)
    record["stages"]["validate"] = {"ok": True, "duration": round(duration, 3)}

    # ---- 2. transcribe ----
    if args.dry_run:
        melody_path = find_melody_json(clip, args.melody_dir)
        if melody_path is None:
            return fail(
                "transcribe",
                "no_melody_json: pass --melody-dir with <clip_stem>.json "
                "(or place a sibling JSON next to the clip)",
            )
        try:
            melody, tempo, chords = load_melody_json(melody_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return fail("transcribe", f"bad melody JSON {melody_path.name}: {exc}")
        if not melody:
            return fail("transcribe", f"melody JSON {melody_path.name} has zero events")
        record["melody_source"] = str(melody_path)
        record["stages"]["transcribe"] = {
            "ok": True,
            "engine": "local_melody_json+ensemble",
            "requested": args.transcriber,
            "melody_events": len(melody),
            "tempo": tempo,
        }
    else:
        # GPU path: model imports happen here so --dry-run never needs them.
        try:
            if args.transcriber == "stars":
                from transformers import AutoProcessor  # noqa: F401  (STARS kernel dep)
                raise RuntimeError("STARS checkpoint not vendored in this environment")
            else:
                raise RuntimeError("ROSVOT checkpoint not vendored in this environment")
        except (ImportError, RuntimeError) as exc:
            return fail("transcribe", f"{args.transcriber} unavailable: {exc}")
        melody, tempo, chords = [], 120.0, []  # unreachable; kept for type clarity

    # ---- ensemble (local, both modes once GPU melody exists) ----
    try:
        merged = merge_rosvot_crepe(melody, None)
    except Exception as exc:  # fail closed, never silent
        return fail("transcribe", f"ensemble crashed: {type(exc).__name__}: {exc}")
    record["stages"]["transcribe"]["ensemble_events"] = len(merged)

    # ---- 3. arrange ----
    try:
        if not args.dry_run and args.arranger == "pop2piano":
            raise RuntimeError("pop2piano checkpoint not vendored in this environment")
        arranged = arrange_for_piano(merged, chords, seed=args.seed)
        arrange_engine = "local_arranger_v2" if (args.dry_run or args.arranger == "rule") else args.arranger
    except (RuntimeError, ValueError) as exc:
        return fail("arrange", str(exc))
    record["stages"]["arrange"] = {
        "ok": True, "engine": arrange_engine,
        "requested": args.arranger, "events": len(arranged),
    }

    # ---- 4. perform ----
    try:
        if not args.dry_run and args.performer == "virtuosonet":
            raise RuntimeError("virtuosonet checkpoint not vendored in this environment")
        performed = apply_performance(arranged, tempo=tempo, seed=args.seed)
        perform_engine = (
            "local_phrase_arch" if (args.dry_run or args.performer == "phrase_arch")
            else args.performer
        )
    except (RuntimeError, ValueError) as exc:
        return fail("perform", str(exc))
    record["stages"]["perform"] = {
        "ok": True, "engine": perform_engine,
        "requested": args.performer, "events": len(performed),
    }

    # ---- 5. gate (structural contract; strict_sequence when reference exists) ----
    problems = validate_score([e for e in performed if e.get("midi") is not None])
    if problems:
        return fail("gate", f"contract violations: {problems[:5]}")
    gate_info: dict = {"ok": True, "contract": "valid", "events": len(performed)}
    if args.reference_dir is not None:
        ref_path = args.reference_dir / f"{stem}.json"
        if ref_path.is_file():
            try:
                import strict_sequence as ss

                ref = ss.load_notes(str(ref_path))
                # Score the performed melody-role events directly.
                est_events = [e for e in performed if e.get("role", "melody") == "melody"]
                est_notes = [ss.Note(float(e["start"]), float(e["end"]), int(e["midi"]))
                             for e in est_events
                             if e.get("midi") is not None and float(e["end"]) > float(e["start"])]
                result = ss.evaluate(ref, sorted(est_notes, key=lambda n: (n.start, n.midi)))
                gate_info.update({
                    "reference": ref_path.name,
                    "f1": round(result["f1"], 4),
                    "recall": round(result["recall"], 4),
                    "precision": round(result["precision"], 4),
                    "missing": result["missing_count"],
                    "extra": result["extra_count"],
                })
            except Exception as exc:
                return fail("gate", f"strict_sequence scoring crashed: {exc}")
        else:
            gate_info["reference"] = None
            gate_info["note"] = "no reference JSON; structural contract only"
    record["stages"]["gate"] = gate_info

    # ---- write per-clip outputs ----
    score_doc = {
        "clip": clip.name,
        "duration": round(duration, 3),
        "tempo": tempo,
        "engine": {
            "transcriber_requested": args.transcriber,
            "arranger_requested": args.arranger,
            "performer_requested": args.performer,
            "dry_run": bool(args.dry_run),
        },
        "events": performed,
    }
    (out_dir / f"{stem}.piano.json").write_text(
        json.dumps(score_doc, indent=2), encoding="utf-8")
    record["score_file"] = f"{stem}.piano.json"
    record["score_events"] = len(performed)
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Kaggle batch stage runner (clips -> piano scores + strict gate).")
    ap.add_argument("--input", required=True, help="Clip file or directory of clips")
    ap.add_argument("--out", required=True, help="Output directory for scores + manifest")
    ap.add_argument("--transcriber", default="stars", choices=["stars", "rosvot"])
    ap.add_argument("--arranger", default="pop2piano", choices=["pop2piano", "rule"])
    ap.add_argument("--performer", default="virtuosonet",
                    choices=["virtuosonet", "phrase_arch"])
    ap.add_argument("--dry-run", action="store_true",
                    help="No GPU/token: validate with ffprobe and run local "
                         "pianist_v2 ensemble+arranger+performance on existing melody JSONs.")
    ap.add_argument("--melody-dir", default=None,
                    help="Directory of <clip_stem>.json melody files (dry-run transcribe source)")
    ap.add_argument("--reference-dir", default=None,
                    help="Directory of <clip_stem>.json adjudicated references for strict scoring")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--max-duration", type=float, default=1200.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not args.dry_run and not token:
        print("KAGGLE_API_TOKEN not set in environment (see kaggle/README.md). "
              "Use --dry-run for the no-GPU local path.", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.melody_dir is not None:
        args.melody_dir = Path(args.melody_dir)
    if args.reference_dir is not None:
        args.reference_dir = Path(args.reference_dir)

    clips = discover_clips(input_path)
    if args.max_clips is not None:
        clips = clips[: args.max_clips]
    if not clips:
        print(f"no audio clips found under {input_path}", file=sys.stderr)
        return 2

    t0 = time.time()
    records = [run_clip(c, args=args, out_dir=out_dir) for c in clips]
    ok = sum(1 for r in records if r["status"] == "ok")
    manifest = {
        "mode": "dry_run" if args.dry_run else "kaggle_gpu",
        "stage_order": STAGE_ORDER,
        "requested": {
            "transcriber": args.transcriber,
            "arranger": args.arranger,
            "performer": args.performer,
        },
        "input": str(input_path.resolve()),
        "seed": args.seed,
        # Presence only. The token VALUE is never written to disk.
        "kaggle_token_present": bool(token),
        "repo_commit": _git_commit(),
        "elapsed_s": round(time.time() - t0, 2),
        "summary": {"clips": len(records), "ok": ok, "failed": len(records) - ok},
        "clips": records,
    }
    manifest_text = json.dumps(manifest, indent=2)
    if token and token in manifest_text:
        # Secret-leak tripwire: fail closed instead of writing the token to disk.
        print("refusing to write manifest: secret value detected in output", file=sys.stderr)
        return 1
    (out_dir / "batch_manifest.json").write_text(manifest_text, encoding="utf-8")
    for r in records:
        detail = r.get("fail_reason", f"{r.get('score_events', 0)} events -> {r.get('score_file')}")
        print(f"[{r['status']}] {r['clip']} ({r.get('duration', '?')}s): {detail}")
    print(f"manifest -> {out_dir}/batch_manifest.json ({ok}/{len(records)} ok)")
    return 0 if ok == len(records) else 1


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=str(REPO_ROOT))
        return proc.stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
