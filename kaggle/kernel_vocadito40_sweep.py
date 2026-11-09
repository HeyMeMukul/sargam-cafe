"""SargamKing Vocadito 40-clip sweep (v3 decoder) on Kaggle GPU.

Dataset: cyb3rfy/vocadito-full (vocadito-full.zip with Audio/, Annotations/).
Pipeline per clip: repo backend/vocal_melody.py (C/Bilawal, MIDI absolute)
  vs A1 notes -> strict_note_metrics (musical_score). Fail-closed per clip.
"""
import csv
import glob
import json
import math
import os
import subprocess
import sys
import zipfile

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

# --- locate + unzip dataset ---
zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
print("zips:", zips, flush=True)
DATA = os.path.join(WORK, "vocadito")
if not os.path.exists(os.path.join(DATA, "Audio")):
    os.makedirs(DATA, exist_ok=True)
    for z in zips:
        print("unzipping", z, flush=True)
        with zipfile.ZipFile(z) as zh:
            zh.extractall(DATA)
    # flatten one level if zip contained tmp/opencode/vocds prefix
    import pathlib
    auds = list(pathlib.Path(DATA).rglob("Audio"))
    if auds:
        print("audio at:", auds[0], flush=True)

import pathlib
aud_dirs = list(pathlib.Path(WORK).rglob("Audio"))
AUD = str(aud_dirs[0]) if aud_dirs else None
print("AUD:", AUD, flush=True)

# --- clone repo for v3 decoder + metrics ---
REPO = os.path.join(WORK, "sargam-cafe")
if not os.path.exists(os.path.join(REPO, "backend", "vocal_melody.py")):
    print("cloning repo...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/HeyMeMukul/sargam-cafe.git", REPO], check=True)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import strict_note_metrics as sm
from strict_sequence import load_notes
from pathlib import Path

print("installing deps...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "demucs==4.1.0", "torchcrepe==0.0.24", "librosa", "soundfile"],
               check=True)

import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)


def csv_to_ref(csv_path):
    ref = []
    with open(csv_path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            try:
                o, f, d = float(row[0]), float(row[1]), float(row[2])
            except (ValueError, IndexError):
                continue
            if f <= 0 or d <= 0:
                continue
            midi = int(round(69 + 12 * math.log2(f / 440.0)))
            ref.append({"start": round(o, 3), "end": round(o + d, 3), "midi": midi})
    return sorted(ref, key=lambda x: x["start"])


NOTES = str(list(pathlib.Path(WORK).rglob("Notes"))[0])
wavs = sorted(glob.glob(os.path.join(AUD, "*.wav")))
print(f"clips: {len(wavs)}", flush=True)
results = []
for wav in wavs:
    stem = os.path.splitext(os.path.basename(wav))[0]  # vocadito_<id>
    cid = stem.split("_")[-1]
    a1 = os.path.join(NOTES, f"vocadito_{cid}_notesA1.csv")
    est_p = os.path.join(WORK, f"{stem}.est.json")
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "backend", "vocal_melody.py"),
                            wav, "C", "--thaat", "Bilawal"],
                           capture_output=True, text=True, timeout=1200)
        est = json.loads(r.stdout)
        mel = est.get("melody", [])
        json.dump(mel, open(est_p, "w"))
        ref = csv_to_ref(a1)
        rpath = os.path.join(WORK, f"{stem}.ref.json")
        json.dump(ref, open(rpath, "w"))
        m = sm.evaluate(load_notes(Path(rpath)), load_notes(Path(est_p)))
        results.append({"clip": stem, "status": "ok", "ref": len(ref), "est": len(mel),
                        "musical": round(m["musical_score"], 4),
                        "opoF1": round(m["onset_pitch_offset_f1"], 4),
                        "P": round(m["onset_precision"], 4), "R": round(m["onset_recall"], 4),
                        "rep": round(m["repeated_note_accuracy"], 4)})
        print(f"[ok] {stem}: ref {len(ref)} est {len(mel)} musical {m['musical_score']:.3f}", flush=True)
    except Exception as e:
        results.append({"clip": stem, "status": "failed", "reason": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"[failed] {stem}: {e}", flush=True)

ok = [r for r in results if r["status"] == "ok"]
mean_mus = sum(r["musical"] for r in ok) / len(ok) if ok else 0
mean_opo = sum(r["opoF1"] for r in ok) / len(ok) if ok else 0
json.dump({"results": results, "ok": len(ok), "total": len(results),
           "mean_musical": round(mean_mus, 4), "mean_opoF1": round(mean_opo, 4)},
          open(os.path.join(WORK, "sweep40.json"), "w"), indent=2)
print(f"done {len(ok)}/{len(results)} mean_musical {mean_mus:.4f} mean_opoF1 {mean_opo:.4f}", flush=True)
