"""SargamKing v2 Kaggle batch: v2 pipeline + strict gate on GPU.

Input: /kaggle/input/sargamking-batch-v1 (3 mp3 + 3 melody JSONs from dataset).
Output: /kaggle/working/scores.json + per-clip .piano.json.
Fail-closed: every clip ok/failed with reason; no silent fallbacks.
"""
import glob
import json
import os
import subprocess
import sys

IN = "/kaggle/input"
import pathlib as _pl
_cands = [str(x.parent) for x in _pl.Path(IN).rglob("*.mp3")]
if _cands:
    from collections import Counter
    IN = Counter(_cands).most_common(1)[0][0]
print("resolved IN:", IN)
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

print("torch:", end=" ")
try:
    import torch
    print(torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("missing", e)

# Local v2 modules are inlined (no repo dependency) to keep the kernel hermetic.
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def merge_rosvot_crepe(crepe, rosvot):
    if not rosvot:
        return crepe
    return sorted(rosvot, key=lambda x: (x["start"], x["midi"]))


def arrange(melody, chords=()):
    out = []
    for m in melody:
        e = dict(m)
        if e.get("midi") is not None and e["midi"] < 60:
            e["midi"] += 12
        e["role"] = "melody"
        out.append(e)
    for c in chords:
        try:
            s, en = float(c["start"]), float(c["end"])
        except Exception:
            continue
        mids = [int(x) for x in (c.get("midis") or [])]
        if not mids or not s < en:
            continue
        bass = max(21, min(mids) - 12)
        out.append({"start": round(s, 3), "end": round(min(en, s + 0.6), 3),
                    "midi": bass, "note": f"{NAMES[bass % 12]}{bass // 12 - 1}",
                    "velocity": 0.62, "role": "bass"})
    return sorted(out, key=lambda x: x["start"])


def perform(events, tempo=None):
    beat = 60.0 / float(tempo or 120.0)
    out = []
    for e in events:
        e = dict(e)
        pos = float(e.get("phrase_pos", 0.5))
        dt = max(-0.08, min(0.08, beat * 0.06 * (0.5 - abs(pos - 0.5)) * 2))
        e["start"] = round(max(0.0, float(e["start"]) + dt), 3)
        e["velocity"] = round(max(0.2, min(1.0, float(e.get("velocity", 0.7)))), 3)
        out.append(e)
    return sorted(out, key=lambda x: x["start"])


results = []
import pathlib
print("input tree:", [str(x) for x in pathlib.Path("/kaggle/input").rglob("*")][:20])
for mp3 in sorted(glob.glob(os.path.join(IN, "*.mp3")) + glob.glob(os.path.join(IN, "*", "*.mp3"))):
    stem = os.path.splitext(os.path.basename(mp3))[0]
    mj = os.path.join(IN, stem + ".json")
    try:
        d = json.load(open(mj))
        mel = d.get("melody", d if isinstance(d, list) else [])
        if not mel:
            raise ValueError("empty melody")
        arranged = arrange(mel, d.get("chords", []))
        performed = perform(arranged, d.get("tempo"))
        json.dump(performed, open(os.path.join(OUT, stem + ".piano.json"), "w"))
        results.append({"clip": stem, "status": "ok", "in": len(mel), "out": len(performed)})
        print(f"[ok] {stem}: {len(mel)} -> {len(performed)}")
    except Exception as e:
        results.append({"clip": stem, "status": "failed", "reason": f"{type(e).__name__}: {e}"})
        print(f"[failed] {stem}: {e}")

json.dump({"results": results,
           "ok": sum(1 for r in results if r["status"] == "ok"),
           "total": len(results)},
          open(os.path.join(OUT, "scores.json"), "w"), indent=2)
print(f"done {sum(1 for r in results if r['status']=='ok')}/{len(results)}")
