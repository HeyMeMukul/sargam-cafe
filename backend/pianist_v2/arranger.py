"""Rule-based piano arrangement: vocal line -> playable piano score.

A literal vocal copy sounds thin. A pianist plays:
- melody (vocal line, lifted into piano register when needed),
- bass (chord roots on strong beats, an octave below),
- inner harmony (quiet chord tones on off-beats, skipped when crowded),
- simple fills (short scale steps into long gaps, capped and quiet).

This is intentionally conservative: no invented melody, no scale-snapping.
Set SARGAM_ARRANGER=v2 to enable; default v1 path is untouched.
"""
from __future__ import annotations

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _name(midi: int) -> str:
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def arrange_for_piano(melody: list[dict], chords: list[dict], seed: int = 42) -> list[dict]:
    """Return arranged piano events with `role` in {melody,bass,harmony,fill}."""
    arranged: list[dict] = []
    # Melody first: ensure piano register (lift anything below C4 up one octave).
    for m in melody:
        e = dict(m)
        if e.get("midi") is not None and e["midi"] < 60:
            e["midi"] = int(e["midi"]) + 12
            e["note"] = _name(int(e["midi"]))
            e["octave"] = int(e["midi"]) // 12 - 1
            e["pitch_class"] = NAMES[int(e["midi"]) % 12]
        e["role"] = "melody"
        e["velocity"] = round(float(e.get("velocity", 0.7)), 3)
        arranged.append(e)

    # Rhythm scaffolding from chords: bass on chord start, harmony mid-chord.
    for c in chords or []:
        try:
            s, en = float(c["start"]), float(c["end"])
        except (TypeError, ValueError):
            continue
        midis = [int(x) for x in (c.get("midis") or []) if 21 <= int(x) <= 108]
        if not midis or not s < en:
            continue
        root = min(midis)
        bass = max(21, root - 12)
        # Bass: strong, short-ish (half the chord or 0.6s cap).
        arranged.append({
            "start": round(s, 3), "end": round(min(en, s + 0.6), 3),
            "midi": bass, "note": _name(bass),
            "octave": bass // 12 - 1, "pitch_class": NAMES[bass % 12],
            "velocity": 0.62, "role": "bass", "source_model": "arranger_v2",
        })
        # Inner harmony: skip if the melody already occupies this window densely.
        crowded = sum(1 for m in melody if m["start"] < (s + en) / 2 < m["end"]) >= 2
        if not crowded and en - s >= 1.0:
            inner = sorted(set(midis))[1:3]  # up to two upper chord tones
            for im in inner:
                arranged.append({
                    "start": round(s + (en - s) / 2, 3), "end": round(min(en, s + (en - s) / 2 + 0.45), 3),
                    "midi": im, "note": _name(im),
                    "octave": im // 12 - 1, "pitch_class": NAMES[im % 12],
                    "velocity": 0.42, "role": "harmony", "source_model": "arranger_v2",
                })

    # Fills: one quiet stepwise lead-in when there is a real gap (>=0.7s).
    mel_sorted = sorted(melody, key=lambda x: x["start"])
    for a, b in zip(mel_sorted, mel_sorted[1:]):
        gap = float(b["start"]) - float(a["end"])
        if 0.7 <= gap <= 2.0 and a.get("midi") is not None and b.get("midi") is not None:
            step = 1 if int(b["midi"]) >= int(a["midi"]) else -1
            f0 = int(a["midi"]) + step
            arranged.append({
                "start": round(float(a["end"]) + gap / 2, 3),
                "end": round(float(a["end"]) + gap / 2 + 0.18, 3),
                "midi": f0, "note": _name(f0),
                "octave": f0 // 12 - 1, "pitch_class": NAMES[f0 % 12],
                "velocity": 0.38, "role": "fill", "source_model": "arranger_v2",
            })
            break  # at most one fill per score in v2.0 (conservative)

    arranged.sort(key=lambda x: (x["start"], x["midi"] if x.get("midi") is not None else 0))
    # Clamp same-role overlaps lightly; melody keeps priority (don't cut melody).
    return arranged
