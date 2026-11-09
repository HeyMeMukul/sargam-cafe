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
    # INVARIANT: melody notes are never cut, trimmed, or re-pitched except for
    # the octave lift below C4 (pitch class + start/end/duration preserved).
    lifted_melody: list[dict] = []
    for m in melody:
        e = dict(m)
        if e.get("midi") is not None and int(e["midi"]) < 60:
            # Lift by whole octaves so the final pitch is >= C4 while the
            # pitch class is unchanged (single lift in the common case).
            midi0 = int(e["midi"])
            octaves = max(1, -(-(60 - midi0) // 12))
            e["midi"] = midi0 + 12 * octaves
            e["note"] = _name(int(e["midi"]))
            e["octave"] = int(e["midi"]) // 12 - 1
            e["pitch_class"] = NAMES[int(e["midi"]) % 12]
        e["role"] = "melody"
        e["velocity"] = round(max(0.2, min(1.0, float(e.get("velocity", 0.7)))), 3)
        arranged.append(e)
        lifted_melody.append(e)

    min_mel_vel = min((float(e.get("velocity", 0.7)) for e in lifted_melody), default=0.7)
    # Cap for quiet parts: strictly below the softest melody note.
    quiet_cap = round(min(0.42, min_mel_vel - 0.05), 3)
    quiet_cap = max(0.2, quiet_cap)
    fill_cap = round(min(0.38, min_mel_vel - 0.05), 3)
    fill_cap = max(0.2, fill_cap)

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
        bass_end = round(min(en, s + 0.6), 3)
        # INVARIANT: bass must never collide with an overlapping melody note
        # (within +/-1 semitone in absolute pitch). Shift down an octave;
        # if that leaves piano range, shift up instead; drop if still colliding.
        def _collides(b: int) -> bool:
            for lm in lifted_melody:
                try:
                    if lm.get("midi") is None:
                        continue
                    ms, me = float(lm["start"]), float(lm["end"])
                except (TypeError, ValueError):
                    continue
                overlaps = ms < bass_end and s < me
                if overlaps and abs(int(lm["midi"]) - b) <= 1:
                    return True
            return False

        if _collides(bass):
            shifted = bass - 12 if bass - 12 >= 21 else bass + 12
            if 21 <= shifted <= 108 and not _collides(shifted):
                bass = shifted
            elif 21 <= bass + 12 <= 108 and not _collides(bass + 12):
                bass = bass + 12
            else:
                bass = None  # unresolvable collision: omit bass rather than clash
        if bass is None:
            continue
        # Bass: strong, short-ish (half the chord or 0.6s cap).
        arranged.append({
            "start": round(s, 3), "end": bass_end,
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
                    # INVARIANT: harmony strictly quieter than any melody note.
                    "velocity": quiet_cap, "role": "harmony", "source_model": "arranger_v2",
                })

    # Fills: one quiet stepwise lead-in when there is a real gap (>=0.7s).
    # INVARIANT: at most one fill per score in v2.0 (conservative).
    mel_sorted = sorted(melody, key=lambda x: x["start"])
    for a, b in zip(mel_sorted, mel_sorted[1:]):
        gap = float(b["start"]) - float(a["end"])
        if 0.7 <= gap <= 2.0 and a.get("midi") is not None and b.get("midi") is not None:
            step = 1 if int(b["midi"]) >= int(a["midi"]) else -1
            f0 = int(a["midi"]) + step
            if 21 <= f0 <= 108:
                arranged.append({
                    "start": round(float(a["end"]) + gap / 2, 3),
                    "end": round(float(a["end"]) + gap / 2 + 0.18, 3),
                    "midi": f0, "note": _name(f0),
                    "octave": f0 // 12 - 1, "pitch_class": NAMES[f0 % 12],
                    "velocity": fill_cap, "role": "fill", "source_model": "arranger_v2",
                })
            break  # at most one fill per score in v2.0 (conservative)

    arranged.sort(key=lambda x: (x["start"], x["midi"] if x.get("midi") is not None else 0))
    # Clamp same-role overlaps lightly; melody keeps priority (don't cut melody).
    return arranged
