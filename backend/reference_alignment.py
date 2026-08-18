"""Optional reference-guided melody alignment.

This module is deliberately opt-in. A supplied sargam/lyric note sequence is
not a scale-correction rule: it is a song-specific annotation that is aligned
to detector events with monotonic dynamic programming. Every rewritten or
synthesized event is marked with reference provenance so the UI/audit trail can
distinguish measured audio fields from guided pitch fields.
"""
from __future__ import annotations

import copy
import math
import re
from statistics import median


DEG = {
    "S": 0, "R": 2, "G": 4, "M": 5, "P": 7, "D": 9, "N": 11,
    "r": 1, "g": 3, "m": 6, "d": 8, "n": 10,
}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_PC = {n: i for i, n in enumerate(NAMES)}
NOTE_PC.update({"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "E#": 5, "B#": 0})


def _note_pc(note: str):
    match = re.match(r"^([A-G](?:#|b)?)", str(note or ""))
    return NOTE_PC.get(match.group(1)) if match else None


def _token(token: str, root_pc: int):
    token = str(token).strip()
    if not token:
        return None
    # Accept S', S'', S, and lower-case komal spellings. Commas are a common
    # notation for lower octave and are treated symmetrically.
    up = token.count("'")
    down = token.count(",")
    body = token.replace("'", "").replace(",", "")
    if not body:
        return None
    degree = body[0]
    if degree not in DEG:
        return None
    return {
        "token": token,
        "pc": (root_pc + DEG[degree]) % 12,
        "relative": DEG[degree] + 12 * (up - down),
    }


def _parse_phrase(phrase, root_pc: int):
    if isinstance(phrase, str):
        tokens = phrase.split()
    elif isinstance(phrase, list):
        tokens = phrase
    else:
        return []
    return [x for x in (_token(t, root_pc) for t in tokens) if x]


def _match_cost(observed, expected, i, j, m, n):
    obs_pc = _note_pc(observed.get("note"))
    if obs_pc is None and observed.get("midi") is not None:
        obs_pc = int(observed["midi"]) % 12
    pitch_cost = 0.0 if obs_pc == expected["pc"] else 2.4
    # A small monotonic timing prior discourages mapping all expected notes to
    # one detector event, while allowing rubato and skipped ornaments.
    oi = i / max(1, m - 1)
    ej = j / max(1, n - 1)
    return pitch_cost + 0.35 * abs(oi - ej)


def _align_indices(observed, expected):
    """Return matched (observed_index, expected_index) pairs via edit DP."""
    m, n = len(observed), len(expected)
    if not m or not n:
        return []
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    prev = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            cur = dp[i][j]
            if not math.isfinite(cur):
                continue
            if i < m and cur + 0.72 < dp[i + 1][j]:
                dp[i + 1][j] = cur + 0.72  # detector-only event / ornament
                prev[i + 1][j] = (i, j, "skip_observed")
            if j < n and cur + 1.65 < dp[i][j + 1]:
                dp[i][j + 1] = cur + 1.65  # reference event not detected
                prev[i][j + 1] = (i, j, "skip_expected")
            if i < m and j < n:
                cost = cur + _match_cost(observed[i], expected[j], i, j, m, n)
                if cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = cost
                    prev[i + 1][j + 1] = (i, j, "match")
    i, j = m, n
    pairs = []
    while i or j:
        step = prev[i][j]
        if step is None:
            break
        pi, pj, kind = step
        if kind == "match":
            pairs.append((pi, pj))
        i, j = pi, pj
    pairs.reverse()
    return pairs


def _root_base_midi(observed, expected, root_pc: int):
    observed_midis = [int(x["midi"]) for x in observed if x.get("midi") is not None]
    if not observed_midis:
        return root_pc + 12 * 4
    rel = [x["relative"] for x in expected]
    # Estimate a tonic register from the observed phrase, then snap to the
    # correct pitch class. This changes octave only; the supplied sequence owns
    # the pitch-class identity.
    rough = median(observed_midis) - median(rel or [0])
    candidates = [root_pc + 12 * k for k in range(1, 8)]
    return min(candidates, key=lambda x: abs(x - rough))


def _set_reference_pitch(event, expected, base_root_midi, root_pc):
    target = int(base_root_midi + expected["relative"])
    # Keep explicit octave marks, but avoid a register jump if an event was
    # matched to a detector note and the reference has no octave mark.
    if event.get("midi") is not None and "'" not in expected["token"] and "," not in expected["token"]:
        target += 12 * round((int(event["midi"]) - target) / 12)
    target = max(21, min(108, target))
    event["midi"] = target
    event["note"] = f"{NAMES[target % 12]}{target // 12 - 1}"
    event["pitch_class"] = NAMES[target % 12]
    event["octave"] = target // 12 - 1
    event["sargam"] = expected["token"]
    event["reference_guided"] = True
    event["reference_token"] = expected["token"]
    event["review_flags"] = sorted(set((event.get("review_flags") or []) + ["reference_guided_pitch"]))
    event["source_pitch"] = "reference"


def align_reference_phrases(melody, reference, root_pc: int):
    """Align configured phrases to detector events and return a new list.

    Reference format::
        {"phrases": [{"start": 2, "end": 9,
                       "sargam": "G P R R G P P D"}, ...]}
    """
    if not isinstance(reference, dict) or not isinstance(reference.get("phrases"), list):
        return melody
    source = [copy.deepcopy(x) for x in melody]
    used = set()
    output = []
    for phrase in reference["phrases"]:
        try:
            start, end = float(phrase["start"]), float(phrase["end"])
        except (KeyError, TypeError, ValueError):
            continue
        expected = _parse_phrase(phrase.get("sargam", phrase.get("notes", [])), root_pc)
        if not expected or end <= start:
            continue
        observed = [x for idx, x in enumerate(source) if idx not in used and float(x.get("start", 0)) < end and float(x.get("end", 0)) > start]
        observed.sort(key=lambda x: float(x.get("start", 0)))
        pairs = _align_indices(observed, expected)
        pair_by_expected = {ej: oi for oi, ej in pairs}
        base_root = _root_base_midi(observed, expected, root_pc)
        matched_observed = set()
        for j, exp in enumerate(expected):
            if j in pair_by_expected:
                oi = pair_by_expected[j]
                event = copy.deepcopy(observed[oi])
                matched_observed.add(oi)
            else:
                # Interpolate only timing for missing notes; the pitch remains
                # explicitly supplied by the reference and is auditable.
                frac0 = j / max(1, len(expected))
                frac1 = (j + 1) / max(1, len(expected))
                event = {
                    "start": start + (end - start) * frac0,
                    "end": start + (end - start) * frac1,
                    "velocity": 0.62,
                    "pitch_confidence": 0.0,
                    "voicing_confidence": 0.0,
                    "render_role": "reference_fill",
                }
            _set_reference_pitch(event, exp, base_root, root_pc)
            if j > 0 and expected[j - 1]["pc"] == exp["pc"]:
                event["retrigger"] = True
                event["articulation"] = "retrigger"
                event["review_flags"] = sorted(set(event.get("review_flags", []) + ["reference_retrigger"]))
            # The annotation window is authoritative for phrase membership.
            # A detector onset can lead the lyric boundary by a few frames; do
            # not let that make the first guided note disappear from playback
            # or evaluation. Preserve the measured timing whenever possible,
            # but clamp each guided event into its annotated window.
            event_start = max(start, min(end - 0.06, float(event.get("start", start))))
            event_end = min(end, max(event_start + 0.06, float(event.get("end", event_start + 0.06))))
            event["start"] = round(event_start, 3)
            event["end"] = round(event_end, 3)
            output.append(event)
        # Detector-only events inside a guided phrase are intentionally omitted;
        # their evidence remains available in the unmodified extraction log.
        used.update(source.index(x) for x in observed if x in source)
    guided_windows = [(float(p.get("start", 0)), float(p.get("end", 0))) for p in reference["phrases"] if isinstance(p, dict)]
    for idx, event in enumerate(source):
        if idx in used:
            continue
        t = float(event.get("start", 0))
        if not any(a <= t < b for a, b in guided_windows):
            output.append(event)
    output.sort(key=lambda x: (float(x.get("start", 0)), int(x.get("midi", 0))))
    return output
