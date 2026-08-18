#!/usr/bin/env python3
"""Chord progression detection using librosa chroma + template matching.

Matches the per-frame chroma (pitch-class energy) against chord templates
(major, minor, dominant 7th, minor 7th) across all 12 roots, and also against
the triads built from the detected thaat/scale. Returns a time-stamped chord
progression the frontend can play as left-hand backing.

Output:
    {
      "root": "F#",
      "thaat": "Bilawal",
      "duration": 29.5,
      "chords": [
        {"start": 0.0, "end": 2.0, "root": "F#", "quality": "major", "midis": [66,70,73]},
        ...
      ]
    }

Usage:
    python chord_detection.py <audio_filepath> <root_note> [--thaat Bilawal]
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_PC = {n: i for i, n in enumerate(PC_TO_NOTE)}

# chord templates: intervals above root (semitone offsets), and label
CHORD_TEMPLATES = {
    'major':  ([0, 4, 7], 1.0),
    'minor':  ([0, 3, 7], 1.0),
    'dim':    ([0, 3, 6], 0.8),
    'maj7':   ([0, 4, 7, 11], 0.7),
    'min7':   ([0, 3, 7, 10], 0.7),
    'dom7':   ([0, 4, 7, 10], 0.7),
}

# triads implied by each thaat (degree -> [root offset, intervals])
THAAT_TRIADS = {
    'Bilawal':  {'I': 0, 'ii': 2, 'iii': 4, 'IV': 5, 'V': 7, 'vi': 9, 'vii': 11},
    'Khamaj':   {'I': 0, 'ii': 2, 'IV': 5, 'V': 7, 'vi': 9, 'bVII': 10},
    'Kafi':     {'i': 0, 'II': 2, 'bIII': 3, 'iv': 5, 'v': 7, 'bVII': 10},
    'Asavari':  {'i': 0, 'II': 2, 'bIII': 3, 'iv': 5, 'v': 7, 'bVI': 8, 'bVII': 10},
    'Bhairavi': {'i': 0, 'bII': 1, 'bIII': 3, 'iv': 5, 'v': 7, 'bVI': 8, 'bVII': 10},
    'Bhairav':  {'I': 0, 'bII': 1, 'III': 4, 'IV': 5, 'V': 7, 'bVI': 8, 'VII': 11},
    'Kalyan':   {'I': 0, 'II': 2, 'III': 4, 'IV': 6, 'V': 7, 'vi': 9, 'vii': 11},
    'Marwa':    {'I': 0, 'bII': 1, 'III': 4, 'IV': 6, 'V': 7, 'vi': 9, 'VII': 11},
}


def normalize_root(root: str) -> int:
    name = root.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_TO_PC[name]


def build_template_vector(root_pc, intervals, weight):
    """A 12-length chroma template for a chord at a given root."""
    vec = np.zeros(12)
    for iv in intervals:
        vec[(root_pc + iv) % 12] = weight
    return vec


def _pc_to_midi(pc, lo, hi):
    """Return the single MIDI note for pitch class `pc` within [lo, hi)."""
    return lo + ((pc - lo) % 12)


def _voiced_inversions(root_pc, ivs):
    """Candidate left-hand voicings (bass + inner tones) for a chord.

    Enumerates the inversions of the chord's interval set and keeps only the
    ones whose lowest (bass) tone is the root or fifth. The bass is voiced in
    MIDI 36-48 while the remaining 1-3 inner tones sit tightly in MIDI 48-60,
    so the accompaniment is spread rather than a dense block.
    """
    pcs = sorted({(root_pc + iv) % 12 for iv in ivs})
    fifth_iv = next((iv for iv in ivs if iv in (6, 7)), 7)
    bass_pcs = {root_pc, (root_pc + fifth_iv) % 12}
    candidates = []
    for r in range(len(pcs)):
        ordered = pcs[r:] + pcs[:r]
        bass_pc = ordered[0]
        if bass_pc not in bass_pcs:
            continue
        bass = _pc_to_midi(bass_pc, 36, 48)
        inner = sorted(_pc_to_midi(pc, 48, 60) for pc in ordered[1:])
        candidates.append([bass] + inner)
    # de-dupe and prefer the root in the bass as the stable default
    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    uniq.sort(key=lambda c: 0 if c[0] % 12 == root_pc else 1)
    return uniq


def _voice_cost(prev, cand):
    """Total absolute semitone movement vs the previous voicing (voice-leading)."""
    a = sorted(prev)
    b = sorted(cand)
    n = min(len(a), len(b))
    cost = float(sum(abs(x - y) for x, y in zip(a[:n], b[:n])))
    if len(b) > n:
        top = a[-1]
        cost += sum(abs(x - top) for x in b[n:])
    elif len(a) > n:
        top = b[-1]
        cost += sum(abs(x - top) for x in a[n:])
    return cost


def detect_chords(audio_filepath: str, root_pc: int, thaat: str,
                  chord_len=2.0):
    """Return a chord progression with left-hand voicing, bass-aware root
    selection, transition smoothing, and per-chord confidence.

    Improvements over the prototype (accuracy review §2.7):
      - Bass/low-frequency evidence nudges the root choice (helps inversions).
      - Viterbi-style transition smoothing penalizes spurious rapid changes.
      - Generated MIDI is voiced in a pianist's LEFT-HAND register (root ~MIDI
        36-55) so the backing does not mask the melody.
    """
    import librosa

    y, sr = librosa.load(audio_filepath, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    times = librosa.times_like(chroma, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Low-frequency (bass) chroma for root/inversion evidence (full-sr CQT,
    # which stays under Nyquist). Weight lower octaves more for bass info.
    bass_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz('C2'))
    bass_chroma = bass_chroma / (bass_chroma.sum(axis=0) + 1e-9)

    # Build candidate template set.
    templates = []
    triad_roots = set(THAAT_TRIADS.get(thaat, {}).values())
    for deg_off in (triad_roots or {0, 5, 7}):
        r = (root_pc + deg_off) % 12
        for quality, (ivs, w) in CHORD_TEMPLATES.items():
            templates.append({'root': r, 'quality': quality,
                              'vec': build_template_vector(r, ivs, w)})
    for r in range(12):
        for quality, (ivs, w) in CHORD_TEMPLATES.items():
            templates.append({'root': r, 'quality': quality,
                              'vec': build_template_vector(r, ivs, w)})
    # dedupe templates
    seen = set()
    uniq = []
    for t in templates:
        k = (t['root'], t['quality'])
        if k not in seen:
            seen.add(k); uniq.append(t)
    templates = uniq

    n_chunks = max(1, int(duration // chord_len) + 1)
    # candidate scores per chunk
    cand_scores = []
    for ci in range(n_chunks):
        t0 = ci * chord_len
        t1 = min((ci + 1) * chord_len, duration)
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            cand_scores.append(None)
            continue
        frame_chroma = chroma[:, mask].mean(axis=1)
        frame_chroma = frame_chroma / (frame_chroma.sum() + 1e-9)
        # bass chroma averaged over the same window
        bmask = np.arange(bass_chroma.shape[1]) * (times[-1] / bass_chroma.shape[1])
        bass_window = bass_chroma[:, (bmask >= t0) & (bmask < t1)].mean(axis=1) if (bmask >= t0).any() else None
        if bass_window is None:
            bass_window = np.ones(12) / 12.0
        bass_window = bass_window / (bass_window.sum() + 1e-9)

        scored = []
        for t in templates:
            sim = float(np.dot(t['vec'], frame_chroma) /
                        (np.linalg.norm(t['vec']) * np.linalg.norm(frame_chroma) + 1e-9))
            # bass support: does the low register favour this chord root?
            root_pc_t = t['root']
            bass_support = 0.6 * bass_window[root_pc_t] + 0.4 * bass_window[(root_pc_t + 7) % 12]
            scored.append(sim + 0.25 * bass_support)
        cand_scores.append(scored)

    # Viterbi smoothing across chunks (penalize transitions to a different chord)
    trans_penalty = 0.08
    n_cands = len(templates)
    if all(c is not None for c in cand_scores):
        dp = [[0.0] * n_cands for _ in range(n_chunks)]
        back = [[-1] * n_cands for _ in range(n_chunks)]
        for k in range(n_cands):
            dp[0][k] = cand_scores[0][k]
        for ci in range(1, n_chunks):
            for k in range(n_cands):
                best_prev = -1
                best_val = -1e9
                for p in range(n_cands):
                    v = dp[ci - 1][p]
                    if (templates[p]['root'], templates[p]['quality']) != (templates[k]['root'], templates[k]['quality']):
                        v -= trans_penalty
                    if v > best_val:
                        best_val = v; best_prev = p
                dp[ci][k] = cand_scores[ci][k] + best_val
                back[ci][k] = best_prev
        # traceback
        last = max(range(n_cands), key=lambda k: dp[n_chunks - 1][k])
        chosen = [0] * n_chunks
        chosen[n_chunks - 1] = last
        for ci in range(n_chunks - 1, 0, -1):
            chosen[ci - 1] = back[ci][chosen[ci]]
    else:
        chosen = [max(range(n_cands), key=lambda k: cand_scores[ci][k])
                  if cand_scores[ci] is not None else -1 for ci in range(n_chunks)]

    chords = []
    prev_voicing = None
    for ci in range(n_chunks):
        t0 = ci * chord_len
        t1 = min((ci + 1) * chord_len, duration)
        if chosen[ci] < 0 or cand_scores[ci] is None:
            prev_voicing = None
            continue
        idx = chosen[ci]
        best = templates[idx]
        ivs, w = CHORD_TEMPLATES[best['quality']]
        # left-hand, voice-led voicing: bass (root/fifth) in MIDI 36-48 plus
        # tight inner tones in 48-60, choosing the inversion that moves least.
        candidates = _voiced_inversions(best['root'], ivs)
        if not candidates:
            candidates = [[_pc_to_midi(best['root'], 36, 48)]]
        if prev_voicing is None:
            midis = candidates[0]
        else:
            midis = min(candidates, key=lambda c: _voice_cost(prev_voicing, c))
        prev_voicing = midis
        # confidence from the normalized template score
        score = cand_scores[ci][idx]
        conf = float(round(min(1.0, max(0.0, (score - 0.5) * 1.5)), 3))
        chords.append({
            'start': round(t0, 2),
            'end': round(t1, 2),
            'root': PC_TO_NOTE[best['root']],
            'quality': best['quality'],
            'midis': sorted(midis),
            'confidence': conf,
        })

    return chords, duration


def main():
    parser = argparse.ArgumentParser(description="Detect chord progression.")
    parser.add_argument("audio_filepath")
    parser.add_argument("root_note")
    parser.add_argument("--thaat", default="Bilawal")
    parser.add_argument("--chord-len", type=float, default=2.0)
    args = parser.parse_args()

    try:
        root_pc = normalize_root(args.root_note)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        chords, duration = detect_chords(args.audio_filepath, root_pc,
                                         args.thaat, args.chord_len)
    except Exception as e:
        print(json.dumps({"error": f"Chord detection failed: {str(e)}"}))
        sys.exit(1)

    print(json.dumps({
        "root": PC_TO_NOTE[root_pc],
        "thaat": args.thaat,
        "duration": round(duration, 2),
        "chords": chords,
    }))


if __name__ == "__main__":
    main()
