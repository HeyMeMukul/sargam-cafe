#!/usr/bin/env python3
"""Robust root/key detection using the Krumhansl-Schmuckler key profiles.

Instead of probing a single intro timestamp (which is fragile), this:
  1. Loads the WHOLE track and isolates the harmonic content (HPSS).
  2. Computes a full-track mean chroma (12 pitch classes).
  3. Correlates it against the Krumhansl-Kessler major/minor key profiles
     (all 12 transpositions) -> the tonic (root) + major/minor family.
  4. Refines the exact thaat by scoring each thaat's interval set against
     the observed pitch-class histogram (in-set energy vs out-of-set energy).

This gives a deterministic, repeatable root note — the same answer every run.

Output:
    {"root": "D#", "thaat": "Kafi", "western_scale": "D# Dorian",
     "confidence": 0.87, "candidates": [...]}

Usage:
    python key_detection.py <audio_filepath>
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_PC = {n: i for i, n in enumerate(PC_TO_NOTE)}

# Krumhansl-Kessler key profiles (correlation strength per scale degree)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

THAATS = {
    "Bilawal": {"intervals": [0, 2, 4, 5, 7, 9, 11], "western": "Major", "family": "major"},
    "Khamaj":  {"intervals": [0, 2, 4, 5, 7, 9, 10], "western": "Mixolydian", "family": "major"},
    "Kalyan":  {"intervals": [0, 2, 4, 6, 7, 9, 11], "western": "Lydian", "family": "major"},
    "Bhairav": {"intervals": [0, 1, 4, 5, 7, 8, 11], "western": "Double Harmonic Major", "family": "major"},
    "Marwa":   {"intervals": [0, 1, 4, 6, 7, 9, 11], "western": "Marwa", "family": "major"},
    "Kafi":    {"intervals": [0, 2, 3, 5, 7, 9, 10], "western": "Dorian", "family": "minor"},
    "Asavari": {"intervals": [0, 2, 3, 5, 7, 8, 10], "western": "Natural Minor", "family": "minor"},
    "Bhairavi":{"intervals": [0, 1, 3, 5, 7, 8, 10], "western": "Phrygian", "family": "minor"},
    "Poorvi":  {"intervals": [0, 1, 4, 6, 7, 8, 11], "western": "Poorvi", "family": "minor"},
    "Todi":    {"intervals": [0, 1, 3, 6, 7, 8, 11], "western": "Todi", "family": "minor"},
}


def mean_chroma(audio_filepath: str):
    import librosa
    y, sr = librosa.load(audio_filepath, sr=22050, mono=True)
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    mean = chroma.mean(axis=1)
    total = mean.sum()
    if total <= 0:
        return None
    return mean / total


def pitch_class_duration(audio_filepath: str):
    """Pitch-class histogram weighted by sustained duration (where the melody
    RESTS / holds). This is the key cue for resolving relative-key ambiguity:
    the tonic is where phrases resolve, so the tonic's pitch class is typically
    the most-held non-neighbor note. Computed from a root-independent pyin
    pitch track. Returns a normalized 12-vector.
    """
    import librosa
    y, sr = librosa.load(audio_filepath, sr=22050, mono=True)
    yv, _ = librosa.effects.hpss(y)
    f0, voiced_flag, voiced_prob = librosa.pyin(yv, fmin=librosa.note_to_hz('C3'),
                                                fmax=librosa.note_to_hz('C6'), sr=sr)
    times = librosa.times_like(f0, sr=sr)
    step = times[1] - times[0] if len(times) > 1 else 0.01
    pc_dur = np.zeros(12)
    valid = voiced_flag & (voiced_prob > 0.4) & ~np.isnan(f0)
    for i in range(len(f0)):
        if not valid[i]:
            continue
        midi = librosa.hz_to_midi(f0[i])
        pc = int(round(midi)) % 12
        pc_dur[pc] += step
    total = pc_dur.sum()
    if total <= 0:
        return np.ones(12) / 12.0
    return pc_dur / total


def bass_pitch_class(audio_filepath: str):
    """Bass (low-frequency) pitch-class histogram from CQT. In tonal music the
    harmony resolves to the tonic in the bass, so a strong tonic/bass presence
    supports a candidate root. Returns a normalized 12-vector.
    """
    import librosa
    y, sr = librosa.load(audio_filepath, sr=22050, mono=True)
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                             n_bins=36, bins_per_octave=12, hop_length=1024))
    freqs = librosa.cqt_frequencies(36, fmin=librosa.note_to_hz('C2'),
                                    bins_per_octave=12)
    bass_pc = np.zeros(12)
    for t in range(cqt.shape[1]):
        col = cqt[:, t]
        if col.max() < 0.01:
            continue
        idx = int(np.argmax(col))
        midi = librosa.hz_to_midi(freqs[idx])
        pc = int(round(midi)) % 12
        bass_pc[pc] += 1
    total = bass_pc.sum()
    if total <= 0:
        return np.ones(12) / 12.0
    return bass_pc / total


def detect_key(audio_filepath: str, mel_dur=None):
    chroma = mean_chroma(audio_filepath)
    if chroma is None:
        return None

    # Multi-signal tonic evidence (accuracy review §2.6 / relative-key fix):
    # - Krumhansl-Schmuckler correlation (full-track harmonic fit)
    # - melodic sustained-duration (where phrases rest -> tonic). Best computed
    #   from the ISOLATED vocal melody (root-independent) and passed in; pyin on
    #   the raw mix is a weak fallback that can be confused by backing instruments.
    # - bass support (harmony resolves to tonic in the bass)
    if mel_dur is None:
        mel_dur = pitch_class_duration(audio_filepath)
    mel_dur = np.asarray(mel_dur, dtype=float)
    if mel_dur.shape[0] != 12 or mel_dur.sum() <= 0:
        mel_dur = np.ones(12) / 12.0
    bass = bass_pitch_class(audio_filepath)

    # Score every (tonic, mode): combined Krumhansl + melodic + bass + scale-fit
    scored_keys = []
    for tonic in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = float(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1])
            mel = float(mel_dur[tonic])
            bs = float(bass[tonic])
            ivs = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10]
            scale = set((tonic + iv) % 12 for iv in ivs)
            in_energy = sum(chroma[pc] for pc in scale)
            out_energy = sum(chroma[pc] for pc in range(12) if pc not in scale)
            scale_fit = in_energy / (in_energy + out_energy + 1e-9)
            # weights: corr is base, melodic-duration is the strongest relative-key
            # disambiguator, bass supports the harmonic root
            total = 0.5 * corr + 0.4 * mel + 0.2 * bs + 0.3 * scale_fit
            scored_keys.append((total, corr, tonic, mode))
    scored_keys.sort(key=lambda r: r[0], reverse=True)
    best_total, best_corr, best_tonic, best_family = scored_keys[0]

    # Refine exact thaat among the winning family by scale-fit
    candidates = [name for name, t in THAATS.items() if t["family"] == best_family]
    scored = []
    for name in candidates:
        ivs = THAATS[name]["intervals"]
        in_set = [(best_tonic + i) % 12 for i in ivs]
        in_energy = sum(chroma[i] for i in in_set)
        out_energy = sum(chroma[i] for i in range(12) if i not in in_set)
        score = in_energy / (in_energy + out_energy + 1e-9)
        scored.append((score, name))
    scored.sort(key=lambda s: s[0], reverse=True)

    thaat = scored[0][1] if scored else candidates[0]
    western = THAATS[thaat]["western"]
    root = PC_TO_NOTE[best_tonic]

    # confidence: margin between best and runner-up combined score
    margin = float(scored_keys[0][0] - scored_keys[1][0])
    confidence = round(float(min(1.0, max(0.0, 0.5 + margin * 3))), 3)

    top = [{"root": PC_TO_NOTE[t], "mode": m, "score": round(float(s), 3)}
           for s, _, t, m in scored_keys[:4]]

    return {
        "root": root,
        "root_pc": best_tonic,
        "thaat": thaat,
        "western_scale": f"{root} {western}",
        "mode": best_family,
        "confidence": confidence,
        "candidates": top,
    }



    # Krumhansl-Schmuckler: best correlation across 12 tonics x 2 modes
    results = []
    for tonic in range(12):
        corr_maj = float(np.corrcoef(chroma, np.roll(MAJOR_PROFILE, tonic))[0, 1])
        corr_min = float(np.corrcoef(chroma, np.roll(MINOR_PROFILE, tonic))[0, 1])
        results.append((corr_maj, tonic, "major"))
        results.append((corr_min, tonic, "minor"))

    results.sort(key=lambda r: r[0], reverse=True)
    best_corr, best_tonic, best_family = results[0]

    # Refine exact thaat among the winning family by histogram fit
    candidates = [name for name, t in THAATS.items() if t["family"] == best_family]
    scored = []
    for name in candidates:
        ivs = THAATS[name]["intervals"]
        in_set = [(best_tonic + i) % 12 for i in ivs]
        in_energy = sum(chroma[i] for i in in_set)
        out_energy = sum(chroma[i] for i in range(12) if i not in in_set)
        # ratio of in-set to total, mildly penalised by out-of-set energy
        score = in_energy / (in_energy + out_energy + 1e-9)
        scored.append((score, name))
    scored.sort(key=lambda s: s[0], reverse=True)

    thaat = scored[0][1] if scored else candidates[0]
    western = THAATS[thaat]["western"]
    root = PC_TO_NOTE[best_tonic]

    # confidence: margin between best and runner-up correlation
    margin = results[0][0] - results[1][0]
    confidence = round(min(1.0, max(0.0, 0.5 + margin * 2)), 3)

    top = [{"root": PC_TO_NOTE[t], "mode": m, "corr": round(c, 3)}
           for c, t, m in results[:4]]

    return {
        "root": root,
        "root_pc": best_tonic,
        "thaat": thaat,
        "western_scale": f"{root} {western}",
        "mode": best_family,
        "confidence": confidence,
        "candidates": top,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect key/root using Krumhansl-Schmuckler.")
    parser.add_argument("audio_filepath")
    parser.add_argument("--melodic-duration-json", help="Optional JSON file with a 12-vector "
                        "pitch-class duration histogram from the isolated vocal melody")
    args = parser.parse_args()

    mel_dur = None
    if args.melodic_duration_json:
        import json as _json
        try:
            mel_dur = _json.load(open(args.melodic_duration_json)).get("pitch_class_duration")
        except Exception:
            mel_dur = None

    result = detect_key(args.audio_filepath, mel_dur=mel_dur)
    if result is None:
        print(json.dumps({"error": "Could not analyze audio."}))
        sys.exit(1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
