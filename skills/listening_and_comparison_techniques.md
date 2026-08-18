# Programmatic Techniques for AI "Listening Skills"
**Audio Engineering & Machine Listening Research Report**

## 1. Pitch Detection Algorithms and Polyphonic Limitations

### Overview of Single-Pitch Estimation (SPE) Algorithms
Standard pitch detection algorithms are highly optimized for monophonic audio (single voice or instrument) but degrade significantly when faced with polyphonic complexities.

- **YIN**: A time-domain algorithm based on autocorrelation. It is fast and efficient but suffers from "octave errors" and struggles with noisy audio.
- **pYIN (Probabilistic YIN)**: Introduces multiple pitch candidates and a Hidden Markov Model (HMM) to decode the most likely pitch trajectory, improving robustness against transient errors.
- **CREPE**: A Convolutional Neural Network (CNN) operating on raw time-domain waveforms. It sets the standard for fine-grained pitch estimation and noise robustness.

### Why They Struggle with Polyphonic Music
These algorithms are architecturally constrained to **Single-Pitch Estimation**. They output a single fundamental frequency ($f_0$) per frame. 
In a polyphonic context (e.g., a vocal melody over a guitar rhythm):
1. **Inability to Separate Voices**: The periodicities of different instruments interfere with one another. Autocorrelation-based methods (YIN/pYIN) become confused by overlapping harmonic series.
2. **Frequency Jumping**: Models like CREPE will attempt to track the most dominant sound. If an accompaniment chord is louder than the target melody, the predicted pitch will jump to the accompaniment's notes, rendering the transcription useless for finding a specific target note.

**Solution**: For polyphonic audio, one must use either **Automatic Music Transcription (AMT)** models (like Spotify's Basic Pitch) or employ **Source Separation** (like Demucs or Spleeter) to isolate the target track before applying pYIN or CREPE.

---

## 2. Chromagrams and Harmonic Matching (Librosa)

To determine if a synthesized note matches a snippet of a song, regardless of octave or timbre, we use **Pitch Class Profiles (PCP)** or **Chromagrams**.

### How Chromagrams Work
A chromagram collapses all octaves into 12 discrete pitch classes (C, C#, D, etc.). This makes it incredibly robust to timbral differences and octave shifts. It represents the "harmonic identity" of the audio frame.

### Implementation Workflow in Python (Librosa)
1. **Harmonic/Percussive Separation**: Transients (drums) pollute pitch-class energy. First, isolate the harmonic component.
2. **CQT Chromagram**: Use the Constant-Q Transform (CQT) instead of standard STFT, as CQT bins align logarithmically with musical pitches.
3. **CENS Features**: For robust matching, Chroma Energy Normalized Statistics (CENS) smooths out local tempo and dynamic variations.

```python
import librosa
import numpy as np

# 1. Load polyphonic audio snippet
y, sr = librosa.load('snippet.wav')

# 2. Separate harmonic content
y_harmonic, _ = librosa.effects.hpss(y)

# 3. Compute CQT Chromagram
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# The result is a 12 x N matrix representing the energy of each pitch class over time.
```

---

## 3. FFT, Spectrograms, and Cross-Correlation

Can we simply cross-correlate a generated C4 with the polyphonic audio?

### The Mechanics
- **Spectrogram Template Matching**: Convert the isolated generated note and the audio snippet into spectrograms. Slide the template (generated note) over the snippet's spectrogram to find the point of maximum 2D cross-correlation.
  
### The Challenges in Polyphonic Audio
Direct cross-correlation rarely works in complex polyphony because:
1. **Spectral Overlap**: The target note's frequencies are buried under the harmonics of other simultaneous notes.
2. **Timbre Mismatch**: The generated note (e.g., a pure sine or basic synth) has a different harmonic envelope than the real instrument in the polyphonic track. This throws off the correlation calculation.

### Modern Solutions
Instead of raw STFT cross-correlation, use:
- **Constant-Q Transform (CQT)** for musically aligned bins.
- **Non-negative Matrix Factorization (NMF)** to decompose the polyphonic spectrogram into individual spectral templates and temporal activations.

---

## 4. Calculating a "Confidence Score"

If an AI agent needs to definitively answer: "Does this snippet contain the note C4?", it requires a mathematical workflow to generate a confidence score (0.0 to 1.0).

### Approach A: The Deep Learning Route (Recommended)
Use a pre-trained AMT model like **Spotify Basic Pitch**.
1. Run the snippet through Basic Pitch.
2. The model outputs a continuous activation map (shape: `[time_frames, 88_piano_keys]`) with values between 0.0 and 1.0.
3. **Confidence Score**: The max activation value for the specific pitch bin (e.g., C4) during the snippet duration.

### Approach B: Custom Signal Processing Pipeline
If building a deterministic matching algorithm without neural nets:

1. **Calculate CQT Spectrogram**: 
   $S = |CQT(y)|$
2. **Target Pitch Mask**: Create a frequency mask targeting the fundamental frequency ($f_0$) of C4 and its first few harmonics ($2f_0, 3f_0$).
3. **Salience Calculation**: Sum the energy within the target mask over the total energy of the frame.

**The Math (Simplified Salience Ratio)**:
For a specific time frame $t$:
$$ Salience(t) = \frac{\sum_{h=1}^{H} S(h \cdot f_0, t)}{\sum_{f} S(f, t)} $$
Where $H$ is the number of harmonics analyzed.

**Match Percentage Workflow**:
1. Calculate the Salience array over all time frames in the snippet.
2. Find the frame with the maximum Salience.
3. Normalize this value against a predefined empirical threshold or background noise floor to clamp it between 0 and 1.
4. **Agent Decision Logic**:
   - `if Confidence > 0.85:` "Yes, this note is 100% correct."
   - `elif Confidence > 0.40:` "The pitch class is present, but it might be part of a chord or background harmony."
   - `else:` "Note not detected."
