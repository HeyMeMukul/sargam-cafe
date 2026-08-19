# Boundary-Aware Note Model Research Matrix

| Approach | Boundary mechanism | Pitch mechanism | Direct test on supplied clip | Practical status | Decision |
|---|---|---|---|---|---|
| Current CREPE path | Generic librosa onsets + 150 ms persistence + handwritten retrigger merge | torchcrepe framewise F0 | 37 browser events; 5 scale disagreements; phrase edit distance 15 | Stable and fast, but structurally weak at note boundaries | Keep as rollback baseline |
| ROSVOT | Multi-scale note segmentation with U-Net/Conformer-style architecture and a trained boundary model | Joint attention-based pitch decoder; adapter emits MIDI intervals | Direct adapter works with available checkpoints: 52 events, 19 retriggers, 4 scale disagreements, phrase edit distance 16 | Best immediate audio-only candidate; threshold sweep 0.60–0.95 did not materially change phrase score | Integrate as an experimental candidate, not default yet |
| VOCANO | Dedicated PyramidNet note-segmentation model after pitch extraction | Patch-CNN pitch contour | Not run: legacy Apex/Google Drive checkpoint dependencies and old PyTorch assumptions | Open-source and conceptually relevant, but installation/checkpoint risk is high | Use as a research reference or isolated future benchmark |
| MusicYOLO | Whole-note object detection on spectrogram regions | Spectral pitch labeling | Not run: old YOLOX/PyTorch stack and Baidu-hosted checkpoints | Strong boundary idea, weak immediate reproducibility | Do not integrate before checkpoint and dependency validation |
| STARS | Hierarchical frame/word/phoneme/note model with note decoder | Joint singing transcription/alignment | Not run: pretrained checkpoints require phoneme/word metadata and Python 3.10/PyTorch 2.4/CUDA 12.8 | Strong conditional model when lyrics/phonemes exist; current Hindi audio-only path lacks required metadata | Future lyrics-informed mode, not current default |

## Direct ROSVOT observations

The existing adapter and checkpoints are operational in the sandbox. ROSVOT emits real MIDI intervals and preserves 19 same-pitch retriggers, which directly addresses one known CREPE failure mode. However, its 52-event output is over-segmented relative to the supplied diagnostic phrase sequence, and the threshold sweep did not improve the sequence score. The correct next experiment is therefore **ROSVOT boundaries plus an evidence-aware decoder**, not blind replacement.

## Recommended implementation order

First, expose ROSVOT as a feature-flagged candidate with explicit provenance and save both ROSVOT and CREPE candidate scores. Second, build a shared candidate decoder that can merge or retain ROSVOT retriggers using pitch continuity, boundary confidence, and duration evidence without scale-forcing. Third, benchmark that candidate on adjudicated annotations. Only then decide whether ROSVOT becomes the default.

The long-term best architecture is a joint boundary/pitch model trained or fine-tuned on the project’s annotation set. No current one-clip phrase score is sufficient to claim that any pretrained model is perfect.

## Sources

- [Robust Singing Voice Transcription Serves Synthesis](https://arxiv.org/html/2405.09940v1)
- [STARS official implementation](https://github.com/gwx314/STARS)
- [VOCANO official implementation](https://github.com/B05901022/VOCANO)
- [MusicYOLO official implementation](https://github.com/itec-hust/MusicYOLO)
- [Note-Level Singing Melody Transcription for Time-Aligned Musical Score Generation](https://arxiv.org/html/2502.12438v1)
