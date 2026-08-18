#!/usr/bin/env python3
"""Dependency-light regression test for optional evidence export."""
import json
import tempfile
from pathlib import Path

import numpy as np

from vocal_melody import write_evidence_json


def main():
    melody = [{"start": 0.0, "end": 0.1, "midi": 60, "note": "C4"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.json"
        result = write_evidence_json(
            path,
            audio_filepath="song.mp3",
            vocals_path="vocals.wav",
            times=np.array([0.0, 0.005]),
            midi=np.array([60.0, np.nan]),
            periodicity=np.array([0.9, 0.1]),
            rms=np.array([0.2, 0.1]),
            onset_env=np.array([1.0, 0.1]),
            brightness=np.array([0.4, 0.3]),
            onsets=[0.0],
            tempo=120.0,
            beats=[0.0],
            melody=melody,
            root="C",
            thaat="Bilawal",
        )
        payload = json.loads(Path(result).read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert len(payload["frames"]) == 2
        assert payload["frames"][0]["midi_crepe"] == 60.0
        assert payload["frames"][1]["midi_crepe"] is None
        assert payload["melody"] == melody
    print("evidence export regression passed")


if __name__ == "__main__":
    main()
