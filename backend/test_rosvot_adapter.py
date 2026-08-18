"""Focused regression test for ROSVOT interval preservation."""
from pathlib import Path
import tempfile

import pretty_midi

from rosvot_adapter import _read_midi_events


def test_read_midi_events_preserves_gaps_and_retriggers():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "output.mid"
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes.append(pretty_midi.Note(100, 61, 0.50, 0.90))
        instrument.notes.append(pretty_midi.Note(95, 61, 0.92, 1.20))
        instrument.notes.append(pretty_midi.Note(90, 63, 1.70, 2.00))
        midi.instruments.append(instrument)
        midi.write(str(path))

        events = _read_midi_events(path)
        assert len(events) == 3
        assert events[0]["start"] == 0.5
        assert events[1]["retrigger"] is True
        assert events[2]["start"] == 1.7


if __name__ == "__main__":
    test_read_midi_events_preserves_gaps_and_retriggers()
    print("ROSVOT MIDI timing regression passed")
