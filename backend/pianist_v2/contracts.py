"""Piano contracts shared by v2 modules and the __SOLO__ frontend path."""
from __future__ import annotations


REQUIRED = ("start", "end", "midi", "note")


def validate_score(events: list[dict]) -> list[str]:
    """Return a list of problems (empty = valid). Never raises."""
    problems: list[str] = []
    for i, e in enumerate(events):
        for k in REQUIRED:
            if e.get(k) is None:
                problems.append(f"event {i}: missing {k}")
        try:
            s, en = float(e.get("start", 0)), float(e.get("end", 0))
            if not s < en:
                problems.append(f"event {i}: non-positive duration")
            if s < 0:
                problems.append(f"event {i}: negative start")
        except (TypeError, ValueError):
            problems.append(f"event {i}: bad timing")
        m = e.get("midi")
        if isinstance(m, int) and not 21 <= m <= 108:
            problems.append(f"event {i}: midi {m} outside piano range")
    for a, b in zip(events, events[1:]):
        try:
            if float(b.get("start", 0)) < float(a.get("start", 0)):
                problems.append("score not sorted by start")
                break
        except (TypeError, ValueError):
            pass
    return problems


class PianoEvent(dict):
    """Thin dict subclass for type clarity; validators accept plain dicts."""
