"""Installable runtime registry for the Sargam Cafe pianist agent."""
from __future__ import annotations

from .audio_tools import (
    audition_phrase,
    get_note_candidates,
    get_track_manifest,
    inspect_boundary_window,
    inspect_pitch_window,
    score_hypothesis,
)
from .tool_registry import ToolRegistry, default_registry


def build_runtime_registry() -> ToolRegistry:
    registry = default_registry()
    registry.bind("get_track_manifest", get_track_manifest)
    registry.bind("get_note_candidates", get_note_candidates)
    registry.bind("inspect_pitch_window", inspect_pitch_window)
    registry.bind("inspect_boundary_window", inspect_boundary_window)
    registry.bind("audition_phrase", audition_phrase)
    registry.bind("score_hypothesis", score_hypothesis)
    return registry
