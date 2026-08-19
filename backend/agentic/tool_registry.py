"""Tool registry for the evidence-to-pianist agent.

The registry describes capabilities and validates call envelopes. Execution is
provided by the specialist adapters in later phases; keeping registration
separate prevents the LLM from inventing arbitrary shell commands as its only
interface to audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    output_artifact: str = ""
    mutating: bool = False


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, spec: ToolSpec, handler: Callable[..., Any] | None = None) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec
        if handler is not None:
            self._handlers[spec.name] = handler

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "required_arguments": list(spec.required_arguments),
                "optional_arguments": list(spec.optional_arguments),
                "output_artifact": spec.output_artifact,
                "mutating": spec.mutating,
            }
            for spec in self._specs.values()
        ]

    def validate_call(self, name: str, arguments: dict[str, Any]) -> None:
        if name not in self._specs:
            raise ValueError(f"unknown tool: {name}")
        spec = self._specs[name]
        missing = [key for key in spec.required_arguments if key not in arguments]
        if missing:
            raise ValueError(f"tool {name} missing arguments: {', '.join(missing)}")
        allowed = set(spec.required_arguments) | set(spec.optional_arguments)
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"tool {name} received unknown arguments: {', '.join(unknown)}")

    def call(self, name: str, **arguments: Any) -> Any:
        self.validate_call(name, arguments)
        if name not in self._handlers:
            raise RuntimeError(f"tool {name} has no installed handler")
        return self._handlers[name](**arguments)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "get_track_manifest",
        "Return audio duration, hashes, stems, and available evidence artifacts.",
        ("audio_path",),
        output_artifact="track_manifest",
    ))
    registry.register(ToolSpec(
        "get_note_candidates",
        "Return baseline note candidates in a requested time window from one or more transcribers.",
        ("audio_path", "start", "end"),
        ("transcribers", "min_confidence"),
        output_artifact="note_candidates",
    ))
    registry.register(ToolSpec(
        "inspect_pitch_window",
        "Return frame pitch tracks, voicing, uncertainty, and alternative octave tracks for a narrow window.",
        ("audio_path", "start", "end"),
        ("trackers", "resolution"),
        output_artifact="pitch_window",
    ))
    registry.register(ToolSpec(
        "inspect_boundary_window",
        "Return onset, offset, energy, spectral-flux, and retrigger evidence around a suspected boundary.",
        ("audio_path", "start", "end"),
        ("resolution",),
        output_artifact="boundary_window",
    ))
    registry.register(ToolSpec(
        "align_lyrics_or_phonemes",
        "Align a supplied lyric/phoneme sequence to the vocal audio and return timing candidates.",
        ("audio_path", "text"),
        ("language", "start", "end"),
        output_artifact="lyric_alignment",
    ))
    registry.register(ToolSpec(
        "get_harmonic_context",
        "Return local chords, tonic alternatives, and harmonic evidence without rewriting melody pitch.",
        ("audio_path", "start", "end"),
        output_artifact="harmonic_context",
    ))
    registry.register(ToolSpec(
        "audition_phrase",
        "Render a proposed phrase with exact event timing and return a comparison artifact for listening.",
        ("audio_path", "events", "start", "end"),
        ("render_profile",),
        output_artifact="audition_comparison",
    ))
    registry.register(ToolSpec(
        "score_hypothesis",
        "Compute strict ordered metrics and evidence-consistency checks for a candidate hypothesis.",
        ("audio_path", "events"),
        ("reference_path",),
        output_artifact="hypothesis_score",
    ))
    registry.register(ToolSpec(
        "commit_hypothesis",
        "Persist an accepted versioned score and its evidence trace after validation.",
        ("trace_id", "hypothesis"),
        output_artifact="committed_score",
        mutating=True,
    ))
    return registry
