"""Typed contracts for the evidence-to-pianist agent loop.

The contracts deliberately separate measured evidence from agent interpretation.
No hypothesis may be committed without evidence references, and every revision
keeps its parent version so decisions remain reversible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DecisionState = Literal["candidate", "accepted", "rejected", "uncertain"]
EventRole = Literal["attack", "sustain", "ornament", "rest"]


@dataclass(frozen=True)
class EvidenceRef:
    """Pointer to a measured artifact or a targeted observation."""

    tool: str
    artifact_id: str
    feature: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    summary: str = ""

    def validate(self) -> None:
        if not self.tool or not self.artifact_id or not self.feature:
            raise ValueError("evidence references require tool, artifact_id, and feature")
        if self.start is not None and self.start < 0:
            raise ValueError("evidence start must be non-negative")
        if self.end is not None and self.start is not None and self.end < self.start:
            raise ValueError("evidence end must not precede start")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("evidence confidence must be in [0, 1]")


@dataclass
class NoteCandidate:
    """One proposed musical event, retaining raw pitch and alternatives."""

    event_id: str
    start: float
    end: float
    midi: int | None
    raw_midi_float: float | None = None
    pitch_confidence: float | None = None
    voicing_confidence: float | None = None
    onset_confidence: float | None = None
    offset_confidence: float | None = None
    velocity: float | None = None
    articulation: str = "normal"
    render_role: EventRole = "attack"
    state: DecisionState = "candidate"
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    agent_reason: str = ""

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("note candidate requires event_id")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("note candidate must have 0 <= start < end")
        if self.midi is not None and not 0 <= self.midi <= 127:
            raise ValueError("MIDI pitch must be in [0, 127]")
        for value in (self.pitch_confidence, self.voicing_confidence,
                      self.onset_confidence, self.offset_confidence, self.velocity):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("confidence and velocity values must be in [0, 1]")
        for ref in self.evidence_refs:
            ref.validate()


@dataclass
class ToolCall:
    """A planned or executed specialist-tool call."""

    call_id: str
    tool: str
    arguments: dict[str, Any]
    rationale: str
    expected_observation: str = ""
    result_artifact_id: str | None = None
    status: Literal["planned", "completed", "failed"] = "planned"
    error: str | None = None


@dataclass
class HypothesisVersion:
    """Versioned score state; revisions never overwrite their parent."""

    hypothesis_id: str
    version: int
    parent_version: int | None
    state: DecisionState
    events: list[NoteCandidate]
    unresolved_questions: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    decision_reason: str = ""

    def validate(self) -> None:
        if self.version < 0:
            raise ValueError("hypothesis version must be non-negative")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent version must precede child version")
        for event in self.events:
            event.validate()
        for previous, current in zip(self.events, self.events[1:]):
            if current.start < previous.start:
                raise ValueError("events must be ordered by start time")


@dataclass
class AgentTrace:
    """Complete audit trail for one agentic transcription run."""

    trace_id: str
    audio_sha256: str
    skill_citations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[HypothesisVersion] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_hypothesis_id: str | None = None

    def validate(self) -> None:
        if not self.trace_id or not self.audio_sha256:
            raise ValueError("trace requires trace_id and audio_sha256")
        for hypothesis in self.hypotheses:
            hypothesis.validate()
        if self.final_hypothesis_id and self.final_hypothesis_id not in {
            h.hypothesis_id for h in self.hypotheses
        }:
            raise ValueError("final hypothesis must exist in trace")


def to_jsonable(value: Any) -> Any:
    """Serialize nested dataclass contracts without losing evidence fields."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
