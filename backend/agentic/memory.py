"""Small JSONL episodic memory for agent traces and musical decisions."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class EpisodicMemory:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("SARGAM_AGENTIC_MEMORY_DIR", "/tmp/sargam_agentic_memory"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "episodes.jsonl"

    def write(self, trace: dict[str, Any], outcome: str = "uncertain", tags: list[str] | None = None) -> dict[str, Any]:
        record = {
            "memory_id": "mem-" + hashlib.sha256(
                (trace.get("trace_id", "") + str(time.time_ns())).encode("utf-8")
            ).hexdigest()[:16],
            "created_at": time.time(),
            "outcome": outcome,
            "tags": tags or [],
            "audio_sha256": trace.get("audio_sha256"),
            "trace": trace,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def retrieve(self, audio_sha256: str | None = None, tags: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
        wanted = set(tags or [])
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.read():
            score = 0
            if audio_sha256 and record.get("audio_sha256") == audio_sha256:
                score += 100
            score += len(wanted & set(record.get("tags", []))) * 10
            if record.get("outcome") == "accepted":
                score += 2
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], -float(item[1].get("created_at", 0))))
        return [record for score, record in scored[:limit] if score > 0]
