#!/usr/bin/env python3
import tempfile
from pathlib import Path

from agentic.memory import EpisodicMemory

with tempfile.TemporaryDirectory() as directory:
    memory = EpisodicMemory(directory)
    record = memory.write({"trace_id": "t1", "audio_sha256": "abc"}, outcome="accepted", tags=["melody", "pianist"])
    assert record["memory_id"].startswith("mem-")
    assert memory.retrieve("abc", ["melody"], limit=1)[0]["memory_id"] == record["memory_id"]
print("agentic memory passed")
