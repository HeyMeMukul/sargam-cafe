#!/usr/bin/env python3
"""Run the tool-using Sargam pianist controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model", default=os.getenv("SARGAM_AGENTIC_MODEL", "gpt-5"))
    parser.add_argument("--max-tool-calls", type=int, default=int(os.getenv("SARGAM_AGENTIC_MAX_TOOL_CALLS", "10")))
    parser.add_argument("--output", default="agentic_transcription.json")
    args = parser.parse_args()

    from openai import OpenAI
    from .controller import PianistAgent

    result = PianistAgent(OpenAI(), model=args.model, max_tool_calls=args.max_tool_calls).run(args.audio)
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "trace_id": result["trace"]["trace_id"],
        "model": args.model,
        "tool_calls": len(result["trace"]["tool_calls"]),
        "skill_citations": len(result["trace"]["skill_citations"]),
        "hypothesis_state": result["hypothesis"]["state"],
        "event_count": len(result["hypothesis"]["events"]),
        "output": str(output.resolve()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
