#!/usr/bin/env python3
import asyncio

from agentic.opencode_adapter import parse_operations, run_opencode_micro_agent

payload = {'operations': [{'op': 'keep', 'reason': 'none', 'evidence_refs': []}], 'state': 'uncertain', 'decision_reason': 'keep', 'unresolved_questions': []}
assert parse_operations('```json\n' + __import__('json').dumps(payload) + '\n```') == payload
assert parse_operations(__import__('json').dumps(payload)) == payload
assert parse_operations('no json') is None

async def fake_runner(cmd, log_callback, cost_tracker=None, max_requests=None, timeout_seconds=None):
    assert max_requests == 8
    assert timeout_seconds == 180.0
    return __import__('json').dumps(payload)

async def fake_log(message):
    pass

result = asyncio.run(run_opencode_micro_agent(
    '/tmp/a.mp3', 2.0, [{'start':0,'end':1,'midi':60}], fake_runner, fake_log, 'model', 2,
    max_requests=8, timeout_seconds=180.0,
))
assert result['promoted'] is False and len(result['events']) == 1
print('opencode adapter passed')
