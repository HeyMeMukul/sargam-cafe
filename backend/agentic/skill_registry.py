"""Skill and knowledge retrieval for the pianist agent.

This is deliberately deterministic and auditable first. It can later be
replaced or augmented by embeddings, but every returned excerpt keeps the
source file and JSON path so the agent cannot cite anonymous prompt prose.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillExcerpt:
    skill_id: str
    source_path: str
    title: str
    purpose: str
    rule: str
    score: float
    matched_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "source_path": self.source_path,
            "title": self.title,
            "purpose": self.purpose,
            "rule": self.rule,
            "score": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
        }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}


def _flatten_rules(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_flatten_rules(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten_rules(item, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        rows.append((prefix, value))
    return rows


class SkillRegistry:
    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._skills: list[dict[str, Any]] = []
        for path in sorted(self.skills_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            data["_source_path"] = str(path)
            data["_skill_id"] = path.stem
            data["_rules"] = _flatten_rules(data)
            self._skills.append(data)

    def retrieve(self, query: str, limit: int = 4) -> list[SkillExcerpt]:
        query_terms = _tokens(query)
        candidates: list[SkillExcerpt] = []
        for skill in self._skills:
            title = str(skill.get("title", skill["_skill_id"]))
            purpose = str(skill.get("purpose", ""))
            text = " ".join([title, purpose] + [value for _, value in skill["_rules"]])
            skill_terms = _tokens(text)
            matched = tuple(sorted(query_terms & skill_terms))
            score = float(len(matched))
            if any(term in title.lower() for term in query_terms):
                score += 2.0
            if score <= 0:
                continue
            rule = ""
            preferred = ("rule", "how_to_use", "prohibitions")
            for key in preferred:
                if isinstance(skill.get(key), str):
                    rule = skill[key]
                    break
            if not rule:
                for path, value in skill["_rules"]:
                    if any(part in path.lower() for part in ("rule", "prohibition", "confidence")):
                        rule = value
                        break
            candidates.append(SkillExcerpt(
                skill_id=skill["_skill_id"],
                source_path=skill["_source_path"],
                title=title,
                purpose=purpose,
                rule=rule,
                score=score,
                matched_terms=matched,
            ))
        candidates.sort(key=lambda item: (-item.score, item.skill_id))
        return candidates[: max(0, limit)]

    def citation_bundle(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        return [excerpt.as_dict() for excerpt in self.retrieve(query, limit)]

    def __len__(self) -> int:
        return len(self._skills)
