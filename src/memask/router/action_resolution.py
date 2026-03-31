from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from memask.models.item import Item

logger = logging.getLogger(__name__)

RESOLVE_UPDATE_SYSTEM = (
    "You are a todo-list assistant. "
    "Given a user instruction and a list of candidate todo items, "
    "determine which item the user wants to update and what the new content should be. "
    "Respond with ONLY a JSON object, no explanation, no markdown:\n"
    '{"target_id": "<id of the item to update>", "new_content": "<updated text>"}\n'
    "If you cannot determine the target, respond with:\n"
    '{"target_id": null, "new_content": null}'
)

RESOLVE_UPDATE_PROMPT = (
    "User instruction: {text}\n\n"
    "Candidate items:\n"
    "{candidates}\n\n"
    "Which item should be updated, and what should the new content be?"
)


@dataclass(frozen=True)
class ActionPlan:
    target_id: str
    new_content: str


def resolve_update(text: str, candidates: list[Item], llm) -> ActionPlan | None:
    if not candidates:
        return None

    formatted = _format_candidates(candidates)
    prompt = RESOLVE_UPDATE_PROMPT.format(text=text, candidates=formatted)

    try:
        raw = llm.generate(prompt, system=RESOLVE_UPDATE_SYSTEM)
        return _parse_action_plan(raw, candidates)
    except Exception:
        logger.debug("resolve_update failed, returning None")
        return None


def _format_candidates(candidates: list[Item]) -> str:
    lines = []
    for item in candidates:
        lines.append(f"- id={item.id} content=\"{item.content}\"")
    return "\n".join(lines)


def _parse_action_plan(raw: str, candidates: list[Item]) -> ActionPlan | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.debug("failed to parse LLM response as JSON: %s", cleaned[:100])
        return None

    target_id = data.get("target_id")
    new_content = data.get("new_content")

    if not target_id or not new_content:
        return None

    valid_ids = {c.id for c in candidates}
    if target_id not in valid_ids:
        return None

    return ActionPlan(target_id=target_id, new_content=new_content)
