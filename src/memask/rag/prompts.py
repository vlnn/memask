from __future__ import annotations

from memask.rag.context import ContextBlock

SYSTEM_MESSAGE = (
    "You are a personal memory assistant. "
    "Answer the user's question based only on the provided context. "
    "If the context does not contain enough information, say so honestly. "
    "When you use information from a specific source, "
    "reference it by its number (e.g. [1], [2]). "
    "Keep answers concise and direct."
)


def build_system_message() -> str:
    return SYSTEM_MESSAGE


def build_answer_prompt(
    query: str,
    context: ContextBlock,
    *,
    session_history: list[dict[str, str]] | None = None,
) -> str:
    parts = []

    parts.append(_context_section(context))

    if session_history:
        parts.append(_session_section(session_history))

    parts.append(_question_section(query))

    return "\n\n".join(parts)


def _context_section(context: ContextBlock) -> str:
    if not context.text:
        return "Context:\nNo relevant notes found."
    return f"Context:\n{context.text}"


def _session_section(history: list[dict[str, str]]) -> str:
    lines = ["Conversation history:"]
    for entry in history:
        role = entry["role"].capitalize()
        lines.append(f"{role}: {entry['content']}")
    return "\n".join(lines)


def _question_section(query: str) -> str:
    return f"Question:\n{query}"
