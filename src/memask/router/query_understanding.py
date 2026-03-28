import re

from memask.router.intents import QueryContext

DATE_PATTERNS = [
    re.compile(r"\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(?:this|last)\s+(?:week|month|morning|evening|night)\b", re.I),
    re.compile(r"\byesterday\b", re.I),
    re.compile(r"\btoday\b", re.I),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I),
]

TOPIC_PATTERNS = [
    re.compile(r"\babout\s+(?:the\s+)?(.+?)(?:\s+(?:yesterday|today|last|this|from)\b|[?.]|$)", re.I),
    re.compile(r"\bon\s+(?:the\s+)?(.+?)(?:\s+(?:yesterday|today|last|this|from)\b|[?.]|$)", re.I),
    re.compile(r"\bfor\s+(?:the\s+)?(.+?)(?:\s+(?:yesterday|today|last|this|from)\b|[?.]|$)", re.I),
]

TYPE_MAP = {
    "todo": "todo",
    "todos": "todo",
    "task": "todo",
    "tasks": "todo",
    "note": "note",
    "notes": "note",
    "decision": "decision",
    "decisions": "decision",
    "guide": "guide",
    "guides": "guide",
}

STATUS_MAP = {
    "pending": "pending",
    "open": "pending",
    "done": "done",
    "completed": "done",
    "complete": "done",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def extract_query_context(text: str) -> QueryContext:
    raw_query = _strip_prefix(text)
    date_hints = _extract_date_hints(raw_query)
    topic = _extract_topic(raw_query)
    type_filter = _extract_type_filter(raw_query)
    status_filter = _extract_status_filter(raw_query)

    return QueryContext(
        date_hints=date_hints,
        topic=topic,
        type_filter=type_filter,
        status_filter=status_filter,
        raw_query=raw_query,
    )


def _strip_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("?", "!", "/")):
        stripped = stripped[1:]
    return stripped.strip()


def _extract_date_hints(text: str) -> list[str]:
    hints = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            hints.append(match.group(0).lower())
    return hints


def _extract_topic(text: str) -> str | None:
    for pattern in TOPIC_PATTERNS:
        match = pattern.search(text)
        if match:
            topic = match.group(1).strip().rstrip("?.,!")
            if topic and len(topic) > 1:
                return topic
    return None


def _extract_type_filter(text: str) -> str | None:
    lower = text.lower()
    for keyword, item_type in TYPE_MAP.items():
        if re.search(rf"\b{keyword}\b", lower):
            return item_type
    return None


def _extract_status_filter(text: str) -> str | None:
    lower = text.lower()
    for keyword, status in STATUS_MAP.items():
        if re.search(rf"\b{keyword}\b", lower):
            return status
    return None
