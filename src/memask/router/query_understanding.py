import re

from memask.router.intents import QueryContext

DATE_PATTERNS = [
    re.compile(r"\b(?:yesterday|today|tomorrow)\b", re.I),
    re.compile(
        r"\blast\s+(?:week|month|year|monday|tuesday|"
        r"wednesday|thursday|friday|saturday|sunday)\b",
        re.I,
    ),
    re.compile(
        r"\bthis\s+(?:week|month|year|morning|afternoon|evening)\b",
        re.I,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(
        r"\b(?:january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b",
        re.I,
    ),
]

TOPIC_PATTERNS = [
    re.compile(
        r"(?:search|looking|look)\s+for\s+(.+?)(?:\s+(?:yesterday|today|last|this|from|since)|\?|$)",
        re.I,
    ),
    re.compile(
        r"(?:notes?|thoughts?|ideas?)\s+(?:about|on|regarding)\s+(.+?)(?:\s+(?:yesterday|today|last|this)|\?|$)",
        re.I,
    ),
    re.compile(
        r"(?:about|regarding|on|re)\s+(.+?)(?:\s+(?:yesterday|today|last|this|from|since)|\?|$)",
        re.I,
    ),
]

TYPE_MAP = {
    "note": "note",
    "notes": "note",
    "todo": "todo",
    "todos": "todo",
    "task": "todo",
    "tasks": "todo",
    "decision": "decision",
    "decisions": "decision",
    "guide": "guide",
    "guides": "guide",
    "url": "url",
    "urls": "url",
    "link": "url",
    "links": "url",
}

STATUS_MAP = {
    "pending": "pending",
    "open": "pending",
    "done": "done",
    "completed": "done",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def extract_query_context(text: str) -> QueryContext:
    raw_query = _strip_prefixes(text)
    return QueryContext(
        date_hints=_extract_date_hints(text),
        topic=_extract_topic(text),
        type_filter=_extract_type_filter(text),
        status_filter=_extract_status_filter(text),
        raw_query=raw_query,
    )


def _strip_prefixes(text: str) -> str:
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
            topic = _strip_articles(topic)
            if topic and len(topic) > 1:
                return topic
    return None


def _strip_articles(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)


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
