import re
from datetime import datetime

from memask.router.date_range import DateRange, resolve as resolve_date
from memask.router.intents import QueryContext

DATE_EXPRESSION_PATTERNS = [
    re.compile(r"\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(?:this|last)\s+(?:week|month|year)\b", re.I),
    re.compile(r"[+-]\d+[dwmy]", re.I),
    re.compile(r"\b(?:today|yesterday|tomorrow)\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]

TRAILING_PREPOSITION_RE = re.compile(
    r"\b(?:from|since|for|in|on|during|until|before|after)\s*$",
    re.I,
)

DATE_HINT_PATTERNS = [
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

SEARCH_PREAMBLES = [
    re.compile(
        r"^what\s+did\s+i\s+(?:write|note|save|record|put|jot|capture)\s+"
        r"(?:down\s+)?(?:about|on|regarding)\s+",
        re.I,
    ),
    re.compile(
        r"^what\s+(?:was|were)\s+(?:my|the|that)\s+"
        r"(?:notes?|thoughts?|ideas?)\s+(?:about|on|regarding)\s+",
        re.I,
    ),
    re.compile(
        r"^what\s+(?:about|was\s+that\s+(?:thing|stuff)\s+about)\s+",
        re.I,
    ),
    re.compile(
        r"^(?:find|search\s+for|look\s+up|look\s+for)\s+"
        r"(?:my\s+)?(?:notes?\s+(?:about|on|regarding)\s+)?",
        re.I,
    ),
    re.compile(
        r"^show\s+me\s+(?:my\s+)?(?:notes?\s+(?:about|on|regarding)\s+)?",
        re.I,
    ),
    re.compile(
        r"^(?:my\s+)?(?:notes?|thoughts?|ideas?)\s+(?:about|on|regarding)\s+",
        re.I,
    ),
    re.compile(r"^anything\s+(?:about|on|regarding)\s+", re.I),
    re.compile(
        r"^(?:what|where|when|how|who|which|why)\s+did\s+i\s+",
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


def extract_query_context(text: str, now: datetime | None = None) -> QueryContext:
    raw_query = _strip_prefixes(text)
    date_range, cleaned_query = _extract_and_resolve_dates(raw_query, now)
    topic = _extract_topic(cleaned_query)
    cleaned_query = _strip_search_preamble(cleaned_query)
    return QueryContext(
        date_hints=_extract_date_hints(text),
        date_range=date_range,
        topic=topic,
        type_filter=_extract_type_filter(text),
        status_filter=_extract_status_filter(text),
        raw_query=cleaned_query,
    )


def _strip_prefixes(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("?", "!", "/")):
        stripped = stripped[1:]
    return stripped.strip()


def _strip_search_preamble(text: str) -> str:
    cleaned = text
    for pattern in SEARCH_PREAMBLES:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip().rstrip("?.,!")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.I)
    return cleaned.strip()


def _extract_and_resolve_dates(text, now=None):
    cleaned = text
    resolved = None

    for pattern in DATE_EXPRESSION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            expr = match.group(0).strip()
            result = resolve_date(expr, now)
            if result:
                resolved = result
                before = cleaned[: match.start()].rstrip()
                after = cleaned[match.end() :].lstrip()
                before = TRAILING_PREPOSITION_RE.sub("", before).rstrip()
                cleaned = f"{before} {after}".strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                break

    return resolved, cleaned


def _extract_date_hints(text: str) -> list[str]:
    hints = []
    for pattern in DATE_HINT_PATTERNS:
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
    for word, type_value in TYPE_MAP.items():
        if re.search(rf"\b{word}\b", lower):
            return type_value
    return None


def _extract_status_filter(text: str) -> str | None:
    lower = text.lower()
    for word, status_value in STATUS_MAP.items():
        if re.search(rf"\b{word}\b", lower):
            return status_value
    return None
