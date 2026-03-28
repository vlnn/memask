import re

from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context

TODO_PREFIX = re.compile(r"^/todo\b", re.I)
TODO_LIST_SUFFIX = re.compile(r"^/todo\s+(?:list|ls)\s*$", re.I)
TODO_DONE_SUFFIX = re.compile(r"^/todo\s+(?:done|complete)\s+", re.I)
TODO_ADD_SUFFIX = re.compile(r"^/todo\s+(?:add\s+)?", re.I)

APP_BANG = re.compile(r"^!", re.I)
APP_SLASH = re.compile(r"^/(?!todo\b)(\w+)", re.I)
QUESTION_PREFIX = re.compile(r"^\?", re.I)

QUESTION_STARTERS = re.compile(
    r"^(?:what|when|where|who|how|which|why)\b.*\??\s*$", re.I,
)

SEARCH_KEYWORDS = re.compile(
    r"^(?:find|search\s+(?:for|my|the|about|in)|look\s+up|show\s+me)\b"
    r"|\bwhat\s+did\s+i\s+(?:note|write|save|record)\b",
    re.I,
)

TODO_KEYWORDS = re.compile(
    r"^(?:remind\s+(?:me\s+)?(?:to\s+|about\s+)?|todo[:\s]|add\s+todo\s)", re.I,
)

TODO_SOFT_KEYWORDS = re.compile(
    r"\b(?:i\s+need\s+to|don'?t\s+forget\s+to|remember\s+to|i\s+have\s+to|i\s+must|i\s+should)\b", re.I,
)


def classify_by_rules(text: str) -> RoutingResult:
    stripped = text.strip()
    lower = stripped.lower()
    query_context = extract_query_context(text)

    if TODO_LIST_SUFFIX.match(stripped):
        return _result(Intent.TODO_LIST, Confidence.HIGH, text, query_context)

    if TODO_DONE_SUFFIX.match(stripped):
        return _result(Intent.TODO_COMPLETE, Confidence.HIGH, text, query_context)

    if TODO_PREFIX.match(stripped):
        return _result(Intent.TODO_CREATE, Confidence.HIGH, text, query_context)

    if APP_BANG.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, query_context)

    if APP_SLASH.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, query_context)

    if QUESTION_PREFIX.match(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, query_context)

    if SEARCH_KEYWORDS.search(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, query_context)

    if QUESTION_STARTERS.match(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, query_context)

    if TODO_KEYWORDS.match(stripped):
        return _result(Intent.TODO_CREATE, Confidence.HIGH, text, query_context)

    if TODO_SOFT_KEYWORDS.search(stripped):
        return _result(Intent.TODO_CREATE, Confidence.MEDIUM, text, query_context)

    if _looks_like_query(lower):
        return _result(Intent.SEARCH, Confidence.LOW, text, query_context)

    return _result(Intent.CAPTURE, Confidence.LOW, text, query_context)


def _looks_like_query(lower: str) -> bool:
    query_signals = [
        r"\bnotes?\s+(?:about|on|from)\b",
        r"\b(?:about|regarding)\s+\w+\s+(?:yesterday|today|last)\b",
    ]
    return any(re.search(p, lower) for p in query_signals)


def _result(
    intent: Intent,
    confidence: Confidence,
    raw_input: str,
    query_context,
) -> RoutingResult:
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        query_context=query_context,
        raw_input=raw_input,
        source="rules",
    )
