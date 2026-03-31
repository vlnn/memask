import re

from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context

TODO_PREFIX = re.compile(r"^/todo\b", re.I)
TODO_BARE = re.compile(r"^/todo\s*$", re.I)
TODO_LIST_SUFFIX = re.compile(r"^/todo\s+(?:list|ls)\s*$", re.I)
TODO_DONE_SUFFIX = re.compile(r"^/todo\s+(?:done|complete)\s+", re.I)
TODO_ADD_SUFFIX = re.compile(r"^/todo\s+(?:add\s+)?", re.I)

DONE_PREFIX = re.compile(r"^/done\s+", re.I)
DONE_BARE = re.compile(r"^/done\s*$", re.I)
UNDONE_PREFIX = re.compile(r"^/undone\s+", re.I)

APP_BANG = re.compile(r"^!", re.I)
APP_SLASH = re.compile(r"^/(?!todo\b|done\b|undone\b)(\w+)", re.I)
QUESTION_PREFIX = re.compile(r"^\?", re.I)

STANDALONE_DATE_OFFSET = re.compile(r"^[+-]\d+[dwmy]$", re.I)
STANDALONE_NAMED_DATE = re.compile(
    r"^(?:today|yesterday|tomorrow"
    r"|(?:this|last)\s+(?:week|month|year)"
    r"|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|\d{4}-\d{2}-\d{2})$",
    re.I,
)

QUESTION_STARTERS = re.compile(
    r"^(?:what|when|where|who|how|which|why)\b.*\??\s*$",
    re.I,
)

SEARCH_KEYWORDS = re.compile(
    r"\b(?:find|search\s+for|look\s+up|show\s+me"
    r"|what\s+did\s+i\s+(?:note|write|save|record))\b",
    re.I,
)

TODO_KEYWORDS = re.compile(
    r"^(?:remind\s+(?:me\s+)?(?:to\s+|about\s+)?"
    r"|todos?[:\s]"
    r"|add\s+todo\b)",
    re.I,
)

TODO_SOFT_KEYWORDS = re.compile(
    r"^(?:i\s+need\s+to|don'?t\s+forget\s+to"
    r"|remember\s+to|i\s+have\s+to"
    r"|i\s+must|i\s+should)\b",
    re.I,
)

NL_COMPLETION = re.compile(
    r"(?:^finished\s+|^i\s+completed\s+|^i\s+already\s+|^mark\s+.+\s+as\s+done$"
    r"|\bis\s+(?:done|complete)\s*$"
    r"|\bis\s+(?:done|complete)\s*[\?!.]?\s*$)",
    re.I,
)

NL_DELETION = re.compile(
    r"(?:^remove\s+.+\s+from\s+my\s+(?:todos?|tasks?|list)"
    r"|^delete\s+(?:the\s+)?(?:todo|task|my\s+todo)\b"
    r"|^cancel\s+the\s+.+\s+(?:todo|task|reminder)"
    r"|^delete\s+my\s+todo\b"
    r"|^remove\s+the\s+.+\s+(?:task|todo)\s+from\s+my\s+list)",
    re.I,
)

NL_TODO_LIST = re.compile(
    r"(?:^what(?:'s| is)\s+on\s+my\s+(?:todo\s+)?list"
    r"|^show\s+(?:me\s+)?my\s+(?:todos?|tasks?)"
    r"|^any\s+pending\s+(?:todos?|tasks?)"
    r"|^what\s+do\s+i\s+need\s+to\s+do"
    r"|^what\s+are\s+my\s+(?:pending\s+)?(?:todos?|tasks?)"
    r"|^show\s+me\s+my\s+(?:todos?|tasks?))",
    re.I,
)

NL_HELP = re.compile(
    r"(?:^what\s+can\s+you\s+do"
    r"|^help\s+me$"
    r"|^how\s+does\s+this\s+work)",
    re.I,
)


def classify_by_rules(text: str) -> RoutingResult:
    stripped = text.strip()
    ctx = extract_query_context(stripped)

    if TODO_LIST_SUFFIX.match(stripped):
        return _result(Intent.TODO_LIST, Confidence.HIGH, text, ctx)

    if TODO_BARE.match(stripped):
        return _result(Intent.TODO_LIST, Confidence.HIGH, text, ctx)

    if TODO_DONE_SUFFIX.match(stripped):
        return _result(Intent.TODO_COMPLETE, Confidence.HIGH, text, ctx)

    if TODO_PREFIX.match(stripped):
        return _result(Intent.TODO_CREATE, Confidence.HIGH, text, ctx)

    if DONE_PREFIX.match(stripped):
        return _result(Intent.TODO_COMPLETE, Confidence.HIGH, text, ctx)

    if DONE_BARE.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, ctx)

    if UNDONE_PREFIX.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, ctx)

    if APP_BANG.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, ctx)

    if APP_SLASH.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.HIGH, text, ctx)

    if QUESTION_PREFIX.match(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, ctx)

    if STANDALONE_DATE_OFFSET.match(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, ctx)

    if STANDALONE_NAMED_DATE.match(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, ctx)

    if TODO_KEYWORDS.match(stripped):
        return _result(Intent.TODO_CREATE, Confidence.HIGH, text, ctx)

    if NL_TODO_LIST.match(stripped):
        return _result(Intent.TODO_LIST, Confidence.MEDIUM, text, ctx)

    if NL_HELP.match(stripped):
        return _result(Intent.APP_COMMAND, Confidence.MEDIUM, text, ctx)

    if NL_COMPLETION.search(stripped):
        return _result(Intent.TODO_COMPLETE, Confidence.MEDIUM, text, ctx)

    if NL_DELETION.match(stripped):
        return _result(Intent.TODO_DELETE, Confidence.MEDIUM, text, ctx)

    if TODO_SOFT_KEYWORDS.match(stripped):
        return _result(Intent.TODO_CREATE, Confidence.MEDIUM, text, ctx)

    if SEARCH_KEYWORDS.search(stripped):
        return _result(Intent.SEARCH, Confidence.HIGH, text, ctx)

    if QUESTION_STARTERS.match(stripped):
        return _result(Intent.SEARCH, Confidence.MEDIUM, text, ctx)

    if _has_query_signals(stripped):
        return _result(Intent.SEARCH, Confidence.LOW, text, ctx)

    return _result(Intent.CAPTURE, Confidence.MEDIUM, text, ctx)


def _has_query_signals(text: str) -> bool:
    lower = text.lower()
    query_signals = [
        r"\b(?:notes?|thoughts?|ideas?)\s+(?:about|on|from)\b",
        r"\b(?:about|regarding)\s+\w+\s+(?:yesterday|today|last)\b",
    ]
    return any(re.search(p, lower) for p in query_signals)


def _result(intent, confidence, raw_input, query_context):
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        query_context=query_context,
        raw_input=raw_input,
        source="rules",
    )
