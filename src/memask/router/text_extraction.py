import re


def extract_todo_content(text: str) -> str:
    stripped = text.strip()

    prefixed = re.match(r"^/todo\s+(?:add\s+)?(.+)$", stripped, re.I)
    if prefixed:
        return prefixed.group(1).strip()

    for pattern in [
        re.compile(r"^remind\s+me\s+(?:to\s+|about\s+)(.+)$", re.I),
        re.compile(r"^remind\s+(?:to\s+|about\s+)(.+)$", re.I),
        re.compile(r"^remind\s+(.+)$", re.I),
        re.compile(r"^todos?[:\s]+(.+)$", re.I),
        re.compile(r"^add\s+todo\s+(.+)$", re.I),
        re.compile(
            r"^(?:i\s+need\s+to|don'?t\s+forget\s+to|remember\s+to|i\s+have\s+to|i\s+must|i\s+should)\s+(.+)$",
            re.I,
        ),
    ]:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()

    return ""


def extract_complete_query(text: str) -> str:
    for pattern in [
        re.compile(r"^/todo\s+(?:done|complete)\s+(.+)$", re.I),
        re.compile(r"^/done\s+(.+)$", re.I),
        re.compile(r"^finished\s+(.+)$", re.I),
        re.compile(r"^i\s+completed\s+(.+)$", re.I),
        re.compile(r"^i\s+already\s+(.+)$", re.I),
        re.compile(r"^mark\s+(.+?)\s+as\s+done$", re.I),
        re.compile(
            r"^(?:my\s+)?task\s+(?:about\s+)?(.+?)"
            r"\s+is\s+(?:done|complete)\s*$",
            re.I,
        ),
        re.compile(r"^(.+?)\s+is\s+(?:done|complete)\s*$", re.I),
    ]:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return ""


def extract_delete_query(text: str) -> str:
    for pattern in [
        re.compile(r"^remove\s+(.+?)\s+from\s+my\s+(?:todos?|tasks?|list)", re.I),
        re.compile(r"^delete\s+(?:the\s+)?(?:todo|task)\s+(?:about\s+)?(.+)$", re.I),
        re.compile(r"^delete\s+(?:my\s+)?(?:todo|task)\s+(?:about\s+)?(.+)$", re.I),
        re.compile(r"^cancel\s+the\s+(.+?)\s+(?:todo|task|reminder)$", re.I),
        re.compile(r"^remove\s+the\s+(.+?)\s+(?:task|todo)\s+from\s+my\s+list$", re.I),
        re.compile(r"^delete\s+(.+)$", re.I),
        re.compile(r"^remove\s+(.+)$", re.I),
    ]:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return ""


UPDATE_PATTERNS = [
    re.compile(r"^(?:change|update)\s+(?:the\s+)?(.+?)\s+(?:todo\s+)?to\s+(.+)$", re.I),
    re.compile(r"^reschedule\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
    re.compile(r"^rename\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
    re.compile(r"^set\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
]


def extract_update_parts(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    for pattern in UPDATE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return stripped, None


def extract_undone_query(text: str) -> str:
    m = re.match(r"^/undone\s+(.+)$", text.strip(), re.I)
    return m.group(1).strip() if m else ""


def extract_command_name(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^[!/](\w+)", stripped)
    if match:
        return match.group(1).lower()
    return stripped.lower()


def extract_list_type(text: str) -> str | None:
    m = re.match(r"^/list\s+(\w+)", text.strip(), re.I)
    if not m:
        return None
    word = m.group(1).lower()
    type_map = {"notes": "note", "note": "note", "todos": "todo", "todo": "todo"}
    return type_map.get(word)


def is_nl_help(text: str) -> bool:
    return bool(re.match(
        r"(?:what\s+can\s+you\s+do|help\s+me|how\s+does\s+this\s+work)",
        text.strip(),
        re.I,
    ))


def split_multi_todo(content: str) -> list[str]:
    normalized = re.sub(r",\s*and\s+", ", ", content)
    if "," not in normalized:
        return [content.strip()]
    parts = [p.strip() for p in normalized.split(",")]
    return [p for p in parts if p]


def synthesis_question(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("?"):
        stripped = stripped[1:].strip()
    return stripped
