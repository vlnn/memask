from dataclasses import dataclass, field

from memask.search.keyword import SearchResult

DEFAULT_TOP_N = 10
DEFAULT_MAX_CHARS = 4000


@dataclass(frozen=True)
class ContextBlock:
    text: str
    sources: list[str] = field(default_factory=list)


def assemble_context(
    results: list[SearchResult],
    query: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ContextBlock:
    if not results:
        return ContextBlock(text="", sources=[])

    selected = results[:top_n]
    blocks = [_format_result(i + 1, r) for i, r in enumerate(selected)]
    sources = [r.item.id for r in selected]

    joined = "\n\n".join(blocks)
    if len(joined) <= max_chars:
        return ContextBlock(text=joined, sources=sources)

    return _truncate(selected, max_chars)


def _format_result(index: int, result: SearchResult) -> str:
    item = result.item
    header = f"[{index}] ({item.type}, {_short_date(item.created_at)}) id: {item.id}"
    parts = [header]
    if item.title:
        parts.append(f"Title: {item.title}")
    parts.append(item.content)
    return "\n".join(parts)


def _format_result_truncated(index: int, result: SearchResult, max_content: int) -> str:
    item = result.item
    header = f"[{index}] ({item.type}, {_short_date(item.created_at)}) id: {item.id}"
    parts = [header]
    if item.title:
        parts.append(f"Title: {item.title}")
    content = item.content
    if len(content) > max_content:
        content = content[:max_content].rstrip() + " [truncated]"
    parts.append(content)
    return "\n".join(parts)


def _truncate(results: list[SearchResult], max_chars: int) -> ContextBlock:
    sources = []
    blocks = []
    remaining = max_chars

    for i, result in enumerate(results):
        header_overhead = _header_size(i + 1, result)
        separator = 2 if blocks else 0
        available = remaining - header_overhead - separator

        if available <= 0:
            break

        block = _format_result_truncated(i + 1, result, available)
        blocks.append(block)
        sources.append(result.item.id)
        remaining -= len(block) + separator

    return ContextBlock(text="\n\n".join(blocks), sources=sources)


def _header_size(index: int, result: SearchResult) -> int:
    item = result.item
    size = len(f"[{index}] ({item.type}, {_short_date(item.created_at)}) id: {item.id}\n")
    if item.title:
        size += len(f"Title: {item.title}\n")
    return size


def _short_date(iso_timestamp: str) -> str:
    return iso_timestamp[:10]
