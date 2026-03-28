from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP_FRACTION = 0.15


@dataclass(frozen=True)
class Chunk:
    item_id: str
    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    item_id: str,
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [Chunk(item_id=item_id, index=0, text=text, start_char=0, end_char=len(text))]

    overlap = int(chunk_size * overlap_fraction)
    step = chunk_size - overlap
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]

        if chunk_text_slice.strip():
            chunks.append(Chunk(
                item_id=item_id,
                index=index,
                text=chunk_text_slice,
                start_char=start,
                end_char=end,
            ))
            index += 1

        if end == len(text):
            break

        start += step

    return chunks


def needs_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    return len(text) > chunk_size
