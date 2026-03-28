import pytest

from memask.search.chunking import Chunk, chunk_text, needs_chunking


class TestNeedsChunking:
    @pytest.mark.parametrize("text,chunk_size,expected", [
        ("short", 500, False),
        ("x" * 500, 500, False),
        ("x" * 501, 500, True),
        ("", 500, False),
    ])
    def test_threshold(self, text, chunk_size, expected):
        assert needs_chunking(text, chunk_size) is expected, (
            f"text of length {len(text)} with chunk_size {chunk_size} should {'need' if expected else 'not need'} chunking"
        )


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("item-1", "hello world", chunk_size=500)
        assert len(chunks) == 1, "short text should produce one chunk"
        assert chunks[0].text == "hello world", "single chunk should contain full text"
        assert chunks[0].item_id == "item-1", "chunk should reference parent item"
        assert chunks[0].index == 0, "single chunk should have index 0"
        assert chunks[0].start_char == 0, "should start at 0"
        assert chunks[0].end_char == 11, "should end at text length"

    def test_exactly_at_threshold(self):
        text = "x" * 500
        chunks = chunk_text("item-1", text, chunk_size=500)
        assert len(chunks) == 1, "text at exact threshold should produce one chunk"

    def test_long_text_multiple_chunks(self):
        text = "word " * 200
        chunks = chunk_text("item-1", text, chunk_size=100, overlap_fraction=0.15)
        assert len(chunks) > 1, "long text should produce multiple chunks"

    def test_chunks_have_sequential_indices(self):
        text = "word " * 200
        chunks = chunk_text("item-1", text, chunk_size=100)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks))), "chunks should have sequential indices"

    def test_chunks_cover_full_text(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 40
        chunks = chunk_text("item-1", text, chunk_size=100, overlap_fraction=0.15)
        covered = set()
        for c in chunks:
            covered.update(range(c.start_char, c.end_char))
        assert covered == set(range(len(text))), "chunks should cover every character"

    def test_chunks_overlap(self):
        text = "x" * 300
        chunks = chunk_text("item-1", text, chunk_size=100, overlap_fraction=0.2)
        for i in range(len(chunks) - 1):
            current_end = chunks[i].end_char
            next_start = chunks[i + 1].start_char
            assert next_start < current_end, (
                f"chunk {i} end ({current_end}) should overlap with chunk {i+1} start ({next_start})"
            )

    def test_empty_text(self):
        chunks = chunk_text("item-1", "")
        assert chunks == [], "empty text should produce no chunks"

    def test_whitespace_only(self):
        chunks = chunk_text("item-1", "   ")
        assert chunks == [], "whitespace-only text should produce no chunks"

    def test_all_chunks_reference_parent(self):
        text = "word " * 200
        chunks = chunk_text("item-42", text, chunk_size=100)
        for c in chunks:
            assert c.item_id == "item-42", "every chunk should reference the parent item"

    def test_custom_overlap(self):
        text = "x" * 400
        small_overlap = chunk_text("item-1", text, chunk_size=100, overlap_fraction=0.1)
        large_overlap = chunk_text("item-1", text, chunk_size=100, overlap_fraction=0.3)
        assert len(large_overlap) > len(small_overlap), (
            "larger overlap should produce more chunks"
        )
