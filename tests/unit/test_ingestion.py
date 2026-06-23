"""
Unit tests for the ingestion pipeline components.

We test each component in isolation using small synthetic inputs
rather than real PDFs. This keeps tests fast and deterministic.
"""

import pytest
from pathlib import Path
from src.ingestion.chunker import TextChunker, TextChunk
from src.ingestion.ner import NERExtractor, NERResult


class TestTextChunker:

    def test_basic_chunking(self):
        """Text is split into chunks of approximately the right size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c.text) <= 50 for c in chunks)

    def test_overlap_exists(self):
        """Adjacent chunks share overlapping content."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "abcdefghijklmnopqrstuvwxyz" * 4
        chunks = chunker.chunk_text(text)
        if len(chunks) >= 2:
            # End of chunk 0 should appear at start of chunk 1
            overlap_text = chunks[0].text[-5:]
            assert chunks[1].text.startswith(overlap_text)

    def test_empty_text_returns_empty_list(self):
        """Empty input produces no chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_short_text_produces_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text("Short text.")
        assert len(chunks) == 1

    def test_chunk_indices_are_sequential(self):
        """Chunk indices increment from 0."""
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_text("word " * 50)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_invalid_overlap_raises(self):
        """Overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=50, chunk_overlap=50)


class TestNERExtractor:

    @pytest.fixture(scope="class")
    def extractor(self):
        """Shared NER extractor (model loading is slow)."""
        return NERExtractor()

    def test_extracts_organisation(self, extractor):
        """Company names are detected as ORG entities."""
        result = extractor.extract("Apple Inc. reported record revenue.", chunk_index=0)
        orgs = result.organisations
        assert any("Apple" in org for org in orgs)

    def test_extracts_money(self, extractor):
        """Monetary values are detected as MONEY entities."""
        result = extractor.extract("Revenue reached $12.4 billion in Q3.", chunk_index=0)
        assert len(result.monetary_values) > 0

    def test_extracts_date(self, extractor):
        """Date expressions are detected."""
        result = extractor.extract("Results for fiscal year 2023 showed growth.", chunk_index=0)
        assert len(result.dates) > 0

    def test_empty_text_returns_empty_entities(self, extractor):
        """Empty input produces a result with no entities."""
        result = extractor.extract("", chunk_index=0)
        assert result.entities == []

    def test_batch_returns_correct_count(self, extractor):
        """Batch processing returns one result per input."""
        texts = [
            "Apple Inc. earned $90 billion.",
            "Microsoft reported growth in 2023.",
            "No entities here at all.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == 3