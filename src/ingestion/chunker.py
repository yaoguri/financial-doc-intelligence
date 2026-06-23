"""
Text chunking module.

Splits extracted document text into overlapping chunks suitable
for embedding and retrieval.

Design decision: we chunk by character count rather than token count
because token counts are model-dependent. At inference time, the
embedding model handles its own truncation. Character-based chunking
is portable across models.

The overlap ensures that sentences spanning chunk boundaries appear
fully in at least one chunk, preventing information loss at edges.
"""

from dataclasses import dataclass
from loguru import logger

from src.config import settings


@dataclass
class TextChunk:
    """A single chunk of text with its position metadata."""
    chunk_index: int        # Position of this chunk within the document
    text: str               # The chunk content
    start_char: int         # Character offset where this chunk begins
    end_char: int           # Character offset where this chunk ends
    page_numbers: list[int] # Which pages this chunk spans (approximate)

    @property
    def char_count(self) -> int:
        return len(self.text)


class TextChunker:
    """
    Splits document text into overlapping fixed-size chunks.

    Uses a sliding window approach:
    - Window size = chunk_size characters
    - Each window advances by (chunk_size - overlap) characters
    - Result: adjacent chunks share 'overlap' characters

    Example with chunk_size=20, overlap=5:
    Text:   "The quick brown fox jumped over the lazy dog"
    Chunk1: "The quick brown fox "     (chars 0-19)
    Chunk2: "fox jumped over the "     (chars 15-34)  <- 5 char overlap
    Chunk3: "the lazy dog"             (chars 30-end)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        # Fall back to config values if not explicitly provided
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.stride = self.chunk_size - self.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )

        logger.debug(
            f"TextChunker initialised: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, stride={self.stride}"
        )

    def chunk_text(
        self,
        text: str,
        page_map: dict[int, int] | None = None,
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: The full document text to chunk.
            page_map: Optional dict mapping char_offset -> page_number,
                      used to tag chunks with their source page numbers.

        Returns:
            List of TextChunk objects.
        """
        if not text.strip():
            logger.warning("Empty text passed to chunker")
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Skip chunks that are only whitespace
            if chunk_text.strip():
                page_numbers = self._get_page_numbers(start, end, page_map)
                chunks.append(TextChunk(
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    page_numbers=page_numbers,
                ))
                chunk_index += 1

            # Advance by stride (chunk_size - overlap)
            start += self.stride

        logger.info(
            f"Chunked {len(text)} chars into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _get_page_numbers(
        self,
        start: int,
        end: int,
        page_map: dict[int, int] | None,
    ) -> list[int]:
        """
        Determine which pages a character range spans.

        Args:
            start: Start character offset.
            end: End character offset.
            page_map: Dict of {char_offset: page_number} boundaries.

        Returns:
            Sorted list of page numbers this chunk touches.
        """
        if not page_map:
            return []

        pages = set()
        for char_offset, page_num in page_map.items():
            if start <= char_offset < end:
                pages.add(page_num)

        return sorted(pages)