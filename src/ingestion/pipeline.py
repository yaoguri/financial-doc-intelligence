"""
Ingestion pipeline — updated for Phase 3 to embed chunks after storage.

Stage sequence:
  1. Parse     — extract text pages from PDF (PDFParser)
  2. Chunk     — split pages into overlapping text chunks (TextChunker)
  3. NER       — extract named entities from chunks (NERExtractor)
  4. Store     — persist document, chunks, entities to PostgreSQL
  5. Embed     — embed chunks via Ollama, persist to embeddings table (Phase 3)

The embedder and embedding_repo are optional so the pipeline remains usable
in Phase 1/2 tests that run without a live Ollama instance.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import TextChunker
from src.ingestion.ner import NERExtractor
from src.ingestion.ocr import OCRProcessor
from src.ingestion.parser import PDFParser
from src.storage.models import Chunk, Document
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingRepository,
    EntityRepository,
)
from src.embeddings.embedder import EmbeddingService, EmbeddingError

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Summary of a completed ingestion run.

    Returned by IngestionPipeline.ingest() so callers have visibility into
    what happened at each stage without parsing log output.
    """
    document_id: str
    total_pages: int
    ocr_pages: int
    total_chunks: int
    embedded_chunks: int = 0
    entity_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow.

    Dependencies are injected so each stage is independently testable and
    swappable. The embedding stage is optional — if embedder or embedding_repo
    are not provided, Stage 5 is skipped and embedded_chunks stays 0.

    Args:
        document_repo: Persists Document records.
        chunk_repo: Persists Chunk records.
        entity_repo: Persists Entity records.
        embedding_repo: Persists Embedding records (optional, Phase 3+).
        embedder: Calls Ollama to produce vectors (optional, Phase 3+).
        chunk_size: Target chunk size in characters (default 1000).
        chunk_overlap: Overlap between adjacent chunks (default 200).
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        entity_repo: EntityRepository,
        embedding_repo: Optional[EmbeddingRepository] = None,
        embedder: Optional[EmbeddingService] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.entity_repo = entity_repo
        self.embedding_repo = embedding_repo
        self.embedder = embedder

        self.parser = PDFParser()
        self.ocr = OCRProcessor()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.ner = NERExtractor()

        self._embedding_enabled = (
            self.embedder is not None and self.embedding_repo is not None
        )
        if not self._embedding_enabled:
            logger.info(
                "Embedding stage disabled — embedder or embedding_repo not provided. "
                "Pass both to enable Phase 3 embedding."
            )

    def ingest(self, pdf_path: Path, source_name: Optional[str] = None) -> IngestionResult:
        """
        Run the full ingestion pipeline on a single PDF.

        Args:
            pdf_path: Path to the PDF file.
            source_name: Optional display name for the document (defaults to filename).

        Returns:
            IngestionResult with counts and any non-fatal error messages.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source_name = source_name or pdf_path.name
        logger.info(f"Starting ingestion: {source_name}")

        # ── Stage 1: Parse ────────────────────────────────────────────────────
        logger.info("Stage 1/5: Parsing PDF")
        pages = self.parser.parse(pdf_path)
        total_pages = len(pages)
        logger.info(f"  Extracted {total_pages} pages")

        # ── Stage 2: OCR (selective — only scanned pages) ────────────────────
        logger.info("Stage 2/5: OCR (scanned pages only)")
        ocr_pages = 0
        processed_pages = []
        for page in pages:
            if page.needs_ocr:
                text = self.ocr.process_page(pdf_path, page.page_number)
                page.text = text
                ocr_pages += 1
            processed_pages.append(page)
        logger.info(f"  OCR applied to {ocr_pages}/{total_pages} pages")

        # ── Stage 3: Chunk ────────────────────────────────────────────────────
        logger.info("Stage 3/5: Chunking text")
        all_text = "\n".join(p.text for p in processed_pages if p.text)
        raw_chunks = self.chunker.chunk(all_text)
        logger.info(f"  Produced {len(raw_chunks)} chunks")

        # ── Stage 4: Store document, chunks, entities ─────────────────────────
        logger.info("Stage 4/5: Storing to database")
        document = self.document_repo.create(
            filename=pdf_path.name,
            source=source_name,
            total_pages=total_pages,
        )

        stored_chunks: list[Chunk] = []
        entity_count = 0
        errors: list[str] = []

        for i, chunk_text in enumerate(raw_chunks):
            try:
                chunk = self.chunk_repo.create(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                )
                stored_chunks.append(chunk)

                # NER per chunk
                entities = self.ner.extract(chunk_text)
                for entity in entities:
                    self.entity_repo.create(
                        chunk_id=chunk.id,
                        text=entity.text,
                        label=entity.label,
                    )
                entity_count += len(entities)

            except Exception as e:
                msg = f"Failed to store chunk {i}: {e}"
                logger.error(msg)
                errors.append(msg)

        logger.info(
            f"  Stored {len(stored_chunks)} chunks, {entity_count} entities"
        )

        # ── Stage 5: Embed (Phase 3) ──────────────────────────────────────────
        embedded_chunks = 0

        if self._embedding_enabled and stored_chunks:
            logger.info("Stage 5/5: Embedding chunks")
            chunk_texts = [c.content for c in stored_chunks]

            try:
                vectors = self.embedder.embed_chunks(chunk_texts)

                for chunk, vector in zip(stored_chunks, vectors):
                    self.embedding_repo.create(
                        chunk_id=chunk.id,
                        embedding=vector,
                    )
                    embedded_chunks += 1

                logger.info(f"  Embedded and stored {embedded_chunks} chunk vectors")

            except EmbeddingError as e:
                msg = f"Embedding stage failed: {e}"
                logger.error(msg)
                errors.append(msg)
                # Non-fatal: document and chunks are already stored.
                # The embedding stage can be re-run separately.

        else:
            logger.info("Stage 5/5: Embedding skipped (not configured)")

        result = IngestionResult(
            document_id=str(document.id),
            total_pages=total_pages,
            ocr_pages=ocr_pages,
            total_chunks=len(stored_chunks),
            embedded_chunks=embedded_chunks,
            entity_count=entity_count,
            errors=errors,
        )

        logger.info(
            f"Ingestion complete: {result.total_chunks} chunks, "
            f"{result.embedded_chunks} embedded, "
            f"{result.entity_count} entities, "
            f"{'OK' if result.success else f'{len(errors)} errors'}"
        )

        return result
