"""
Ingestion pipeline orchestrator.

Coordinates PDF parsing, OCR, chunking, and NER into a single
callable pipeline. This is the only entry point the rest of the
application needs — callers don't interact with individual
components directly.

Design principle: each stage is independently testable and
swappable. The pipeline is a thin coordinator, not a monolith.
"""

from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from src.ingestion.parser import PDFParser, ParsedDocument
from src.ingestion.ocr import OCRProcessor
from src.ingestion.chunker import TextChunker, TextChunk
from src.ingestion.ner import NERExtractor, NERResult


@dataclass
class IngestionResult:
    """Complete result of processing a single PDF document."""
    file_path: Path
    document: ParsedDocument
    chunks: list[TextChunk]
    ner_results: list[NERResult]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_entities(self) -> int:
        return sum(len(r.entities) for r in self.ner_results)


class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline.

    Stages:
    1. Parse PDF (extract text from digital pages)
    2. OCR (process scanned pages that had no text layer)
    3. Chunk (split full text into overlapping windows)
    4. NER (extract named entities from each chunk)

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.process(Path("apple_10k_2023.pdf"))
        print(f"Produced {result.total_chunks} chunks")
    """

    def __init__(self):
        self.parser = PDFParser()
        self.ocr = OCRProcessor()
        self.chunker = TextChunker()
        self.ner = NERExtractor()
        logger.info("IngestionPipeline initialised")

    def process(self, file_path: Path) -> IngestionResult:
        """
        Process a single PDF through the full ingestion pipeline.

        Args:
            file_path: Path to the PDF file to process.

        Returns:
            IngestionResult with all extracted content.
        """
        file_path = Path(file_path)
        logger.info(f"Starting ingestion: {file_path.name}")

        # Stage 1: Parse PDF
        logger.info("Stage 1/4: Parsing PDF")
        document = self.parser.parse(file_path)

        # Stage 2: OCR any scanned pages
        scanned_pages = [p for p in document.pages if p.is_scanned]
        if scanned_pages:
            logger.info(f"Stage 2/4: OCR on {len(scanned_pages)} scanned pages")
            scanned_page_numbers = [p.page_number for p in scanned_pages]
            ocr_results = self.ocr.process_scanned_pages(file_path, scanned_page_numbers)

            # Replace empty scanned page text with OCR output
            for page in document.pages:
                if page.is_scanned and page.page_number in ocr_results:
                    page.text = ocr_results[page.page_number]
        else:
            logger.info("Stage 2/4: No scanned pages detected, skipping OCR")

        # Stage 3: Chunk the full document text
        logger.info("Stage 3/4: Chunking text")
        full_text = document.full_text
        chunks = self.chunker.chunk_text(full_text)

        # Stage 4: Extract named entities from each chunk
        logger.info("Stage 4/4: Extracting named entities")
        chunk_texts = [chunk.text for chunk in chunks]
        ner_results = self.ner.extract_batch(chunk_texts)

        result = IngestionResult(
            file_path=file_path,
            document=document,
            chunks=chunks,
            ner_results=ner_results,
        )

        logger.info(
            f"Ingestion complete: {result.total_chunks} chunks, "
            f"{result.total_entities} entities, "
            f"{document.scanned_pages} OCR pages"
        )

        return result