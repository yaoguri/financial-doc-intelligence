"""
PDF text extraction module.

Handles digital PDFs (text layer present) using pdfplumber.
Routes scanned PDFs to the OCR processor automatically.

Design decision: we detect scanned pages by checking if pdfplumber
extracts less than a minimum character threshold. If a page looks
"empty" to pdfplumber, it's almost certainly a scanned image.
"""

from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
import pdfplumber


# If a page yields fewer characters than this, treat it as scanned
MIN_CHARS_FOR_TEXT_PAGE = 50


@dataclass
class ParsedPage:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    is_scanned: bool
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text.strip())


@dataclass
class ParsedDocument:
    """Represents the fully extracted content of a PDF document."""
    file_path: Path
    pages: list[ParsedPage]
    total_pages: int
    scanned_pages: int

    @property
    def full_text(self) -> str:
        """Concatenate all page texts into a single string."""
        return "\n\n".join(
            page.text for page in self.pages if page.text.strip()
        )

    @property
    def is_primarily_scanned(self) -> bool:
        """True if more than half the pages required OCR."""
        return self.scanned_pages > (self.total_pages / 2)


class PDFParser:
    """
    Extracts text from digital PDFs using pdfplumber.

    For scanned pages (detected by low character count), sets
    is_scanned=True so the calling pipeline can route them
    to OCRProcessor instead.
    """

    def __init__(self, min_chars: int = MIN_CHARS_FOR_TEXT_PAGE):
        self.min_chars = min_chars

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF file and extract text from all pages.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument containing text from all pages.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the file is not a PDF.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

        logger.info(f"Parsing PDF: {file_path.name}")
        pages = []
        scanned_count = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(f"Total pages: {total_pages}")

            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                is_scanned = len(text.strip()) < self.min_chars

                if is_scanned:
                    scanned_count += 1
                    logger.debug(f"Page {i}: detected as scanned (chars: {len(text.strip())})")
                else:
                    logger.debug(f"Page {i}: extracted {len(text.strip())} chars")

                pages.append(ParsedPage(
                    page_number=i,
                    text=text,
                    is_scanned=is_scanned,
                ))

        logger.info(
            f"Parsing complete: {total_pages} pages, "
            f"{scanned_count} scanned, "
            f"{total_pages - scanned_count} digital"
        )

        return ParsedDocument(
            file_path=file_path,
            pages=pages,
            total_pages=total_pages,
            scanned_pages=scanned_count,
        )