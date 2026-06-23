"""
OCR processing module.

Handles scanned PDF pages by converting them to images
and running Tesseract OCR to extract text.

Design decision: we process page-by-page rather than the whole
document at once. This allows partial recovery if one page fails,
and keeps memory usage bounded for large documents.
"""

from pathlib import Path
from loguru import logger
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from src.config import settings


class OCRProcessor:
    """
    Extracts text from scanned PDF pages using Tesseract OCR.

    Converts each PDF page to an image at 300 DPI (high enough
    for accurate OCR, low enough to keep memory reasonable),
    then runs Tesseract on each image.
    """

    def __init__(self):
        # Point pytesseract at the Windows Tesseract binary
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
        logger.info(f"OCR processor initialised. Tesseract: {settings.tesseract_path}")

    def process_page_image(self, image: Image.Image) -> str:
        """
        Run OCR on a single PIL image.

        Args:
            image: PIL Image of a PDF page.

        Returns:
            Extracted text string.
        """
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",  # Assume uniform block of text
        )
        return text.strip()

    def process_pdf(self, file_path: Path) -> list[str]:
        """
        Convert a PDF to images and OCR every page.

        Args:
            file_path: Path to the scanned PDF.

        Returns:
            List of extracted text strings, one per page.

        Note:
            This is slow — expect 5-30 seconds per page depending
            on hardware. For production, this would run async or
            in a background worker queue.
        """
        file_path = Path(file_path)
        logger.info(f"Starting OCR for: {file_path.name}")

        # Convert PDF pages to images at 300 DPI
        images = convert_from_path(
            file_path,
            dpi=300,
            fmt="PNG",
        )

        logger.info(f"Converted {len(images)} pages to images")
        results = []

        for i, image in enumerate(images, start=1):
            logger.debug(f"OCR processing page {i}/{len(images)}")
            try:
                text = self.process_page_image(image)
                results.append(text)
                logger.debug(f"Page {i}: extracted {len(text)} chars via OCR")
            except Exception as e:
                # Don't fail the whole document if one page fails
                logger.warning(f"OCR failed on page {i}: {e}")
                results.append("")

        logger.info(f"OCR complete: {len(results)} pages processed")
        return results

    def process_scanned_pages(
        self,
        file_path: Path,
        page_numbers: list[int],
    ) -> dict[int, str]:
        """
        OCR only specific pages of a PDF (those flagged as scanned).

        More efficient than processing the entire document when
        most pages are digital and only a few are scanned images.

        Args:
            file_path: Path to the PDF.
            page_numbers: 1-indexed list of pages to OCR.

        Returns:
            Dict mapping page_number -> extracted text.
        """
        file_path = Path(file_path)
        logger.info(f"Selective OCR: pages {page_numbers} of {file_path.name}")

        images = convert_from_path(
            file_path,
            dpi=300,
            fmt="PNG",
        )

        results = {}
        for page_num in page_numbers:
            idx = page_num - 1  # Convert to 0-indexed
            if idx >= len(images):
                logger.warning(f"Page {page_num} out of range")
                continue
            try:
                text = self.process_page_image(images[idx])
                results[page_num] = text
            except Exception as e:
                logger.warning(f"OCR failed on page {page_num}: {e}")
                results[page_num] = ""

        return results