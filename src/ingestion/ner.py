"""
Named Entity Recognition module.

Extracts structured entities from financial document text using spaCy.
Entities are stored alongside chunks to enable filtered retrieval
(e.g. "only search Apple filings") and structured analysis.

Entity types we care about in financial documents:
- ORG: Companies, institutions ("Apple Inc.", "Federal Reserve")
- MONEY: Monetary values ("$12.4 billion", "€500M")
- DATE: Temporal references ("fiscal year 2023", "Q3")
- PERCENT: Percentage figures ("revenue grew 14%")
- GPE: Countries, cities ("United States", "China")
- PERSON: Named individuals ("Tim Cook", "Janet Yellen")
"""

from dataclasses import dataclass
from loguru import logger
import spacy
from spacy.language import Language


# Entity types relevant to financial document analysis
FINANCIAL_ENTITY_TYPES = {"ORG", "MONEY", "DATE", "PERCENT", "GPE", "PERSON"}


@dataclass
class ExtractedEntity:
    """A single named entity extracted from text."""
    text: str           # The entity text as it appears ("Apple Inc.")
    label: str          # Entity type ("ORG", "MONEY", etc.)
    start_char: int     # Character offset in source text
    end_char: int       # Character offset in source text
    confidence: float   # spaCy doesn't provide this natively; set to 1.0


@dataclass
class NERResult:
    """NER results for a single text chunk."""
    chunk_index: int
    entities: list[ExtractedEntity]

    @property
    def organisations(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "ORG"]

    @property
    def monetary_values(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "MONEY"]

    @property
    def dates(self) -> list[str]:
        return [e.text for e in self.entities if e.label == "DATE"]


class NERExtractor:
    """
    Extracts named entities from text using spaCy.

    Uses the small English model (en_core_web_sm) for speed.
    For higher accuracy on financial text, en_core_web_trf
    (transformer-based) would be the production choice,
    at the cost of ~10x slower inference.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model}")
        self.nlp: Language = spacy.load(model)
        logger.info("spaCy model loaded")

    def extract(self, text: str, chunk_index: int = 0) -> NERResult:
        """
        Extract named entities from a single text chunk.

        Args:
            text: Text to analyse.
            chunk_index: Index of the chunk this text came from.

        Returns:
            NERResult containing all detected entities.
        """
        # spaCy has a max text length limit — truncate gracefully
        if len(text) > self.nlp.max_length:
            logger.warning(
                f"Text exceeds spaCy max length ({self.nlp.max_length}), truncating"
            )
            text = text[:self.nlp.max_length]

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ not in FINANCIAL_ENTITY_TYPES:
                continue

            entities.append(ExtractedEntity(
                text=ent.text.strip(),
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0,
            ))

        logger.debug(
            f"Chunk {chunk_index}: found {len(entities)} entities "
            f"({len(set(e.label for e in entities))} types)"
        )

        return NERResult(chunk_index=chunk_index, entities=entities)

    def extract_batch(self, texts: list[str]) -> list[NERResult]:
        """
        Extract entities from multiple chunks efficiently.

        Uses spaCy's pipe() for batch processing, which is
        significantly faster than calling extract() in a loop
        for large document sets.

        Args:
            texts: List of text chunks to process.

        Returns:
            List of NERResult objects, one per input text.
        """
        logger.info(f"Batch NER on {len(texts)} chunks")
        results = []

        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=32)):
            entities = []
            for ent in doc.ents:
                if ent.label_ not in FINANCIAL_ENTITY_TYPES:
                    continue
                entities.append(ExtractedEntity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=1.0,
                ))
            results.append(NERResult(chunk_index=i, entities=entities))

        logger.info(f"Batch NER complete: {len(results)} results")
        return results