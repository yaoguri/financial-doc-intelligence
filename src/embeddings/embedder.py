"""
Embedding service for the Financial Document Intelligence Platform.

Wraps the Ollama nomic-embed-text model and applies the correct task
prefixes for asymmetric retrieval:
  - Chunks get "search_document:" prefix at indexing time
  - Queries get "search_query:" prefix at retrieval time

This asymmetry is baked into nomic-embed-text's training objective.
Skipping the prefixes degrades retrieval quality measurably.
"""

import logging
from typing import Optional

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

# nomic-embed-text task prefixes for asymmetric retrieval
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(Exception):
    """Raised when the Ollama embedding API call fails."""
    pass


class EmbeddingService:
    """
    Calls the Ollama /api/embeddings endpoint to produce text embeddings.

    Design decisions:
    - base_url and model are injected at construction time (not hardcoded)
      so unit tests can point at a mock server and swap models without
      touching this file.
    - embed_chunks and embed_query are separate methods (not one embed())
      because they must apply different task prefixes. Collapsing them
      would require the caller to know about prefixes, leaking a model-
      specific detail out of the service layer.
    - Ollama does not support batch embedding in a single API call, so
      embed_chunks loops internally. The caller sees a clean batch interface.
    - timeout is configurable — nomic-embed-text on CPU can be slow for
      long chunks.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_embed_model
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def _embed_single(self, text: str) -> list[float]:
        """
        Call the Ollama embeddings endpoint for a single text string.

        This is the internal primitive. Public methods (embed_chunks,
        embed_query) apply prefixes and call this.

        Raises EmbeddingError on any HTTP or API-level failure so callers
        get a clean domain exception rather than a raw httpx error.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Ollama returned HTTP {e.response.status_code} for model "
                f"'{self.model}': {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

        data = response.json()

        if "embedding" not in data:
            raise EmbeddingError(
                f"Ollama response missing 'embedding' key. Got: {list(data.keys())}"
            )

        return data["embedding"]

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document chunks for indexing.

        Applies the "search_document:" prefix to each text before embedding.
        This is the correct prefix for nomic-embed-text at index time.

        Args:
            texts: Raw chunk texts (no prefix — this method adds it).

        Returns:
            List of embedding vectors, one per input text, in the same order.

        Raises:
            EmbeddingError: If any Ollama call fails.
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        embeddings = []
        for i, text in enumerate(texts):
            prefixed = DOCUMENT_PREFIX + text
            logger.debug(f"Embedding chunk {i + 1}/{len(texts)} ({len(text)} chars)")
            vector = self._embed_single(prefixed)
            embeddings.append(vector)

        logger.info(f"Embedded {len(embeddings)} chunks with model '{self.model}'")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single user query for retrieval.

        Applies the "search_query:" prefix before embedding.
        This is the correct prefix for nomic-embed-text at query time.

        The separate method (vs embed_chunks) enforces the prefix discipline
        at the type level — callers cannot accidentally cross the prefixes.

        Args:
            text: The raw user query string.

        Returns:
            A single embedding vector.

        Raises:
            EmbeddingError: If the Ollama call fails.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Query text must not be empty")

        prefixed = QUERY_PREFIX + text
        logger.debug(f"Embedding query ({len(text)} chars)")
        vector = self._embed_single(prefixed)
        logger.info(f"Embedded query with model '{self.model}'")
        return vector

    def close(self) -> None:
        """Close the underlying HTTP client. Call when done with the service."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
