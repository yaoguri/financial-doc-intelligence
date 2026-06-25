"""
Retrieval service for the Financial Document Intelligence Platform.

Given a user query, this service:
  1. Embeds the query using EmbeddingService (with search_query: prefix)
  2. Calls EmbeddingRepository.find_similar for vector similarity search
  3. Optionally filters by cosine distance threshold
  4. Returns ranked (Chunk, distance) tuples for the RAG pipeline

Design principle: the retriever is a coordinator, not a policy-maker.
It does not decide what to do with low-scoring chunks — it returns scores
so the caller (RAG pipeline) can apply whatever threshold policy it needs.
"""

import logging
from typing import Optional

from src.embeddings.embedder import EmbeddingService
from src.storage.models import Chunk
from src.storage.repositories import ChunkRepository, EmbeddingRepository

logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Coordinates query embedding and vector similarity search.

    Dependencies are injected at construction time so this class is
    fully testable without a live Ollama instance or database:
    - In unit tests: pass mock embedder + mock repos
    - In integration tests: pass real embedder + real repos with live DB session

    Args:
        embedder: EmbeddingService instance for encoding the query.
        embedding_repo: EmbeddingRepository for vector similarity search.
        chunk_repo: ChunkRepository for fetching full chunk data if needed.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        embedding_repo: EmbeddingRepository,
        chunk_repo: ChunkRepository,
    ):
        self.embedder = embedder
        self.embedding_repo = embedding_repo
        self.chunk_repo = chunk_repo

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Find the top-k most relevant chunks for a query.

        Process:
          1. Embed the query with the search_query: prefix
          2. Run cosine similarity search via pgvector
          3. Optionally filter results by distance threshold
          4. Return (Chunk, cosine_distance) tuples, best first

        Cosine distance interpretation:
          0.0 = identical vectors (perfect match)
          1.0 = orthogonal vectors (completely unrelated)
          2.0 = opposite vectors (rare in practice for text)

        Typical useful range: 0.0–0.5. Chunks above 0.6–0.7 are usually noise.

        Args:
            query: The user's natural language question.
            k: Number of results to return (default 5).
            score_threshold: If set, exclude chunks with cosine distance
                             greater than this value. None means return all k.
                             Recommended: 0.5 for production, None for dev/debug.

        Returns:
            List of (Chunk, cosine_distance) tuples, sorted ascending by distance
            (best match first). May be shorter than k if threshold filtering applies.

        Raises:
            EmbeddingError: If the Ollama embedding call fails.
            ValueError: If query is empty or k < 1.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        logger.info(f"Retrieving top-{k} chunks for query: '{query[:80]}...'")

        # Step 1: embed the query
        query_vector = self.embedder.embed_query(query)
        logger.debug(f"Query embedded to vector of dimension {len(query_vector)}")

        # Step 2: vector similarity search
        results: list[tuple[Chunk, float]] = self.embedding_repo.find_similar(
            query_vector=query_vector,
            limit=k,
        )
        logger.debug(f"find_similar returned {len(results)} results")

        # Step 3: optional threshold filtering
        if score_threshold is not None:
            before = len(results)
            results = [
                (chunk, distance)
                for chunk, distance in results
                if distance <= score_threshold
            ]
            filtered = before - len(results)
            if filtered > 0:
                logger.info(
                    f"Filtered {filtered} chunks above distance threshold "
                    f"{score_threshold} (kept {len(results)}/{before})"
                )

        # Step 4: log score distribution for observability
        if results:
            distances = [d for _, d in results]
            logger.info(
                f"Retrieved {len(results)} chunks — "
                f"best: {distances[0]:.4f}, worst: {distances[-1]:.4f}"
            )
        else:
            logger.warning("No chunks retrieved — query may be out of distribution")

        return results
