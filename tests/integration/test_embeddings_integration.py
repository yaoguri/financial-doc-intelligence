"""
Integration tests for Phase 3 — Embeddings + Retrieval.

These tests require:
  - Live Ollama with nomic-embed-text pulled
  - Live Docker PostgreSQL on port 5433

Run with:
    python -m pytest tests/integration/test_embeddings.py -v

Skipped automatically if Ollama is not reachable.
"""

import pytest
import httpx

from src.embeddings.embedder import EmbeddingService
from src.retrieval.retriever import RetrieverService
from src.storage.database import get_db
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingRepository,
)
from src.config import get_settings  # noqa: F401 — used in test methods


# ── Skip guard — don't fail CI if Ollama isn't running ───────────────────────

def ollama_is_available() -> bool:
    """Return True if Ollama is reachable at the configured base URL."""
    try:
        settings = get_settings()
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_is_available(),
    reason="Ollama not running — skipping embedding integration tests",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_session():
    """Live DB session with rollback isolation (same pattern as Phase 2)."""
    with get_db() as session:
        yield session
        session.rollback()


@pytest.fixture
def embedder():
    """Real EmbeddingService using live Ollama."""
    service = EmbeddingService()
    yield service
    service.close()


@pytest.fixture
def repos(db_session):
    """All repositories wired to the test session."""
    return {
        "document": DocumentRepository(db_session),
        "chunk": ChunkRepository(db_session),
        "embedding": EmbeddingRepository(db_session),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@requires_ollama
class TestEmbeddingServiceIntegration:

    def test_embed_query_returns_vector_of_expected_dimension(self, embedder):
        vector = embedder.embed_query("What was the company's revenue in 2023?")
        # nomic-embed-text produces 768-dimensional vectors
        assert isinstance(vector, list)
        assert len(vector) == 768
        assert all(isinstance(v, float) for v in vector)

    def test_embed_chunks_returns_correct_count(self, embedder):
        texts = [
            "Revenue for fiscal year 2023 was $42 billion.",
            "Operating expenses increased by 8% year-over-year.",
            "Net income margin improved to 22%.",
        ]
        vectors = embedder.embed_chunks(texts)
        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)

    def test_query_and_document_vectors_differ(self, embedder):
        """
        The same text encoded with different prefixes should produce
        different vectors. This validates the asymmetric encoding is active.
        """
        text = "revenue increased significantly"
        doc_vector = embedder.embed_chunks([text])[0]
        query_vector = embedder.embed_query(text)
        # They should not be identical — prefixes shift the vectors
        assert doc_vector != query_vector


@requires_ollama
class TestRetrievalRoundTrip:
    """
    Full round-trip: store chunks → embed → retrieve with a query.

    This is the most important integration test — it validates the entire
    Phase 3 pipeline end to end.
    """

    def test_retrieve_finds_relevant_chunk(self, embedder, repos, db_session):
        doc_repo = repos["document"]
        chunk_repo = repos["chunk"]
        embedding_repo = repos["embedding"]

        settings = get_settings()

        # 1. Create a document
        doc = doc_repo.create(
            filename="test_annual_report.pdf",
            file_hash="aabbcc112233",
            page_count=5,
        )

        # 2. Store chunks with varying content
        chunk_texts = [
            "Apple Inc reported revenue of $394 billion for fiscal year 2023.",
            "The board of directors approved a $90 billion share buyback program.",
            "Research and development expenditure reached $29.9 billion.",
            "iPhone sales accounted for 52% of total revenue.",
            "Services segment revenue grew 16% year-over-year to $85 billion.",
        ]

        stored_chunks = []
        for i, text in enumerate(chunk_texts):
            chunk = chunk_repo.create(
                document_id=doc.id,
                text=text,
                chunk_index=i,
            )
            stored_chunks.append(chunk)

        # 3. Embed and store all chunks
        vectors = embedder.embed_chunks(chunk_texts)
        for chunk, vector in zip(stored_chunks, vectors):
            embedding_repo.create(
                chunk_id=chunk.id,
                model_name=settings.ollama_embed_model,
                vector=vector,
            )

        db_session.flush()  # make rows visible within this transaction

        # 4. Retrieve with a semantically relevant query
        retriever = RetrieverService(
            embedder=embedder,
            embedding_repo=embedding_repo,
            chunk_repo=chunk_repo,
        )

        results = retriever.retrieve("What was Apple's total revenue?", k=3)

        # 5. Validate
        assert len(results) > 0

        # The revenue chunk should be nearest (lowest distance)
        best_chunk, best_distance = results[0]
        assert best_distance < 0.5, (
            f"Best match distance {best_distance:.4f} is too high — "
            "expected a semantically close result under 0.5"
        )
        # The most relevant chunk should mention revenue — check whichever
        # attribute your Chunk model exposes (text or content)
        chunk_text = getattr(best_chunk, "text", None) or getattr(best_chunk, "content", "")
        assert "revenue" in chunk_text.lower()

    def test_score_threshold_excludes_low_relevance_chunks(self, embedder, repos, db_session):
        doc_repo = repos["document"]
        chunk_repo = repos["chunk"]
        embedding_repo = repos["embedding"]

        settings = get_settings()

        doc = doc_repo.create(
            filename="test_threshold.pdf",
            file_hash="ddeeff445566",
            page_count=1,
        )

        # One highly relevant chunk, one totally unrelated
        chunk_texts = [
            "Net profit margin for 2023 was 21.4 percent.",
            "The company held its annual summer picnic in July.",
        ]

        stored_chunks = []
        for i, text in enumerate(chunk_texts):
            chunk = chunk_repo.create(
                document_id=doc.id, text=text, chunk_index=i
            )
            stored_chunks.append(chunk)

        vectors = embedder.embed_chunks(chunk_texts)
        for chunk, vector in zip(stored_chunks, vectors):
            embedding_repo.create(
                chunk_id=chunk.id,
                model_name=settings.ollama_embed_model,
                vector=vector,
            )

        db_session.flush()

        retriever = RetrieverService(
            embedder=embedder,
            embedding_repo=embedding_repo,
            chunk_repo=chunk_repo,
        )

        # Tight threshold — should filter out the picnic chunk
        results = retriever.retrieve(
            "What was the profit margin?",
            k=5,
            score_threshold=0.4,
        )

        assert len(results) >= 1
        for chunk, distance in results:
            assert distance <= 0.4
            chunk_text = getattr(chunk, "text", None) or getattr(chunk, "content", "")
            assert "profit" in chunk_text.lower() or "margin" in chunk_text.lower()
