"""
Unit tests for Phase 3 — Embeddings + Retrieval.

These tests run without a live Ollama instance or database.
The httpx HTTP client is mocked at the boundary so all real service
logic (prefix application, error handling, input validation) is exercised.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.embeddings.embedder import EmbeddingService, EmbeddingError, DOCUMENT_PREFIX, QUERY_PREFIX
from src.retrieval.retriever import RetrieverService
from src.storage.models import Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5]


def make_mock_response(vector: list[float] = None):
    """Build a mock httpx response that returns an embedding vector."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": vector or FAKE_VECTOR}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def make_chunk(content: str = "test content", chunk_index: int = 0) -> Chunk:
    """Build a minimal Chunk ORM object for testing."""
    chunk = Chunk()
    chunk.id = "test-chunk-id"
    chunk.content = content
    chunk.chunk_index = chunk_index
    return chunk


# ── EmbeddingService — happy path ─────────────────────────────────────────────

class TestEmbeddingServiceEmbedChunks:

    def test_embed_chunks_returns_list_of_vectors(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            result = service.embed_chunks(["chunk one", "chunk two"])

        assert len(result) == 2
        assert result[0] == FAKE_VECTOR
        assert result[1] == FAKE_VECTOR

    def test_embed_chunks_applies_document_prefix(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            service.embed_chunks(["hello world"])

        call_payload = mock_client.post.call_args[1]["json"]
        assert call_payload["prompt"].startswith(DOCUMENT_PREFIX)
        assert "hello world" in call_payload["prompt"]

    def test_embed_chunks_does_not_apply_query_prefix(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            service.embed_chunks(["revenue grew 12%"])

        call_payload = mock_client.post.call_args[1]["json"]
        assert not call_payload["prompt"].startswith(QUERY_PREFIX)

    def test_embed_chunks_calls_api_once_per_chunk(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            service.embed_chunks(["a", "b", "c"])

        assert mock_client.post.call_count == 3

    def test_embed_chunks_raises_on_empty_list(self):
        service = EmbeddingService(base_url="http://fake", model="fake-model")
        with pytest.raises(ValueError, match="must not be empty"):
            service.embed_chunks([])


class TestEmbeddingServiceEmbedQuery:

    def test_embed_query_returns_vector(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            result = service.embed_query("what was revenue in 2023?")

        assert result == FAKE_VECTOR

    def test_embed_query_applies_query_prefix(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            service.embed_query("what was revenue in 2023?")

        call_payload = mock_client.post.call_args[1]["json"]
        assert call_payload["prompt"].startswith(QUERY_PREFIX)

    def test_embed_query_does_not_apply_document_prefix(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.return_value = make_mock_response()

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            service.embed_query("what was revenue?")

        call_payload = mock_client.post.call_args[1]["json"]
        assert not call_payload["prompt"].startswith(DOCUMENT_PREFIX)

    def test_embed_query_raises_on_empty_string(self):
        service = EmbeddingService(base_url="http://fake", model="fake-model")
        with pytest.raises(ValueError, match="must not be empty"):
            service.embed_query("")

    def test_embed_query_raises_on_whitespace(self):
        service = EmbeddingService(base_url="http://fake", model="fake-model")
        with pytest.raises(ValueError, match="must not be empty"):
            service.embed_query("   ")


# ── EmbeddingService — error handling ────────────────────────────────────────

class TestEmbeddingServiceErrors:

    def test_raises_embedding_error_on_http_error(self):
        import httpx

        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_response
            )

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            with pytest.raises(EmbeddingError, match="HTTP 500"):
                service.embed_query("test")

    def test_raises_embedding_error_on_connection_failure(self):
        import httpx

        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            with pytest.raises(EmbeddingError, match="Could not reach Ollama"):
                service.embed_query("test")

    def test_raises_embedding_error_on_missing_embedding_key(self):
        with patch("src.embeddings.embedder.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            bad_response = MagicMock()
            bad_response.json.return_value = {"error": "model not found"}
            bad_response.raise_for_status.return_value = None
            mock_client.post.return_value = bad_response

            service = EmbeddingService(base_url="http://fake", model="fake-model")
            with pytest.raises(EmbeddingError, match="missing 'embedding' key"):
                service.embed_query("test")


# ── RetrieverService ──────────────────────────────────────────────────────────

class TestRetrieverService:

    def _make_retriever(self, query_vector=None, similar_results=None):
        """Build a RetrieverService with fully mocked dependencies."""
        mock_embedder = MagicMock(spec=EmbeddingService)
        mock_embedder.embed_query.return_value = query_vector or FAKE_VECTOR

        mock_embedding_repo = MagicMock()
        mock_embedding_repo.find_similar.return_value = similar_results or []

        mock_chunk_repo = MagicMock()

        retriever = RetrieverService(
            embedder=mock_embedder,
            embedding_repo=mock_embedding_repo,
            chunk_repo=mock_chunk_repo,
        )
        return retriever, mock_embedder, mock_embedding_repo

    def test_retrieve_embeds_query_and_calls_find_similar(self):
        chunk = make_chunk("Apple revenue was $100B")
        retriever, mock_embedder, mock_embedding_repo = self._make_retriever(
            similar_results=[(chunk, 0.15)]
        )

        results = retriever.retrieve("what was Apple revenue?", k=3)

        mock_embedder.embed_query.assert_called_once_with("what was Apple revenue?")
        mock_embedding_repo.find_similar.assert_called_once_with(
            query_vector=FAKE_VECTOR, limit=3
        )
        assert len(results) == 1
        assert results[0][0] is chunk
        assert results[0][1] == 0.15

    def test_retrieve_returns_all_results_when_no_threshold(self):
        chunks = [(make_chunk(f"chunk {i}"), float(i) * 0.1) for i in range(5)]
        retriever, _, _ = self._make_retriever(similar_results=chunks)

        results = retriever.retrieve("test query", k=5, score_threshold=None)
        assert len(results) == 5

    def test_retrieve_filters_by_score_threshold(self):
        chunk_good = make_chunk("relevant content")
        chunk_bad = make_chunk("unrelated content")
        results_from_db = [(chunk_good, 0.2), (chunk_bad, 0.8)]

        retriever, _, _ = self._make_retriever(similar_results=results_from_db)
        results = retriever.retrieve("test query", k=5, score_threshold=0.5)

        assert len(results) == 1
        assert results[0][0] is chunk_good

    def test_retrieve_raises_on_empty_query(self):
        retriever, _, _ = self._make_retriever()
        with pytest.raises(ValueError, match="must not be empty"):
            retriever.retrieve("")

    def test_retrieve_raises_on_invalid_k(self):
        retriever, _, _ = self._make_retriever()
        with pytest.raises(ValueError, match="k must be >= 1"):
            retriever.retrieve("test", k=0)

    def test_retrieve_returns_empty_list_when_no_matches(self):
        retriever, _, _ = self._make_retriever(similar_results=[])
        results = retriever.retrieve("obscure query no one knows about")
        assert results == []
