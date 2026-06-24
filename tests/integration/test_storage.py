"""
Integration tests for the storage layer.

These tests run against the REAL Docker database — not mocks.
That's what makes them integration tests rather than unit tests.

Before running: ensure Docker is up and the database is reachable.
    docker compose up -d

Run with:
    python -m pytest tests/integration/test_storage.py -v

Each test class uses a db_session fixture that rolls back all changes
after each test. This means tests are isolated — one test's writes
don't pollute the next test's reads — without needing to recreate
the database schema between tests.
"""

import uuid
import pytest
from sqlalchemy import text

from src.storage.database import SessionLocal, check_database_connection, engine
from src.storage.models import Base, Chunk, Document, Embedding, Entity
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingRepository,
    EntityRepository,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """
    Create all tables before the test session starts.

    scope="session" means this runs once for the entire test run, not once
    per test. Creating tables is idempotent (checkfirst=True is the default
    for create_all), so it's safe to run even if tables already exist.

    In a CI environment, this ensures the schema exists before any test runs.
    """
    Base.metadata.create_all(bind=engine)
    yield
    # We do NOT drop tables after tests — we want to inspect the database
    # manually after runs during development. In CI, the container is ephemeral.


@pytest.fixture
def db_session():
    """
    Provide a database session that rolls back after each test.

    This is the standard pattern for integration tests with SQLAlchemy:
    1. Begin a real transaction
    2. Run the test inside it
    3. Roll back — as if the test never happened

    Result: tests are fully isolated without recreating tables between runs.
    Much faster than truncating tables, and no risk of test order dependencies.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def doc_repo(db_session):
    return DocumentRepository(db_session)


@pytest.fixture
def chunk_repo(db_session):
    return ChunkRepository(db_session)


@pytest.fixture
def embedding_repo(db_session):
    return EmbeddingRepository(db_session)


@pytest.fixture
def entity_repo(db_session):
    return EntityRepository(db_session)


@pytest.fixture
def sample_document(doc_repo):
    """Create and return a sample document for use in other tests."""
    return doc_repo.create(
        filename="apple_10k_2023.pdf",
        file_hash="abc123def456",
        page_count=120,
        file_size_bytes=2_048_000,
    )


@pytest.fixture
def sample_chunk(chunk_repo, sample_document):
    """Create and return a sample chunk for use in other tests."""
    return chunk_repo.create(
        document_id=sample_document.id,
        chunk_index=0,
        text="Apple Inc. reported revenue of $394.3 billion for fiscal year 2023.",
        page_number=1,
        char_start=0,
        char_end=68,
    )


# ---------------------------------------------------------------------------
# Database connectivity
# ---------------------------------------------------------------------------

class TestDatabaseConnection:
    def test_database_is_reachable(self):
        """Verify we can connect to the Docker PostgreSQL instance."""
        assert check_database_connection() is True

    def test_pgvector_extension_is_enabled(self, db_session):
        """Verify the pgvector extension was loaded by init.sql."""
        result = db_session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        assert result == "vector", "pgvector extension not found — check docker/init.sql"


# ---------------------------------------------------------------------------
# DocumentRepository tests
# ---------------------------------------------------------------------------

class TestDocumentRepository:
    def test_create_document(self, doc_repo):
        doc = doc_repo.create(
            filename="test.pdf",
            file_hash="hash_001",
            page_count=10,
            file_size_bytes=500_000,
        )
        assert doc.id is not None
        assert doc.filename == "test.pdf"
        assert doc.processed is False

    def test_get_by_id(self, doc_repo, sample_document):
        fetched = doc_repo.get_by_id(sample_document.id)
        assert fetched is not None
        assert fetched.filename == sample_document.filename

    def test_get_by_id_returns_none_for_missing(self, doc_repo):
        result = doc_repo.get_by_id(uuid.uuid4())
        assert result is None

    def test_get_by_hash_detects_duplicate(self, doc_repo, sample_document):
        """This is the duplicate-detection mechanism."""
        existing = doc_repo.get_by_hash(sample_document.file_hash)
        assert existing is not None
        assert existing.id == sample_document.id

    def test_get_by_hash_returns_none_for_new_file(self, doc_repo):
        result = doc_repo.get_by_hash("hash_that_does_not_exist")
        assert result is None

    def test_mark_processed(self, doc_repo, sample_document):
        assert sample_document.processed is False
        doc_repo.mark_processed(sample_document.id)
        refreshed = doc_repo.get_by_id(sample_document.id)
        assert refreshed.processed is True

    def test_list_all(self, doc_repo, sample_document):
        docs = doc_repo.list_all()
        ids = [d.id for d in docs]
        assert sample_document.id in ids

    def test_delete_document(self, doc_repo, sample_document):
        deleted = doc_repo.delete(sample_document.id)
        assert deleted is True
        assert doc_repo.get_by_id(sample_document.id) is None

    def test_delete_nonexistent_returns_false(self, doc_repo):
        result = doc_repo.delete(uuid.uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# ChunkRepository tests
# ---------------------------------------------------------------------------

class TestChunkRepository:
    def test_create_chunk(self, chunk_repo, sample_document):
        chunk = chunk_repo.create(
            document_id=sample_document.id,
            chunk_index=0,
            text="Revenue grew 8% year-over-year.",
            page_number=5,
        )
        assert chunk.id is not None
        assert chunk.chunk_index == 0
        assert chunk.document_id == sample_document.id

    def test_bulk_create_chunks(self, chunk_repo, sample_document):
        chunks_data = [
            {
                "document_id": sample_document.id,
                "chunk_index": i,
                "text": f"Chunk number {i} with financial content.",
                "page_number": i + 1,
            }
            for i in range(5)
        ]
        chunks = chunk_repo.bulk_create(chunks_data)
        assert len(chunks) == 5
        assert all(c.id is not None for c in chunks)

    def test_get_by_document_returns_ordered(self, chunk_repo, sample_document):
        chunks_data = [
            {"document_id": sample_document.id, "chunk_index": i, "text": f"text {i}"}
            for i in range(3)
        ]
        chunk_repo.bulk_create(chunks_data)
        fetched = chunk_repo.get_by_document(sample_document.id)
        indices = [c.chunk_index for c in fetched]
        assert indices == sorted(indices)

    def test_count_by_document(self, chunk_repo, sample_document):
        chunks_data = [
            {"document_id": sample_document.id, "chunk_index": i, "text": f"text {i}"}
            for i in range(4)
        ]
        chunk_repo.bulk_create(chunks_data)
        count = chunk_repo.count_by_document(sample_document.id)
        assert count == 4


# ---------------------------------------------------------------------------
# EmbeddingRepository tests
# ---------------------------------------------------------------------------

class TestEmbeddingRepository:
    def test_create_embedding(self, embedding_repo, sample_chunk):
        vector = [0.1] * 768  # fake 768-dim vector
        emb = embedding_repo.create(
            chunk_id=sample_chunk.id,
            model_name="nomic-embed-text",
            vector=vector,
        )
        assert emb.id is not None
        assert emb.model_name == "nomic-embed-text"
        assert len(emb.vector) == 768

    def test_get_by_chunk_id(self, embedding_repo, sample_chunk):
        vector = [0.2] * 768
        created = embedding_repo.create(
            chunk_id=sample_chunk.id,
            model_name="nomic-embed-text",
            vector=vector,
        )
        fetched = embedding_repo.get_by_chunk_id(sample_chunk.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_find_similar_returns_results(
        self, chunk_repo, embedding_repo, sample_document
    ):
        """
        Verify the vector similarity search returns results in distance order.

        We insert three chunks with distinct vectors, then query with a vector
        close to chunk 0. We expect chunk 0 to rank first (smallest distance).
        """
        # Create three chunks with distinct embeddings
        chunks_data = [
            {"document_id": sample_document.id, "chunk_index": i, "text": f"text {i}"}
            for i in range(3)
        ]
        chunks = chunk_repo.bulk_create(chunks_data)

        vectors = [
            [1.0] + [0.0] * 767,   # chunk 0: points along dimension 0
            [0.0, 1.0] + [0.0] * 766,  # chunk 1: points along dimension 1
            [0.0, 0.0, 1.0] + [0.0] * 765,  # chunk 2: points along dimension 2
        ]
        for chunk, vector in zip(chunks, vectors):
            embedding_repo.create(
                chunk_id=chunk.id,
                model_name="nomic-embed-text",
                vector=vector,
            )

        # Query vector is closest to chunk 0
        query = [0.9] + [0.0] * 767
        results = embedding_repo.find_similar(query, limit=3)

        assert len(results) == 3
        # First result should be chunk 0 (most similar)
        closest_chunk, distance = results[0]
        assert closest_chunk.id == chunks[0].id
        assert distance < results[1][1]  # smaller distance = more similar

    def test_delete_by_chunk_id(self, embedding_repo, sample_chunk):
        embedding_repo.create(
            chunk_id=sample_chunk.id,
            model_name="nomic-embed-text",
            vector=[0.5] * 768,
        )
        deleted = embedding_repo.delete_by_chunk_id(sample_chunk.id)
        assert deleted is True
        assert embedding_repo.get_by_chunk_id(sample_chunk.id) is None


# ---------------------------------------------------------------------------
# EntityRepository tests
# ---------------------------------------------------------------------------

class TestEntityRepository:
    def test_bulk_create_entities(self, entity_repo, sample_chunk):
        entities_data = [
            {"chunk_id": sample_chunk.id, "text": "Apple Inc.", "label": "ORG"},
            {"chunk_id": sample_chunk.id, "text": "$394.3 billion", "label": "MONEY"},
            {"chunk_id": sample_chunk.id, "text": "fiscal year 2023", "label": "DATE"},
        ]
        entities = entity_repo.bulk_create(entities_data)
        assert len(entities) == 3
        assert all(e.id is not None for e in entities)

    def test_get_by_chunk_id(self, entity_repo, sample_chunk):
        entities_data = [
            {"chunk_id": sample_chunk.id, "text": "Apple Inc.", "label": "ORG"},
        ]
        entity_repo.bulk_create(entities_data)
        fetched = entity_repo.get_by_chunk_id(sample_chunk.id)
        assert len(fetched) == 1
        assert fetched[0].text == "Apple Inc."

    def test_get_by_label(self, entity_repo, sample_chunk):
        entities_data = [
            {"chunk_id": sample_chunk.id, "text": "Apple Inc.", "label": "ORG"},
            {"chunk_id": sample_chunk.id, "text": "$394.3 billion", "label": "MONEY"},
        ]
        entity_repo.bulk_create(entities_data)
        money_entities = entity_repo.get_by_label("MONEY")
        labels = {e.label for e in money_entities}
        assert labels == {"MONEY"}

    def test_search_by_text_case_insensitive(self, entity_repo, sample_chunk):
        entities_data = [
            {"chunk_id": sample_chunk.id, "text": "Apple Inc.", "label": "ORG"},
        ]
        entity_repo.bulk_create(entities_data)
        results = entity_repo.search_by_text("apple")  # lowercase
        assert any(e.text == "Apple Inc." for e in results)


# ---------------------------------------------------------------------------
# Cascade delete test
# ---------------------------------------------------------------------------

class TestCascadeDeletes:
    def test_deleting_document_cascades_to_chunks_and_entities(
        self, doc_repo, chunk_repo, entity_repo, db_session
    ):
        """
        Verify that deleting a document removes all downstream data.

        This tests the ON DELETE CASCADE constraints defined on the foreign keys.
        If the constraints are wrong, orphan rows would remain after deletion.
        """
        doc = doc_repo.create(
            filename="cascade_test.pdf",
            file_hash="cascade_hash_999",
        )
        chunk = chunk_repo.create(
            document_id=doc.id,
            chunk_index=0,
            text="Some financial content here.",
        )
        entity_repo.bulk_create([
            {"chunk_id": chunk.id, "text": "Revenue", "label": "MONEY"},
        ])

        doc_id = doc.id
        chunk_id = chunk.id

        doc_repo.delete(doc_id)

        # Chunk should be gone
        assert chunk_repo.get_by_id(chunk_id) is None
        # Entity query should return empty
        assert entity_repo.get_by_chunk_id(chunk_id) == []
