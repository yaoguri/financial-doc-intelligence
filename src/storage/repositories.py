"""
Repository classes for the Financial Document Intelligence Platform.

The Repository Pattern enforces a strict boundary: ALL database access goes
through these classes. No raw SQLAlchemy queries anywhere else in the codebase.

Why this matters:
- Code outside these classes is forbidden from knowing HOW data is stored
- Enables mocking in unit tests (swap real repo for a fake one)
- Enables future storage backend changes (change one file, not twenty)
- Makes intent readable: chunk_repo.find_similar(...) reads like English

One repository class per table. Each takes a Session in __init__ and uses
it for all operations — the session's lifecycle is managed by the caller
(via get_db()), never by the repository itself.
"""

import uuid
from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from src.storage.models import Chunk, Document, Embedding, Entity


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------

class DocumentRepository:
    """All database operations for the documents table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        filename: str,
        file_hash: str,
        page_count: int | None = None,
        file_size_bytes: int | None = None,
    ) -> Document:
        """
        Insert a new document row and return the created object.

        Flushes (not commits) so the id is populated before we return.
        The caller's session (via get_db) controls when the commit happens.
        """
        doc = Document(
            filename=filename,
            file_hash=file_hash,
            page_count=page_count,
            file_size_bytes=file_size_bytes,
            processed=False,
        )
        self.db.add(doc)
        self.db.flush()  # sends INSERT to DB, populates doc.id, but doesn't commit
        return doc

    def get_by_id(self, document_id: uuid.UUID) -> Optional[Document]:
        """Fetch a document by primary key. Returns None if not found."""
        return self.db.get(Document, document_id)

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """
        Check if a file has already been ingested.

        This is the duplicate-detection query. Before ingesting any PDF,
        the pipeline calls this. If it returns a Document, we skip processing.
        """
        stmt = select(Document).where(Document.file_hash == file_hash)
        return self.db.execute(stmt).scalar_one_or_none()

    def mark_processed(self, document_id: uuid.UUID) -> None:
        """
        Flip the processed flag to True.

        Called by the pipeline only after all chunks, embeddings, and entities
        have been successfully committed. If ingestion fails partway through,
        processed stays False — making the document easy to find and retry.
        """
        doc = self.get_by_id(document_id)
        if doc:
            doc.processed = True
            self.db.flush()

    def list_all(self) -> list[Document]:
        """Return all documents, ordered by upload date descending."""
        stmt = select(Document).order_by(Document.upload_date.desc())
        return list(self.db.execute(stmt).scalars().all())

    def list_unprocessed(self) -> list[Document]:
        """
        Return documents where processed=False.

        Useful for a retry job that re-ingests any documents that failed
        partway through the pipeline.
        """
        stmt = select(Document).where(Document.processed == False)
        return list(self.db.execute(stmt).scalars().all())

    def delete(self, document_id: uuid.UUID) -> bool:
        """
        Delete a document and all its associated data.

        CASCADE constraints on chunks → embeddings/entities mean deleting
        a document automatically deletes everything downstream. Returns True
        if a row was deleted, False if the document didn't exist.
        """
        doc = self.get_by_id(document_id)
        if doc:
            self.db.delete(doc)
            self.db.flush()
            return True
        return False


# ---------------------------------------------------------------------------
# ChunkRepository
# ---------------------------------------------------------------------------

class ChunkRepository:
    """All database operations for the chunks table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        document_id: uuid.UUID,
        chunk_index: int,
        text: str,
        page_number: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Chunk:
        """Insert a single chunk row."""
        chunk = Chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            page_number=page_number,
            char_start=char_start,
            char_end=char_end,
        )
        self.db.add(chunk)
        self.db.flush()
        return chunk

    def bulk_create(self, chunks_data: list[dict]) -> list[Chunk]:
        """
        Insert multiple chunks efficiently in a single flush.

        Expects a list of dicts with keys matching Chunk constructor args.
        Much faster than calling create() in a loop for large documents —
        one round trip to the database instead of N.

        Example input:
            [
                {"document_id": uuid, "chunk_index": 0, "text": "...", "page_number": 1},
                {"document_id": uuid, "chunk_index": 1, "text": "...", "page_number": 1},
            ]
        """
        chunks = [Chunk(**data) for data in chunks_data]
        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def get_by_id(self, chunk_id: uuid.UUID) -> Optional[Chunk]:
        """Fetch a chunk by primary key."""
        return self.db.get(Chunk, chunk_id)

    def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """
        Return all chunks for a document, ordered by position.

        This is the primary retrieval path when re-embedding a document
        or inspecting what was ingested.
        """
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(self.db.execute(stmt).scalars().all())

    def count_by_document(self, document_id: uuid.UUID) -> int:
        """Return the number of chunks for a document."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(Chunk).where(
            Chunk.document_id == document_id
        )
        return self.db.execute(stmt).scalar_one()


# ---------------------------------------------------------------------------
# EmbeddingRepository
# ---------------------------------------------------------------------------

class EmbeddingRepository:
    """All database operations for the embeddings table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        chunk_id: uuid.UUID,
        model_name: str,
        vector: list[float],
    ) -> Embedding:
        """Insert a single embedding row."""
        embedding = Embedding(
            chunk_id=chunk_id,
            model_name=model_name,
            vector=vector,
        )
        self.db.add(embedding)
        self.db.flush()
        return embedding

    def bulk_create(self, embeddings_data: list[dict]) -> list[Embedding]:
        """
        Insert multiple embeddings in a single flush.

        Same efficiency rationale as ChunkRepository.bulk_create.
        """
        embeddings = [Embedding(**data) for data in embeddings_data]
        self.db.add_all(embeddings)
        self.db.flush()
        return embeddings

    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[Embedding]:
        """Fetch the embedding for a specific chunk."""
        stmt = select(Embedding).where(Embedding.chunk_id == chunk_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def find_similar(
    	self,
    	query_vector: list[float],
    	limit: int = 5,
    	model_name: str | None = None,
	) -> list[tuple[Chunk, float]]:
    	from sqlalchemy import cast, Float, text
    	from pgvector.sqlalchemy import Vector

    	query_vec = cast(query_vector, Vector(768))
    	distance_expr = Embedding.vector.op("<=>")(query_vec)

    	stmt = (
        	select(
            	Chunk,
            	distance_expr.cast(Float).label("distance"),  # ← explicit Float cast
        	)
        	.join(Embedding, Chunk.id == Embedding.chunk_id)
        	.order_by(distance_expr)
        	.limit(limit)
    	)

    	if model_name:
        	stmt = stmt.where(Embedding.model_name == model_name)

    	results = self.db.execute(stmt).all()
    	return [(row.Chunk, float(row.distance)) for row in results]

    def delete_by_chunk_id(self, chunk_id: uuid.UUID) -> bool:
        """
        Delete an embedding so it can be replaced (re-embedding workflow).

        When you upgrade embedding models, you delete the old embedding and
        insert a new one rather than updating in place. Cleaner audit trail.
        """
        stmt = delete(Embedding).where(Embedding.chunk_id == chunk_id)
        result = self.db.execute(stmt)
        self.db.flush()
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# EntityRepository
# ---------------------------------------------------------------------------

class EntityRepository:
    """All database operations for the entities table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def bulk_create(self, entities_data: list[dict]) -> list[Entity]:
        """
        Insert multiple entities in a single flush.

        Entities almost always come in batches (multiple per chunk),
        so bulk_create is the primary write path, not a convenience method.
        """
        entities = [Entity(**data) for data in entities_data]
        self.db.add_all(entities)
        self.db.flush()
        return entities

    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> list[Entity]:
        """Return all entities extracted from a specific chunk."""
        stmt = select(Entity).where(Entity.chunk_id == chunk_id)
        return list(self.db.execute(stmt).scalars().all())

    def get_by_label(
        self,
        label: str,
        document_id: uuid.UUID | None = None,
    ) -> list[Entity]:
        """
        Return all entities of a given type (e.g. all MONEY entities).

        Optional document_id filter scopes to a single document.
        This powers future features like "show me all revenue figures
        mentioned in this annual report."
        """
        stmt = (
            select(Entity)
            .join(Chunk, Entity.chunk_id == Chunk.id)
            .where(Entity.label == label)
        )
        if document_id:
            stmt = stmt.where(Chunk.document_id == document_id)

        return list(self.db.execute(stmt).scalars().all())

    def search_by_text(self, search_text: str) -> list[Entity]:
        """
        Case-insensitive text search across entity values.

        Uses PostgreSQL ILIKE for case-insensitive pattern matching.
        Useful for finding all mentions of a specific company or figure.
        """
        stmt = select(Entity).where(Entity.text.ilike(f"%{search_text}%"))
        return list(self.db.execute(stmt).scalars().all())
