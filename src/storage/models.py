"""
SQLAlchemy ORM models for the Financial Document Intelligence Platform.

Design principles:
- One class per table, mirroring the schema exactly
- UUIDs as primary keys (no sequential integer IDs that leak row counts)
- All timestamps are timezone-aware (TIMESTAMPTZ in PostgreSQL)
- Relationships defined with back_populates for bidirectional navigation
- pgvector's Vector type used directly for the embeddings column
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


# ---------------------------------------------------------------------------
# Base class — all models inherit from this
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base.

    Using the modern DeclarativeBase (SQLAlchemy 2.0+) rather than the legacy
    declarative_base() function. This gives us better type hints and IDE support.
    All models inherit from this class so Alembic can discover them automatically.
    """
    pass


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

class Document(Base):
    """
    One row per uploaded PDF file.

    The file_hash column (SHA-256 of file contents) is the duplicate-detection
    mechanism. Before inserting, the pipeline checks whether this hash already
    exists — if so, the file has already been processed and we skip it.

    The processed flag is set to True only after all chunks, embeddings, and
    entities have been successfully written. This gives us a clean way to find
    and retry failed ingestion runs.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    upload_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # PostgreSQL sets this, not Python
    )
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationship: one document has many chunks
    # cascade="all, delete-orphan" means deleting a document deletes its chunks
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} processed={self.processed}>"


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class Chunk(Base):
    """
    One row per text chunk produced by the TextChunker.

    chunk_index is the zero-based position of this chunk within its document.
    Together with document_id, it uniquely identifies a chunk's position.

    char_start and char_end record the character offsets within the original
    extracted text. This allows us to highlight source passages in the UI later.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    embedding: Mapped["Embedding | None"] = relationship(
        "Embedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,  # one-to-one: each chunk has at most one embedding
    )
    entities: Mapped[list["Entity"]] = relationship(
        "Entity",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} index={self.chunk_index} doc={self.document_id}>"


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

class Embedding(Base):
    """
    One row per chunk, storing the vector representation.

    Separated from Chunk so we can re-embed with a different model without
    touching chunk text. The model_name column records which embedding model
    produced this vector — critical for debugging and future model upgrades.

    Vector(768) matches nomic-embed-text's output dimension. If you switch
    models, you'd need a migration to change this dimension.

    The UniqueConstraint on chunk_id enforces the one-to-one relationship at
    the database level, not just the application level. Defense in depth.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_embeddings_chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationship
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<Embedding id={self.id} model={self.model_name!r} chunk={self.chunk_id}>"


# ---------------------------------------------------------------------------
# Entity model
# ---------------------------------------------------------------------------

class Entity(Base):
    """
    One row per named entity extracted from a chunk by spaCy NER.

    Multiple entities per chunk are expected and normal. A single earnings
    transcript chunk might contain several ORG, MONEY, and DATE entities.

    start_char and end_char are offsets within the chunk text (not the full
    document), matching spaCy's span offsets directly.
    """

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationship
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="entities")

    def __repr__(self) -> str:
        return f"<Entity id={self.id} label={self.label!r} text={self.text!r}>"
