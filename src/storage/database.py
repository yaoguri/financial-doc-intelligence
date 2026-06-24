"""
Database connection and session management.

This module owns exactly one responsibility: producing database sessions.
No other module in the codebase should create engines or manage connections.

Key design decisions:
- Connection pool sized for a single-process app (pool_size=5)
- Sessions are context-managed — always closed after use, never leaked
- get_db() is a generator so FastAPI can use it as a dependency directly
- The engine is a module-level singleton — created once, reused always
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings


def build_engine() -> Engine:
    """
    Create the SQLAlchemy engine with connection pooling configured.

    pool_size=5       — keep 5 connections open and ready
    max_overflow=10   — allow up to 10 extra connections under load
    pool_pre_ping=True — test each connection before use (handles stale connections
                         after Docker restarts or network blips)
    echo=False        — set to True temporarily if you need to debug SQL queries
    """
    settings = get_settings()

    engine = create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )

    return engine


# Module-level singletons — created once when this module is first imported
engine: Engine = build_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,   # we manage transactions explicitly
    autoflush=False,    # we flush explicitly before queries that need fresh data
    expire_on_commit=False,  # keep attribute values accessible after commit
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that provides a database session and guarantees cleanup.

    Usage in application code:
        with get_db() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_id(some_id)

    Usage as FastAPI dependency (wrap it):
        def get_db_dependency():
            with get_db() as db:
                yield db

    The try/except/finally block ensures:
    - On success: session is committed and closed
    - On any exception: session is rolled back and closed
    This means callers never need to think about cleanup.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Verify the database is reachable. Used in health checks and startup.

    Returns True if connection succeeds, raises on failure.
    """
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
