"""
Setup smoke tests.
Run these after initial environment setup to verify all components
are reachable and correctly configured.
"""

import pytest
from src.config import settings


def test_settings_load():
    """Config module can read .env values."""
    assert settings.postgres_user == "finuser"
    assert settings.chunk_size == 512
    assert "postgresql" in settings.database_url


def test_database_connection():
    """PostgreSQL container is reachable and pgvector is installed."""
    import psycopg2
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    assert cur.fetchone()[0] == 1

    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    result = cur.fetchone()
    assert result is not None, "pgvector extension not found"

    conn.close()


def test_ollama_reachable():
    """Ollama server is running and the configured model exists."""
    import ollama
    response = ollama.list()
    models = response.get("models", [])
    model_names = [m.get("name", "") for m in models]
    assert any(settings.ollama_model in name for name in model_names), \
        f"Model '{settings.ollama_model}' not found. Run: ollama pull {settings.ollama_model}"


def test_tesseract_available():
    """Tesseract binary is accessible at the configured path."""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
    version = pytesseract.get_tesseract_version()
    assert version is not None