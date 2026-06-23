"""
Central configuration module.

All settings are loaded from environment variables (or .env file).
Never import settings directly from os.environ elsewhere in the codebase —
always import from this module. This gives you one place to change,
validate, and document every configurable value.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")
    postgres_db: str = Field(..., description="PostgreSQL database name")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # Tesseract
    tesseract_path: str = Field(
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    # Chunking
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # Retrieval
    top_k_retrieval: int = Field(default=5)

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def database_url(self) -> str:
        """Constructs the full SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Single instance imported everywhere else in the codebase
settings = Settings()