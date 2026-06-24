"""
Alembic environment configuration.

Key decisions:
- Pulls database URL from src.config (same source as the app)
- Imports Base.metadata so Alembic can detect model changes automatically
- Supports both offline (SQL script) and online (live DB) migration modes
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import your models' Base so Alembic can see the schema
from src.storage.models import Base

# Import settings the same way the app does
from src.config import get_settings

# Alembic Config object — provides .ini file values
config = context.config

# Set up Python logging from the .ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is what enables autogenerate (alembic revision --autogenerate)
# Alembic compares this metadata against the live DB to find differences
target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection.
    Useful for generating SQL scripts to review before applying.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations against the live database.
    This is the normal mode — what 'alembic upgrade head' uses.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # no pooling for migrations — one connection, then done
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()