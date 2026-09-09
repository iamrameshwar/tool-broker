from __future__ import annotations

import os
import uuid

import pytest

DSN = os.environ.get("TOOLBROKER_PG_DSN", "")


def _reachable(dsn: str) -> bool:
    if not dsn:
        return False
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as connection:
            connection.execute("SELECT 1")
    except Exception:
        return False
    return True


LIVE = _reachable(DSN)

requires_postgres = pytest.mark.skipif(
    not LIVE,
    reason=(
        "no reachable Postgres; set TOOLBROKER_PG_DSN, e.g.\n"
        "  docker run -d -e POSTGRES_PASSWORD=toolbroker -e POSTGRES_DB=toolbroker \\\n"
        "    -p 55432:5432 pgvector/pgvector:pg16"
    ),
)


@pytest.fixture
def dsn() -> str:
    """Connection string for the test database."""
    return DSN


@pytest.fixture
def table() -> str:
    """A unique table name, so tests cannot leak state into one another."""
    return f"ts_{uuid.uuid4().hex[:16]}"
