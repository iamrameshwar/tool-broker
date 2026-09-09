"""PgVectorStore against the shared store contract, on a real Postgres."""

from __future__ import annotations

import pytest
from toolbroker_pgvector import PgVectorStore

from toolbroker.testing import StoreConformanceSuite

from .conftest import requires_postgres


@requires_postgres
class TestPgVectorStore(StoreConformanceSuite):
    """No overrides, no exemptions.

    Each test gets its own table: several backends hand out a process-wide
    shared handle, so a fixture has to isolate itself to return a genuinely
    empty store.
    """

    @pytest.fixture
    def store(self, dsn, table):
        created = PgVectorStore(dim=self.DIM, dsn=dsn, table=table)
        yield created
        created.drop()
        created.close()
