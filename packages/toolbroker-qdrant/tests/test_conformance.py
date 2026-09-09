"""QdrantStore must satisfy the same contract the in-memory store does."""

from __future__ import annotations

import pytest
from toolbroker_qdrant import QdrantStore

from toolbroker.testing import StoreConformanceSuite


class TestQdrantStore(StoreConformanceSuite):
    """The whole point of the conformance suite: no overrides, no exemptions."""

    @pytest.fixture
    def store(self):
        # In-process Qdrant. No server, no container, no network.
        return QdrantStore(dim=self.DIM, location=":memory:")
