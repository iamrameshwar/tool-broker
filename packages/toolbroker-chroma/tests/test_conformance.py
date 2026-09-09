"""ChromaStore must satisfy the same contract every other store does."""

from __future__ import annotations

import uuid

import pytest
from toolbroker_chroma import ChromaStore

from toolbroker.testing import StoreConformanceSuite


class TestChromaStore(StoreConformanceSuite):
    """No overrides, no exemptions.

    The only concession is a unique collection name per test. Chroma's
    ``EphemeralClient()`` returns a process-wide shared client, so two stores
    built with the same collection name in one process are the same collection —
    correct behaviour, but it means the fixture has to isolate itself to hand
    back a genuinely empty store.
    """

    @pytest.fixture
    def store(self):
        return ChromaStore(dim=self.DIM, collection=f"conformance-{uuid.uuid4().hex}")
