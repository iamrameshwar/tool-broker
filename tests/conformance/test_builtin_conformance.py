"""The built-in components must pass the same contracts plugins do."""

from __future__ import annotations

import pytest

from toolbroker.adapters import AnthropicAdapter, OpenAIAdapter
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.store.memory import InMemoryStore
from toolbroker.testing import (
    AdapterConformanceSuite,
    EmbedderConformanceSuite,
    StoreConformanceSuite,
)

pytestmark = pytest.mark.conformance


class TestInMemoryStore(StoreConformanceSuite):
    @pytest.fixture
    def store(self):
        return InMemoryStore(self.DIM)


class TestOpenAIAdapter(AdapterConformanceSuite):
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()


class TestAnthropicAdapter(AdapterConformanceSuite):
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()


class TestHashingEmbedder(EmbedderConformanceSuite):
    @pytest.fixture
    def embedder(self):
        return HashingEmbedder(dim=256)


class TestFastEmbedEmbedder(EmbedderConformanceSuite):
    # A hosted or ONNX model is not always bit-identical run to run.
    SIMILARITY_TOLERANCE = 0.9999

    @pytest.fixture
    def embedder(self):
        fastembed = pytest.importorskip("fastembed")
        del fastembed
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder()
