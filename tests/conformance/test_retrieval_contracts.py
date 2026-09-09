"""The built-ins are held to the contracts third parties are handed.

Both existing suites had holes that only surfaced when somebody wrote a real
plugin against them — the store suite missed "an empty allow-set matches
nothing", the adapter suite could not test any adapter that binds a callable.
Running every built-in through these on the day they are written is the cheapest
way to find the next hole.
"""

from __future__ import annotations

import pytest

from toolbroker.index.embedders import HashingEmbedder
from toolbroker.observability.tracing import NullTracer, OTelTracer
from toolbroker.retrieve.hybrid import HybridRetriever
from toolbroker.retrieve.keyword import KeywordRetriever
from toolbroker.retrieve.rerank import UsageBooster
from toolbroker.retrieve.resilient import ResilientRetriever
from toolbroker.retrieve.semantic import SemanticRetriever
from toolbroker.store.memory import InMemoryStore
from toolbroker.testing import (
    RerankerConformanceSuite,
    RetrieverConformanceSuite,
    TracerConformanceSuite,
)
from toolbroker.types import ToolRecord
from toolbroker.usage import UsageTracker

DIM = 128


@pytest.fixture
def populated_store(request):
    """A store holding the suite's fixture tools, embedded consistently."""
    embedder = HashingEmbedder(dim=DIM)
    store = InMemoryStore(dim=DIM)
    tools = request.getfixturevalue("tools")
    texts = [f"{tool.name} {tool.description}" for tool in tools]
    vectors = embedder.embed(texts)
    store.upsert(
        [
            ToolRecord(tool=tool, text=text, vector=tuple(vector))
            for tool, text, vector in zip(tools, texts, vectors, strict=True)
        ]
    )
    return store


# --- retrievers ------------------------------------------------------------


class TestSemanticRetriever(RetrieverConformanceSuite):
    @pytest.fixture
    def retriever(self, populated_store):
        return SemanticRetriever(populated_store, HashingEmbedder(dim=DIM))


class TestKeywordRetriever(RetrieverConformanceSuite):
    @pytest.fixture
    def retriever(self, populated_store):
        return KeywordRetriever(populated_store)


class TestHybridRetriever(RetrieverConformanceSuite):
    @pytest.fixture
    def retriever(self, populated_store):
        return HybridRetriever(
            [
                SemanticRetriever(populated_store, HashingEmbedder(dim=DIM)),
                KeywordRetriever(populated_store),
            ]
        )


class TestResilientRetrieverHealthy(RetrieverConformanceSuite):
    """A wrapper must be indistinguishable from its primary when nothing fails."""

    @pytest.fixture
    def retriever(self, populated_store):
        return ResilientRetriever(
            SemanticRetriever(populated_store, HashingEmbedder(dim=DIM)),
            fallback=KeywordRetriever(populated_store),
            on_error="fallback",
        )


class TestResilientRetrieverDegraded(RetrieverConformanceSuite):
    """And the degraded path must satisfy the same contract, not a weaker one."""

    @pytest.fixture
    def retriever(self, populated_store):
        class Down:
            def retrieve(self, query, k, filters=None):
                raise ConnectionError("store unreachable")

        return ResilientRetriever(
            Down(), fallback=KeywordRetriever(populated_store), on_error="fallback"
        )


# --- rerankers -------------------------------------------------------------


class TestUsageBooster(RerankerConformanceSuite):
    @pytest.fixture
    def reranker(self):
        tracker = UsageTracker()
        tracker.record("bench/tool_1")
        return UsageBooster(tracker, weight=0.1)


class TestUsageBoosterWithNoHistory(RerankerConformanceSuite):
    """The first request after a cold start, where nothing has been recorded."""

    @pytest.fixture
    def reranker(self):
        return UsageBooster(UsageTracker(), weight=0.1)


# --- tracers ---------------------------------------------------------------


class TestNullTracer(TracerConformanceSuite):
    @pytest.fixture
    def tracer(self):
        return NullTracer()


class TestOTelTracer(TracerConformanceSuite):
    """Passes whether or not the OpenTelemetry API is installed."""

    @pytest.fixture
    def tracer(self):
        return OTelTracer()
