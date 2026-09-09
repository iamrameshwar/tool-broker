from __future__ import annotations

import pytest

from toolbroker import Filters, RiskTier, Tool
from toolbroker.errors import DimensionMismatchError
from toolbroker.store.memory import InMemoryStore
from toolbroker.types import ToolRecord


def record(name: str, vector, **kwargs) -> ToolRecord:
    return ToolRecord(
        tool=Tool(name=name, namespace="test", **kwargs), text=name, vector=tuple(vector)
    )


def test_upsert_and_len(store):
    store.upsert([record("a", [1.0] + [0.0] * 127)])
    assert len(store) == 1


def test_upsert_replaces_by_id(store):
    store.upsert([record("a", [1.0] + [0.0] * 127, description="first")])
    store.upsert([record("a", [0.0, 1.0] + [0.0] * 126, description="second")])
    assert len(store) == 1
    assert store.get("test/a").tool.description == "second"


def test_search_ranks_by_similarity(store):
    store.upsert(
        [
            record("close", [1.0] + [0.0] * 127),
            record("far", [0.0] * 127 + [1.0]),
        ]
    )
    hits = store.search([1.0] + [0.0] * 127, k=2)
    assert [hit.tool.name for hit in hits] == ["close", "far"]
    assert hits[0].score > hits[1].score


def test_search_applies_filters_before_ranking(store):
    store.upsert(
        [
            record("risky", [1.0] + [0.0] * 127, risk=RiskTier.CRITICAL),
            record("safe", [0.9, 0.1] + [0.0] * 126, risk=RiskTier.LOW),
        ]
    )
    hits = store.search([1.0] + [0.0] * 127, k=1, filters=Filters(max_risk=RiskTier.LOW))
    assert [hit.tool.name for hit in hits] == ["safe"]


def test_filtering_happens_before_k_is_applied(store):
    # The point of pre-filtering: asking for 1 low-risk tool returns 1, even
    # though the highest-scoring tool overall is filtered out.
    store.upsert(
        [record(f"high{i}", [1.0] + [0.0] * 127, risk=RiskTier.HIGH) for i in range(5)]
        + [record("low", [0.5] * 128, risk=RiskTier.LOW)]
    )
    hits = store.search([1.0] + [0.0] * 127, k=1, filters=Filters(max_risk=RiskTier.LOW))
    assert len(hits) == 1


def test_delete_removes_and_reports_count(store):
    store.upsert([record("a", [1.0] + [0.0] * 127), record("b", [0.0, 1.0] + [0.0] * 126)])
    assert store.delete(["test/a"]) == 1
    assert store.get("test/a") is None
    assert len(store) == 1


def test_delete_keeps_search_working(store):
    store.upsert([record("a", [1.0] + [0.0] * 127), record("b", [0.0, 1.0] + [0.0] * 126)])
    store.delete(["test/a"])
    hits = store.search([0.0, 1.0] + [0.0] * 126, k=5)
    assert [hit.tool.name for hit in hits] == ["b"]


def test_dimension_mismatch_is_rejected(store):
    with pytest.raises(DimensionMismatchError):
        store.upsert([record("a", [1.0, 0.0])])
    with pytest.raises(DimensionMismatchError):
        store.search([1.0, 0.0], k=1)


def test_empty_store_returns_nothing(store):
    assert store.search([1.0] + [0.0] * 127, k=5) == []


def test_zero_k_returns_nothing(store):
    store.upsert([record("a", [1.0] + [0.0] * 127)])
    assert store.search([1.0] + [0.0] * 127, k=0) == []


def test_ties_break_deterministically(store):
    vector = [1.0] + [0.0] * 127
    store.upsert([record("zebra", vector), record("alpha", vector), record("mango", vector)])
    names = [hit.tool.name for hit in store.search(vector, k=3)]
    assert names == ["alpha", "mango", "zebra"]


def test_clear_empties_the_store(store):
    store.upsert([record("a", [1.0] + [0.0] * 127)])
    store.clear()
    assert len(store) == 0


def test_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="positive"):
        InMemoryStore(0)
