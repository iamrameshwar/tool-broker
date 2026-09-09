"""Behaviour specific to the Qdrant backend, beyond the shared contract."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models
from toolbroker_qdrant import QdrantStore
from toolbroker_qdrant.store import build_filter, point_id

from toolbroker import Filters, RiskTier, Tool, ToolBroker
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.testing import make_records, make_tools
from toolbroker.types import ToolRecord

DIM = 128


@pytest.fixture
def store():
    return QdrantStore(dim=DIM, location=":memory:")


@pytest.fixture
def populated(store):
    store.upsert(make_records(make_tools(20), dim=DIM))
    return store


def record(name: str, vector, **kwargs) -> ToolRecord:
    return ToolRecord(
        tool=Tool(name=name, namespace="test", **kwargs), text=name, vector=tuple(vector)
    )


# -- point ids ------------------------------------------------------------


def test_point_ids_are_deterministic():
    # A tool must land on the same point across processes, or an upsert would
    # duplicate rather than replace.
    assert point_id("billing/issue_refund") == point_id("billing/issue_refund")


def test_different_tools_get_different_points():
    assert point_id("a/b") != point_id("a/c")


def test_upsert_replaces_rather_than_duplicating(store):
    store.upsert([record("a", [1.0] + [0.0] * (DIM - 1), description="first")])
    store.upsert([record("a", [1.0] + [0.0] * (DIM - 1), description="second")])
    assert len(store) == 1
    assert store.get("test/a").tool.description == "second"


# -- construction ---------------------------------------------------------


def test_rejects_non_positive_dim():
    with pytest.raises(ValueError, match="positive"):
        QdrantStore(dim=0)


def test_rejects_unknown_distance():
    with pytest.raises(ValueError, match="distance must be"):
        QdrantStore(dim=DIM, distance="hamming")


def test_accepts_a_shared_client():
    client = QdrantClient(location=":memory:")
    store = QdrantStore(dim=DIM, client=client, collection="shared")
    assert store.client is client


def test_reopening_a_collection_keeps_its_data():
    client = QdrantClient(location=":memory:")
    first = QdrantStore(dim=DIM, client=client, collection="persist")
    first.upsert(make_records(make_tools(3), dim=DIM))

    second = QdrantStore(dim=DIM, client=client, collection="persist")
    assert len(second) == 3


def test_dimension_change_on_an_existing_collection_is_refused():
    # Silently querying a collection built by a different embedder returns
    # confident nonsense, which is worse than failing.
    client = QdrantClient(location=":memory:")
    QdrantStore(dim=DIM, client=client, collection="fixed")
    with pytest.raises(StoreError, match="different embedder"):
        QdrantStore(dim=DIM * 2, client=client, collection="fixed")


def test_recreate_wipes_the_collection():
    client = QdrantClient(location=":memory:")
    first = QdrantStore(dim=DIM, client=client, collection="wipe")
    first.upsert(make_records(make_tools(3), dim=DIM))

    second = QdrantStore(dim=DIM, client=client, collection="wipe", recreate=True)
    assert len(second) == 0


def test_wrong_vector_width_is_rejected(store):
    with pytest.raises(DimensionMismatchError):
        store.upsert([record("a", [1.0, 0.0])])
    with pytest.raises(DimensionMismatchError):
        store.search([1.0, 0.0], k=1)


# -- filter translation ---------------------------------------------------


def test_empty_filters_translate_to_none():
    assert build_filter(None) is None
    assert build_filter(Filters()) is None


def test_namespace_filter_translates():
    built = build_filter(Filters(namespaces=frozenset({"billing"})))
    assert built.must[0].key == "namespace"


def test_tags_all_becomes_one_condition_per_tag():
    built = build_filter(Filters(tags_all=frozenset({"read", "safe"})))
    assert len(built.must) == 2


def test_exclusions_go_into_must_not():
    built = build_filter(Filters(exclude_tags=frozenset({"internal"})))
    assert built.must is None
    assert built.must_not[0].key == "tags"


def test_max_risk_becomes_an_ordinal_range():
    built = build_filter(Filters(max_risk=RiskTier.MEDIUM))
    condition = built.must[0]
    assert condition.key == "risk_level"
    assert condition.range.lte == 1.0


def test_risk_range_is_enforced_server_side(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    hits = populated.search(query, k=20, filters=Filters(max_risk=RiskTier.LOW))
    assert hits
    assert all(hit.tool.risk is RiskTier.LOW for hit in hits)


def test_combined_filters_intersect(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    hits = populated.search(
        query,
        k=20,
        filters=Filters(tags_any=frozenset({"even"}), max_risk=RiskTier.MEDIUM),
    )
    assert all("even" in hit.tool.tags for hit in hits)
    assert all(hit.tool.risk <= RiskTier.MEDIUM for hit in hits)


def test_filter_matching_nothing_returns_nothing(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    assert populated.search(query, k=5, filters=Filters(namespaces=frozenset({"absent"}))) == []


# -- round-tripping -------------------------------------------------------


def test_tool_metadata_survives_the_round_trip(store):
    tool = Tool(
        name="issue_refund",
        namespace="billing",
        description="Refund a payment",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
        tags=frozenset({"write", "money"}),
        risk=RiskTier.HIGH,
        required_scopes=frozenset({"payments:write"}),
    )
    store.upsert([ToolRecord(tool=tool, text="refund", vector=tuple([0.1] * DIM))])
    restored = store.get("billing/issue_refund").tool
    assert restored == tool


def test_vectors_survive_the_round_trip(store):
    vector = tuple([0.5] + [0.0] * (DIM - 1))
    store.upsert([record("a", vector)])
    stored = store.get("test/a").vector
    assert len(stored) == DIM
    assert stored[0] == pytest.approx(1.0, abs=1e-5)  # Qdrant normalises for cosine


def test_all_records_is_ordered_deterministically(populated):
    first = [record.id for record in populated.all_records()]
    second = [record.id for record in populated.all_records()]
    assert first == second == sorted(first)


def test_ties_break_by_tool_id(store):
    vector = [1.0] + [0.0] * (DIM - 1)
    store.upsert([record("zebra", vector), record("alpha", vector), record("mango", vector)])
    names = [hit.tool.name for hit in store.search(vector, k=3)]
    assert names == ["alpha", "mango", "zebra"]


def test_corrupt_payload_is_reported(store):
    store.client.upsert(
        collection_name=store.collection,
        points=[models.PointStruct(id=point_id("x/y"), vector=[0.0] * DIM, payload={})],
        wait=True,
    )
    with pytest.raises(StoreError, match="missing its 'tool' payload"):
        store.get("x/y")


# -- integration ----------------------------------------------------------


def test_drop_in_replacement_for_the_default_store():
    embedder = HashingEmbedder(dim=DIM)
    broker = ToolBroker(
        embedder=embedder,
        store=QdrantStore(dim=DIM, location=":memory:", collection="e2e"),
        cache_embeddings=False,
    )
    broker.index(
        [
            Tool(name="issue_refund", namespace="billing", description="Refund a payment"),
            Tool(name="check_inventory", namespace="shop", description="Stock level for a SKU"),
        ]
    )
    assert broker.select("refund a payment", k=1).tool_ids == ("billing/issue_refund",)


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_STORES, available, resolve

    assert "qdrant" in available(GROUP_STORES)
    assert resolve(GROUP_STORES, "qdrant") is QdrantStore
