"""Behaviour specific to the Chroma backend, beyond the shared contract."""

from __future__ import annotations

import uuid

import pytest
from toolbroker_chroma import ChromaStore
from toolbroker_chroma.store import build_where, tag_key

from toolbroker import Filters, RiskTier, Tool, ToolBroker
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.testing import make_records, make_tools
from toolbroker.types import ToolRecord

DIM = 128


def fresh(**kwargs) -> ChromaStore:
    kwargs.setdefault("collection", f"test-{uuid.uuid4().hex}")
    kwargs.setdefault("dim", DIM)
    return ChromaStore(**kwargs)


@pytest.fixture
def store():
    return fresh()


@pytest.fixture
def populated(store):
    store.upsert(make_records(make_tools(20), dim=DIM))
    return store


def record(name: str, vector, **kwargs) -> ToolRecord:
    return ToolRecord(
        tool=Tool(name=name, namespace="test", **kwargs), text=name, vector=tuple(vector)
    )


# -- construction ---------------------------------------------------------


def test_rejects_non_positive_dim():
    with pytest.raises(ValueError, match="positive"):
        fresh(dim=0)


def test_rejects_unknown_space():
    with pytest.raises(ValueError, match="space must be"):
        fresh(space="hamming")


@pytest.mark.parametrize("name", ["ab", "-leading", "trailing-", "has spaces", "sym!bols"])
def test_invalid_collection_names_fail_at_construction(name):
    # Chroma's own error for this arrives late and reads badly; catch it here.
    with pytest.raises(StoreError, match="invalid Chroma collection name"):
        ChromaStore(dim=DIM, collection=name)


def test_dimension_change_on_an_existing_collection_is_refused():
    name = f"fixed-{uuid.uuid4().hex}"
    ChromaStore(dim=DIM, collection=name)
    with pytest.raises(StoreError, match="different embedder"):
        ChromaStore(dim=DIM * 2, collection=name)


def test_recreate_wipes_the_collection():
    name = f"wipe-{uuid.uuid4().hex}"
    first = ChromaStore(dim=DIM, collection=name)
    first.upsert(make_records(make_tools(3), dim=DIM))
    assert len(ChromaStore(dim=DIM, collection=name, recreate=True)) == 0


def test_persistent_client_writes_to_disk(tmp_path):
    name = f"disk-{uuid.uuid4().hex}"
    first = ChromaStore(dim=DIM, path=str(tmp_path), collection=name)
    first.upsert(make_records(make_tools(4), dim=DIM))

    second = ChromaStore(dim=DIM, path=str(tmp_path), collection=name)
    assert len(second) == 4


def test_wrong_vector_width_is_rejected(store):
    with pytest.raises(DimensionMismatchError):
        store.upsert([record("a", [1.0, 0.0])])
    with pytest.raises(DimensionMismatchError):
        store.search([1.0, 0.0], k=1)


# -- where translation ----------------------------------------------------


def test_empty_filters_translate_to_none():
    assert build_where(None) is None
    assert build_where(Filters()) is None


def test_tag_keys_are_sanitised():
    assert tag_key("read-only") == "tag_read_only"
    assert tag_key("a.b/c") == "tag_a_b_c"


def test_single_clause_is_not_wrapped_in_and():
    assert build_where(Filters(namespaces=frozenset({"billing"}))) == {
        "namespace": {"$in": ["billing"]}
    }


def test_multiple_clauses_are_joined_with_and():
    built = build_where(Filters(namespaces=frozenset({"a"}), max_risk=RiskTier.LOW))
    assert set(built) == {"$and"}
    assert len(built["$and"]) == 2


def test_tags_any_becomes_an_or():
    built = build_where(Filters(tags_any=frozenset({"read", "write"})))
    assert built == {"$or": [{"tag_read": True}, {"tag_write": True}]}


def test_single_tag_any_skips_the_or():
    assert build_where(Filters(tags_any=frozenset({"read"}))) == {"tag_read": True}


@pytest.mark.parametrize(
    "filters",
    [
        Filters(tags_any=frozenset()),
        Filters(namespaces=frozenset()),
        Filters(tool_ids=frozenset()),
    ],
)
def test_empty_allow_sets_match_nothing(filters):
    # Dropping the clause would silently widen the query to everything, which
    # is the dangerous direction for something used as a guard rail.
    built = build_where(filters)
    assert built == {"tool_id": {"$eq": "//toolbroker-never-matches"}}


def test_empty_tags_any_returns_no_hits(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    assert populated.search(query, k=5, filters=Filters(tags_any=frozenset())) == []


def test_exclude_tags_uses_ne_so_missing_keys_still_match():
    assert build_where(Filters(exclude_tags=frozenset({"internal"}))) == {
        "tag_internal": {"$ne": True}
    }


def test_exclusion_keeps_tools_that_never_had_the_tag(store):
    store.upsert(
        [
            record("tagged", [1.0] + [0.0] * (DIM - 1), tags=frozenset({"internal"})),
            record("untagged", [0.9, 0.1] + [0.0] * (DIM - 2)),
        ]
    )
    hits = store.search(
        [1.0] + [0.0] * (DIM - 1), k=5, filters=Filters(exclude_tags=frozenset({"internal"}))
    )
    assert [hit.tool.name for hit in hits] == ["untagged"]


def test_max_risk_uses_an_ordinal_comparison():
    assert build_where(Filters(max_risk=RiskTier.MEDIUM)) == {"risk_level": {"$lte": 1}}


def test_filters_run_before_k(populated):
    # The contract: ask for 3 low-risk tools, get 3 -- not whatever survives
    # filtering the global top 3.
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    hits = populated.search(query, k=3, filters=Filters(max_risk=RiskTier.LOW))
    assert len(hits) == 3
    assert all(hit.tool.risk is RiskTier.LOW for hit in hits)


# -- scoring and round-tripping -------------------------------------------


def test_identical_vector_scores_near_one(store):
    vector = [1.0] + [0.0] * (DIM - 1)
    store.upsert([record("a", vector)])
    assert store.search(vector, k=1)[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_ordered_best_first(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3 topic 3")
    scores = [hit.score for hit in populated.search(query, k=10)]
    assert scores == sorted(scores, reverse=True)


def test_tool_survives_the_round_trip(store):
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
    assert store.get("billing/issue_refund").tool == tool


def test_tool_with_no_tags_round_trips(store):
    store.upsert([record("plain", [0.2] * DIM)])
    assert store.get("test/plain").tool.tags == frozenset()


def test_all_records_is_ordered_deterministically(populated):
    first = [r.id for r in populated.all_records()]
    assert first == sorted(first)
    assert first == [r.id for r in populated.all_records()]


def test_ties_break_by_tool_id(store):
    vector = [1.0] + [0.0] * (DIM - 1)
    store.upsert([record("zebra", vector), record("alpha", vector), record("mango", vector)])
    assert [h.tool.name for h in store.search(vector, k=3)] == ["alpha", "mango", "zebra"]


def test_k_larger_than_the_collection_is_clamped(store):
    # Chroma errors rather than clamping when n_results exceeds the count.
    store.upsert([record("only", [1.0] + [0.0] * (DIM - 1))])
    assert len(store.search([1.0] + [0.0] * (DIM - 1), k=50)) == 1


def test_corrupt_metadata_is_reported(store):
    store.collection.upsert(
        ids=["x/y"], embeddings=[[0.0] * DIM], metadatas=[{"tool_id": "x/y"}], documents=["d"]
    )
    with pytest.raises(StoreError, match="missing its 'tool_json'"):
        store.get("x/y")


# -- integration ----------------------------------------------------------


def test_drop_in_replacement_for_the_default_store():
    broker = ToolBroker(embedder=HashingEmbedder(dim=DIM), store=fresh(), cache_embeddings=False)
    broker.index(
        [
            Tool(name="issue_refund", namespace="billing", description="Refund a payment"),
            Tool(name="check_inventory", namespace="shop", description="Stock level for a SKU"),
        ]
    )
    assert broker.select("refund a payment", k=1).tool_ids == ("billing/issue_refund",)


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_STORES, available, resolve

    assert "chroma" in available(GROUP_STORES)
    assert resolve(GROUP_STORES, "chroma") is ChromaStore
