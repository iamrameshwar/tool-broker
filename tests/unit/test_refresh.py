"""Incremental refresh: cost, correctness, and the outage safety valve."""

from __future__ import annotations

import pytest

from toolbroker import RiskTier, Tool, ToolBroker
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources.base import BaseSource


class CountingEmbedder:
    """Wraps the hashing embedder and counts how many texts it embedded."""

    def __init__(self, dim: int = 128) -> None:
        self._inner = HashingEmbedder(dim=dim)
        self.embedded = 0

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def id(self) -> str:
        return "counting"

    def embed(self, texts):
        self.embedded += len(texts)
        return self._inner.embed(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


class MutableSource(BaseSource):
    """A source whose tool list the test controls."""

    def __init__(self, source_id: str, tools=()):
        super().__init__(source_id)
        self.tools = list(tools)
        self.error: Exception | None = None
        self.calls = 0

    def _discover(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.tools)


def tool(name: str, namespace: str = "srv", **kwargs) -> Tool:
    kwargs.setdefault("description", f"Does {name}.")
    return Tool(name=name, namespace=namespace, **kwargs)


@pytest.fixture
def embedder():
    return CountingEmbedder()


@pytest.fixture
def source():
    return MutableSource("srv", [tool("alpha"), tool("beta"), tool("gamma")])


@pytest.fixture
def broker(embedder, source):
    catalogue = ToolBroker(embedder=embedder, cache_embeddings=False)
    catalogue.add_source(source)
    catalogue.refresh()
    embedder.embedded = 0  # ignore the initial index
    return catalogue


# -- cost -----------------------------------------------------------------


def test_unchanged_catalogue_embeds_nothing(broker, embedder):
    report = broker.refresh()
    assert embedder.embedded == 0
    assert report.unchanged == 3
    assert not report.changed


def test_adding_one_tool_embeds_one(broker, embedder, source):
    source.tools.append(tool("delta"))
    report = broker.refresh()
    assert embedder.embedded == 1
    assert report.added == ("srv/delta",)
    assert report.unchanged == 3


def test_changing_a_description_re_embeds_only_that_tool(broker, embedder, source):
    source.tools[0] = tool("alpha", description="Completely different now.")
    report = broker.refresh()
    assert embedder.embedded == 1
    assert report.updated == ("srv/alpha",)


def test_metadata_only_change_updates_without_embedding(broker, embedder, source):
    # Risk and scopes are read by policy, so the record must be rewritten --
    # but the indexed text is unchanged, so the vector is still correct.
    source.tools[0] = tool("alpha", risk=RiskTier.CRITICAL)
    report = broker.refresh()
    assert report.updated == ("srv/alpha",)
    assert embedder.embedded == 0
    assert broker.get("srv/alpha").risk is RiskTier.CRITICAL


def test_metadata_change_keeps_the_tool_searchable(broker, source):
    source.tools[0] = tool("alpha", risk=RiskTier.CRITICAL)
    broker.refresh()
    assert "srv/alpha" in broker.select("does alpha", k=3).tool_ids


# -- correctness ----------------------------------------------------------


def test_removed_tools_are_pruned(broker, source):
    source.tools = source.tools[:1]
    report = broker.refresh()
    assert report.removed == ("srv/beta", "srv/gamma")
    assert len(broker) == 1


def test_prune_can_be_disabled(broker, source):
    source.tools = source.tools[:1]
    report = broker.refresh(prune=False)
    assert report.removed == ()
    assert len(broker) == 3


def test_added_tools_are_immediately_selectable(broker, source):
    source.tools.append(tool("refund_payment", description="Refund a customer payment."))
    broker.refresh()
    assert "srv/refund_payment" in broker.select("refund a customer payment", k=2).tool_ids


def test_removed_tools_stop_being_selectable(broker, source):
    source.tools = [t for t in source.tools if t.name != "beta"]
    broker.refresh()
    assert "srv/beta" not in broker.select("does beta", k=5).tool_ids


def test_report_totals_add_up(broker, source):
    source.tools = [tool("alpha"), tool("beta", description="new"), tool("delta")]
    report = broker.refresh()
    assert report.total == len(broker)
    assert report.summary().startswith("+1 -1 ~1 =1")


# -- the outage safety valve ----------------------------------------------


def test_a_failing_source_does_not_delete_its_tools(broker, source):
    # A thirty second outage must not silently strip capabilities from every
    # agent with no error anywhere the caller would look.
    source.error = RuntimeError("connection refused")
    report = broker.refresh()
    assert report.removed == ()
    assert len(broker) == 3


def test_a_failing_source_is_reported(broker, source):
    source.error = RuntimeError("connection refused")
    report = broker.refresh()
    assert report.failed_sources == (("srv", "connection refused"),)
    assert "1 source(s) failed" in report.summary()


def test_a_healthy_source_still_prunes_when_another_fails(embedder):
    good = MutableSource("good", [tool("a", namespace="good"), tool("b", namespace="good")])
    bad = MutableSource("bad", [tool("x", namespace="bad")])
    catalogue = ToolBroker(embedder=embedder, cache_embeddings=False)
    catalogue.add_source(good)
    catalogue.add_source(bad)
    catalogue.refresh()

    bad.error = RuntimeError("down")
    good.tools = good.tools[:1]
    report = catalogue.refresh()

    assert report.removed == ("good/b",)  # the healthy source pruned
    assert catalogue.get("bad/x") is not None  # the failed one did not


def test_recovery_after_an_outage(broker, source):
    source.error = RuntimeError("down")
    broker.refresh()
    source.error = None
    source.tools = source.tools[:1]
    report = broker.refresh()
    assert report.removed == ("srv/beta", "srv/gamma")


def test_discover_still_raises_for_callers_who_want_it(broker, source):
    source.error = RuntimeError("down")
    with pytest.raises(RuntimeError):
        broker.discover()


# -- bookkeeping ----------------------------------------------------------


def test_last_refresh_is_recorded(broker):
    assert broker.last_refresh is not None
    report = broker.refresh()
    assert broker.last_refresh is report


def test_refresh_marks_the_catalogue_indexed(embedder, source):
    catalogue = ToolBroker(embedder=embedder, cache_embeddings=False)
    catalogue.add_source(source)
    catalogue.refresh()
    assert len(catalogue.select("does alpha", k=1)) == 1


def test_refresh_on_an_empty_catalogue(embedder):
    catalogue = ToolBroker(embedder=embedder, cache_embeddings=False)
    report = catalogue.refresh()
    assert not report.changed
    assert report.total == 0


def test_duplicate_ids_collapse_during_sync(broker, source):
    source.tools.append(tool("alpha", description="a duplicate id"))
    report = broker.refresh()
    assert report.updated == ("srv/alpha",)
    assert len(broker) == 3
