"""The async entry points, and selecting for turn five rather than its sentence."""

from __future__ import annotations

import asyncio

import pytest

from toolbroker import Tool, ToolBroker, aio
from toolbroker.conversation import Conversation, Turn, contextual_query
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources.base import BaseSource

CATALOGUE = [
    Tool(name="issue_refund", namespace="billing", description="Refund a customer payment."),
    Tool(name="track_shipment", namespace="shipping", description="Locate a parcel in transit."),
    Tool(name="get_stock_level", namespace="inventory", description="Units on hand for a SKU."),
]


class Server(BaseSource):
    """A source whose tools and behaviour a test controls."""

    def __init__(self, source_id: str, tools: list[Tool], *, fail: bool = False) -> None:
        super().__init__(source_id)
        self.tools = list(tools)
        self.fail = fail
        self.calls = 0

    def _discover(self) -> list[Tool]:
        self.calls += 1
        if self.fail:
            raise ConnectionError(f"{self.id} unreachable")
        return list(self.tools)


class AsyncServer(Server):
    """A source with a native async path, which must be preferred."""

    def __init__(self, source_id: str, tools: list[Tool]) -> None:
        super().__init__(source_id, tools)
        self.async_calls = 0

    async def adiscover(self) -> list[Tool]:
        self.async_calls += 1
        await asyncio.sleep(0)
        return list(self.tools)


def _broker() -> ToolBroker:
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(CATALOGUE)
    return broker


# --- aselect ---------------------------------------------------------------


async def test_aselect_matches_select():
    """The point of not forking the logic: the two cannot disagree."""
    broker = _broker()
    assert (await aio.aselect(broker, "refund a payment", k=2)).tool_ids == broker.select(
        "refund a payment", k=2
    ).tool_ids


async def test_aselect_passes_arguments_through():
    broker = _broker()
    selection = await aio.aselect(broker, "refund a payment", k=1, agent=None)
    assert len(selection.hits) == 1


async def test_aselect_does_not_block_the_event_loop():
    """A slow retriever must not stop other coroutines from running."""
    broker = _broker()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0)
            ticks += 1

    await asyncio.gather(aio.aselect(broker, "refund", k=1), ticker())
    assert ticks == 5


async def test_concurrent_selects_all_succeed():
    broker = _broker()
    results = await asyncio.gather(
        *(aio.aselect(broker, "refund a payment", k=1) for _ in range(10))
    )
    assert all(result.tool_ids for result in results)


async def test_aselect_for_renders():
    broker = _broker()
    rendered, selection = await aio.aselect_for(broker, "openai", "refund a payment", k=1)
    assert len(rendered) == 1
    assert selection.tool_ids


# --- adiscover -------------------------------------------------------------


async def test_adiscover_collects_from_every_source():
    sources = [Server("a", CATALOGUE[:1]), Server("b", CATALOGUE[1:])]
    tools, failures = await aio.adiscover(sources)
    assert len(tools) == 3
    assert failures == []


async def test_adiscover_prefers_a_native_async_source():
    source = AsyncServer("a", CATALOGUE)
    await aio.adiscover([source])
    assert source.async_calls == 1
    assert source.calls == 0


async def test_a_failing_source_does_not_fail_the_batch():
    """One unreachable server must not empty the catalogue."""
    sources = [Server("good", CATALOGUE[:1]), Server("bad", [], fail=True)]
    tools, failures = await aio.adiscover(sources)
    assert len(tools) == 1
    assert [source_id for source_id, _ in failures] == ["bad"]


async def test_discovery_is_concurrent():
    started = 0
    peak = 0

    class Slow(BaseSource):
        def __init__(self, source_id: str) -> None:
            super().__init__(source_id)

        def _discover(self) -> list[Tool]:
            nonlocal started, peak
            started += 1
            peak = max(peak, started)
            import time

            time.sleep(0.02)
            started -= 1
            return []

    await aio.adiscover([Slow(f"s{i}") for i in range(4)], concurrency=4)
    assert peak > 1, "sources were contacted one at a time"


async def test_concurrency_is_bounded():
    started = 0
    peak = 0

    class Slow(BaseSource):
        def __init__(self, source_id: str) -> None:
            super().__init__(source_id)

        def _discover(self) -> list[Tool]:
            nonlocal started, peak
            started += 1
            peak = max(peak, started)
            import time

            time.sleep(0.02)
            started -= 1
            return []

    await aio.adiscover([Slow(f"s{i}") for i in range(8)], concurrency=2)
    assert peak <= 2


# --- aindex / arefresh -----------------------------------------------------


async def test_aindex_indexes():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_source(Server("a", CATALOGUE))
    await aio.aindex(broker)
    assert len(broker.tools()) == 3


async def test_aindex_raises_when_everything_failed():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_source(Server("a", [], fail=True))
    with pytest.raises(RuntimeError, match="every source failed"):
        await aio.aindex(broker)


async def test_arefresh_applies_only_the_diff():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    source = Server("a", CATALOGUE)
    broker.add_source(source)
    await aio.aindex(broker)

    source.tools = [*CATALOGUE, Tool(name="new", namespace="x", description="A new tool.")]
    report = await aio.arefresh(broker)
    assert report.added == ("x/new",)
    assert report.embedded == 1


async def test_arefresh_never_prunes_a_source_that_failed():
    """The rule the synchronous path already holds: an outage strips nothing."""
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    good = Server("good", CATALOGUE[:1])
    bad = Server("bad", CATALOGUE[1:])
    broker.add_source(good)
    broker.add_source(bad)
    await aio.aindex(broker)

    bad.fail = True
    report = await aio.arefresh(broker)
    assert report.removed == ()
    assert len(broker.tools()) == 3
    assert [source_id for source_id, _ in report.failed_sources] == ["bad"]


# --- conversation ----------------------------------------------------------


def test_a_single_turn_is_just_the_query():
    assert contextual_query([("user", "refund a customer")], weight=1) == "refund a customer"


def test_history_is_folded_in():
    query = contextual_query(
        [("user", "what is in the Berlin warehouse"), ("user", "check stock there")],
        weight=1,
    )
    assert "Berlin" in query
    assert "check stock there" in query


def test_the_current_turn_is_weighted():
    query = contextual_query([("user", "the Berlin warehouse"), ("user", "check stock")], weight=3)
    assert query.count("check stock") == 3
    assert query.count("the Berlin warehouse") == 1


def test_weight_one_disables_the_repetition():
    query = contextual_query([("user", "a"), ("user", "b")], weight=1)
    assert query.count("b") == 1


def test_assistant_turns_are_excluded_by_default():
    query = contextual_query(
        [("user", "a question"), ("assistant", "a long generated answer"), ("user", "and now")],
    )
    assert "generated" not in query


def test_assistant_turns_can_be_included():
    query = contextual_query(
        [("user", "a question"), ("assistant", "a long generated answer"), ("user", "and now")],
        include_assistant=True,
    )
    assert "generated" in query


def test_history_depth_is_bounded():
    turns = [("user", f"turn {index}") for index in range(6)]
    query = contextual_query(turns, max_turns=2, weight=1)
    assert "turn 3" in query and "turn 4" in query
    assert "turn 0" not in query


def test_zero_history_is_the_bare_query():
    turns = [("user", "context"), ("user", "the ask")]
    assert contextual_query(turns, max_turns=0, weight=1) == "the ask"


def test_an_overlong_turn_is_dropped_from_history():
    """A pasted document is not context; it swamps the actual question."""
    turns = [("user", "x" * 500), ("user", "the ask")]
    assert "x" * 500 not in contextual_query(turns, weight=1)


def test_empty_input_is_an_empty_query():
    assert contextual_query([]) == ""


def test_blank_turns_are_ignored():
    assert contextual_query([("user", "   "), ("user", "the ask")], weight=1) == "the ask"


def test_turn_objects_and_tuples_agree():
    pairs = [("user", "context"), ("user", "the ask")]
    objects = [Turn("user", "context"), Turn("user", "the ask")]
    assert contextual_query(pairs) == contextual_query(objects)


def test_conversation_accumulates_and_chains():
    conversation = Conversation().add_user("a").add_assistant("b").add_user("c")
    assert len(conversation) == 3
    assert conversation.latest == "c"


def test_conversation_query_matches_the_function():
    conversation = Conversation(max_turns=1, weight=2)
    conversation.add_user("the Berlin warehouse").add_user("check stock")
    assert conversation.query() == contextual_query(conversation.turns, max_turns=1, weight=2)


def test_latest_is_empty_before_any_user_turn():
    assert Conversation().add_assistant("hello").latest == ""


def test_history_reaches_retrieval():
    """The mechanism: terms from an earlier turn are retrievable in a later one.

    Deliberately not a quality claim. These tests use the lexical embedder, so a
    *semantic* rescue cannot be shown here however the wording is picked — the
    earlier turn has to share a literal term for anything to move. Proving that
    context helps needs a real embedder, and `bench/multiturn.py` measures it
    there: recall goes 0.400 to 0.633 at a thousand tools.

    So this uses "SKU", which exactly one tool's description contains. If the
    history reaches retrieval at all, that tool wins; if it does not, nothing
    connects the two turns.
    """
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(CATALOGUE)

    conversation = Conversation()
    conversation.add_user("what is the SKU count in Berlin")
    conversation.add_user("check it")

    assert "SKU" in conversation.query()
    assert broker.select(conversation.query(), k=1).tool_ids == ("inventory/get_stock_level",)
    assert broker.select("check it", k=1).tool_ids != ("inventory/get_stock_level",)
