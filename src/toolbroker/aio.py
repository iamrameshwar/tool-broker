"""Async entry points, for the frameworks that are async.

Agent frameworks are overwhelmingly async — LangGraph, the OpenAI Agents SDK,
anything behind FastAPI — and until now the hot path was not. Every ``select()``
blocked the event loop, and a caller who noticed had to wrap it in a thread
themselves. The mismatch was already inside the library: MCP discovery is
natively async and the catalogue consuming it was not.

    from toolbroker import aio

    selection = await aio.aselect(broker, "refund a customer", k=5)
    report = await aio.arefresh(broker)

Three things this deliberately does and does not do.

**It does not fork the selection logic.** ``aselect`` runs the same ``select``
off the event loop rather than reimplementing it. Hooks, policy, exclusions and
timings stay one implementation, so an async caller can never drift into a
different set of rules from a sync one — and the tests assert the two return
equal selections rather than merely that both worked.

**It does not pretend retrieval is async.** No retriever in the tree does
non-blocking I/O, so ``aselect`` uses :func:`asyncio.to_thread`. That solves the
real problem — a blocked event loop — without inventing API surface for a native
path nobody has implemented yet. When a store plugin grows one, this is where it
gets used.

**It does make discovery concurrent**, because that is where the wall-clock time
actually goes. Thirty MCP servers contacted one after another is thirty
round-trips in series; the bound exists so a large catalogue does not open every
connection at once without being asked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .observability import get_logger, span
from .types import Selection, Tool

if TYPE_CHECKING:
    from .catalog import ToolBroker
    from .index.indexer import IndexReport, RefreshReport

logger = get_logger("aio")

#: Sources contacted at once during discovery.
DEFAULT_CONCURRENCY = 8


async def aselect(broker: ToolBroker, query: str, **kwargs: Any) -> Selection:
    """Await a selection without blocking the event loop.

    Takes the same arguments as :meth:`~toolbroker.catalog.ToolBroker.select`
    and returns the same :class:`~toolbroker.types.Selection`.
    """
    with span("toolbroker.aselect", k=kwargs.get("k")):
        return await asyncio.to_thread(lambda: broker.select(query, **kwargs))


async def aselect_for(
    broker: ToolBroker,
    adapter: str | Any,
    query: str,
    **kwargs: Any,
) -> tuple[Any, Selection]:
    """Select and render, awaited. Rendering is pure CPU, so it stays inline."""
    selection = await aselect(broker, query, **kwargs)
    return broker.render(adapter, selection), selection


async def adiscover(
    sources: Sequence[Any],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[list[Tool], list[tuple[str, str]]]:
    """Discover from every source concurrently.

    A source exposing ``adiscover`` is awaited natively; one exposing only
    ``discover`` is run in a worker thread, so a blocking HTTP client costs a
    thread rather than the loop.

    Returns:
        ``(tools, failures)``, where ``failures`` is ``(source_id, error)``. A
        source that fails does not fail the batch — the same reason
        :meth:`~toolbroker.catalog.ToolBroker.refresh` refuses to prune it.
    """
    limiter = asyncio.Semaphore(max(1, concurrency))

    async def one(source: Any) -> tuple[str, list[Tool] | Exception]:
        async with limiter:
            try:
                if hasattr(source, "adiscover"):
                    return source.id, list(await source.adiscover())
                return source.id, list(await asyncio.to_thread(source.discover))
            except Exception as exc:
                return source.id, exc

    gathered = await asyncio.gather(*(one(source) for source in sources))

    tools: list[Tool] = []
    failures: list[tuple[str, str]] = []
    for source_id, outcome in gathered:
        if isinstance(outcome, Exception):
            logger.warning("source %s failed during discovery: %s", source_id, outcome)
            failures.append((source_id, str(outcome)))
        else:
            tools.extend(outcome)
    return tools, failures


async def aindex(
    broker: ToolBroker,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> IndexReport:
    """Discover concurrently, then index off the event loop."""
    tools, failures = await adiscover(broker.sources, concurrency=concurrency)
    if failures and not tools:
        raise RuntimeError(f"every source failed during discovery: {failures}")
    return await asyncio.to_thread(broker.index, tools)


async def arefresh(
    broker: ToolBroker,
    *,
    prune: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> RefreshReport:
    """Re-discover concurrently and apply only the diff.

    Sources that failed are reported rather than pruned, exactly as in the
    synchronous path: a thirty-second outage must not silently strip an agent's
    capabilities.
    """
    tools, failures = await adiscover(broker.sources, concurrency=concurrency)
    failed = {source_id for source_id, _ in failures}
    answered = [source.id for source in broker.sources if source.id not in failed]
    return await asyncio.to_thread(
        lambda: broker.sync_tools(
            tools,
            prune=prune,
            known_sources=answered,
            failed_sources=failures,
        )
    )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "adiscover",
    "aindex",
    "arefresh",
    "aselect",
    "aselect_for",
]
