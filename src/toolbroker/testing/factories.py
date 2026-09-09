"""Deterministic test data.

Shared by the core test suite and by plugin authors, so a bug reproduced
against one store can be replayed against another without rewriting fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..index.embedders.hashing import HashingEmbedder
from ..types import RiskTier, Tool, ToolRecord


def make_tool(
    name: str,
    description: str = "",
    *,
    namespace: str = "test",
    tags: Iterable[str] = (),
    risk: RiskTier = RiskTier.LOW,
    **kwargs: object,
) -> Tool:
    """Build a tool with sensible test defaults."""
    return Tool(
        name=name,
        namespace=namespace,
        description=description,
        tags=frozenset(tags),
        risk=risk,
        **kwargs,  # type: ignore[arg-type]
    )


def make_tools(count: int, *, namespace: str = "test") -> list[Tool]:
    """Build ``count`` distinct tools with varied risk and tags.

    Deliberately heterogeneous: a store that only ever sees identical tools
    will pass filter tests it should fail.
    """
    tiers = list(RiskTier)
    return [
        make_tool(
            f"tool_{index:03d}",
            f"Tool number {index} which handles topic {index % 7}",
            namespace=namespace,
            tags={f"topic_{index % 7}", "even" if index % 2 == 0 else "odd"},
            risk=tiers[index % len(tiers)],
        )
        for index in range(count)
    ]


def make_records(tools: Sequence[Tool], dim: int = 128) -> list[ToolRecord]:
    """Embed ``tools`` with the offline hashing embedder."""
    embedder = HashingEmbedder(dim=dim)
    texts = [f"{tool.name} {tool.description}" for tool in tools]
    vectors = embedder.embed(texts)
    return [
        ToolRecord(tool=tool, text=text, vector=tuple(vector))
        for tool, text, vector in zip(tools, texts, vectors, strict=True)
    ]
