"""The six interfaces every component implements.

These are ``Protocol`` classes, not base classes, on purpose: a contributor can
satisfy one with any object that has the right methods, including objects that
already exist in their codebase. Nothing here imports a framework.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ._vectors import Vector
    from .types import Filters, Hit, PolicyResult, Tool, ToolRecord


@runtime_checkable
class Source(Protocol):
    """Where tool definitions come from: MCP servers, OpenAPI, Python functions."""

    @property
    def id(self) -> str:
        """Stable identifier, stamped onto every tool this source yields."""
        ...

    def discover(self) -> Iterable[Tool]:
        """Yield the tools currently exposed by this source."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors produced."""
        ...

    @property
    def id(self) -> str:
        """Identifier including model name, used as a cache and index key."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Separate from :meth:`embed` because asymmetric models prefix queries and
        documents differently.
        """
        ...


@runtime_checkable
class Store(Protocol):
    """Holds tool records and answers nearest-neighbour queries."""

    @property
    def dim(self) -> int:
        """Dimensionality this store was built for."""
        ...

    def upsert(self, records: Sequence[ToolRecord]) -> None:
        """Insert or replace records, keyed by tool id."""
        ...

    def delete(self, tool_ids: Sequence[str]) -> int:
        """Remove records by tool id, returning how many were removed."""
        ...

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return the ``k`` closest records that satisfy ``filters``."""
        ...

    def get(self, tool_id: str) -> ToolRecord | None:
        """Return one record by tool id, or ``None``."""
        ...

    def all_records(self) -> Sequence[ToolRecord]:
        """Return every stored record. Used by lexical retrievers and the CLI."""
        ...

    def clear(self) -> None:
        """Drop everything."""
        ...

    def __len__(self) -> int:
        """Number of stored records."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Turns a query into ranked candidates."""

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return up to ``k`` scored candidates for ``query``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates using a more expensive signal than the retriever."""

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Return up to ``k`` reordered hits."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Decides which retrieved tools an agent is actually allowed to see."""

    def evaluate(
        self,
        hits: Sequence[Hit],
        *,
        agent: str | None = None,
        scopes: frozenset[str] = frozenset(),
        query: str = "",
        tenant: str | None = None,
    ) -> PolicyResult:
        """Apply rules to ``hits`` and record every decision made."""
        ...


@runtime_checkable
class Adapter(Protocol):
    """Renders tools into the shape a specific framework expects."""

    @property
    def id(self) -> str:
        """Identifier of the target framework."""
        ...

    def render(self, tools: Sequence[Tool]) -> Any:
        """Return the framework-native representation of ``tools``."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Opens spans around the stages worth timing in production.

    Deliberately one method. Every APM vendor can satisfy it, and a core that
    asked for more would be picking a winner among them.
    """

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Any]:
        """Return a context manager covering the named operation."""
        ...
