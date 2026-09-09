"""Shared source behaviour."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from ..observability import get_logger
from ..protocols import Source
from ..types import Tool

logger = get_logger("sources")


class BaseSource(ABC):
    """Convenience base that stamps ``source_id`` onto every discovered tool.

    Implementing :class:`~toolbroker.protocols.Source` directly is fine; this
    just removes the boilerplate.
    """

    def __init__(self, source_id: str) -> None:
        """Store the source identifier."""
        self._id = source_id

    @property
    def id(self) -> str:
        """Stable identifier for this source."""
        return self._id

    @abstractmethod
    def _discover(self) -> Iterable[Tool]:
        """Yield tools without worrying about the ``source_id`` stamp."""

    def discover(self) -> list[Tool]:
        """Return tools, each stamped with this source's id.

        Eager rather than lazy: a generator would defer connection errors to
        whenever the caller happened to iterate, which puts the traceback a
        long way from the cause.
        """
        return [
            tool if tool.source_id == self._id else tool.model_copy(update={"source_id": self._id})
            for tool in self._discover()
        ]

    def __repr__(self) -> str:
        """Show the class and id."""
        return f"{type(self).__name__}(id={self._id!r})"


class CompositeSource(BaseSource):
    """Presents several sources as one, skipping any that fail.

    One unreachable MCP server should degrade the catalogue, not break indexing
    entirely. Failures are logged and surfaced on :attr:`failures`.
    """

    def __init__(self, sources: Sequence[Source], source_id: str = "composite") -> None:
        """Wrap ``sources``."""
        super().__init__(source_id)
        self._sources = list(sources)
        self.failures: list[tuple[str, Exception]] = []

    def _discover(self) -> Iterable[Tool]:
        self.failures.clear()
        for source in self._sources:
            try:
                yield from source.discover()
            except Exception as exc:
                self.failures.append((source.id, exc))
                logger.warning(
                    "source failed during discovery; continuing without it",
                    extra={"source_id": source.id, "error": str(exc)},
                )

    def discover(self) -> list[Tool]:
        """Return tools from every child, preserving each child's own source id."""
        return list(self._discover())
