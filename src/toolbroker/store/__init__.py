"""Vector stores. In-memory ships in core; servers ship as plugins."""

from .memory import InMemoryStore

__all__ = ["InMemoryStore"]
