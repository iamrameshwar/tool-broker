"""Exception hierarchy.

Every error raised by ToolBroker derives from :class:`ToolBrokerError`, so callers
can guard an entire integration with a single ``except`` without swallowing
unrelated failures.
"""

from __future__ import annotations


class ToolBrokerError(Exception):
    """Base class for every error raised by this library."""


class ConfigurationError(ToolBrokerError):
    """A component was constructed or wired with invalid settings."""


class PluginError(ConfigurationError):
    """A plugin could not be resolved, imported, or instantiated."""


class SourceError(ToolBrokerError):
    """A tool source failed to discover tools."""


class EmbeddingError(ToolBrokerError):
    """An embedder failed to produce vectors."""


class StoreError(ToolBrokerError):
    """A vector store rejected a read or write."""


class DimensionMismatchError(StoreError):
    """A vector's dimensionality does not match the store's."""

    def __init__(self, expected: int, actual: int) -> None:
        """Record both dimensionalities in the message."""
        super().__init__(f"expected vectors of dimension {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


class RetrievalError(ToolBrokerError):
    """A retriever failed to produce results."""


class PolicyError(ToolBrokerError):
    """A policy could not be evaluated."""


class PolicyViolationError(PolicyError):
    """A hard policy constraint was violated and the caller asked to fail loudly."""


class AdapterError(ToolBrokerError):
    """A tool could not be rendered into a target framework's shape."""


class NotIndexedError(ToolBrokerError):
    """A query was issued before the catalogue was indexed."""
