"""Tracing, with the provider swappable.

Tracing is how a team answers "why did this agent get those tools" in
production, and the answer to "which tracing vendor" is never the same twice.
So the core depends on a :class:`~toolbroker.protocols.Tracer` protocol — one
method returning a context manager — and ships two implementations: an
OpenTelemetry one that activates when the API is importable, and a no-op.

Anything else plugs in the same way a store or an embedder does:

    from toolbroker.observability import set_tracer

    set_tracer(MyDatadogTracer())

or declaratively, resolved through the ``toolbroker.tracers`` entry-point
group::

    observability:
      tracer:
        name: my_pkg.tracing:DatadogTracer
        options: {service: agent-gateway}

It must never become a hard dependency, so the default degrades to a no-op
rather than an ImportError, and a tracer that raises is disabled rather than
allowed to take a request down with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .logging import get_logger

if TYPE_CHECKING:
    from ..protocols import Tracer

logger = get_logger("tracing")


class NullTracer:
    """Records nothing. The default when no tracing backend is available."""

    @property
    def enabled(self) -> bool:
        """Always ``False``; there is nothing behind this tracer."""
        return False

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Do nothing, cheaply."""
        yield


class OTelTracer:
    """Emits OpenTelemetry spans, if the API is installed."""

    def __init__(self, service: str = "toolbroker") -> None:
        """Acquire a tracer, degrading to disabled when OTel is absent."""
        self._tracer: Any | None = None
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer(service)
        except ImportError:  # pragma: no cover - depends on extras
            self._tracer = None

    @property
    def enabled(self) -> bool:
        """Whether the OpenTelemetry API was importable."""
        return self._tracer is not None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Open an OpenTelemetry span, or do nothing if unavailable."""
        if self._tracer is None:
            yield
            return
        with self._tracer.start_as_current_span(name) as current:
            for key, value in attributes.items():
                if value is not None:
                    current.set_attribute(f"toolbroker.{key}", value)
            yield


_active: Tracer = OTelTracer() if OTelTracer().enabled else NullTracer()


def set_tracer(tracer: Tracer | None) -> None:
    """Install ``tracer`` as the process-wide tracer, or ``None`` to disable."""
    global _active
    _active = tracer if tracer is not None else NullTracer()


def get_tracer() -> Tracer:
    """Return the active tracer."""
    return _active


def tracing_enabled() -> bool:
    """Whether the active tracer records anything."""
    return bool(getattr(_active, "enabled", True))


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Open a span on the active tracer.

    A tracer that raises is replaced with the no-op rather than allowed to fail
    the request. Observability is not worth an outage, and a vendor SDK that
    throws on a misconfigured endpoint is a real way to cause one.
    """
    tracer = _active
    if isinstance(tracer, NullTracer):
        yield
        return
    try:
        manager = tracer.span(name, **attributes)
        entered = manager.__enter__()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tracer failed to open a span; disabling tracing", extra={"error": str(exc)})
        set_tracer(None)
        yield
        return
    try:
        yield entered
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tracer failed to close a span", extra={"error": str(exc)})
