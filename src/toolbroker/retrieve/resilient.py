"""What happens when the vector store is unreachable.

Until now, nothing. A Qdrant restart or a Postgres failover raised out of
:meth:`~toolbroker.catalog.ToolBroker.select` and the agent was handed no tools
at all — which does not look like an outage to a language model, it looks like
a world with no capabilities in it. The model apologises and the user is told
the thing cannot be done.

The right behaviour is a decision the operator has to make, not one this
library should make for them:

* ``"fail"`` — raise. Correct when a wrong answer is worse than no answer.
* ``"empty"`` — return nothing, but do it deliberately, so the caller can
  detect it and say "temporarily unavailable" rather than hallucinate.
* ``"fallback"`` — try a second retriever. A lexical one over the same store's
  cached records needs no external service, so it survives exactly the outage
  that takes the vector search down.

    pipeline = ResilientRetriever(
        primary,
        fallback=KeywordRetriever(store),
        on_error="fallback",
    )

The rule that makes this safe
-----------------------------

**A degraded path returns fewer or worse tools. It never returns tools the
policy would have refused.** This class sits *inside* retrieval, upstream of
:class:`~toolbroker.policy.engine.PolicyEngine`, so every result it produces —
primary, fallback or empty — is filtered by exactly the same policy on exactly
the same code path. There is no branch that skips it, and the tests assert
that, because "graceful degradation" that quietly widens access is the shape of
a real privilege-escalation bug rather than a resilience feature.
"""

from __future__ import annotations

from typing import Literal

from ..observability import get_logger, span
from ..protocols import Retriever
from ..types import Filters, Hit

logger = get_logger("retrieve.resilient")

#: What to do when the primary retriever raises.
OnError = Literal["fail", "empty", "fallback"]


class ResilientRetriever:
    """Wraps a retriever with a configured answer to "the store is down"."""

    def __init__(
        self,
        primary: Retriever,
        *,
        fallback: Retriever | None = None,
        on_error: OnError = "fail",
    ) -> None:
        """Configure the degradation behaviour.

        Args:
            primary: The retriever used when everything is healthy.
            fallback: Used when ``on_error`` is ``"fallback"``. A lexical
                retriever is the useful choice: it needs no external service,
                so it survives the outage that took the primary down.
            on_error: ``"fail"`` re-raises, ``"empty"`` returns nothing,
                ``"fallback"`` tries ``fallback``.

        Raises:
            ValueError: If ``on_error`` is ``"fallback"`` with no fallback.
        """
        if on_error == "fallback" and fallback is None:
            raise ValueError('on_error="fallback" needs a fallback retriever')
        self._primary = primary
        self._fallback = fallback
        self._on_error = on_error
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """Whether the last retrieval had to fall back or return empty.

        Worth alerting on. A silently degraded retriever answers every request
        with worse tools and nothing fails, which is the failure mode that
        survives longest in production.
        """
        return self._degraded

    @property
    def primary(self) -> Retriever:
        """The healthy-path retriever."""
        return self._primary

    @property
    def fallback(self) -> Retriever | None:
        """The degraded-path retriever, if configured."""
        return self._fallback

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Retrieve, degrading as configured if the primary raises."""
        try:
            hits = self._primary.retrieve(query, k, filters)
        except Exception as exc:
            return self._degrade(exc, query, k, filters)
        self._degraded = False
        return hits

    def _degrade(
        self,
        exc: Exception,
        query: str,
        k: int,
        filters: Filters | None,
    ) -> list[Hit]:
        """Apply the configured response to a primary failure."""
        if self._on_error == "fail":
            raise exc

        self._degraded = True
        if self._on_error == "empty":
            logger.error(
                "retrieval failed; returning no tools",
                extra={"error": str(exc), "mode": "empty"},
            )
            return []

        if self._fallback is None:  # pragma: no cover - __init__ guarantees this
            raise exc
        logger.error(
            "retrieval failed; falling back",
            extra={"error": str(exc), "mode": "fallback"},
        )
        with span("toolbroker.retrieve.fallback", k=k):
            try:
                return self._fallback.retrieve(query, k, filters)
            except Exception as fallback_exc:
                # Both stages are down. Raising the original is more useful:
                # it names the thing that actually broke, and the fallback's
                # failure is usually a consequence of the same outage.
                logger.error(
                    "fallback retrieval also failed",
                    extra={"error": str(fallback_exc)},
                )
                raise exc from fallback_exc


def resilient(
    primary: Retriever,
    fallback: Retriever | None = None,
    *,
    on_error: OnError = "fallback",
) -> ResilientRetriever:
    """Convenience constructor with the useful default."""
    return ResilientRetriever(primary, fallback=fallback, on_error=on_error)


__all__ = ["OnError", "ResilientRetriever", "resilient"]
