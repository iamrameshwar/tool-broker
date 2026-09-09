"""Observation and mutation points around each pipeline stage.

Hooks are how a team injects behaviour we did not anticipate — redacting a
description before indexing, boosting tools their users actually click, logging
every denial to their own audit system — without forking the core.

Two kinds:

* **Observers** (``on_*``) see the stage's result and return nothing.
* **Transformers** (``transform_*``) return a replacement value; returning
  ``None`` leaves the value unchanged, and returning :data:`DROP` removes the
  item entirely.

``None`` and :data:`DROP` are deliberately different. A handler that only cares
about some tools returns ``None`` for the rest, and that must not be mistaken
for "delete this tool" — the failure mode would be a handler silently emptying
a catalogue.

Handlers run in registration order. A failing handler is logged and skipped
rather than taking down the retrieval, unless ``strict=True``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Final, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
Handler = Callable[..., Any]


class _Drop(Enum):
    """Sentinel type for :data:`DROP`.

    An ``Enum`` with one member, because that gives a true singleton, correct
    static narrowing, and ``is`` comparison without hand-written ``__new__``
    machinery.
    """

    TOKEN = "DROP"

    def __repr__(self) -> str:
        """Render as ``DROP``."""
        return "DROP"

    def __bool__(self) -> bool:
        """Always falsy, so a truthiness check also treats it as absent."""
        return False


DROP: Final = _Drop.TOKEN
"""Return this from a transformer to remove the item from the pipeline."""


class Event(str, Enum):
    """Points in the pipeline a handler can attach to."""

    BEFORE_DISCOVERY = "before_discovery"
    AFTER_DISCOVERY = "after_discovery"
    TRANSFORM_TOOL = "transform_tool"
    TRANSFORM_INDEX_TEXT = "transform_index_text"
    AFTER_INDEXING = "after_indexing"
    BEFORE_RETRIEVAL = "before_retrieval"
    TRANSFORM_QUERY = "transform_query"
    TRANSFORM_HITS = "transform_hits"
    AFTER_RETRIEVAL = "after_retrieval"
    BEFORE_POLICY = "before_policy"
    AFTER_POLICY = "after_policy"
    AFTER_SELECTION = "after_selection"


class HookManager:
    """Registry and dispatcher for hook handlers."""

    def __init__(self, *, strict: bool = False) -> None:
        """Create an empty manager.

        Args:
            strict: Re-raise handler exceptions instead of logging and skipping.
        """
        self._handlers: dict[Event, list[Handler]] = defaultdict(list)
        self._strict = strict

    def register(self, event: Event, handler: Handler) -> None:
        """Attach ``handler`` to ``event``."""
        self._handlers[event].append(handler)

    def unregister(self, event: Event, handler: Handler) -> bool:
        """Detach ``handler``; return whether it was attached."""
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)
            return True
        return False

    def on(self, event: Event) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`register`."""

        def decorator(handler: Handler) -> Handler:
            self.register(event, handler)
            return handler

        return decorator

    def handlers(self, event: Event) -> Sequence[Handler]:
        """Return the handlers attached to ``event``."""
        return tuple(self._handlers.get(event, ()))

    def emit(self, event: Event, **kwargs: Any) -> None:
        """Run every observer for ``event``, ignoring return values."""
        for handler in self._handlers.get(event, ()):
            try:
                handler(**kwargs)
            except Exception:
                if self._strict:
                    raise
                logger.exception("hook %s handler %r failed", event.value, handler)

    def transform(self, event: Event, value: T, **kwargs: Any) -> T:
        """Chain transformers for ``event``, threading ``value`` through each.

        For events where dropping is meaningless — a query, a piece of index
        text, a finished selection. A handler that returns :data:`DROP` here is
        buggy, so it is logged and ignored rather than silently emptying the
        pipeline.
        """
        result = self._run(event, value, kwargs)
        if isinstance(result, _Drop):
            logger.warning(
                "hook %s returned DROP, which this event does not support; ignoring",
                event.value,
            )
            return value
        return result

    def transform_or_drop(self, event: Event, value: T, **kwargs: Any) -> T | _Drop:
        """Like :meth:`transform`, but a handler may return :data:`DROP`.

        Used for per-item events such as :attr:`Event.TRANSFORM_TOOL`, where
        removing the item is a legitimate outcome. Later handlers do not run on
        a dropped item.
        """
        return self._run(event, value, kwargs)

    def _run(self, event: Event, value: T, kwargs: dict[str, Any]) -> T | _Drop:
        """Chain handlers, returning the final value or :data:`DROP`."""
        current: T = value
        for handler in self._handlers.get(event, ()):
            try:
                result = handler(current, **kwargs)
            except Exception:
                if self._strict:
                    raise
                logger.exception("hook %s handler %r failed", event.value, handler)
                continue
            # isinstance, not `is DROP`: comparing against an Any-typed
            # handler result widens the sentinel back to Any for the checker.
            if isinstance(result, _Drop):
                return result
            if result is not None:
                current = cast("T", result)
        return current
