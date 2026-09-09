from __future__ import annotations

import pytest

from toolbroker.hooks import DROP, Event, HookManager


def test_observers_receive_the_payload():
    seen = []
    manager = HookManager()
    manager.register(Event.AFTER_RETRIEVAL, lambda **kw: seen.append(kw["query"]))
    manager.emit(Event.AFTER_RETRIEVAL, query="hello")
    assert seen == ["hello"]


def test_transformers_chain_in_registration_order():
    manager = HookManager()
    manager.register(Event.TRANSFORM_QUERY, lambda value: value + " one")
    manager.register(Event.TRANSFORM_QUERY, lambda value: value + " two")
    assert manager.transform(Event.TRANSFORM_QUERY, "start") == "start one two"


def test_returning_none_leaves_the_value_alone():
    manager = HookManager()
    manager.register(Event.TRANSFORM_QUERY, lambda value: None)
    assert manager.transform(Event.TRANSFORM_QUERY, "unchanged") == "unchanged"


def test_returning_drop_removes_the_item():
    manager = HookManager()
    manager.register(Event.TRANSFORM_TOOL, lambda value: DROP)
    assert manager.transform_or_drop(Event.TRANSFORM_TOOL, "anything") is DROP


def test_drop_short_circuits_later_handlers():
    calls = []
    manager = HookManager()
    manager.register(Event.TRANSFORM_TOOL, lambda value: DROP)
    manager.register(Event.TRANSFORM_TOOL, lambda value: calls.append(value))
    manager.transform_or_drop(Event.TRANSFORM_TOOL, "x")
    assert calls == []


def test_plain_transform_ignores_drop():
    # Dropping a query or a finished selection is meaningless, so a handler
    # that asks for it is buggy: log and carry on rather than empty the
    # pipeline silently.
    manager = HookManager()
    manager.register(Event.TRANSFORM_QUERY, lambda value: DROP)
    assert manager.transform(Event.TRANSFORM_QUERY, "unchanged") == "unchanged"


def test_a_broken_handler_is_logged_and_skipped():
    manager = HookManager()

    def broken(value):
        raise RuntimeError("boom")

    manager.register(Event.TRANSFORM_QUERY, broken)
    manager.register(Event.TRANSFORM_QUERY, lambda value: value + "!")
    assert manager.transform(Event.TRANSFORM_QUERY, "x") == "x!"


def test_strict_mode_reraises():
    manager = HookManager(strict=True)

    def broken(value):
        raise RuntimeError("boom")

    manager.register(Event.TRANSFORM_QUERY, broken)
    with pytest.raises(RuntimeError):
        manager.transform(Event.TRANSFORM_QUERY, "x")


def test_decorator_registration():
    manager = HookManager()

    @manager.on(Event.TRANSFORM_QUERY)
    def upper(value):
        return value.upper()

    assert manager.transform(Event.TRANSFORM_QUERY, "x") == "X"


def test_unregister():
    manager = HookManager()

    def handler(value):
        return "changed"

    manager.register(Event.TRANSFORM_QUERY, handler)
    assert manager.unregister(Event.TRANSFORM_QUERY, handler) is True
    assert manager.unregister(Event.TRANSFORM_QUERY, handler) is False
    assert manager.transform(Event.TRANSFORM_QUERY, "x") == "x"


def test_drop_is_falsy():
    assert not DROP
    assert repr(DROP) == "DROP"
