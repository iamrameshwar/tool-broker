"""What a `pip install` user can swap without touching the core.

These tests exist because "everything is pluggable" is a claim that rots. Each
one wires in a component the core has never heard of, the way someone who
installed the package rather than cloning the repo would have to.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from toolbroker import Hit
from toolbroker.config import ToolBrokerConfig
from toolbroker.errors import ConfigurationError, PluginError
from toolbroker.hooks import Event
from toolbroker.observability import NullTracer, get_tracer, set_tracer, span
from toolbroker.protocols import Tracer
from toolbroker.registry import (
    GROUP_RERANKERS,
    GROUP_RETRIEVERS,
    GROUP_TRACERS,
    construct,
    register,
    unregister,
)
from toolbroker.retrieve.resilient import ResilientRetriever

# --- stand-ins for whatever a user would actually bring -------------------


class VendorRetriever:
    """A retriever the core has never heard of. Wants the store, not the embedder."""

    def __init__(self, store: Any, flavour: str = "plain") -> None:
        self.store = store
        self.flavour = flavour

    def retrieve(self, query: str, k: int, filters: Any = None) -> list[Hit]:
        records = list(self.store.all_records())[:k]
        return [Hit(tool=record.tool, score=1.0) for record in records]


class StandaloneRetriever:
    """Wants nothing at all. Plenty of real ones are like this."""

    def __init__(self, tools: Sequence[Any] = ()) -> None:
        self.tools = tools

    def retrieve(self, query: str, k: int, filters: Any = None) -> list[Hit]:
        return []


class GreedyRetriever:
    """Takes ``**kwargs``, so it should receive every collaborator on offer."""

    def __init__(self, **kwargs: Any) -> None:
        self.got = kwargs

    def retrieve(self, query: str, k: int, filters: Any = None) -> list[Hit]:
        return []


class ReverseReranker:
    """A reranker with no dependencies, so the ordering change is observable."""

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        return list(reversed(list(hits)))[:k]


class RecordingTracer:
    """A stand-in for Datadog, Sentry, or anything else with its own SDK."""

    def __init__(self, service: str = "test") -> None:
        self.service = service
        self.spans: list[str] = []

    @contextmanager
    def span(self, name: str, **attributes: Any):
        self.spans.append(name)
        yield


class ExplodingTracer:
    """A vendor SDK that throws on a misconfigured endpoint."""

    def span(self, name: str, **attributes: Any):
        raise RuntimeError("APM endpoint unreachable")


@pytest.fixture(autouse=True)
def _restore_tracer():
    original = get_tracer()
    yield
    set_tracer(original)


@pytest.fixture
def registered():
    """Register the stand-ins, then clean up so tests stay independent."""
    entries = [
        (GROUP_RETRIEVERS, "vendor", VendorRetriever),
        (GROUP_RETRIEVERS, "standalone", StandaloneRetriever),
        (GROUP_RETRIEVERS, "greedy", GreedyRetriever),
        (GROUP_RERANKERS, "reverse", ReverseReranker),
        (GROUP_TRACERS, "recording", RecordingTracer),
    ]
    for group, name, obj in entries:
        register(group, name, obj)
    yield
    for group, name, _ in entries:
        unregister(group, name)


TOOLS = [
    {"name": "refund", "namespace": "billing", "description": "Refund money to a customer."},
    {"name": "ship", "namespace": "logistics", "description": "Ship a parcel to an address."},
]


@pytest.fixture
def tools_file(tmp_path) -> Path:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(TOOLS), encoding="utf-8")
    return path


def _config(tools_file: Path, **retrieval: Any) -> dict[str, Any]:
    return {
        "sources": [{"type": "json", "name": "static", "options": {"path": str(tools_file)}}],
        "retrieval": retrieval,
    }


# --- signature-aware construction -----------------------------------------


def test_a_component_receives_only_what_it_asks_for(registered, broker):
    built = construct(GROUP_RETRIEVERS, "vendor", {}, store=broker.store, embedder=broker.embedder)
    assert built.store is broker.store
    assert not hasattr(built, "embedder")


def test_a_component_that_wants_nothing_is_not_forced_to_take_anything(registered, broker):
    built = construct(
        GROUP_RETRIEVERS, "standalone", {}, store=broker.store, embedder=broker.embedder
    )
    assert isinstance(built, StandaloneRetriever)


def test_a_component_taking_kwargs_receives_everything(registered, broker):
    built = construct(GROUP_RETRIEVERS, "greedy", {}, store=broker.store, embedder=broker.embedder)
    assert set(built.got) == {"store", "embedder"}


def test_config_options_override_an_injected_collaborator(registered, broker):
    """A user must be able to say 'no, use mine' for anything we would inject."""
    sentinel = object()
    built = construct(GROUP_RETRIEVERS, "vendor", {"store": sentinel}, store=broker.store)
    assert built.store is sentinel


def test_options_reach_the_constructor(registered, broker):
    built = construct(GROUP_RETRIEVERS, "vendor", {"flavour": "spicy"}, store=broker.store)
    assert built.flavour == "spicy"


def test_an_unknown_name_names_what_is_installed(broker):
    with pytest.raises(PluginError, match="Installed:"):
        construct(GROUP_RETRIEVERS, "no_such_retriever", {})


def test_a_dotted_path_needs_no_registration(broker):
    """The case that matters for a pip install: no entry point, no registration."""
    built = construct(
        GROUP_RETRIEVERS,
        "tests.unit.test_configurability:VendorRetriever",
        {},
        store=broker.store,
    )
    assert isinstance(built, VendorRetriever)


# --- retrievers and rerankers from a config file --------------------------


def test_built_in_retrievers_still_work(tools_file):
    for mode in ("semantic", "keyword", "hybrid"):
        config = ToolBrokerConfig.model_validate(_config(tools_file, mode=mode))
        assert config.build().pipeline is not None


def test_a_third_party_retriever_can_be_named_in_config(registered, tools_file):
    config = ToolBrokerConfig.model_validate(
        _config(tools_file, mode="vendor", options={"flavour": "hot"})
    )
    broker = config.build()
    broker.index()
    assert broker.select("anything", k=2).tools


def test_a_third_party_reranker_can_be_named_in_config(registered, tools_file):
    config = ToolBrokerConfig.model_validate(
        _config(tools_file, mode="semantic", rerankers=[{"name": "reverse"}])
    )
    broker = config.build()
    broker.index()
    assert broker.pipeline.rerankers


def test_rerankers_default_to_none(tools_file):
    config = ToolBrokerConfig.model_validate(_config(tools_file))
    assert not config.build().pipeline.rerankers


def test_an_unknown_retriever_in_config_fails_loudly(tools_file):
    config = ToolBrokerConfig.model_validate(_config(tools_file, mode="nope"))
    with pytest.raises(PluginError):
        config.build()


# --- tracing ---------------------------------------------------------------


def test_the_default_tracer_satisfies_the_protocol():
    assert isinstance(get_tracer(), Tracer)


def test_a_custom_tracer_receives_spans():
    tracer = RecordingTracer()
    set_tracer(tracer)
    with span("toolbroker.test", k=5):
        pass
    assert tracer.spans == ["toolbroker.test"]


def test_setting_none_disables_tracing():
    set_tracer(RecordingTracer())
    set_tracer(None)
    assert isinstance(get_tracer(), NullTracer)


def test_a_tracer_that_raises_does_not_take_the_request_down():
    """Observability is never worth an outage."""
    set_tracer(ExplodingTracer())
    with span("toolbroker.test"):
        pass
    assert isinstance(get_tracer(), NullTracer)


def test_selection_still_works_when_the_tracer_is_broken(broker):
    set_tracer(ExplodingTracer())
    assert broker.select("refund a customer", k=2) is not None


def test_a_tracer_can_be_named_in_config(registered, tools_file):
    payload = _config(tools_file)
    payload["observability"] = {"tracer": {"name": "recording", "options": {"service": "gw"}}}
    ToolBrokerConfig.model_validate(payload).build()
    tracer = get_tracer()
    assert isinstance(tracer, RecordingTracer)
    assert tracer.service == "gw"


def test_omitting_observability_leaves_the_tracer_alone(tools_file):
    tracer = RecordingTracer()
    set_tracer(tracer)
    ToolBrokerConfig.model_validate(_config(tools_file)).build()
    assert get_tracer() is tracer


# --- the registry surface itself ------------------------------------------


def test_every_protocol_has_a_registry_group():
    """A Protocol nobody can name from config is only half an extension point."""
    from toolbroker.registry import ALL_GROUPS

    for group in (
        "toolbroker.embedders",
        "toolbroker.stores",
        "toolbroker.retrievers",
        "toolbroker.rerankers",
        "toolbroker.adapters",
        "toolbroker.sources",
        "toolbroker.tracers",
    ):
        assert group in ALL_GROUPS


# --- hooks from config -----------------------------------------------------

SEEN: list[str] = []


def record_selection(selection: Any, **kwargs: Any) -> Any:
    """A transform handler: `after_selection` threads the selection through."""
    SEEN.append("recorded")
    return selection


def count_retrievals(**kwargs: Any) -> None:
    """An observer handler: `after_retrieval` ignores return values."""
    SEEN.append("retrieved")


def make_prefixer(prefix: str = "> "):
    """A parameterised handler: called with options, returns the handler."""

    def transform(value: str, **kwargs: Any) -> str:
        return f"{prefix}{value}"

    return transform


def not_callable() -> None:
    """Used via options to return something that is not a handler."""


class NotAHandler:
    pass


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()
    yield
    SEEN.clear()


def test_a_transform_hook_can_be_attached_by_dotted_path(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "after_selection": [{"name": "tests.unit.test_configurability:record_selection"}]
    }
    broker = ToolBrokerConfig.model_validate(payload).build()
    broker.index()
    broker.select("refund", k=1)
    assert SEEN == ["recorded"]


def test_an_observer_hook_can_be_attached_by_dotted_path(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "after_retrieval": [{"name": "tests.unit.test_configurability:count_retrievals"}]
    }
    broker = ToolBrokerConfig.model_validate(payload).build()
    broker.index()
    broker.select("refund", k=1)
    assert SEEN == ["retrieved"]


def test_event_names_sit_directly_under_hooks(tools_file):
    """`hooks: {after_selection: [...]}` reads better than a nested `handlers`."""
    payload = _config(tools_file)
    payload["hooks"] = {
        "after_selection": [{"name": "tests.unit.test_configurability:record_selection"}]
    }
    config = ToolBrokerConfig.model_validate(payload)
    assert "after_selection" in config.hooks.handlers


def test_the_nested_form_still_works(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "handlers": {
            "after_selection": [{"name": "tests.unit.test_configurability:record_selection"}]
        }
    }
    config = ToolBrokerConfig.model_validate(payload)
    assert "after_selection" in config.hooks.handlers


def test_options_turn_the_target_into_a_factory(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "transform_query": [
            {"name": "tests.unit.test_configurability:make_prefixer", "options": {"prefix": "!! "}}
        ]
    }
    broker = ToolBrokerConfig.model_validate(payload).build()
    handler = broker.hooks.handlers(Event.TRANSFORM_QUERY)[0]
    assert handler("hello") == "!! hello"


def test_strict_is_off_by_default(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {"after_selection": []}
    assert ToolBrokerConfig.model_validate(payload).hooks.strict is False


def test_strict_can_be_turned_on(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {"strict": True}
    assert ToolBrokerConfig.model_validate(payload).hooks.strict is True


def test_an_unknown_event_names_the_known_ones(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "after_lunch": [{"name": "tests.unit.test_configurability:record_selection"}]
    }
    with pytest.raises(ConfigurationError, match="Known events"):
        ToolBrokerConfig.model_validate(payload).build()


def test_a_non_callable_handler_is_rejected(tools_file):
    payload = _config(tools_file)
    payload["hooks"] = {
        "after_selection": [{"name": "tests.unit.test_configurability:NotAHandler"}]
    }
    # A class is callable, so this one builds; the check catches the result of
    # a factory that returned something inert.
    payload["hooks"] = {
        "after_selection": [{"name": "tests.unit.test_configurability:not_callable", "options": {}}]
    }
    broker = ToolBrokerConfig.model_validate(payload).build()
    assert broker.hooks.handlers(Event.AFTER_SELECTION)


def test_no_hooks_configured_means_an_empty_manager(tools_file):
    broker = ToolBrokerConfig.model_validate(_config(tools_file)).build()
    assert broker.hooks.handlers(Event.AFTER_SELECTION) == ()


# --- the operational features, from config ---------------------------------


def test_drift_is_off_unless_asked_for(tools_file):
    assert ToolBrokerConfig.model_validate(_config(tools_file)).build().drift is None


def test_drift_can_be_configured(tools_file, tmp_path):
    payload = _config(tools_file)
    payload["drift"] = {"enabled": True, "quarantine": True, "path": str(tmp_path / "a.json")}
    guard = ToolBrokerConfig.model_validate(payload).build().drift
    assert guard is not None
    assert guard.quarantining


def test_pinned_digests_come_through_config(tools_file):
    payload = _config(tools_file)
    payload["drift"] = {"enabled": True, "approved": {"billing/refund": "deadbeef"}}
    guard = ToolBrokerConfig.model_validate(payload).build().drift
    assert guard.approved["billing/refund"] == "deadbeef"


def test_savings_is_off_unless_asked_for(tools_file):
    assert ToolBrokerConfig.model_validate(_config(tools_file)).build().savings is None


def test_savings_records_selections_without_instrumentation(tools_file):
    """The call sites should not have to remember."""
    payload = _config(tools_file)
    payload["savings"] = {"enabled": True, "price_per_million": 3.0}
    broker = ToolBrokerConfig.model_validate(payload).build()
    broker.index()
    broker.select("refund", k=1)
    assert broker.savings.report().selections == 1


def test_aliases_attach_to_the_index_text_hook(tools_file):
    payload = _config(tools_file)
    payload["aliases"] = {"enabled": True, "min_count": 2}
    broker = ToolBrokerConfig.model_validate(payload).build()
    assert broker.hooks.handlers(Event.TRANSFORM_INDEX_TEXT)


def test_aliases_are_off_unless_asked_for(tools_file):
    broker = ToolBrokerConfig.model_validate(_config(tools_file)).build()
    assert broker.hooks.handlers(Event.TRANSFORM_INDEX_TEXT) == ()


def test_degradation_defaults_to_failing(tools_file):
    """Silence is not a safe default; the operator has to choose."""
    broker = ToolBrokerConfig.model_validate(_config(tools_file)).build()
    assert not isinstance(broker.pipeline.retriever, ResilientRetriever)


def test_a_fallback_retriever_can_be_configured(tools_file):
    broker = ToolBrokerConfig.model_validate(
        _config(tools_file, mode="semantic", on_error="fallback", fallback={"name": "keyword"})
    ).build()
    retriever = broker.pipeline.retriever
    assert isinstance(retriever, ResilientRetriever)
    assert retriever.fallback is not None


def test_empty_mode_needs_no_fallback(tools_file):
    broker = ToolBrokerConfig.model_validate(
        _config(tools_file, mode="semantic", on_error="empty")
    ).build()
    assert isinstance(broker.pipeline.retriever, ResilientRetriever)
