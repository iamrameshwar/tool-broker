from __future__ import annotations

import pytest

from toolbroker import Tool
from toolbroker.adapters import AnthropicAdapter, OpenAIAdapter, get_adapter
from toolbroker.adapters.raw import flatten_name
from toolbroker.errors import AdapterError


def test_openai_shape():
    rendered = OpenAIAdapter().render([Tool(name="search", description="Find things")])
    assert rendered[0]["type"] == "function"
    assert rendered[0]["function"]["description"] == "Find things"


def test_anthropic_shape():
    rendered = AnthropicAdapter().render([Tool(name="search", description="Find things")])
    assert rendered[0]["name"] == "search"
    assert rendered[0]["input_schema"]["type"] == "object"


def test_namespace_is_flattened_because_providers_reject_slashes():
    assert flatten_name(Tool(name="search", namespace="github")) == "github__search"


def test_default_namespace_is_omitted():
    assert flatten_name(Tool(name="search")) == "search"


def test_illegal_characters_are_replaced():
    assert flatten_name(Tool(name="search.things", namespace="my org")) == "my_org__search_things"


def test_long_names_are_truncated_to_the_provider_limit():
    tool = Tool(name="x" * 40, namespace="y" * 40)
    assert len(flatten_name(tool)) <= 64


def test_empty_schema_becomes_a_valid_object():
    rendered = OpenAIAdapter().render([Tool(name="ping")])
    assert rendered[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_strict_mode_marks_everything_required():
    tool = Tool(
        name="search",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    rendered = OpenAIAdapter(strict=True).render([tool])
    assert rendered[0]["function"]["strict"] is True
    assert rendered[0]["function"]["parameters"]["required"] == ["q"]
    assert rendered[0]["function"]["parameters"]["additionalProperties"] is False


def test_names_can_be_resolved_back_to_tools():
    adapter = OpenAIAdapter()
    adapter.render([Tool(name="search", namespace="github")])
    assert adapter.resolve_name("github__search").id == "github/search"


def test_unknown_name_raises_with_the_known_ones():
    adapter = OpenAIAdapter()
    adapter.render([Tool(name="search")])
    with pytest.raises(AdapterError, match="search"):
        adapter.resolve_name("nope")


def test_colliding_flattened_names_are_rejected():
    # `a/b__c` and `a__b/c` both flatten to `a__b__c`. Silently dropping one
    # would make the model unable to call a tool it can see.
    adapter = OpenAIAdapter()
    with pytest.raises(AdapterError, match="collide"):
        adapter.render([Tool(name="b__c", namespace="a"), Tool(name="c", namespace="a__b")])


def test_collision_error_names_both_tools():
    adapter = OpenAIAdapter()
    with pytest.raises(AdapterError, match="a/b__c"):
        adapter.render([Tool(name="b__c", namespace="a"), Tool(name="c", namespace="a__b")])


def test_callable_is_returned_for_local_tools():
    def handler() -> None:
        """Do a thing."""

    adapter = OpenAIAdapter()
    adapter.render([Tool(name="run", metadata={"callable": handler})])
    assert adapter.callable_for("run") is handler


def test_remote_tools_report_where_they_live():
    adapter = OpenAIAdapter()
    adapter.render([Tool(name="run", source_id="mcp:github")])
    with pytest.raises(AdapterError, match="mcp:github"):
        adapter.callable_for("run")


def test_adapters_resolve_by_name():
    assert get_adapter("openai").id == "openai"
    assert get_adapter("anthropic").id == "anthropic"
