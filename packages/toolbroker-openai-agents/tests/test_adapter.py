"""OpenAI Agents SDK rendering."""

from __future__ import annotations

import asyncio
import json

import pytest
from toolbroker_openai_agents import OpenAIAgentsAdapter

from toolbroker import RiskTier, Tool, ToolBroker
from toolbroker.errors import AdapterError
from toolbroker.index.embedders import HashingEmbedder


def echo(message: str) -> str:
    """Echo a message back.

    Args:
        message: What to echo.
    """
    return message


async def aecho(message: str) -> str:
    """Echo a message back, asynchronously.

    Args:
        message: What to echo.
    """
    return message


def local(name: str = "echo", func=echo, **kwargs) -> Tool:
    return Tool(name=name, metadata={"callable": func}, **kwargs)


def test_renders_one_function_tool_per_tool():
    rendered = OpenAIAgentsAdapter().render([local(), local("other")])
    assert len(rendered) == 2
    assert rendered[0].name == "echo"


def test_description_and_schema_are_carried_through():
    tool = Tool(
        name="search",
        description="Search the docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        metadata={"callable": echo},
    )
    rendered = OpenAIAgentsAdapter().render([tool])[0]
    assert rendered.description == "Search the docs"
    assert rendered.params_json_schema["properties"]["q"]["type"] == "string"


def test_empty_schema_becomes_a_valid_object():
    rendered = OpenAIAgentsAdapter().render([local()])[0]
    assert rendered.params_json_schema == {"type": "object", "properties": {}}


def test_strict_mode_is_off_by_default():
    # Most real MCP and OpenAPI schemas do not satisfy strict mode, so turning
    # it on by default would reject perfectly good tools at runtime.
    assert OpenAIAgentsAdapter().render([local()])[0].strict_json_schema is False


def test_strict_mode_marks_everything_required():
    tool = Tool(
        name="search",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["q"],
        },
        metadata={"callable": echo},
    )
    rendered = OpenAIAgentsAdapter(strict=True).render([tool])[0]
    assert rendered.strict_json_schema is True
    # The SDK re-runs its own strict coercion, which keeps declaration order.
    assert set(rendered.params_json_schema["required"]) == {"limit", "q"}
    assert rendered.params_json_schema["additionalProperties"] is False


def invoke(rendered, payload: str):
    return asyncio.run(rendered.on_invoke_tool(None, payload))


def test_invoking_calls_the_local_function():
    rendered = OpenAIAgentsAdapter().render([local()])[0]
    assert invoke(rendered, json.dumps({"message": "hi"})) == "hi"


def test_async_functions_are_awaited():
    rendered = OpenAIAgentsAdapter().render([local(func=aecho)])[0]
    assert invoke(rendered, json.dumps({"message": "hi"})) == "hi"


def test_empty_arguments_are_allowed():
    def ping() -> str:
        """Return pong."""
        return "pong"

    rendered = OpenAIAgentsAdapter().render([local("ping", ping)])[0]
    assert invoke(rendered, "") == "pong"


def test_invalid_json_names_the_tool():
    # A bare JSONDecodeError from inside the run loop is near-impossible to trace.
    rendered = OpenAIAgentsAdapter().render([local()])[0]
    with pytest.raises(AdapterError, match="invalid JSON arguments for tool 'echo'"):
        invoke(rendered, "{not json")


def test_non_object_arguments_are_rejected():
    rendered = OpenAIAgentsAdapter().render([local()])[0]
    with pytest.raises(AdapterError, match="expected a JSON object"):
        invoke(rendered, "[1, 2]")


def test_remote_tool_without_an_invoker_fails_loudly():
    remote = Tool(name="remote", source_id="mcp:github")
    with pytest.raises(AdapterError, match="no local callable"):
        OpenAIAgentsAdapter().render([remote])


def test_remote_tool_uses_the_invoker():
    remote = Tool(name="remote", source_id="mcp:github")
    calls: list[tuple[str, dict]] = []

    def invoker(tool, kwargs):
        calls.append((tool.id, kwargs))
        return "forwarded"

    rendered = OpenAIAgentsAdapter(invoker=invoker).render([remote])[0]
    assert invoke(rendered, json.dumps({"a": 1})) == "forwarded"
    assert calls == [("default/remote", {"a": 1})]


def test_async_invoker_is_awaited():
    remote = Tool(name="remote", source_id="mcp:github")

    async def invoker(tool, kwargs):
        return "awaited"

    rendered = OpenAIAgentsAdapter(invoker=invoker).render([remote])[0]
    assert invoke(rendered, "{}") == "awaited"


def test_skip_uninvokable_drops_remote_tools():
    tools = [local(), Tool(name="remote", source_id="mcp:github")]
    assert len(OpenAIAgentsAdapter(skip_uninvokable=True).render(tools)) == 1


def test_names_are_bare_by_default():
    tool = local("search", namespace="docs")
    assert OpenAIAgentsAdapter().render([tool])[0].name == "search"


def test_qualified_names_include_the_namespace():
    tool = local("search", namespace="docs")
    assert OpenAIAgentsAdapter(use_qualified_names=True).render([tool])[0].name == "docs__search"


def test_duplicate_names_across_namespaces_are_rejected():
    # The SDK has no notion of namespaces, so the model would see a duplicate.
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    with pytest.raises(AdapterError, match="use_qualified_names"):
        OpenAIAgentsAdapter().render(tools)


def test_qualified_names_resolve_the_collision():
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    rendered = OpenAIAgentsAdapter(use_qualified_names=True).render(tools)
    assert {t.name for t in rendered} == {"docs__search", "code__search"}


def test_names_resolve_back_to_tools():
    adapter = OpenAIAgentsAdapter(use_qualified_names=True)
    adapter.render([local("search", namespace="docs")])
    assert adapter.resolve_name("docs__search").id == "docs/search"


def test_unknown_name_lists_the_known_ones():
    adapter = OpenAIAgentsAdapter()
    adapter.render([local()])
    with pytest.raises(AdapterError, match="echo"):
        adapter.resolve_name("nope")


def test_end_to_end_through_the_catalogue():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([echo])
    broker.index()
    rendered, selection = broker.select_for("openai-agents", "echo a message back", k=1)
    assert len(rendered) == 1
    assert selection.tool_ids == ("python/echo",)


def test_risk_metadata_is_still_reachable_via_the_catalogue():
    tool = local("wipe", risk=RiskTier.CRITICAL)
    adapter = OpenAIAgentsAdapter()
    adapter.render([tool])
    assert adapter.resolve_name("wipe").risk is RiskTier.CRITICAL


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_ADAPTERS, available, resolve

    assert "openai-agents" in available(GROUP_ADAPTERS)
    assert resolve(GROUP_ADAPTERS, "openai-agents") is OpenAIAgentsAdapter
