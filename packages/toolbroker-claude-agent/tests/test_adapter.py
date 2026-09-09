"""Claude Agent SDK rendering."""

from __future__ import annotations

import asyncio

import pytest
from toolbroker_claude_agent import ClaudeAgentAdapter, default_result_formatter

from toolbroker import Tool, ToolBroker
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


def run(sdk_tool, arguments):
    return asyncio.run(sdk_tool.handler(arguments))


# -- rendering ------------------------------------------------------------


def test_renders_one_sdk_tool_per_tool():
    rendered = ClaudeAgentAdapter().render([local(), local("other")])
    assert len(rendered) == 2
    assert rendered[0].name == "echo"


def test_description_and_schema_are_carried_through():
    tool = Tool(
        name="search",
        description="Search the docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        metadata={"callable": echo},
    )
    rendered = ClaudeAgentAdapter().render([tool])[0]
    assert rendered.description == "Search the docs"
    assert rendered.input_schema["properties"]["q"]["type"] == "string"


def test_empty_schema_becomes_a_valid_object():
    assert ClaudeAgentAdapter().render([local()])[0].input_schema == {
        "type": "object",
        "properties": {},
    }


def test_empty_selection_renders_empty():
    assert ClaudeAgentAdapter().render([]) == []


# -- invocation and result shaping ----------------------------------------


def test_result_is_wrapped_as_mcp_content():
    # Handlers must return content blocks, not bare values, or the SDK rejects
    # the response.
    rendered = ClaudeAgentAdapter().render([local()])[0]
    assert run(rendered, {"message": "hi"}) == {"content": [{"type": "text", "text": "hi"}]}


def test_async_functions_are_awaited():
    rendered = ClaudeAgentAdapter().render([local(func=aecho)])[0]
    assert run(rendered, {"message": "hi"})["content"][0]["text"] == "hi"


def test_none_becomes_empty_content():
    def nothing() -> None:
        """Return nothing."""

    rendered = ClaudeAgentAdapter().render([local("nothing", nothing)])[0]
    assert run(rendered, {}) == {"content": []}


def test_a_result_that_is_already_mcp_shaped_passes_through():
    payload = {"content": [{"type": "image", "data": "...", "mimeType": "image/png"}]}

    def screenshot() -> dict:
        """Take a screenshot."""
        return payload

    rendered = ClaudeAgentAdapter().render([local("screenshot", screenshot)])[0]
    assert run(rendered, {}) == payload


def test_custom_result_formatter():
    rendered = ClaudeAgentAdapter(
        result_formatter=lambda value: {"content": [{"type": "text", "text": f"<{value}>"}]}
    ).render([local()])[0]
    assert run(rendered, {"message": "hi"})["content"][0]["text"] == "<hi>"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {"content": []}),
        ("text", {"content": [{"type": "text", "text": "text"}]}),
        (42, {"content": [{"type": "text", "text": "42"}]}),
        ({"content": []}, {"content": []}),
        ({"other": 1}, {"content": [{"type": "text", "text": "{'other': 1}"}]}),
    ],
)
def test_default_formatter(value, expected):
    assert default_result_formatter(value) == expected


def test_non_dict_arguments_are_tolerated():
    def ping() -> str:
        """Return pong."""
        return "pong"

    rendered = ClaudeAgentAdapter().render([local("ping", ping)])[0]
    assert run(rendered, None)["content"][0]["text"] == "pong"


# -- remote tools ---------------------------------------------------------


def test_remote_tool_without_an_invoker_fails_loudly():
    with pytest.raises(AdapterError, match="no local callable"):
        ClaudeAgentAdapter().render([Tool(name="remote", source_id="mcp:github")])


def test_remote_tool_uses_the_invoker():
    calls: list[tuple[str, dict]] = []

    def invoker(tool, kwargs):
        calls.append((tool.id, kwargs))
        return "forwarded"

    rendered = ClaudeAgentAdapter(invoker=invoker).render(
        [Tool(name="remote", source_id="mcp:github")]
    )[0]
    assert run(rendered, {"a": 1})["content"][0]["text"] == "forwarded"
    assert calls == [("default/remote", {"a": 1})]


def test_skip_uninvokable_drops_remote_tools():
    tools = [local(), Tool(name="remote", source_id="mcp:github")]
    assert len(ClaudeAgentAdapter(skip_uninvokable=True).render(tools)) == 1


# -- names and permissions ------------------------------------------------


def test_duplicate_names_across_namespaces_are_rejected():
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    with pytest.raises(AdapterError, match="use_qualified_names"):
        ClaudeAgentAdapter().render(tools)


def test_qualified_names_resolve_the_collision():
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    rendered = ClaudeAgentAdapter(use_qualified_names=True).render(tools)
    assert {t.name for t in rendered} == {"docs__search", "code__search"}


def test_allowed_tool_names_match_the_sdk_permission_format():
    # Getting this string wrong means the model sees a tool and is then refused
    # permission to call it, which reads like a bug in the model.
    tools = [local("search"), local("refund")]
    adapter = ClaudeAgentAdapter()
    assert adapter.allowed_tool_names(tools) == [
        "mcp__toolbroker__search",
        "mcp__toolbroker__refund",
    ]


def test_allowed_tool_names_honour_the_server_name():
    assert ClaudeAgentAdapter().allowed_tool_names([local()], server="mytools") == [
        "mcp__mytools__echo"
    ]


def test_allowed_tool_names_agree_with_rendered_names():
    tools = [local("search", namespace="docs"), local("refund", namespace="billing")]
    adapter = ClaudeAgentAdapter(use_qualified_names=True)
    rendered = {t.name for t in adapter.render(tools)}
    allowed = {name.removeprefix("mcp__toolbroker__") for name in adapter.allowed_tool_names(tools)}
    assert rendered == allowed


def test_names_resolve_back_to_tools():
    adapter = ClaudeAgentAdapter(use_qualified_names=True)
    adapter.render([local("search", namespace="docs")])
    assert adapter.resolve_name("docs__search").id == "docs/search"


def test_unknown_name_lists_the_known_ones():
    adapter = ClaudeAgentAdapter()
    adapter.render([local()])
    with pytest.raises(AdapterError, match="echo"):
        adapter.resolve_name("nope")


# -- server construction --------------------------------------------------


def test_create_server_returns_an_sdk_server_config():
    config = ClaudeAgentAdapter().create_server([local()], name="tools")
    assert config["type"] == "sdk"
    assert config["name"] == "tools"


def test_create_server_carries_the_tools():
    config = ClaudeAgentAdapter().create_server([local(), local("other")])
    server = config["instance"]
    assert server is not None


# -- integration ----------------------------------------------------------


def test_end_to_end_through_the_catalogue():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([echo])
    broker.index()
    rendered, selection = broker.select_for("claude-agent", "echo a message back", k=1)
    assert len(rendered) == 1
    assert selection.tool_ids == ("python/echo",)


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_ADAPTERS, available, resolve

    assert "claude-agent" in available(GROUP_ADAPTERS)
    assert resolve(GROUP_ADAPTERS, "claude-agent") is ClaudeAgentAdapter
