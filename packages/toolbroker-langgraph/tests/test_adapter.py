"""LangChain / LangGraph rendering."""

from __future__ import annotations

import pytest
from toolbroker_langgraph import LangGraphAdapter

from toolbroker import Tool
from toolbroker.errors import AdapterError


def echo(message: str) -> str:
    """Echo a message back.

    Args:
        message: What to echo.
    """
    return message


def test_local_callable_is_wired_through():
    tool = Tool(name="echo", description="Echo a message", metadata={"callable": echo})
    rendered = LangGraphAdapter().render([tool])
    assert rendered[0].invoke({"message": "hi"}) == "hi"


def test_metadata_carries_the_toolbroker_id():
    tool = Tool(name="echo", namespace="demo", metadata={"callable": echo})
    assert LangGraphAdapter().render([tool])[0].metadata["toolbroker_id"] == "demo/echo"


def test_remote_tool_without_an_invoker_fails_loudly():
    # Producing a tool that raises at call time would be worse than failing here.
    remote = Tool(name="remote", source_id="mcp:github")
    with pytest.raises(AdapterError, match="no local callable"):
        LangGraphAdapter().render([remote])


def test_remote_tool_uses_the_invoker():
    remote = Tool(name="remote", source_id="mcp:github", description="Remote tool")
    calls: list[tuple[str, dict]] = []

    def invoker(tool, kwargs):
        calls.append((tool.id, kwargs))
        return "forwarded"

    rendered = LangGraphAdapter(invoker=invoker).render([remote])
    assert rendered[0].invoke({"anything": 1}) == "forwarded"
    assert calls[0][0] == "default/remote"


def test_skip_uninvokable_drops_remote_tools():
    tools = [
        Tool(name="local", metadata={"callable": echo}),
        Tool(name="remote", source_id="mcp:github"),
    ]
    rendered = LangGraphAdapter(skip_uninvokable=True).render(tools)
    assert len(rendered) == 1


def test_risk_is_exposed_in_metadata():
    from toolbroker import RiskTier, Tool

    tool = Tool(name="wipe", risk=RiskTier.CRITICAL, metadata={"callable": echo})
    assert LangGraphAdapter().render([tool])[0].metadata["risk"] == "critical"


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_ADAPTERS, available, resolve

    assert "langgraph" in available(GROUP_ADAPTERS)
    assert resolve(GROUP_ADAPTERS, "langgraph") is LangGraphAdapter
