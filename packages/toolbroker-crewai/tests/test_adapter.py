"""CrewAI rendering: schema conversion, sync and async, name handling."""

from __future__ import annotations

import asyncio

import pytest
from toolbroker_crewai import CrewAIAdapter

from toolbroker import Tool, ToolBroker
from toolbroker.errors import AdapterError
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources.python_fn import tool_from_function


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
    """Build a tool the way a real source would: schema derived from the function."""
    kwargs.setdefault("namespace", "default")
    return tool_from_function(func, name=name, source_id="test", **kwargs)


# -- rendering ------------------------------------------------------------


def test_renders_one_tool_each():
    rendered = CrewAIAdapter().render([local(), local("other")])
    assert len(rendered) == 2
    assert rendered[0].name == "echo"


def test_description_is_carried_through():
    # CrewAI rewrites description into its own "Tool Name / Arguments /
    # Description" format, so the original has to be looked for inside it.
    assert "Echo a message back." in CrewAIAdapter().render([local()])[0].description


def test_empty_description_gets_a_fallback():
    # CrewAI needs a description; an empty one makes the tool invisible to the
    # agent's reasoning.
    tool = Tool(name="ping", description="", metadata={"callable": echo})
    assert CrewAIAdapter().render([tool])[0].description


def test_empty_selection_renders_empty():
    assert CrewAIAdapter().render([]) == []


def test_result_as_answer_flag():
    assert CrewAIAdapter(result_as_answer=True).render([local()])[0].result_as_answer is True


# -- schema conversion ----------------------------------------------------


def test_args_schema_is_a_pydantic_model():
    from pydantic import BaseModel

    rendered = CrewAIAdapter().render([local()])[0]
    assert issubclass(rendered.args_schema, BaseModel)
    assert "message" in rendered.args_schema.model_fields


def test_required_and_optional_are_distinguished():
    tool = Tool(
        name="search",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["q"],
        },
        metadata={"callable": echo},
    )
    model = CrewAIAdapter().render([tool])[0].args_schema
    assert model.model_fields["q"].is_required()
    assert not model.model_fields["limit"].is_required()


def test_awkward_parameter_names_are_aliased():
    # `order-id` is not a Python identifier; `schema` would shadow pydantic.
    tool = Tool(
        name="refund",
        input_schema={
            "type": "object",
            "properties": {"order-id": {"type": "string"}, "schema": {"type": "string"}},
            "required": ["order-id"],
        },
        metadata={"callable": echo},
    )
    model = CrewAIAdapter().render([tool])[0].args_schema
    assert model.model_fields["order_id"].alias == "order-id"
    assert model.model_fields["schema_"].alias == "schema"
    assert model(**{"order-id": "o1"}).order_id == "o1"


def test_tool_with_no_parameters_renders():
    rendered = CrewAIAdapter().render([Tool(name="ping", metadata={"callable": echo})])[0]
    assert rendered.args_schema is not None


# -- invocation -----------------------------------------------------------


def test_sync_tool_runs():
    assert CrewAIAdapter().render([local()])[0].run(message="hi") == "hi"


def test_async_tool_runs_through_crewai():
    # CrewAI's own run() awaits a coroutine returned by _run, which is exactly
    # why ToolBinding.call hands it back rather than swallowing it.
    assert CrewAIAdapter().render([local(func=aecho)])[0].run(message="hi") == "hi"


def test_async_entry_point_awaits():
    rendered = CrewAIAdapter().render([local(func=aecho)])[0]
    assert asyncio.run(rendered._arun(message="hi")) == "hi"


def test_sync_function_through_the_async_entry_point():
    rendered = CrewAIAdapter().render([local()])[0]
    assert asyncio.run(rendered._arun(message="hi")) == "hi"


# -- remote tools ---------------------------------------------------------


def test_remote_tool_without_an_invoker_fails_loudly():
    with pytest.raises(AdapterError, match="no local callable"):
        CrewAIAdapter().render([Tool(name="remote", source_id="mcp:github")])


def test_remote_tool_uses_the_invoker():
    calls: list[tuple[str, dict]] = []

    def invoker(tool, kwargs):
        calls.append((tool.id, kwargs))
        return "forwarded"

    rendered = CrewAIAdapter(invoker=invoker).render(
        [Tool(name="remote", description="A remote tool.", source_id="mcp:github")]
    )[0]
    assert rendered.run(a=1) == "forwarded"
    assert calls == [("default/remote", {"a": 1})]


def test_skip_uninvokable_drops_remote_tools():
    tools = [local(), Tool(name="remote", source_id="mcp:github")]
    assert len(CrewAIAdapter(skip_uninvokable=True).render(tools)) == 1


# -- names ----------------------------------------------------------------


def test_duplicate_names_across_namespaces_are_rejected():
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    with pytest.raises(AdapterError, match="use_qualified_names"):
        CrewAIAdapter().render(tools)


def test_qualified_names_resolve_the_collision():
    tools = [local("search", namespace="docs"), local("search", namespace="code")]
    rendered = CrewAIAdapter(use_qualified_names=True).render(tools)
    assert {t.name for t in rendered} == {"docs__search", "code__search"}


def test_names_resolve_back_to_tools():
    adapter = CrewAIAdapter(use_qualified_names=True)
    adapter.render([local("search", namespace="docs")])
    assert adapter.resolve_name("docs__search").id == "docs/search"


def test_unknown_name_lists_the_known_ones():
    adapter = CrewAIAdapter()
    adapter.render([local()])
    with pytest.raises(AdapterError, match="echo"):
        adapter.resolve_name("nope")


# -- integration ----------------------------------------------------------


def test_end_to_end_through_the_catalogue():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.add_functions([echo])
    broker.index()
    rendered, selection = broker.select_for("crewai", "echo a message back", k=1)
    assert len(rendered) == 1
    assert selection.tool_ids == ("python/echo",)


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_ADAPTERS, available, resolve

    assert "crewai" in available(GROUP_ADAPTERS)
    assert resolve(GROUP_ADAPTERS, "crewai") is CrewAIAdapter
