from __future__ import annotations

import json

import pytest

from toolbroker.errors import SourceError
from toolbroker.sources import CompositeSource, PythonFunctionSource, StaticJSONSource
from toolbroker.sources.python_fn import tool_from_function
from toolbroker.types import RiskTier


def sample(query: str, limit: int = 10, verbose: bool = False) -> list[str]:
    """Search the catalogue for matching records.

    Longer explanation that should not end up in the summary.

    Args:
        query: What to search for.
        limit: How many results to return.
        verbose: Include debug output.

    Returns:
        Matching record ids.
    """
    return []


def test_description_comes_from_the_docstring_summary():
    tool = tool_from_function(sample)
    assert tool.description == "Search the catalogue for matching records."


def test_schema_marks_only_defaultless_params_required():
    tool = tool_from_function(sample)
    assert tool.input_schema["required"] == ["query"]
    assert tool.input_schema["properties"]["limit"]["default"] == 10


def test_param_descriptions_come_from_the_args_section():
    tool = tool_from_function(sample)
    assert tool.input_schema["properties"]["query"]["description"] == "What to search for."


def test_types_map_to_json_schema():
    tool = tool_from_function(sample)
    assert tool.input_schema["properties"]["limit"]["type"] == "integer"
    assert tool.input_schema["properties"]["verbose"]["type"] == "boolean"


def test_callable_is_carried_in_metadata():
    assert tool_from_function(sample).metadata["callable"] is sample


def test_undocumented_function_still_works():
    def bare(x):
        return x

    tool = tool_from_function(bare)
    assert tool.description == ""
    assert tool.input_schema["required"] == ["x"]


def test_varargs_are_skipped():
    def variadic(a: int, *args, **kwargs) -> None:
        """Do something."""

    assert list(tool_from_function(variadic).input_schema["properties"]) == ["a"]


def test_non_callable_is_rejected():
    with pytest.raises(SourceError):
        tool_from_function("not a function")


def test_source_stamps_its_id_and_namespace():
    source = PythonFunctionSource([sample], source_id="mine", namespace="tools")
    tool = source.discover()[0]
    assert tool.source_id == "mine"
    assert tool.id == "tools/sample"


def test_decorator_registers_and_returns_the_function():
    source = PythonFunctionSource()

    @source.tool(risk=RiskTier.HIGH)
    def dangerous(target: str) -> None:
        """Break something."""

    tools = list(source.discover())
    assert len(tools) == 1
    assert tools[0].risk is RiskTier.HIGH
    assert dangerous("x") is None


def test_static_json_reads_a_list(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([{"name": "a", "description": "first"}]))
    tools = list(StaticJSONSource(path).discover())
    assert tools[0].name == "a"


def test_static_json_reads_a_tools_key(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"tools": [{"name": "a"}]}))
    assert len(list(StaticJSONSource(path).discover())) == 1


def test_static_json_reads_jsonl(tmp_path):
    path = tmp_path / "tools.jsonl"
    path.write_text('{"name": "a"}\n{"name": "b"}\n')
    assert len(list(StaticJSONSource(path).discover())) == 2


def test_missing_file_raises_source_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        list(StaticJSONSource(tmp_path / "nope.json").discover())


def test_invalid_json_raises_source_error(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text("{not json")
    with pytest.raises(SourceError, match="not valid JSON"):
        list(StaticJSONSource(path).discover())


def test_invalid_tool_names_the_offending_index(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([{"name": "ok"}, {"name": "bad/name"}]))
    with pytest.raises(SourceError, match=r"\[1\]"):
        list(StaticJSONSource(path).discover())


class BrokenSource:
    id = "broken"

    def discover(self):
        raise RuntimeError("server is down")


def test_composite_survives_a_failing_source():
    composite = CompositeSource([BrokenSource(), PythonFunctionSource([sample])])
    tools = list(composite.discover())
    assert [tool.name for tool in tools] == ["sample"]
    assert composite.failures[0][0] == "broken"
