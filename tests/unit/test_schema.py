"""JSON Schema to pydantic conversion."""

from __future__ import annotations

from typing import Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from toolbroker.schema import model_from_schema, python_type, safe_field_name


def build(properties: dict, required: list[str] | None = None) -> type[BaseModel]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return model_from_schema(schema, name="Args")


# -- names ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "alias"),
    [
        ("query", "query", None),
        ("order-id", "order_id", "order-id"),
        ("order.id", "order_id", "order.id"),
        ("2fa", "f_2fa", "2fa"),
        ("class", "class_", "class"),
        ("schema", "schema_", "schema"),
        ("model_config", "model_config_", "model_config"),
        ("", "field", ""),
    ],
)
def test_field_names(raw, expected, alias):
    assert safe_field_name(raw) == (expected, alias)


def test_aliases_accept_the_wire_name():
    model = build({"order-id": {"type": "string"}}, ["order-id"])
    assert model(**{"order-id": "o1"}).order_id == "o1"


def test_aliases_also_accept_the_python_name():
    # populate_by_name, so callers that already normalised are not punished.
    model = build({"order-id": {"type": "string"}}, ["order-id"])
    assert model(order_id="o1").order_id == "o1"


def test_serialisation_uses_the_wire_name():
    model = build({"order-id": {"type": "string"}}, ["order-id"])
    assert model(order_id="o1").model_dump(by_alias=True)["order-id"] == "o1"


# -- types ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("json_type", "expected"),
    [("string", str), ("integer", int), ("number", float), ("boolean", bool)],
)
def test_primitives(json_type, expected):
    assert build({"x": {"type": json_type}}, ["x"]).model_fields["x"].annotation is expected


def test_arrays_are_typed():
    assert (
        build({"tags": {"type": "array", "items": {"type": "string"}}}, ["tags"])
        .model_fields["tags"]
        .annotation
        == list[str]
    )


def test_untyped_array():
    assert build({"tags": {"type": "array"}}, ["tags"]).model_fields["tags"].annotation == list[Any]


def test_enums_become_literals():
    annotation = build({"mode": {"enum": ["a", "b"]}}, ["mode"]).model_fields["mode"].annotation
    assert get_origin(annotation) is Literal
    assert set(get_args(annotation)) == {"a", "b"}


def test_nested_objects_become_nested_models():
    model = build(
        {"meta": {"type": "object", "properties": {"note": {"type": "string"}}}}, ["meta"]
    )
    nested = model.model_fields["meta"].annotation
    assert issubclass(nested, BaseModel)
    assert model(meta={"note": "hi"}).meta.note == "hi"


def test_object_without_properties_is_a_plain_dict():
    assert (
        build({"meta": {"type": "object"}}, ["meta"]).model_fields["meta"].annotation
        == dict[str, Any]
    )


def test_any_of_becomes_a_union():
    annotation = (
        build({"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}, ["x"])
        .model_fields["x"]
        .annotation
    )
    assert get_origin(annotation) is Union
    assert set(get_args(annotation)) == {str, int}


def test_nullable_type_list():
    annotation = build({"x": {"type": ["string", "null"]}}, ["x"]).model_fields["x"].annotation
    assert type(None) in get_args(annotation)


def test_unknown_type_degrades_to_any():
    # A construct we cannot map must not drop the tool.
    assert build({"x": {"type": "quaternion"}}, ["x"]).model_fields["x"].annotation is Any


def test_deeply_nested_schema_terminates():
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(50):
        schema = {"type": "object", "properties": {"child": schema}}
    assert model_from_schema(schema, name="Deep") is not None


def test_recursion_guard_returns_any():
    assert python_type({"type": "string"}, depth=99) is Any


# -- required and defaults ------------------------------------------------


def test_required_fields_are_required():
    model = build({"q": {"type": "string"}}, ["q"])
    assert model.model_fields["q"].is_required()
    with pytest.raises(ValidationError):
        model()


def test_optional_fields_admit_none():
    # An optional field with no default has to accept omission, which is what
    # the schema says is allowed.
    model = build({"limit": {"type": "integer"}})
    assert model().limit is None


def test_defaults_are_preserved():
    assert build({"limit": {"type": "integer", "default": 10}})().limit == 10


def test_descriptions_are_preserved():
    model = build({"q": {"type": "string", "description": "The query."}}, ["q"])
    assert model.model_fields["q"].description == "The query."


# -- degenerate input -----------------------------------------------------


@pytest.mark.parametrize(
    "schema", [None, {}, {"type": "object"}, {"type": "object", "properties": {}}]
)
def test_empty_schemas_give_an_empty_model(schema):
    model = model_from_schema(schema, name="Empty")
    assert issubclass(model, BaseModel)
    assert model.model_fields == {}


def test_non_dict_property_spec_is_tolerated():
    assert (
        "x"
        in model_from_schema(
            {"type": "object", "properties": {"x": "not a dict"}}, name="Odd"
        ).model_fields
    )


def test_extra_fields_are_allowed():
    # A schema is a description of a real API, not necessarily a complete one.
    assert build({"q": {"type": "string"}}, ["q"])(q="a", surprise=1).surprise == 1


def test_model_names_are_class_safe():
    assert model_from_schema({}, name="my-tool args").__name__.isidentifier()
