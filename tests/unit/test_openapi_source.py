from __future__ import annotations

import json

import pytest

from toolbroker import CostTier, RiskTier
from toolbroker.errors import SourceError
from toolbroker.sources.openapi import OpenAPISource, operation_name

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Pet Store", "version": "1.0.0"},
    "servers": [{"url": "https://api.petstore.example/v1"}],
    "paths": {
        "/pets": {
            "parameters": [
                {
                    "name": "tenant",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Tenant identifier.",
                }
            ],
            "get": {
                "operationId": "listPets",
                "summary": "List pets",
                "description": "Return every pet in the store.",
                "tags": ["Pets"],
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer"},
                        "description": "How many to return.",
                    }
                ],
            },
            "post": {
                "summary": "Create a pet",
                "security": [{"oauth2": ["pets:write"]}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                    },
                },
            },
        },
        "/pets/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Fetch one pet",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            },
            "delete": {
                "summary": "Delete a pet",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            },
        },
        "/legacy": {"get": {"operationId": "legacy", "deprecated": True, "summary": "Old"}},
    },
    "components": {
        "schemas": {
            "Pet": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "The pet's name."},
                    "tag": {"type": "string"},
                },
            }
        }
    },
}


@pytest.fixture
def tools():
    return {tool.name: tool for tool in OpenAPISource(SPEC).discover()}


def test_namespace_defaults_to_a_slug_of_the_title(tools):
    assert all(tool.namespace == "pet_store" for tool in tools.values())


def test_operation_id_becomes_the_tool_name(tools):
    assert "listPets" in tools
    assert "getPet" in tools


def test_missing_operation_id_derives_a_readable_name(tools):
    # POST /pets and DELETE /pets/{petId} have no operationId.
    assert "post_pets" in tools
    assert "delete_pets_by_petId" in tools


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("get", "/users/{id}/orders", "get_users_by_id_orders"),
        ("post", "/", "post"),
        ("get", "/a-b/c.d", "get_a_b_c_d"),
    ],
)
def test_derived_names(method, path, expected):
    assert operation_name(method, path, None) == expected


def test_summary_and_description_are_combined(tools):
    assert tools["listPets"].description == "List pets Return every pet in the store."


def test_query_parameters_land_in_the_schema(tools):
    schema = tools["listPets"].input_schema
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["limit"]["description"] == "How many to return."


def test_path_level_parameters_are_inherited_by_every_operation(tools):
    assert "tenant" in tools["listPets"].input_schema["properties"]
    assert "tenant" in tools["post_pets"].input_schema["properties"]


def test_required_parameters_are_marked_required(tools):
    assert "tenant" in tools["listPets"].input_schema["required"]
    assert "limit" not in tools["listPets"].input_schema.get("required", [])


def test_path_parameters_are_required_and_located(tools):
    tool = tools["getPet"]
    assert "petId" in tool.input_schema["required"]
    assert tool.metadata["http"]["parameter_in"]["petId"] == "path"


def test_request_body_is_flattened_into_the_schema(tools):
    schema = tools["post_pets"].input_schema
    assert schema["properties"]["name"]["description"] == "The pet's name."
    assert "name" in schema["required"]


def test_body_fields_are_recorded_so_a_caller_can_rebuild_the_request(tools):
    http = tools["post_pets"].metadata["http"]
    assert set(http["body_fields"]) == {"name", "tag"}
    assert http["parameter_in"]["name"] == "body"
    assert http["parameter_in"]["tenant"] == "header"


def test_refs_are_resolved(tools):
    assert "$ref" not in json.dumps(tools["post_pets"].input_schema)


def test_http_metadata_carries_method_path_and_server(tools):
    http = tools["getPet"].metadata["http"]
    assert http["method"] == "GET"
    assert http["path"] == "/pets/{petId}"
    assert http["server"] == "https://api.petstore.example/v1"


def test_risk_is_inferred_from_the_http_method(tools):
    assert tools["listPets"].risk is RiskTier.LOW
    assert tools["post_pets"].risk is RiskTier.MEDIUM
    assert tools["delete_pets_by_petId"].risk is RiskTier.HIGH


def test_security_scopes_become_required_scopes(tools):
    assert tools["post_pets"].required_scopes == frozenset({"pets:write"})
    assert tools["listPets"].required_scopes == frozenset()


def test_spec_tags_become_tool_tags(tools):
    assert "pets" in tools["listPets"].tags


def test_deprecated_operations_are_skipped_by_default(tools):
    assert "legacy" not in tools


def test_deprecated_can_be_included():
    names = {tool.name for tool in OpenAPISource(SPEC, include_deprecated=True).discover()}
    assert "legacy" in names


def test_methods_can_be_restricted():
    names = {tool.name for tool in OpenAPISource(SPEC, include_methods=["get"]).discover()}
    assert names == {"listPets", "getPet"}


def test_source_id_and_overrides():
    source = OpenAPISource(SPEC, namespace="pets", tags=["external"], default_cost=CostTier.HIGH)
    tool = next(iter(source.discover()))
    assert source.id == "openapi:pets"
    assert tool.namespace == "pets"
    assert "external" in tool.tags
    assert tool.cost is CostTier.HIGH


def test_cyclic_refs_do_not_hang():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Cyclic"},
        "paths": {
            "/node": {
                "post": {
                    "operationId": "createNode",
                    "requestBody": {
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Node"}}
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/components/schemas/Node"}},
                }
            }
        },
    }
    tools = list(OpenAPISource(spec).discover())
    assert tools[0].name == "createNode"


def test_unresolvable_ref_degrades_instead_of_crashing():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Broken"},
        "paths": {
            "/x": {
                "get": {
                    "operationId": "x",
                    "parameters": [{"$ref": "#/components/parameters/Missing"}],
                }
            }
        },
    }
    assert len(list(OpenAPISource(spec).discover())) == 1


def test_non_object_body_becomes_a_single_field():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Bulk"},
        "paths": {
            "/bulk": {
                "post": {
                    "operationId": "bulk",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                }
            }
        },
    }
    tool = next(iter(OpenAPISource(spec).discover()))
    assert tool.input_schema["properties"]["body"]["type"] == "array"
    assert tool.metadata["http"]["parameter_in"]["body"] == "body"


def test_body_field_colliding_with_a_query_param_is_suffixed():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Collide"},
        "paths": {
            "/x": {
                "post": {
                    "operationId": "x",
                    "parameters": [{"name": "id", "in": "query", "schema": {"type": "string"}}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"id": {"type": "integer"}},
                                }
                            }
                        }
                    },
                }
            }
        },
    }
    tool = next(iter(OpenAPISource(spec).discover()))
    assert tool.input_schema["properties"]["id"]["type"] == "string"
    assert tool.input_schema["properties"]["body_id"]["type"] == "integer"


def test_duplicate_derived_names_are_disambiguated():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Dup"},
        "paths": {
            "/x": {
                "get": {"operationId": "same", "summary": "a"},
                "post": {"operationId": "same", "summary": "b"},
            }
        },
    }
    names = [tool.name for tool in OpenAPISource(spec).discover()]
    assert len(set(names)) == 2


def test_loads_from_a_json_file(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(SPEC))
    assert len(list(OpenAPISource(path).discover())) == 4


def test_loads_from_a_yaml_file(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(SPEC))
    assert len(list(OpenAPISource(path).discover())) == 4


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        OpenAPISource(tmp_path / "absent.json")


def test_document_without_paths_is_rejected():
    with pytest.raises(SourceError, match="no 'paths'"):
        list(OpenAPISource({"openapi": "3.0.0", "info": {"title": "Empty"}}).discover())


def test_end_to_end_selection_over_an_api():
    from toolbroker import ToolBroker
    from toolbroker.index.embedders import HashingEmbedder

    broker = ToolBroker(embedder=HashingEmbedder(dim=256), cache_embeddings=False)
    broker.add_source(OpenAPISource(SPEC))
    broker.index()
    # Lexical wording on purpose: this asserts the source integrates with the
    # catalogue, not that the offline embedder understands synonyms.
    selection = broker.select("delete a pet", k=1)
    assert selection.tool_ids == ("pet_store/delete_pets_by_petId",)
