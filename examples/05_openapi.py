"""Turning an OpenAPI spec into a searchable, policy-governed tool catalogue.

python examples/05_openapi.py
"""

from toolbroker import MaxRisk, PolicyEngine, RiskTier, ToolBroker
from toolbroker.sources import OpenAPISource

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Warehouse API", "version": "2.1.0"},
    "servers": [{"url": "https://api.warehouse.example/v2"}],
    "paths": {
        "/shipments": {
            "get": {
                "operationId": "listShipments",
                "summary": "List shipments",
                "description": "Return shipments, most recently created first.",
                "tags": ["Shipments"],
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Filter by shipment status.",
                    },
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                ],
            },
            "post": {
                "operationId": "createShipment",
                "summary": "Create a shipment",
                "security": [{"oauth2": ["shipments:write"]}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["destination", "items"],
                                "properties": {
                                    "destination": {
                                        "type": "string",
                                        "description": "Destination address.",
                                    },
                                    "items": {"type": "array", "items": {"type": "string"}},
                                },
                            }
                        }
                    },
                },
            },
        },
        "/shipments/{shipmentId}": {
            "delete": {
                "operationId": "cancelShipment",
                "summary": "Cancel a shipment that has not left the warehouse",
                "parameters": [
                    {
                        "name": "shipmentId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
            }
        },
        "/inventory/{sku}": {
            "get": {
                "operationId": "getStockLevel",
                "summary": "Get the stock level for a SKU",
                "parameters": [
                    {"name": "sku", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
            }
        },
    },
}

broker = ToolBroker()
broker.add_source(OpenAPISource(SPEC, namespace="warehouse"))
broker.index()

print("discovered:")
for tool in broker.tools():
    http = tool.metadata["http"]
    scopes = ", ".join(sorted(tool.required_scopes)) or "-"
    print(
        f"  {tool.id:<34} {http['method']:<7}{http['path']:<26} "
        f"risk={tool.risk.value:<7} scopes={scopes}"
    )

print()
selection = broker.select("check the stock level for a SKU", k=1)
print("query -> ", list(selection.tool_ids))

# Risk is inferred from the HTTP verb, so a read-only agent is one rule away.
broker.set_policy(PolicyEngine([MaxRisk(RiskTier.LOW)]))
readonly = broker.select("cancel that shipment", k=3)
print("read-only agent ->", list(readonly.tool_ids))
for exclusion in readonly.exclusions:
    if exclusion.stage.value == "policy":
        print(f"  blocked {exclusion.tool_id}: {exclusion.reason}")

print()
print("everything needed to actually issue the request is in metadata:")
create = broker.get("warehouse/createShipment")
print(" ", create.metadata["http"])
