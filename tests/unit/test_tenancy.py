"""One deployment, many customers. Isolation has to fail closed."""

from __future__ import annotations

import json

from toolbroker import PolicyEngine, RequireScopes, TenantIsolation, Tool, ToolBroker
from toolbroker.config import ToolBrokerConfig
from toolbroker.index.embedders import HashingEmbedder

ACME = Tool(
    name="export_ledger",
    namespace="finance",
    description="Export the general ledger.",
    metadata={"tenant": "acme"},
)
GLOBEX = Tool(
    name="export_ledger",
    namespace="globex",
    description="Export the general ledger.",
    metadata={"tenant": "globex"},
)
SHARED = Tool(name="get_time", namespace="util", description="Return the current time.")

CATALOGUE = [ACME, GLOBEX, SHARED]


def _broker(*rules) -> ToolBroker:
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(CATALOGUE)
    broker.set_policy(PolicyEngine(rules))
    return broker


# --- the isolation boundary ------------------------------------------------


def test_a_tenant_sees_its_own_tools():
    broker = _broker(TenantIsolation())
    ids = broker.select("export the ledger", k=5, tenant="acme").tool_ids
    assert "finance/export_ledger" in ids


def test_a_tenant_never_sees_another_tenants_tools():
    broker = _broker(TenantIsolation())
    ids = broker.select("export the ledger", k=5, tenant="acme").tool_ids
    assert "globex/export_ledger" not in ids


def test_the_isolation_is_symmetric():
    broker = _broker(TenantIsolation())
    ids = broker.select("export the ledger", k=5, tenant="globex").tool_ids
    assert "globex/export_ledger" in ids
    assert "finance/export_ledger" not in ids


def test_a_request_with_no_tenant_sees_no_tenant_owned_tools():
    """Fail closed. An undeclared tenant must not mean 'all tenants'."""
    broker = _broker(TenantIsolation())
    ids = broker.select("export the ledger", k=5).tool_ids
    assert "finance/export_ledger" not in ids
    assert "globex/export_ledger" not in ids


def test_an_unknown_tenant_sees_nothing_owned():
    broker = _broker(TenantIsolation())
    ids = broker.select("export the ledger", k=5, tenant="nobody").tool_ids
    assert "finance/export_ledger" not in ids
    assert "globex/export_ledger" not in ids


def test_shared_tools_are_visible_by_default():
    broker = _broker(TenantIsolation())
    assert "util/get_time" in broker.select("what time is it", k=5, tenant="acme").tool_ids


def test_shared_tools_can_be_switched_off():
    """Safer where forgetting to tag a tool is the likely mistake."""
    broker = _broker(TenantIsolation(shared_tools=False))
    assert "util/get_time" not in broker.select("what time is it", k=5, tenant="acme").tool_ids


def test_the_tenant_key_is_configurable():
    tool = Tool(name="a", namespace="x", description="A tool.", metadata={"org_id": "acme"})
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index([tool])
    broker.set_policy(PolicyEngine([TenantIsolation(tenant_key="org_id")]))
    assert broker.select("a tool", k=5, tenant="acme").tool_ids == ("x/a",)
    assert broker.select("a tool", k=5, tenant="other").tool_ids == ()


def test_a_non_string_tenant_value_still_isolates():
    """Metadata comes from MCP servers; it is not guaranteed to be a string."""
    tool = Tool(name="a", namespace="x", description="A tool.", metadata={"tenant": 42})
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index([tool])
    broker.set_policy(PolicyEngine([TenantIsolation()]))
    assert broker.select("a tool", k=5, tenant="42").tool_ids == ("x/a",)
    assert broker.select("a tool", k=5, tenant="43").tool_ids == ()


def test_the_exclusion_says_why():
    broker = _broker(TenantIsolation())
    selection = broker.select("export the ledger", k=5, tenant="acme")
    reasons = {e.tool_id: e.reason for e in selection.exclusions}
    assert "another tenant" in reasons.get("globex/export_ledger", "")


# --- why scopes were not enough --------------------------------------------


def test_scopes_alone_leak_an_untagged_tool():
    """RequireScopes abstains for a tool with none, which is fail-open."""
    untagged = Tool(name="secret_export", namespace="finance", description="Export secrets.")
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index([untagged])
    broker.set_policy(PolicyEngine([RequireScopes()]))
    assert broker.select("export secrets", k=5).tool_ids == ("finance/secret_export",)


def test_tenant_isolation_decides_rather_than_abstains():
    """The same tool, under isolation with shared tools off, is invisible."""
    untagged = Tool(name="secret_export", namespace="finance", description="Export secrets.")
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index([untagged])
    broker.set_policy(PolicyEngine([TenantIsolation(shared_tools=False)]))
    assert broker.select("export secrets", k=5, tenant="acme").tool_ids == ()


# --- from config -----------------------------------------------------------


def _config(tmp_path, **agent) -> ToolBrokerConfig:
    tools = tmp_path / "tools.json"
    tools.write_text(
        json.dumps([tool.model_dump(mode="json") for tool in CATALOGUE]), encoding="utf-8"
    )
    return ToolBrokerConfig.model_validate(
        {
            "sources": [{"type": "json", "name": "cat", "options": {"path": str(tools)}}],
            "policy": {"agents": {"tenant_app": agent}},
        }
    )


def test_tenancy_is_off_unless_asked_for(tmp_path):
    broker = _config(tmp_path, max_tools=5).build()
    broker.index()
    ids = broker.select("export the ledger", k=5, agent="tenant_app").tool_ids
    assert "finance/export_ledger" in ids


def test_tenancy_can_be_switched_on_from_config(tmp_path):
    broker = _config(tmp_path, tenant_isolation=True).build()
    broker.index()
    ids = broker.select("export the ledger", k=5, agent="tenant_app", tenant="acme").tool_ids
    assert "finance/export_ledger" in ids
    assert "globex/export_ledger" not in ids


def test_shared_tools_can_be_disabled_from_config(tmp_path):
    broker = _config(tmp_path, tenant_isolation=True, shared_tools=False).build()
    broker.index()
    ids = broker.select("what time is it", k=5, agent="tenant_app", tenant="acme").tool_ids
    assert "util/get_time" not in ids


def test_the_tenant_key_can_be_set_from_config(tmp_path):
    config = _config(tmp_path, tenant_isolation=True, tenant_key="org_id")
    engine = config.policy.build().for_agent("tenant_app")
    rule = next(r for r in engine.rules if r.name == "tenant_isolation")
    assert rule.tenant_key == "org_id"
