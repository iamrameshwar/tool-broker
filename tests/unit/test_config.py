from __future__ import annotations

import json
import math

import pytest

from toolbroker.config import ToolBrokerConfig, load
from toolbroker.errors import ConfigurationError
from toolbroker.types import Decision, RiskTier

YAML = """
embedder:
  name: hashing
  options: {dim: 64}
retrieval:
  mode: hybrid
policy:
  default_k: 3
  agents:
    support:
      max_tools: 2
      max_risk: low
      deny: ["*/delete_*"]
    ops:
      max_risk: critical
"""


def write(tmp_path, text, name="toolbroker.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_yaml_round_trip(tmp_path):
    config = ToolBrokerConfig.from_file(write(tmp_path, YAML))
    assert config.embedder.name == "hashing"
    assert config.retrieval.mode == "hybrid"
    assert config.policy.agents["support"].max_tools == 2


def test_build_produces_a_working_catalogue(tmp_path, sample_tools):
    broker = load(write(tmp_path, YAML))
    broker.index(sample_tools)
    assert broker.embedder.dim == 64
    assert len(broker.select("stock", k=3)) <= 3


def test_agent_policies_are_wired(tmp_path, sample_tools):
    broker = load(write(tmp_path, YAML))
    broker.index(sample_tools)

    support = broker.select("erase the customer record", k=5, agent="support")
    assert "test/delete_customer" not in support.tool_ids
    assert len(support) <= 2

    ops = broker.select("erase the customer record", k=5, agent="ops")
    assert "test/delete_customer" in ops.tool_ids


def test_json_config_is_accepted(tmp_path):
    path = write(tmp_path, json.dumps({"policy": {"default_k": 7}}), "config.json")
    assert ToolBrokerConfig.from_file(path).policy.default_k == 7


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        ToolBrokerConfig.from_file(tmp_path / "absent.yaml")


def test_unknown_key_is_an_error_not_a_silent_default(tmp_path):
    # A typo in a policy file must fail loudly; the failure mode otherwise is
    # an agent quietly running with no restrictions.
    with pytest.raises(Exception, match="typoed_key"):
        ToolBrokerConfig.from_file(write(tmp_path, "typoed_key: true"))


def test_policy_section_builds_rules():
    section = ToolBrokerConfig.model_validate(
        {
            "policy": {
                "agents": {
                    "a": {
                        "max_tools": 3,
                        "max_risk": "medium",
                        "deny": ["x/*"],
                        "deny_tags": ["destructive"],
                        "min_score": 0.2,
                        "default": "deny",
                    }
                }
            }
        }
    )
    engine = section.policy.agents["a"].build()
    assert len(engine.rules) == 4
    assert len(engine.selection_rules) == 2


def test_dry_run_flows_through_to_the_engine():
    config = ToolBrokerConfig.model_validate({"policy": {"default": {"dry_run": True}}})
    assert config.policy.default.build().dry_run is True


def test_risk_tier_parses_from_a_string():
    config = ToolBrokerConfig.model_validate({"policy": {"default": {"max_risk": "high"}}})
    assert config.policy.default.max_risk is RiskTier.HIGH


def test_default_decision_parses():
    config = ToolBrokerConfig.model_validate({"policy": {"default": {"default": "deny"}}})
    engine = config.policy.default.build()
    assert engine.evaluate([]).hits == ()
    assert config.policy.default.default == "deny"
    assert Decision.DENY.value == "deny"


def test_static_json_source_from_config(tmp_path):
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps([{"name": "alpha", "description": "first tool"}]))
    config = ToolBrokerConfig.model_validate(
        {
            "embedder": {"name": "hashing", "options": {"dim": 64}},
            "sources": [{"type": "json", "name": "fixtures", "options": {"path": str(tools_file)}}],
        }
    )
    broker = config.build()
    broker.index()
    assert broker.get("default/alpha") is not None


def test_usage_is_off_by_default():
    broker = ToolBrokerConfig.model_validate(
        {"embedder": {"name": "hashing", "options": {"dim": 64}}}
    ).build()
    assert broker.usage is None


def test_usage_can_be_enabled():
    broker = ToolBrokerConfig.model_validate(
        {
            "embedder": {"name": "hashing", "options": {"dim": 64}},
            "usage": {"enabled": True, "weight": 0.2, "half_life_days": 7},
        }
    ).build()
    assert broker.usage is not None
    assert broker.usage.half_life == 7 * 86400.0


def test_usage_counts_are_restored_from_disk(tmp_path):
    from toolbroker import UsageTracker

    path = tmp_path / "usage.json"
    seed = UsageTracker(half_life=math.inf)
    seed.record("shop/search", 5)
    seed.save(path)

    broker = ToolBrokerConfig.model_validate(
        {
            "embedder": {"name": "hashing", "options": {"dim": 64}},
            "usage": {"enabled": True, "path": str(path), "half_life_days": 3650},
        }
    ).build()
    assert broker.usage.count("shop/search") == pytest.approx(5.0, rel=1e-3)


def test_usage_weight_is_bounded_by_validation():
    with pytest.raises(Exception, match="less than or equal to 1"):
        ToolBrokerConfig.model_validate({"usage": {"enabled": True, "weight": 5.0}})


def test_strict_agents_is_off_by_default():
    assert ToolBrokerConfig().policy.build().strict is False


def test_strict_agents_can_be_enabled(sample_tools):
    broker = ToolBrokerConfig.model_validate(
        {
            "embedder": {"name": "hashing", "options": {"dim": 64}},
            "policy": {"strict_agents": True, "agents": {"support": {"max_tools": 2}}},
        }
    ).build()
    broker.index(sample_tools)
    assert broker.select("customer", k=5, agent="typo").tool_ids == ()
    assert broker.select("customer", k=5, agent="support").tool_ids
