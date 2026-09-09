from __future__ import annotations

import pytest

from toolbroker import RiskTier, ToolBroker, classify_by_name
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.risk import always
from toolbroker.sources import PythonFunctionSource


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("delete_customer", RiskTier.HIGH),
        ("remove_file", RiskTier.HIGH),
        ("purge_cache", RiskTier.HIGH),
        ("terminate_instance", RiskTier.HIGH),
        ("wipe_disk", RiskTier.HIGH),
        ("create_invoice", RiskTier.MEDIUM),
        ("send_email", RiskTier.MEDIUM),
        ("restart_service", RiskTier.MEDIUM),
        ("deploy_release", RiskTier.MEDIUM),
        ("list_orders", RiskTier.LOW),
        ("get_user", RiskTier.LOW),
        ("search_docs", RiskTier.LOW),
    ],
)
def test_classification_by_leading_verb(name, expected):
    assert classify_by_name(name) is expected


def test_case_is_ignored():
    assert classify_by_name("DELETE_Customer") is RiskTier.HIGH


def test_description_does_not_influence_the_result():
    # Prose is far too easy to mislead on; only the verb is read.
    assert classify_by_name("list_users", "This will DELETE and DESTROY everything") is RiskTier.LOW


def test_unknown_verb_is_low():
    assert classify_by_name("frobnicate_widget") is RiskTier.LOW


def test_always_helper():
    classifier = always(RiskTier.CRITICAL)
    assert classifier("list_orders") is RiskTier.CRITICAL


def delete_customer(customer_id: str) -> None:
    """Erase a customer record."""


def list_orders() -> list[str]:
    """List all orders."""
    return []


def test_python_source_does_not_infer_by_default():
    # Code you own should carry an explicit risk; a guess would be surprising.
    tools = {t.name: t for t in PythonFunctionSource([delete_customer, list_orders]).discover()}
    assert tools["delete_customer"].risk is RiskTier.LOW


def test_python_source_can_opt_into_inference():
    source = PythonFunctionSource([delete_customer, list_orders], risk_classifier=classify_by_name)
    tools = {t.name: t for t in source.discover()}
    assert tools["delete_customer"].risk is RiskTier.HIGH
    assert tools["list_orders"].risk is RiskTier.LOW


def test_explicit_risk_beats_the_classifier():
    source = PythonFunctionSource(risk_classifier=always(RiskTier.CRITICAL))
    tool = source.add(list_orders, risk=RiskTier.LOW)
    assert tool.risk is RiskTier.LOW


def test_classifier_sees_the_overridden_name():
    seen: list[str] = []

    def spy(name: str, description: str = "") -> RiskTier:
        seen.append(name)
        return RiskTier.LOW

    PythonFunctionSource(risk_classifier=spy).add(list_orders, name="drop_everything")
    assert seen == ["drop_everything"]


def test_add_functions_passes_the_classifier_through():
    broker = ToolBroker(embedder=HashingEmbedder(dim=64), cache_embeddings=False)
    broker.add_functions([delete_customer], risk_classifier=classify_by_name)
    broker.index()
    assert broker.get("python/delete_customer").risk is RiskTier.HIGH


def test_mcp_module_alias_still_resolves():
    # Kept so existing imports from the MCP source keep working.
    from toolbroker.sources.mcp import default_risk_classifier

    assert default_risk_classifier is classify_by_name
