from __future__ import annotations

import pytest

from toolbroker import Tool
from toolbroker.index.enrich import (
    EnrichmentConfig,
    build_index_text,
    describe_schema,
    humanize,
    tokenize,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("get_user_by_id", "get user by id"),
        ("getUserById", "get user by id"),
        ("search-orders", "search orders"),
        ("plain", "plain"),
    ],
)
def test_identifiers_are_split_into_words(raw, expected):
    assert humanize(raw) == expected


def test_index_text_includes_the_name_and_description():
    text = build_index_text(Tool(name="search_orders", description="Find orders"))
    assert "search orders" in text
    assert "Find orders" in text


def test_name_weight_repeats_the_name():
    config = EnrichmentConfig(name_weight=3)
    text = build_index_text(Tool(name="refund"), config)
    assert text.count("refund") == 3


def test_zero_weight_omits_the_field():
    config = EnrichmentConfig(name_weight=0)
    assert "refund" not in build_index_text(Tool(name="refund"), config)


def test_parameter_names_are_searchable():
    tool = Tool(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {"invoice_id": {"type": "string", "description": "The invoice"}},
            "required": ["invoice_id"],
        },
    )
    text = build_index_text(tool)
    assert "invoice id" in text
    assert "The invoice" in text
    assert "[required]" in text


def test_tags_and_examples_are_included():
    tool = Tool(name="a", tags=frozenset({"billing"}), examples=("refund order 42",))
    text = build_index_text(tool)
    assert "billing" in text
    assert "refund order 42" in text


def test_parameters_are_capped():
    schema = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string"} for i in range(50)},
    }
    summary = describe_schema(schema, max_parameters=3)
    assert summary.count(";") == 2


def test_empty_schema_produces_nothing():
    assert describe_schema({}) == ""


def test_tokenizer_splits_identifiers():
    assert tokenize("getUserById and search_orders") == [
        "get",
        "user",
        "by",
        "id",
        "and",
        "search",
        "orders",
    ]


def test_default_name_weight_is_one():
    # Measured, not assumed: name_weight=1 beat 2 and 3 for both the lexical
    # and the semantic embedder at every catalogue size (bench/ablate.py).
    # Changing this default should come with new numbers.
    assert EnrichmentConfig().name_weight == 1


def test_default_still_includes_the_name():
    # Dropping the name entirely costs the lexical embedder ~0.066 recall.
    text = build_index_text(Tool(name="issue_refund", description="Refund a payment"))
    assert "issue refund" in text
    assert text.count("issue_refund") == 1
