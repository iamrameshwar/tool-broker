"""Config files get committed; the secrets in them must not be."""

from __future__ import annotations

import json

import pytest

from toolbroker.config import ToolBrokerConfig
from toolbroker.errors import ConfigurationError
from toolbroker.interpolate import interpolate

ENV = {"DSN": "postgresql://localhost/db", "EMPTY": "", "PORT": "5432"}


def test_a_plain_string_is_untouched():
    assert interpolate("no variables here", env=ENV) == "no variables here"


def test_a_required_variable_is_substituted():
    assert interpolate("${DSN}", env=ENV) == "postgresql://localhost/db"


def test_a_variable_inside_a_larger_string():
    assert interpolate("host=${DSN};ssl=on", env=ENV) == "host=postgresql://localhost/db;ssl=on"


def test_several_variables_in_one_string():
    assert interpolate("${DSN}:${PORT}", env=ENV) == "postgresql://localhost/db:5432"


def test_a_default_is_used_when_unset():
    assert interpolate("${NOPE:-fallback}", env=ENV) == "fallback"


def test_a_set_variable_beats_its_default():
    assert interpolate("${DSN:-fallback}", env=ENV) == "postgresql://localhost/db"


def test_an_empty_variable_counts_as_set():
    """Matching the shell: exported-but-empty is not the same as unset."""
    assert interpolate("${EMPTY:-fallback}", env=ENV) == ""


def test_an_empty_default_is_allowed():
    assert interpolate("${NOPE:-}", env=ENV) == ""


def test_a_missing_variable_is_an_error_not_a_literal():
    """The quiet failure this prevents: a DSN of the literal text '${DSN}'."""
    with pytest.raises(ConfigurationError, match="MISSING"):
        interpolate("${MISSING}", env=ENV)


def test_the_error_names_where_it_appeared():
    with pytest.raises(ConfigurationError, match=r"store\.options\.dsn"):
        interpolate({"store": {"options": {"dsn": "${MISSING}"}}}, env=ENV)


def test_the_error_suggests_a_default():
    with pytest.raises(ConfigurationError, match=r"\$\{MISSING:-some-value\}"):
        interpolate("${MISSING}", env=ENV)


def test_every_missing_variable_is_reported_at_once():
    with pytest.raises(ConfigurationError) as exc:
        interpolate("${ONE}/${TWO}", env=ENV)
    assert "ONE" in str(exc.value) and "TWO" in str(exc.value)


def test_a_double_dollar_escapes_to_a_literal():
    assert interpolate("$${DSN}", env=ENV) == "${DSN}"


def test_nested_structures_are_walked():
    payload = {"a": ["${DSN}", {"b": "${PORT}"}], "c": 7, "d": None, "e": True}
    assert interpolate(payload, env=ENV) == {
        "a": ["postgresql://localhost/db", {"b": "5432"}],
        "c": 7,
        "d": None,
        "e": True,
    }


def test_non_strings_are_left_alone():
    assert interpolate(7, env=ENV) == 7
    assert interpolate(None, env=ENV) is None
    assert interpolate(1.5, env=ENV) == 1.5


def test_keys_are_not_substituted():
    """Only values. A variable in a key would make the schema unpredictable."""
    assert interpolate({"${DSN}": "value"}, env=ENV) == {"${DSN}": "value"}


def test_a_lone_dollar_sign_is_not_a_reference():
    assert interpolate("costs $5, or ${PORT}", env=ENV) == "costs $5, or 5432"


def test_an_unclosed_brace_is_left_alone():
    assert interpolate("${UNCLOSED", env=ENV) == "${UNCLOSED"


def test_config_files_are_interpolated_on_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_TEST_MODEL", "my-model")
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps([{"name": "a", "description": "A tool."}]), encoding="utf-8")
    config = tmp_path / "toolbroker.json"
    config.write_text(
        json.dumps(
            {
                "embedder": {"name": "hashing", "options": {"dim": 64}},
                "sources": [{"type": "json", "name": "s", "options": {"path": "${TB_TEST_TOOLS}"}}],
                "enrichment": {"name_weight": 1},
                "policy": {"default_k": 5},
                "usage": {"enabled": False},
                "retrieval": {"mode": "semantic"},
                "cache_embeddings": False,
                "observability": {},
                "hooks": {},
                "store": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TB_TEST_TOOLS", str(tools))
    loaded = ToolBrokerConfig.from_file(config)
    assert loaded.sources[0].options["path"] == str(tools)


def test_loading_fails_loudly_when_a_variable_is_unset(tmp_path):
    config = tmp_path / "toolbroker.json"
    config.write_text(
        json.dumps({"store": {"name": "pgvector", "options": {"dsn": "${TB_ABSENT_DSN}"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="TB_ABSENT_DSN"):
        ToolBrokerConfig.from_file(config)
