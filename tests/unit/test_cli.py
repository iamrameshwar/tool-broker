from __future__ import annotations

import json

import pytest

from toolbroker.cli.main import main

CONFIG = """
embedder:
  name: hashing
  options: {dim: 64}
sources:
  - type: json
    name: fixtures
    options: {path: "%s"}
policy:
  default_k: 2
"""

TOOLS = [
    {"name": "search_orders", "description": "Find recent orders for a customer"},
    {"name": "check_inventory", "description": "How many units of a SKU are in stock"},
    {"name": "delete_customer", "description": "Erase a customer record"},
]


@pytest.fixture
def config_path(tmp_path):
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps(TOOLS))
    config = tmp_path / "toolbroker.yaml"
    config.write_text(CONFIG % tools_file)
    return config


def test_index_reports_what_it_found(config_path, capsys):
    assert main(["-c", str(config_path), "index"]) == 0
    assert "indexed 3 tools" in capsys.readouterr().out


def test_index_json_output(config_path, capsys):
    assert main(["-c", str(config_path), "index", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tools"] == 3


def test_index_can_dump_discovered_tools(config_path, tmp_path, capsys):
    out = tmp_path / "dump.json"
    main(["-c", str(config_path), "index", "--out", str(out)])
    assert len(json.loads(out.read_text())) == 3


def test_query_prints_the_full_trace(config_path, capsys):
    assert main(["-c", str(config_path), "query", "units in stock", "-k", "1"]) == 0
    output = capsys.readouterr().out
    assert "selected:" in output
    assert "considered 3 tools" in output


def test_query_json_output(config_path, capsys):
    main(["-c", str(config_path), "query", "units in stock", "-k", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_k"] == 1


def test_query_can_render_through_an_adapter(config_path, capsys):
    main(["-c", str(config_path), "query", "stock", "-k", "1", "--render", "openai"])
    assert "rendered for openai" in capsys.readouterr().out


def test_bench_scores_a_query_set(config_path, tmp_path, capsys):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "units in stock", "expected": ["default/check_inventory"]}) + "\n"
    )
    assert main(["-c", str(config_path), "bench", str(queries), "-k", "2"]) == 0
    assert "recall@2" in capsys.readouterr().out


def test_missing_config_is_an_error_not_a_traceback(capsys):
    assert main(["query", "anything"]) == 1
    assert "no config supplied" in capsys.readouterr().err


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_calibrate_prints_a_report(config_path, capsys):
    assert main(["-c", str(config_path), "calibrate"]) == 0
    output = capsys.readouterr().out
    assert "blocks noise" in output
    assert "top-1 score when nothing should match" in output


def test_calibrate_json_output(config_path, capsys):
    assert main(["-c", str(config_path), "calibrate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["catalogue_size"] == 3
    assert payload["suggestions"]


def test_calibrate_with_samples_shows_the_cost(config_path, tmp_path, capsys):
    samples = tmp_path / "samples.txt"
    samples.write_text("units in stock\nfind recent orders\n")
    main(["-c", str(config_path), "calibrate", "--samples", str(samples)])
    output = capsys.readouterr().out
    assert "samples emptied" in output
    assert "samples thinned" in output


def test_doctor_prints_a_report(config_path, capsys):
    assert main(["-c", str(config_path), "doctor"]) == 0
    output = capsys.readouterr().out
    assert "CLOSEST PAIRS" in output
    assert "phrased unlike anything a user would type" in output


def test_doctor_json_output(config_path, capsys):
    assert main(["-c", str(config_path), "doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["catalogue_size"] == 3
    assert len(payload["findings"]) == 3
    assert payload["margin"] is None


def test_doctor_flags_no_twins_without_an_explicit_margin(config_path, capsys):
    main(["-c", str(config_path), "doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert not any("twinned" in f["issues"] for f in payload["findings"])


def test_doctor_margin_is_opt_in(config_path, capsys):
    main(["-c", str(config_path), "doctor", "--margin", "0.99", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["margin"] == 0.99
    assert any("twinned" in f["issues"] for f in payload["findings"])


def test_doctor_with_samples_reports_coverage(config_path, tmp_path, capsys):
    samples = tmp_path / "samples.txt"
    samples.write_text("units in stock\n")
    main(["-c", str(config_path), "doctor", "--samples", str(samples), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_count"] == 1


def test_calibrate_accepts_custom_noise(config_path, tmp_path, capsys):
    noise = tmp_path / "noise.txt"
    noise.write_text("# a comment is ignored\nzzz aaa\n\nqqq www\n")
    main(["-c", str(config_path), "calibrate", "--noise", str(noise), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["noise_scores"]) == 2
