from __future__ import annotations

import json

import anyio

from notebook_to_pipeline.cli import main


def test_cli_analyze_json(nb_factory, capsys):
    nb = nb_factory(["y = x + 1", "x = 1"])
    code = main(["analyze", str(nb), "--json", "--strict"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["summary"]["predicts_top_to_bottom_failure"] is True


def test_cli_analyze_text(nb_factory, capsys):
    nb = nb_factory(["import pandas as pd", "df = pd.read_csv('a.csv')"])
    assert main(["analyze", str(nb)]) == 0
    text = capsys.readouterr().out
    assert "Proposed stages" in text and "load.py" in text and "Next:" in text


def test_cli_missing_file(capsys, tmp_path):
    assert main(["analyze", str(tmp_path / "nope.ipynb")]) == 2


def test_mcp_server_lists_tools():
    from notebook_to_pipeline.mcp_server import build_server

    server = build_server()

    async def names():
        tools = await server.list_tools()
        return sorted(t.name for t in tools)

    assert anyio.run(names) == [
        "analyze_notebook",
        "capture_reference",
        "scaffold_pipeline",
        "verify_pipeline",
        "write_report",
    ]


def test_cli_setup_json_is_valid_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))  # no agent CLIs on PATH
    (tmp_path / ".cursor").mkdir()
    assert main(["setup", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert any("Cursor" in a["description"] for a in out["actions"])


def test_mcp_analyze_is_compact(nb_factory):
    from notebook_to_pipeline.analysis import analyze
    from notebook_to_pipeline.mcp_server import _compact_analysis

    nb = nb_factory(["import pandas as pd", "df = pd.read_csv('a.csv')", "n = len(df)"])
    out = _compact_analysis(analyze(nb))
    assert "cells" not in out and "edges" not in out
    assert out["suggested_artifacts"] == ["df", "n"]
    assert out["next_steps"] and "capture_reference" in out["next_steps"][0]
