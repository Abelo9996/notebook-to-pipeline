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
    assert "Proposed stages:" in text and "load" in text


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
