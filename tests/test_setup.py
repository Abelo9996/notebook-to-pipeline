from __future__ import annotations

import json
import os
import stat
import tomllib

import pytest

from notebook_to_pipeline.setup_agents import NAME, run_setup


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir()
    (home / ".cursor").mkdir()
    (home / ".codex" / "config.toml").write_text('model = "x"\n')
    (home / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "foo"}}}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "claude.log"
    registered = tmp_path / "registered"
    claude = bin_dir / "claude"
    claude.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        f'if [ "$2" = "get" ]; then [ -f "{registered}" ] && exit 0 || exit 1; fi\n'
        f'if [ "$2" = "add" ]; then : > "{registered}"; fi\n'
        "exit 0\n"
    )
    claude.chmod(claude.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_dir))
    return home, log


def test_dry_run_changes_nothing(fake_home):
    home, log = fake_home
    lines = []
    res = run_setup(yes=False, interactive=False, out=lines.append)
    assert res["dry_run"] is True and res["applied"] == []
    assert not (home / ".claude" / "skills").exists()
    assert "notebook-to-pipeline" not in (home / ".codex" / "config.toml").read_text()
    assert "add" not in (log.read_text() if log.exists() else "")
    assert any("register MCP server with Codex" in line for line in lines)


def test_apply_then_idempotent(fake_home):
    home, log = fake_home
    res = run_setup(yes=True, out=lambda s: None)
    assert all("error" not in a for a in res["applied"]), res
    assert "mcp add --scope user notebook-to-pipeline -- uvx notebook-to-pipeline mcp" in log.read_text()
    cfg = tomllib.loads((home / ".codex" / "config.toml").read_text())
    assert cfg["model"] == "x"
    assert cfg["mcp_servers"][NAME] == {"command": "uvx", "args": [NAME, "mcp"]}
    cur = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert cur["mcpServers"]["other"] == {"command": "foo"}
    assert cur["mcpServers"][NAME]["args"] == [NAME, "mcp"]
    for d in (".claude", ".codex"):
        assert (home / d / "skills" / NAME / "SKILL.md").read_text().startswith("---")
    backups = [p for p in os.listdir(home / ".codex") if ".bak-nb2p-" in p]
    assert backups
    second = run_setup(yes=True, out=lambda s: None)
    assert second["applied"] == []


def test_project_mode_writes_mcp_json(fake_home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    run_setup(yes=True, project=proj, out=lambda s: None)
    data = json.loads((proj / ".mcp.json").read_text())
    assert data["mcpServers"][NAME]["command"] == "uvx"
