"""`nb2p setup`: register the MCP server with Claude Code, Codex and Cursor and install the skill.

Shows the plan first. Nothing is changed without --yes (or an interactive yes).
Every file it edits is backed up first. Running it twice changes nothing the second time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NAME = "notebook-to-pipeline"
DEFAULT_COMMAND = ["uvx", NAME, "mcp"]


@dataclass
class Action:
    target: str
    description: str
    kind: str  # "command", "edit", "copy", "skip"
    detail: str = ""
    apply: Any = field(default=None, repr=False)

    def as_dict(self) -> dict[str, str]:
        return {"target": self.target, "kind": self.kind, "description": self.description,
                "detail": self.detail}


def skill_source() -> Path:
    here = Path(__file__).resolve().parent
    packaged = here / "_skill" / "SKILL.md"
    if packaged.exists():
        return packaged
    repo = here.parents[1] / "skills" / NAME / "SKILL.md"
    if repo.exists():
        return repo
    raise FileNotFoundError("SKILL.md not found in the package or the source tree")


def _backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_name(f"{path.name}.bak-nb2p-{stamp}")
    shutil.copy2(path, bak)
    return bak


def _skill_action(dest_dir: Path, label: str) -> Action:
    src = skill_source()
    dest = dest_dir / NAME / "SKILL.md"
    content = src.read_text(encoding="utf-8")
    if dest.exists() and dest.read_text(encoding="utf-8") == content:
        return Action(str(dest), f"{label} skill already up to date", "skip")

    def apply() -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        note = ""
        if dest.exists():
            note = f" (backup: {_backup(dest)})"
        dest.write_text(content, encoding="utf-8")
        return f"wrote {dest}{note}"

    verb = "update" if dest.exists() else "install"
    return Action(str(dest), f"{verb} {label} skill", "copy", f"copy {src.name} to {dest}", apply)


def plan(home: Path | None = None, *, command: list[str] | None = None,
         project: Path | None = None) -> list[Action]:
    home = home or Path.home()
    command = command or DEFAULT_COMMAND
    actions: list[Action] = []

    # Claude Code
    claude_bin = shutil.which("claude")
    claude_dir = home / ".claude"
    if project is not None:
        mcp_json = project / ".mcp.json"
        data = json.loads(mcp_json.read_text()) if mcp_json.exists() else {}
        entry = {"command": command[0], "args": command[1:]}
        if data.get("mcpServers", {}).get(NAME) == entry:
            actions.append(Action(str(mcp_json), "Claude Code project server already registered", "skip"))
        else:
            def apply_project(mcp_json=mcp_json, entry=entry) -> str:
                cur = json.loads(mcp_json.read_text()) if mcp_json.exists() else {}
                note = f" (backup: {_backup(mcp_json)})" if mcp_json.exists() else ""
                cur.setdefault("mcpServers", {})[NAME] = entry
                mcp_json.write_text(json.dumps(cur, indent=2) + "\n")
                return f"wrote {mcp_json}{note}"

            actions.append(Action(str(mcp_json), "register MCP server in the project .mcp.json", "edit",
                                  json.dumps({"mcpServers": {NAME: entry}}), apply_project))
    elif claude_bin:
        exists = subprocess.run([claude_bin, "mcp", "get", NAME], capture_output=True, text=True)
        if exists.returncode == 0:
            actions.append(Action("claude mcp", "Claude Code server already registered", "skip"))
        else:
            cmd = [claude_bin, "mcp", "add", "--scope", "user", NAME, "--", *command]

            def apply_claude(cmd=cmd) -> str:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"`{' '.join(cmd)}` failed: {r.stderr.strip()}")
                return f"ran {' '.join(cmd[1:])}"

            actions.append(Action("claude mcp", "register MCP server with Claude Code (user scope)", "command",
                                  "claude " + " ".join(cmd[1:]), apply_claude))
    elif claude_dir.exists():
        actions.append(Action("claude mcp", "Claude Code config found but the `claude` CLI is not on PATH", "skip",
                              "run: claude mcp add --scope user " + NAME + " -- " + " ".join(command)))
    if claude_dir.exists():
        actions.append(_skill_action(claude_dir / "skills", "Claude Code"))

    # Codex
    codex_dir = home / ".codex"
    if codex_dir.exists() or shutil.which("codex"):
        cfg = codex_dir / "config.toml"
        current = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        try:
            parsed = tomllib.loads(current) if current else {}
        except tomllib.TOMLDecodeError as exc:
            parsed = None
            actions.append(Action(str(cfg), "Codex config is not valid TOML, leaving it alone", "skip", str(exc)))
        if parsed is not None:
            if NAME in parsed.get("mcp_servers", {}):
                actions.append(Action(str(cfg), "Codex server already registered", "skip"))
            else:
                block = (f'\n[mcp_servers.{NAME}]\ncommand = "{command[0]}"\n'
                         f"args = {json.dumps(command[1:])}\n")

                def apply_codex(cfg=cfg, block=block) -> str:
                    cfg.parent.mkdir(parents=True, exist_ok=True)
                    note = f" (backup: {_backup(cfg)})" if cfg.exists() else ""
                    text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
                    if text and not text.endswith("\n"):
                        text += "\n"
                    cfg.write_text(text + block, encoding="utf-8")
                    return f"appended [mcp_servers.{NAME}] to {cfg}{note}"

                actions.append(Action(str(cfg), "register MCP server with Codex", "edit", block.strip(), apply_codex))
        actions.append(_skill_action(codex_dir / "skills", "Codex"))

    # Cursor
    cursor_dir = home / ".cursor"
    if cursor_dir.exists():
        cfg = cursor_dir / "mcp.json"
        entry = {"command": command[0], "args": command[1:]}
        try:
            data = json.loads(cfg.read_text()) if cfg.exists() and cfg.read_text().strip() else {}
        except json.JSONDecodeError as exc:
            data = None
            actions.append(Action(str(cfg), "Cursor mcp.json is not valid JSON, leaving it alone", "skip", str(exc)))
        if data is not None:
            if data.get("mcpServers", {}).get(NAME) == entry:
                actions.append(Action(str(cfg), "Cursor server already registered", "skip"))
            else:
                def apply_cursor(cfg=cfg, entry=entry) -> str:
                    cur = json.loads(cfg.read_text()) if cfg.exists() and cfg.read_text().strip() else {}
                    note = f" (backup: {_backup(cfg)})" if cfg.exists() else ""
                    cur.setdefault("mcpServers", {})[NAME] = entry
                    cfg.write_text(json.dumps(cur, indent=2) + "\n")
                    return f"wrote {cfg}{note}"

                actions.append(Action(str(cfg), "register MCP server with Cursor", "edit",
                                      json.dumps({"mcpServers": {NAME: entry}}), apply_cursor))
    return actions


def run_setup(*, yes: bool = False, home: Path | None = None, command: list[str] | None = None,
              project: Path | None = None, interactive: bool | None = None,
              out=print) -> dict[str, Any]:
    actions = plan(home, command=command, project=project)
    todo = [a for a in actions if a.apply is not None]
    out("notebook-to-pipeline setup plan:")
    if not actions:
        out("  No Claude Code, Codex or Cursor installation found. Nothing to do.")
    for a in actions:
        mark = "skip" if a.apply is None else a.kind
        out(f"  [{mark}] {a.description}: {a.target}")
        if a.detail and a.apply is not None:
            for line in a.detail.splitlines():
                out(f"        {line}")
    result: dict[str, Any] = {"actions": [a.as_dict() for a in actions], "applied": [], "dry_run": True}
    if not todo:
        out("Nothing to change.")
        result["dry_run"] = False
        return result
    if not yes:
        if interactive is None:
            interactive = sys.stdin.isatty()
        if interactive:
            answer = input("Apply these changes? [y/N] ").strip().lower()
            yes = answer in {"y", "yes"}
        if not yes:
            out("No changes made. Re-run with --yes to apply.")
            return result
    result["dry_run"] = False
    for a in todo:
        try:
            msg = a.apply()
            result["applied"].append({"target": a.target, "result": msg})
            out(f"  done: {msg}")
        except Exception as exc:
            result["applied"].append({"target": a.target, "error": str(exc)})
            out(f"  failed: {a.target}: {exc}")
    return result
