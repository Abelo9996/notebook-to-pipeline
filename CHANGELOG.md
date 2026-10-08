# Changelog

## 0.1.0 (unreleased)

First version.

- `analyze`: def/use graph from the notebook source (Python AST after IPython's own input
  transforms), hidden-state findings (out-of-order and hidden executions, use before definition,
  names from deleted cells, stale saved outputs, cross-cell mutation, estimators shared between
  pipelines, unseeded randomness, network and shell access) and a proposed split into
  load/clean/features/train/evaluate/report.
- `capture`: runs the notebook top to bottom in a fresh kernel, records chosen variables with
  content hashes and summaries plus files written, compares saved text outputs with the fresh run,
  reports shared objects found at runtime, and optionally repeats the run to find unstable outputs.
- `verify`: runs a pipeline (`file.py:func`, `module:func`, a script or a module, or a shell command)
  and compares every artifact with tolerances for floats and exact checks for everything else.
- `scaffold`: package with one module per stage, `pipeline.py`, an equivalence test, Makefile,
  `pyproject.toml` with pinned versions and a GitHub Actions workflow.
- `report`: Markdown and JSON evidence.
- `setup`: registers the MCP server with Claude Code, Codex and Cursor and installs the skill.
- MCP server over stdio with the same five operations.
