# AGENTS.md

Notes for coding agents and contributors working on this repository. The workflow for agents that
*use* the tool is in `skills/notebook-to-pipeline/SKILL.md`.

## Layout

```
src/notebook_to_pipeline/
  notebook.py      load .ipynb, run IPython's input transformer, record magics
  analysis.py      AST def/use graph, hidden-state findings, stage proposal, suggested artifacts
  execution.py     interpreter choice, fresh-kernel execution (nbclient), file change tracking
  capture.py       `capture`: reference run, saved-output comparison, runtime findings, --repeat
  verify.py        `verify`: run the pipeline, collect artifacts, compare, verdict
  report.py        report.md / report.json
  scaffold.py      starting layout for the refactored project
  setup_agents.py  `setup` for Claude Code, Codex, Cursor
  mcp_server.py    MCP tools (same operations as the CLI)
  cli.py           argparse CLI, entry points `nb2p` and `notebook-to-pipeline`
  runtime/         scripts that run inside the user's interpreter (stdlib only, Python 3.8+)
    nb2p_probe.py    serialize variables with hashes and summaries
    nb2p_compare.py  tolerant comparison of two captures
    nb2p_runner.py   import or run the pipeline, then call the probe
    nb2p_figures.py  record every matplotlib figure as a PNG (kernel and pipeline)
examples/          real notebooks, hand-written pipelines and committed evidence
skills/            agent skill installed by `setup`
```

## Rules

- Tests: `uv run pytest -q` (offline, under 2 minutes). Add a test with every behavior change.
- Lint: `uv run ruff check src tests`.
- No em dashes or en dashes anywhere. No invented numbers in docs: quote real runs.
- Keep the JSON output stable: agents parse it. Add fields rather than renaming them.
- MCP tool docstrings are the agent's documentation: say when to use the tool, what the arguments
  mean and what to call next. Results end with `next_steps`.
