# Changelog

## 0.2.0 (2026-10-23)

- `nb2p pipeline` runs the notebook twice by default (`--repeat 2`), so outputs that change between runs (unseeded randomness, solver nondeterminism) are flagged and left out instead of failing the check. `.py` scripts are accepted as input, split on `# %%` cells.

One command from a notebook to an output-locked pipeline, and the agent refactor the headline
always implied.

- `nb2p pipeline <notebook>` runs analyze, capture, scaffold, verify and report in one step and
  ends with a one-screen Reproducibility Report: an EQUIVALENT / NOT EQUIVALENT badge, how many
  outputs were locked, the findings ranked by category (hidden state, shared-object mutation,
  nondeterminism, unpinned deps), the generated package path and the exact test that locks the
  outputs. A notebook that does not run top to bottom stops after capture with NO REFERENCE and
  the failing cell. There is a matching `pipeline` MCP tool.
- `nb2p pipeline --refactor` drives your installed agent CLI (`claude -p` or `codex exec`, `--agent`
  to choose) to refactor the scaffold into proper stages, with the output-lock test as the
  guardrail: it verifies after every pass, feeds the differences back on a change, and never keeps a
  refactor that alters the outputs (it restores the last equivalent version). It prints the plan and
  the cost and spends nothing without `--yes`; `--max-rounds` bounds the passes. On the scikit-learn
  digits example, `claude`'s first pass changed an output and was rejected, and the second was
  equivalent, for $0.26.
- Hidden state without execution counts: a new `redefined_across_cells` finding flags a name defined
  from scratch in more than one cell and read across cells, a cell-order hazard that needs no saved
  execution counts, so cleared and agent-generated notebooks get a real finding instead of a bland
  one. Use-before-definition, names from deleted cells and cross-cell mutation already worked
  without counts.
- Stage proposal on non-linear notebooks: a name only ever bound inside a loop or branch and never
  read downstream (a loop leftover or scratch variable) is no longer suggested as an output to
  compare.
- A shareable card (SVG and PNG) summarising a run: the notebook, the verdict, how many outputs were
  locked, how many hidden-state issues were caught and a check-yours command. It is rendered with
  resvg-py, a pure wheel, so no system libraries are needed.
- Validated on three public notebooks the repository had not used (a pandas-cookbook analysis and
  two scikit-learn gallery examples), each with one `pipeline` command and two with `--refactor`;
  see the README.

## 0.1.3

### Fixed

- Require mcp 1.30 or newer. The old floor (`mcp>=1.2`) allowed releases that no longer import with current pydantic; the MCP server's tests pass on 1.30.0 and 2.3.0.

## 0.1.2

The equivalence test that `scaffold` writes now works in the user's project with nothing extra
installed. In 0.1.1 it imported `notebook_to_pipeline`, so the project needed notebook-to-pipeline
and ipykernel as dev dependencies; both real agent sessions tripped on that, and one skipped the test.

- `tests/test_equivalence.py` runs `nb2p verify` through uv (`uv tool run --from
  notebook-to-pipeline==<version that wrote it> nb2p`), with the project's interpreter passed as
  `--python`, so the pipeline still runs with the project's packages. The project needs only
  pytest. It finds uv on `PATH` or from the `uv` package on PyPI (`pip install uv`). Without uv it
  skips with a message saying how to fix it, except under `CI`, where it fails: a skipped
  equivalence test would look like a pass. `NB2P_COMMAND` runs nb2p some other way.
- `scaffold` detects whether the project uses uv or pip (`uv.lock`, requirements files,
  `pyproject.toml`, how `.venv` was made; `--env uv|pip` to choose) and writes the matching pieces:
  a `dev = ["pytest>=8"]` dependency group, or `requirements.txt` plus `requirements-dev.txt`
  (pytest and uv); a Makefile that calls the pinned `uvx notebook-to-pipeline==<version>`; and a
  workflow using `astral-sh/setup-uv` or `actions/setup-python`, with the Python version of the
  reference run. When it keeps an existing `pyproject.toml` it prints the `uv add` commands still
  needed. The result has `env`, `install_command` and `test_command`. It also writes a
  `.gitignore` for `.nb2p/`, `.nb2p-verify/` and `.venv/`.
- `examples/scaffold_e2e.sh` checks all of this end to end in fresh uv and pip projects with the
  example notebooks: install, generated test passes, a deliberate change to a pipeline output
  fails it, and the generated workflow's steps pass in a clean copy.
- Values a cell only displayed as its last expression (`df.describe()`, a score) are captured as
  `displayed_cell_<n>` when they are data (DataFrames, Series, arrays, scalars, containers, fitted
  models). Plot handles, values that are the same object as a variable, and unknown objects are
  left out. The scaffold draft keeps them (`displayed_cell_7 = df.describe()`) and returns them,
  so they are compared. A pipeline that does not return one gets `not_returned` for it, which is
  listed in the verdict's "not counted" part instead of failing.

## 0.1.1

Fixes from a fresh-user audit and a real Claude Code session driving the MCP server.

- Figures: every matplotlib figure the notebook draws (inline or saved) is re-rendered as a PNG at
  72 dpi during `capture`, and `verify` compares it pixel by pixel with the figures the pipeline
  draws, matched by content first and then in order. If the pipeline draws no figures they are
  reported as not compared and the verdict says so. Saved PNG files are compared by decoded pixels
  instead of bytes; SVG, PDF and JPEG files are compared by bytes. `--no-figures` turns this off.
- The capture kernel now uses Jupyter's default inline matplotlib backend instead of Agg, so each
  cell's figures are closed when the cell ends, as in Jupyter. Before, a pandas `.plot()` in one
  cell could draw onto the previous cell's figure during capture.
- `scaffold` closes figures after each plotting cell, so the draft draws the same figures as the
  notebook (on the pandas-cookbook example the draft drew 3 figures instead of 5 without it).
- `verify` and `report` write to a hidden `.nb2p-verify/` beside the reference when the reference
  is not under `.nb2p/`, instead of putting `verify.json` and `candidate/` next to it (in the audit
  session the agent had to delete them from `tests/`).
- Printed values: `verify` checks which lines the notebook printed also appear in the pipeline's
  output (for example a printed accuracy) and lists the missing ones. It is reported, not counted
  in the verdict.
- `capture` failing with `ModuleNotFoundError` now says how to fix it (`--python`, a project
  `.venv`, or `uvx --with <package>`), with the pip name for common imports such as `sklearn`.
- Every CLI command and MCP tool result ends with next steps.
- MCP: tool descriptions say when to use each tool, what the arguments mean and what to call next.
  `analyze_notebook` returns a compact result (findings, stages, suggested artifacts) instead of the
  full cell graph. `scaffold_pipeline` no longer needs `out`. The Python `verify_pipeline`
  documents its keyword arguments.
- `analyze` explains each proposed stage with the calls that put a cell there (for example
  "fits a model (`LogisticRegressionCV`)") instead of internal scores.
- `scaffold`: `--out` defaults to the notebook's directory, the Makefile points at the notebook
  with a relative path, and `ipykernel` moved from the runtime dependencies to the dev group.
- `setup --json` printed the text plan before the JSON; it now prints only JSON. Setup installs
  the Claude Code skill whenever the `claude` CLI is found and ends with what to do next.
- The skill tells agents to use the MCP tools first and `uvx notebook-to-pipeline` when `nb2p` is
  not installed (in the audit session the agent tried `nb2p` first and got "command not found").
- CI also checks `ruff format`.
- `SOURCE_DATE_EPOCH=0` is set for notebook and pipeline runs, so PDF and SVG files written by
  matplotlib carry a fixed date.

## 0.1.0

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
