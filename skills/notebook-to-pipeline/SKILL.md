---
name: notebook-to-pipeline
description: Turn a Jupyter notebook into a tested, reproducible Python pipeline and prove the outputs did not change. Use when asked to refactor, productionize, modularize or "clean up" a .ipynb, convert a notebook to scripts or a package, or check whether a notebook actually runs top to bottom.
---

# notebook-to-pipeline

You write the refactor. The tool runs the notebook, records what it produces, runs your pipeline and
compares the two with exact checks and numeric tolerances. Your claim of "same results" must come from
its verdict, not from reading the code.

**How to call it.** Use the MCP tools when they are available: `analyze_notebook`,
`capture_reference`, `verify_pipeline`, `scaffold_pipeline`, `write_report`. Otherwise use the CLI
with the same steps (`analyze`, `capture`, `verify`, `scaffold`, `report`): `nb2p <command>` if it is
installed, else `uvx notebook-to-pipeline <command>`. Add `--json` for machine-readable output. Every
result ends with next steps; follow them.

## Workflow

1. **Analyze first** (`analyze_notebook`). Nothing runs.
   - Read every `error` finding. `use_before_def` and `undefined_name` mean a clean top-to-bottom run
     will fail. `stale_output` and `out_of_order_execution` mean the saved outputs may not be what the
     code produces today.
   - `cross_cell_mutation` and `shared_estimator`: an object changed in place in one cell and read in
     another, or one estimator inside two pipelines. Your split must keep that behavior or change it
     on purpose and say so.
   - Use `stages` as a starting point for modules, not as an instruction. Check `reorder_notes`.

2. **Capture before you change anything** (`capture_reference`).
   - Runs the notebook top to bottom in a fresh kernel and saves the reference: chosen variables,
     files written, every matplotlib figure (as PNG) and what it printed.
   - Use the project's interpreter: a `.venv` next to the notebook is picked up automatically,
     otherwise pass `python` (`--python`). If capture fails with `ModuleNotFoundError`, read `hint`.
   - If it fails for any other reason, that is a finding in itself. Tell the user the failing cell and
     error. Do not "fix" the notebook silently. If the user wants a fix, make the smallest change, say
     exactly what you changed, and capture again.
   - Choose artifacts (`artifacts=[...]`, `--artifacts a,b,c`) when the default (top-level
     assignments) is too broad. Prefer final DataFrames, model metrics, fitted models and written files.
   - If the analysis reports unseeded randomness, use `repeat=2`. Unstable artifacts cannot be
     verified; say so instead of loosening tolerances.

3. **Refactor in small steps.** Optionally start from `scaffold_pipeline`, which pastes the notebook
   code into one function per stage, plus `pipeline.py:run()`, an equivalence test, a Makefile and a
   CI workflow. Then improve one stage at a time.
   - The entry point returns a dict of `{artifact name: value}` (`src/pkg/pipeline.py:run`), or leaves
     the artifacts as module globals (`file.py`).
   - Keep the plots in the verified run (draw them with matplotlib; saving them is optional): figures
     are then compared pixel by pixel. If the pipeline draws no figures they are reported as not
     compared.
   - The generated `tests/test_equivalence.py` runs this tool through uv, pinned to the version that
     wrote it, so the project needs only pytest (not notebook-to-pipeline, not ipykernel). Run the
     `install_command` and `test_command` from the scaffold result, for example `uv sync` then
     `uv run pytest -q`, or `python -m pip install -r requirements-dev.txt` then
     `python -m pytest -q`. The test must pass on the draft before you change anything. If it is
     skipped because uv is missing, say so and use `verify_pipeline` instead.
   - Values a cell only displayed as its last expression (`df.describe()`, a score) are captured as
     `displayed_cell_<n>`, and the draft keeps them as variables with those names. Keep returning
     them; if you drop one, `verify` lists it as `not_returned` and you tell the user.

4. **Verify after every step** (`verify_pipeline`, `pipeline="src/pkg/pipeline.py:run"`,
   `reference=<reference_dir>`, `cwd=<project root>`).
   - Verdicts: `equivalent`, `differs` (first differences shown), `pipeline_failed`,
     `reference_invalid`, `inconclusive`.
   - On `differs`, read the first differences and fix the pipeline. Do not widen `rtol`/`atol`, turn on
     `ignore_row_order`, `ignore_index` or `check_dtype=false`, or drop artifacts just to get a pass. If
     one of those is genuinely right (for example the old row order was arbitrary), tell the user which
     option you used and why.
   - If an artifact was renamed, map it: `rename={"old_name": "new_name"}` (`--rename old=new`).
   - `printed` lists lines the notebook printed that the pipeline did not. It is not counted in the
     verdict, but a printed metric that matters should be returned as an artifact or printed the same way.

5. **Never change logic to make outputs match without saying so.** If the notebook has a bug and the
   user wants it fixed, the outputs will change on purpose. Capture the fixed notebook as a new
   reference and state the change and its effect in plain words.

6. **Report** (`write_report`). Writes `report.md` and `report.json` with the notebook hash, versions,
   top-to-bottom result, hidden-state findings, per-artifact and per-figure comparison and limits. Give
   the user the verdict and the path to the report. Quote numbers from it, not from memory.

## What to tell the user

- The verdict and how many artifacts, files and figures were compared.
- Any hidden-state finding that changes how they should trust the original notebook.
- Anything you could not verify (unstable artifacts, figures the pipeline does not draw, displayed
  values the pipeline does not return, values shown with `display()` or as rich output only).
