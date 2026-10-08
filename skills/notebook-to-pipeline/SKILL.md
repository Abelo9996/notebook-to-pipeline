---
name: notebook-to-pipeline
description: Turn a Jupyter notebook into a tested, reproducible Python pipeline and prove the outputs did not change. Use when asked to refactor, productionize, modularize or "clean up" a .ipynb, convert a notebook to scripts or a package, or check whether a notebook actually runs top to bottom.
---

# notebook-to-pipeline

You write the refactor. The tool runs the notebook, records what it produces, runs your pipeline and
compares the two with exact checks and numeric tolerances. Your claim of "same results" must come from
its verdict, not from reading the code.

Tools are available as MCP tools (`analyze_notebook`, `capture_reference`, `verify_pipeline`,
`scaffold_pipeline`, `write_report`) or as the CLI `nb2p` (same names: `analyze`, `capture`, `verify`,
`scaffold`, `report`). Add `--json` to any CLI command for machine-readable output.

## Workflow

1. **Analyze first.** `nb2p analyze notebook.ipynb`
   - Read every `error` finding. `use_before_def` and `undefined_name` mean a clean top-to-bottom run
     will fail. `stale_output` and `out_of_order_execution` mean the saved outputs may not be what the
     code produces today.
   - Note `cross_cell_mutation` findings: an object changed in place in one cell and read in another.
     Your split must keep that order or make the change explicit.
   - Use `stages` as a starting point for modules, not as an instruction. Check `reorder_notes`.

2. **Capture before you change anything.** `nb2p capture notebook.ipynb`
   - This runs the notebook top to bottom in a fresh kernel and saves the reference outputs.
   - If it fails, that is a finding in itself. Report it to the user with the failing cell and error.
     Do not "fix" the notebook silently. If the user wants a fix, make the smallest change, say exactly
     what you changed, and capture again.
   - Pick the artifacts that matter with `--artifacts a,b,c` when the default list (top-level
     assignments) is too broad or misses something. Prefer final DataFrames, model metrics, fitted
     models and written files.
   - If the analysis reports unseeded randomness, run `--repeat 2`. Unstable artifacts cannot be
     verified; say so instead of loosening tolerances.
   - Use the project's interpreter: `--python .venv/bin/python` (a `.venv` next to the notebook is
     picked up automatically).

3. **Refactor in small steps.** Optionally start from `nb2p scaffold notebook.ipynb --out <dir>`,
   which pastes the notebook code into one function per stage, plus `pipeline.py:run()`, an
   equivalence test, a Makefile and a CI workflow. Then improve one stage at a time.
   - The pipeline entry point should return a dict of `{artifact name: value}` (`file.py:run`), or
     leave the artifacts as module globals (`file.py`).

4. **Verify after every step.**
   `nb2p verify --pipeline src/pkg/pipeline.py:run --reference <capture dir or notebook> --cwd <dir>`
   - Verdicts: `equivalent`, `differs` (first differences shown), `pipeline_failed`,
     `reference_invalid`, `inconclusive`.
   - On `differs`, read the first differences and fix the pipeline. Do not widen `--rtol`/`--atol`,
     turn on `--ignore-row-order`, `--ignore-index` or `--no-check-dtype`, or drop artifacts just to
     get a pass. If one of those is genuinely right (for example the old row order was arbitrary), tell
     the user which option you used and why.
   - If an artifact was renamed, map it: `--rename old_name=new_name`.

5. **Never change logic to make outputs match without saying so.** If the notebook has a bug and the
   user wants it fixed, the outputs will change on purpose. Capture the fixed notebook as a new
   reference and state the change and its effect in plain words.

6. **Report.** `nb2p report --reference <capture dir>` writes `report.md` and `report.json` with the
   notebook hash, versions, top-to-bottom result, hidden-state findings, per-artifact comparison and
   limits. Give the user the verdict and the path to the report. Quote numbers from it, not from memory.

## What to tell the user

- The verdict and how many artifacts and files were compared.
- Any hidden-state finding that changes how they should trust the original notebook.
- Anything you could not verify (unstable artifacts, figures, values not stored in variables).
