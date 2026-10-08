# notebook-to-pipeline

Your agent turns a messy Jupyter notebook into a tested, reproducible pipeline, and proves the outputs didn't change.

```
uvx notebook-to-pipeline analyze analysis.ipynb
uvx notebook-to-pipeline capture analysis.ipynb
uvx notebook-to-pipeline verify --pipeline src/analysis/pipeline.py:run --reference analysis.ipynb
```

> Not on PyPI yet. Until the first release, run it straight from GitHub by replacing `uvx notebook-to-pipeline` with
> `uvx --from git+https://github.com/Abelo9996/notebook-to-pipeline notebook-to-pipeline`. `setup` registers `uvx notebook-to-pipeline mcp`, so it works once the
> package is on PyPI.

`analyze` reads the notebook and finds hidden-state problems. `capture` runs it top to bottom in a
fresh kernel and records what it produces. `verify` runs the refactored pipeline and compares every
output: exact for integers, strings and hashes, with a tolerance for floats, and column by column
for DataFrames. The short command is `nb2p`.

## A real run

The "Importance of Feature Scaling" notebook from the scikit-learn 1.9.1 example gallery, captured
with `nb2p capture plot_scaling_importance.ipynb --out evidence/reference --repeat 2`
(output trimmed in the artifact list only):

```
Python 3.12.13: ~/Downloads/notebook-to-pipeline/examples/.venv/bin/python (virtualenv found at ~/Downloads/notebook-to-pipeline/examples/.venv)
Top-to-bottom run: OK, 7/7 code cells in 7.111 s
Captured 26 artifacts:
  X                            dataframe  a676a863527a
  y                            series     55c53e167556
  ...
  y_proba                      ndarray    260abb10bbbb
  y_proba_scaled               ndarray    71b105b5e066
[warning] shared_object: After the run, `pca`, `unscaled_clf[0]`, `scaled_clf[1]` are one and the same sklearn.decomposition._pca.PCA object. Fitting or changing it through one name changed it for all of them.
[info] identical_artifacts: `pca`, `scaled_pca` have identical content after the run. If the notebook treats them as different results, check for shared objects or a step that was meant to differ.
Determinism check (2 runs): every artifact reproduced
Reference: evidence/reference
```

The notebook builds its "unscaled" and "standardized" pipelines around the same `PCA` object, so
fitting the second one refits the PCA inside the first. A hand-written pipeline that keeps this
behavior verifies as EQUIVALENT on all 26 artifacts. Giving the unscaled pipeline its own PCA is a
deliberate change, and `verify` shows exactly what it touches (3 of the 26 rows shown):

```
pca                      estimator  FAIL differs    first difference at pca['fitted']['components_'][0,0]: reference 0.13443022714615663, candidate 0.001763429172014044, 46 differences in total
y_pred                   ndarray    FAIL differs    first difference at y_pred[2]: reference 0, candidate 1, 34 differences in total
y_pred_scaled            ndarray    PASS identical  hash match
Verdict: DIFFERS (4 of 26 compared outputs differ)
```

The unscaled test accuracy the example prints, 35.19%, becomes 74.07% with its own PCA; the
standardized pipeline stays at 96.30%. The mechanical first draft from `nb2p scaffold` also
verifies as EQUIVALENT on this notebook (26 of 26) and on the pandas-cookbook one (6 of 6).
All of this, plus a notebook broken by hidden state, is in [examples/](examples/README.md) with the
full reports.

## How it works

- **analyze** parses each cell with Python's `ast` after IPython's own input transformer (so
  `%magics` and `!shell` lines are understood), and builds a def/use graph across cells, including
  in-place mutation (`df.dropna(inplace=True)`, `model.fit(...)`, `x.append(...)`, item and
  attribute assignment) and mutation through functions defined in the notebook. Findings:
  use before definition, names that only a deleted cell defined, out-of-order and hidden
  executions from the saved execution counts, saved outputs computed from a different definition
  than a clean run would use, cross-cell mutation, estimators shared between scikit-learn
  pipelines, unseeded randomness, network and shell access. It proposes a split into
  load, clean, features, train, evaluate and report with each stage's inputs and outputs.
- **capture** starts a new Jupyter kernel (`nbclient` + `ipykernel`, over a Unix socket on macOS and Linux) on your
  project's interpreter, runs every cell in order and stops at the first error. It then saves the
  chosen variables with a content hash and a summary (shape, dtypes, column stats, fitted
  attributes), records files the notebook wrote, compares the saved text outputs with the fresh
  run, reports objects reachable under several names, and with `--repeat N` reruns to find
  outputs that change between runs.
- **verify** runs the pipeline in the same interpreter (`file.py:func` or `module:func` returning a
  dict, or a script whose globals hold the results), saves the same artifacts and compares them in
  that interpreter, so pandas, numpy and scikit-learn objects load with the versions that made them.
  Fitted estimators are compared by parameters and fitted attributes. Every failure shows the
  first differences with their path, for example `temperature.index[0]` or
  `pca['fitted']['components_'][0,0]`.
- **scaffold** writes `src/<package>/` with one module per proposed stage (the notebook code pasted
  into functions as a first draft), `pipeline.py:run()`, `tests/test_equivalence.py`, a Makefile,
  a `pyproject.toml` pinned to the captured versions and a GitHub Actions workflow.
- **report** writes `report.md` and `report.json`: notebook hash, interpreter and package versions,
  the top-to-bottom result, hidden-state findings, the per-artifact table, the verdict and the
  limits.

The interpreter is chosen in this order: `--python`, `$NB2P_PYTHON`, a `.venv` next to the
notebook or in a parent directory, then the one running `nb2p`. With no project venv,
`uvx --with pandas --with scikit-learn notebook-to-pipeline capture ...` works too.

Exit codes: `capture` 0 when the notebook runs, 3 when it fails. `verify` 0 equivalent, 1 differs,
3 pipeline failed, 4 reference invalid.

## Setup for agents

```
uvx notebook-to-pipeline setup          # shows what it would change
uvx notebook-to-pipeline setup --yes    # applies it
```

It detects Claude Code, Codex and Cursor and registers the MCP server (`uvx notebook-to-pipeline mcp`)
with each: `claude mcp add --scope user`, a `[mcp_servers.notebook-to-pipeline]` table in
`~/.codex/config.toml`, an entry in `~/.cursor/mcp.json`. It copies the agent skill to
`~/.claude/skills/` and `~/.codex/skills/`. Files are backed up before they are edited and a second
run changes nothing. `--project DIR` writes a project `.mcp.json` instead.

MCP tools: `analyze_notebook`, `capture_reference`, `verify_pipeline`, `scaffold_pipeline`,
`write_report`. The skill (`skills/notebook-to-pipeline/SKILL.md`) tells the agent to capture
before changing anything, verify after every step, and never change logic to make outputs match
without saying so.

## What it can't do

- It compares variables and files, not plots. Figure files are listed but not compared, and a value
  that was only displayed, never stored in a variable, is not compared.
- Static analysis does not follow `exec`, `eval`, `%run`, imports of local modules or aliases
  (`b = a; b.append(1)`). Shared objects of that kind are caught at runtime only if both names
  are captured.
- Equivalence is shown for this data in this environment. A different input file or library
  version can still change results.
- Outputs that change from run to run (unseeded randomness, timings) cannot be verified. `--repeat`
  finds them; it does not fix them.
- The stage proposal is a heuristic starting point and `scaffold` produces a mechanical draft. The
  refactor itself is the agent's job.
- Plain Python kernels only. No R or Julia notebooks, no Spark or remote kernels.

## Privacy and safety

Everything runs on your machine. The tool makes no network calls and calls no LLM; the agent you
already use does the refactoring. `capture` and `verify` execute the notebook and the pipeline with
your user's permissions, exactly as running them yourself would. References are stored as pickles,
so only verify against capture directories you created (see SECURITY.md). Evidence files replace
your home directory with `~`.

## License

MIT. The example notebooks keep their own licenses (CC BY-SA 4.0 and BSD 3-Clause), noted in each
example's `NOTICE.txt`.
