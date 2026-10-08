# Examples

Three notebooks, each taken through the full flow: `analyze`, `capture`, a pipeline written by
hand, `verify`, `report`. Everything under `evidence/` was produced by `run_all.sh` on an Apple M4
MacBook (macOS, Python 3.12.13) with the versions pinned in `requirements.txt`, using
notebook-to-pipeline 0.1.1. The pickled values behind each capture are not committed (see
`.gitignore`); the figures each capture recorded are, under `evidence/reference/figures/`. Run
`./run_all.sh` to regenerate everything.

```
cd examples
./run_all.sh            # creates examples/.venv from requirements.txt on first use
```

## pandas-cookbook-snowiest-month

Source: chapter 6 of Julia Evans' [pandas-cookbook](https://github.com/jvns/pandas-cookbook),
"String Operations: Which month was the snowiest", with its `weather_2012.csv`. CC BY-SA 4.0,
see `NOTICE.txt`. The notebook is unchanged apart from its file name.

What the tool found before running anything: the saved execution counts are out of order
(`In [4]` sits after `In [10]`) and run up to 21 with only 13 executed cells saved, so 8 executions
are not in the file. The fresh run completed (13 of 13 code cells). Of the 10 cells with saved text
output, 4 matched the fresh run and 6 differed; all 6 differences come from newer library versions
(matplotlib's `Axes` repr and pandas' `M` to `ME` frequency alias), not from the data.

The pipeline is in `snowiest/` (load, features, pipeline). It replaces the deprecated
`resample("M").apply(np.median)` with `resample("ME").median()`.

- `evidence/verify.json`: EQUIVALENT, 6 of 6 artifacts identical by hash. The notebook draws 5
  figures; this pipeline does not plot, so they are listed as not compared.
- `evidence/caught-mistake/verify.json`: `mistakes.py` swaps month-end bins for month-start bins
  (`"MS"`). Values stay the same but every label moves to the first of the month. Verdict DIFFERS,
  3 of 6 artifacts, first difference `temperature.index[0]`: `2012-01-31` vs `2012-01-01`.
- `evidence/scaffold-draft/verify.json`: the mechanical draft written by `nb2p scaffold` (notebook
  cells pasted into one function per stage) is also EQUIVALENT, 11 of 11: the 6 artifacts and the
  5 figures, pixel for pixel. The draft closes figures after each plotting cell the way Jupyter does;
  without that, pandas draws several cells' plots onto one figure: an earlier draft drew 3 figures
  instead of 5, and `verify` reported 3 of the 5 as different or missing.
- `evidence/report.md`: the full report.

## sklearn-feature-scaling

Source: "Importance of Feature Scaling" from the scikit-learn 1.9.1 example gallery
(`plot_scaling_importance.ipynb`), BSD 3-Clause, see `NOTICE.txt`. Uses the wine dataset bundled
with scikit-learn. The notebook is unchanged.

What the tool found:

- `analyze` (static): `pca` is a step of both `unscaled_clf` and `scaled_clf`. scikit-learn
  pipelines keep the object, not a copy.
- `capture` (runtime): after the run, `pca`, `unscaled_clf[0]` and `scaled_clf[1]` are the same
  PCA object, and `pca` has the same content hash as `scaled_pca` (the PCA fitted on standardized
  data). Fitting `scaled_clf` refit the PCA inside `unscaled_clf`, so the "unscaled" pipeline
  predicts with a PCA fitted on standardized data.

The pipeline in `scaling/` reproduces the notebook first, shared PCA included:

- `evidence/verify.json`: EQUIVALENT, 26 of 26 artifacts identical by hash (DataFrames, arrays,
  fitted estimators compared by parameters and fitted attributes). The 3 figures are not compared
  because this pipeline does not plot, and none of the 10 lines the notebook prints are printed by
  it (reported, not counted).

Then, as a separate and declared change, `run_fixed()` gives the unscaled pipeline its own PCA:

- `evidence/intentional-fix/verify.json`: DIFFERS in exactly 4 of 26 artifacts: `pca`,
  `unscaled_clf`, `y_pred` (34 of 54 test predictions change) and `y_proba`. Everything on the
  standardized side is identical.
- `evidence/metrics.txt` and `evidence/intentional-fix/metrics.txt`: test accuracy of the unscaled
  pipeline is 0.3519 with the shared PCA and 0.7407 with its own PCA. The standardized pipeline
  scores 0.9630 in both. The notebook's own printout from the fresh run (35.19%, kept in
  `evidence/reference/capture.json` under cell 14) matches the shared-PCA number.
- `evidence/scaffold-draft/verify.json`: the `nb2p scaffold` draft is EQUIVALENT, 29 of 29 (26
  artifacts and 3 figures), and prints all 10 lines the notebook printed.
- `evidence/report.md`: the full report.

## broken-hidden-state

`iris_session.ipynb` is made by `make_broken_notebook.py`, which replays a messy session in a real
kernel (cells run out of order, one cell re-run) and then deletes a helper cell, the way notebooks
end up in practice. The saved outputs and execution counts are real. MIT, like the rest of this
repository.

- `analyze` reports 2 errors before running anything: cell 4 reads `features`, which no cell
  defines (it lived in the deleted cell), and cell 5 reads `X_scaled`, which is defined further
  down in cell 6.
- `capture` fails at cell 4 with `NameError: name 'features' is not defined`, the failure the
  static analysis predicted.
- `verify` returns REFERENCE_INVALID (exit code 4): there is nothing trustworthy to compare a
  pipeline against until the notebook runs top to bottom.

## The scaffolded project, end to end

`scaffold_e2e.sh` checks that the project `nb2p scaffold` writes works as generated, in four fresh
directories outside this repository: the scikit-learn notebook in an empty folder (uv, following
the README quickstart), the pandas-cookbook notebook in an existing uv project (`uv init`,
`uv add`), and both notebooks in a plain `python -m venv` + pip project where nothing inside the
project can see a global uv. In each one it captures the notebook, scaffolds, runs the install
command the scaffold prints, runs the generated test, changes one pipeline output on purpose
(`random_state=42` to `7` in the scikit-learn split, the monthly median temperature to a mean in
the pandas one) and runs the test again, then copies the project without its virtualenv and runs
the `run:` steps of the generated workflow with `CI=true`.

The run with 0.1.2 built locally (`uv build && FIND_LINKS=dist ./scaffold_e2e.sh`, empty uv cache,
Apple M4 MacBook, uv 0.12.5) passed all 16 checks: install, generated test passes, generated test
fails after the change, workflow steps pass, for each of the four projects. After the change the
test failed with `Verdict: DIFFERS (18 of 29 compared outputs differ)` on the scikit-learn notebook
and `Verdict: DIFFERS (4 of 15 compared outputs differ)` on the pandas-cookbook one. The 15 there
are the 6 variables, 4 values cells only displayed (`weather_2012[:5]` and three more) and the 5
figures.

After the release, the same script with 0.1.2 from PyPI (`./scaffold_e2e.sh`, again with an empty
uv cache) passed the same 16 checks, and the repository's CI runs it on ubuntu-latest for every push.

Running the generated workflow on GitHub itself, with references captured on the M4 MacBook (two
generated projects pushed to temporary branches of this repository, since deleted):

- pandas-cookbook, pip project, `ubuntu-latest`: passed. All 5 figures matched pixel for pixel
  across the two machines.
- scikit-learn, uv project: the workflow ran as written (setup-uv, `uv sync`, the test fetched
  notebook-to-pipeline 0.1.2 from PyPI) and failed with `Verdict: DIFFERS (1 of 29 compared outputs
  differ)`, on `ubuntu-latest` and again on `macos-latest` (arm64). The one difference is
  `unscaled_clf`: the `LogisticRegressionCV` regularization path fitted on unscaled features
  differs around the fifth significant digit, for example `coefs_paths_[0,0,4,0,2]` is
  `-0.5303939995299989` in the reference and `-0.5303995773530218` on ubuntu. Predictions match,
  and the three figures matched pixel for pixel. Locally the result is the same with 1, 3 or 10
  BLAS threads, so it comes from the CPU, not from threading. A reference captured on the CI
  machine itself passes (that is what the CI job above does).
