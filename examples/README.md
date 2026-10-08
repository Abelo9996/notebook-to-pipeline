# Examples

Three notebooks, each taken through the full flow: `analyze`, `capture`, a pipeline written by
hand, `verify`, `report`. Everything under `evidence/` was produced by `run_all.sh` on an Apple M4
MacBook (macOS, Python 3.12.13) with the versions pinned in `requirements.txt`. The pickled values
behind each capture are not committed (see `.gitignore`); run `./run_all.sh` to regenerate them.

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

- `evidence/verify.json`: EQUIVALENT, 6 of 6 artifacts identical by hash.
- `evidence/caught-mistake/verify.json`: `mistakes.py` swaps month-end bins for month-start bins
  (`"MS"`). Values stay the same but every label moves to the first of the month. Verdict DIFFERS,
  3 of 6 artifacts, first difference `temperature.index[0]`: `2012-01-31` vs `2012-01-01`.
- `evidence/scaffold-draft/verify.json`: the mechanical draft written by `nb2p scaffold` (notebook
  cells pasted into one function per stage) is also EQUIVALENT, 6 of 6.
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
  fitted estimators compared by parameters and fitted attributes).

Then, as a separate and declared change, `run_fixed()` gives the unscaled pipeline its own PCA:

- `evidence/intentional-fix/verify.json`: DIFFERS in exactly 4 of 26 artifacts: `pca`,
  `unscaled_clf`, `y_pred` (34 of 54 test predictions change) and `y_proba`. Everything on the
  standardized side is identical.
- `evidence/metrics.txt` and `evidence/intentional-fix/metrics.txt`: test accuracy of the unscaled
  pipeline is 0.3519 with the shared PCA and 0.7407 with its own PCA. The standardized pipeline
  scores 0.9630 in both. The notebook's own printout from the fresh run (35.19%, kept in
  `evidence/reference/capture.json` under cell 14) matches the shared-PCA number.
- `evidence/scaffold-draft/verify.json`: the `nb2p scaffold` draft is EQUIVALENT, 26 of 26.
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
