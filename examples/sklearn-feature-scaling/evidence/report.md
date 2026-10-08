# notebook-to-pipeline report: plot_scaling_importance.ipynb

**Verdict: EQUIVALENT: the pipeline reproduces every compared notebook output.**

Reason: all 26 compared outputs match.

Generated 2026-10-08T11:10:10+00:00 by notebook-to-pipeline 0.1.0.

## Notebook

- File: `plot_scaling_importance.ipynb`
- sha256: `46f8e2b46e0d6895c18e026b170b3747313a0aaa8fb35b35fdfb2b513e87ba8f`
- Cells: 15 total, 7 code

## Top-to-bottom run in a fresh kernel

- Command: `nb2p capture plot_scaling_importance.ipynb --out evidence/reference --repeat 2`
- Python 3.12.13 (virtualenv found at ~/Downloads/notebook-to-pipeline/examples/.venv), ipykernel 7.4.0
- Packages: matplotlib 3.9.2, numpy 2.1.2, pandas 2.2.3, scikit-learn 1.9.1, ipykernel 7.4.0
- Result: **ran to completion**, 7 of 7 code cells in 7.111 s

## Hidden-state findings

0 error(s), 2 warning(s), 1 info. Cell numbers count every cell from the top, markdown included; `In [n]` is the saved execution count.

| Severity | Kind | Finding |
|---|---|---|
| warning | cross_cell_mutation | `scaler` (defined in cell 4) is changed in place in cell 6 (.fit_transform()) and later read by cell 12. That dependency is invisible in a def/use view: keep the order or make the change explicit. |
| warning | shared_estimator | `pca` is a step of `scaled_clf` and `unscaled_clf` at the same time. Pipelines keep the object itself, not a copy, so fitting one refits `pca` inside the other too. Use separate instances (or sklearn.base.clone) unless sharing is intended. |
| info | no_execution_history | The notebook has no saved execution counts (outputs were cleared or it was never run), so execution-order checks were skipped. |

Found during the fresh run (not visible in the source alone):

| Severity | Kind | Finding |
|---|---|---|
| warning | shared_object | After the run, `pca`, `unscaled_clf[0]`, `scaled_clf[1]` are one and the same sklearn.decomposition._pca.PCA object. Fitting or changing it through one name changed it for all of them. |
| info | identical_artifacts | `pca`, `scaled_pca` have identical content after the run. If the notebook treats them as different results, check for shared objects or a step that was meant to differ. |

## Reference artifacts

| Name | Kind | Summary | sha256 (first 12) |
|---|---|---|---|
| `X` | dataframe | 178 rows x 13 cols | `a676a863527a` |
| `y` | series | length 178, int64 | `55c53e167556` |
| `scaler` | estimator | StandardScaler, 6 fitted attrs | `8bd36618caed` |
| `X_train` | dataframe | 124 rows x 13 cols | `030b3e519f9b` |
| `X_test` | dataframe | 54 rows x 13 cols | `3cbaf12fa6d0` |
| `y_train` | series | length 124, int64 | `c735c0c2a9c1` |
| `y_test` | series | length 54, int64 | `a67b80f29e10` |
| `scaled_X_train` | dataframe | 124 rows x 13 cols | `e3cc23bf74ef` |
| `X_plot` | dataframe | 178 rows x 2 cols | `1746fc3b85fa` |
| `X_plot_scaled` | dataframe | 178 rows x 2 cols | `1de5351eeaac` |
| `clf` | estimator | KNeighborsClassifier, 7 fitted attrs | `87f305553b96` |
| `pca` | estimator | PCA, 10 fitted attrs | `326d417decaa` |
| `scaled_pca` | estimator | PCA, 10 fitted attrs | `326d417decaa` |
| `X_train_transformed` | ndarray | shape (124, 2), float64 | `d782b740b77a` |
| `X_train_std_transformed` | ndarray | shape (124, 2), float64 | `827753f6cc79` |
| `first_pca_component` | dataframe | 13 rows x 2 cols | `c0dbf0082902` |
| `target_classes` | container | range(0, 3) | `6d7a5f360991` |
| `colors` | container | ('blue', 'red', 'green') | `37a006c86b67` |
| `markers` | container | ('^', 's', 'o') | `9735e28fe7f8` |
| `Cs` | ndarray | shape (20,), float64 | `b5fb4179bc9a` |
| `unscaled_clf` | estimator | Pipeline, 0 fitted attrs | `0a1beb698d9c` |
| `scaled_clf` | estimator | Pipeline, 0 fitted attrs | `ccc4b8e99a54` |
| `y_pred` | ndarray | shape (54,), int64 | `98b7695012dd` |
| `y_pred_scaled` | ndarray | shape (54,), int64 | `6c9cb696ba0a` |
| `y_proba` | ndarray | shape (54, 3), float64 | `260abb10bbbb` |
| `y_proba_scaled` | ndarray | shape (54, 3), float64 | `71b105b5e066` |

Determinism check: the notebook was run 2 times. Every artifact was reproduced.

## Pipeline verification

- Command: `nb2p verify --pipeline scaling/pipeline.py:run --reference evidence/reference --out evidence`
- Pipeline run: ok in 3.164 s
- Packages: numpy 2.1.2, pandas 2.2.3, scikit-learn 1.9.1
- Tolerance: rtol=1e-07, atol=1e-10; ignore row order: False, ignore column order: False, ignore index: False, check dtype: True

| Artifact | Kind | Result | Detail |
|---|---|---|---|
| `X` | dataframe | PASS (identical) | hash match |
| `y` | series | PASS (identical) | hash match |
| `scaler` | estimator | PASS (identical) | hash match |
| `X_train` | dataframe | PASS (identical) | hash match |
| `X_test` | dataframe | PASS (identical) | hash match |
| `y_train` | series | PASS (identical) | hash match |
| `y_test` | series | PASS (identical) | hash match |
| `scaled_X_train` | dataframe | PASS (identical) | hash match |
| `X_plot` | dataframe | PASS (identical) | hash match |
| `X_plot_scaled` | dataframe | PASS (identical) | hash match |
| `clf` | estimator | PASS (identical) | hash match |
| `pca` | estimator | PASS (identical) | hash match |
| `scaled_pca` | estimator | PASS (identical) | hash match |
| `X_train_transformed` | ndarray | PASS (identical) | hash match |
| `X_train_std_transformed` | ndarray | PASS (identical) | hash match |
| `first_pca_component` | dataframe | PASS (identical) | hash match |
| `target_classes` | container | PASS (identical) | hash match |
| `colors` | container | PASS (identical) | hash match |
| `markers` | container | PASS (identical) | hash match |
| `Cs` | ndarray | PASS (identical) | hash match |
| `unscaled_clf` | estimator | PASS (identical) | hash match |
| `scaled_clf` | estimator | PASS (identical) | hash match |
| `y_pred` | ndarray | PASS (identical) | hash match |
| `y_pred_scaled` | ndarray | PASS (identical) | hash match |
| `y_proba` | ndarray | PASS (identical) | hash match |
| `y_proba_scaled` | ndarray | PASS (identical) | hash match |

## Proposed module split

| Stage | Cells | Inputs | Outputs |
|---|---|---|---|
| load | 4 | - | X, X_test, X_train, scaled_X_train, scaler, y, y_test, y_train |
| train | 6, 8, 12 | X, X_train, scaled_X_train, scaler, y, y_train | X_train_std_transformed, X_train_transformed, scaled_clf, unscaled_clf |
| evaluate | 14 | X_test, scaled_clf, unscaled_clf, y_test | - |
| report | 10 | X_train_std_transformed, X_train_transformed, y_train | - |

## Limits

- Static analysis reads cell source only. It does not follow `exec`, `eval`, `%run`, imports of local modules or mutation through aliases (`b = a; b.append(1)`).
- Mutation through notebook-defined functions is tracked one level deep; mutation inside third-party code is only known for common method names (`fit`, `append`, `inplace=True`, ...).
- Only the variables listed in the capture are compared. Anything the notebook displayed but did not keep in a variable is not compared.
- Figure files are listed but not compared. Plots are not compared at all.
- Equivalence is checked on this data, in this environment. A different input file or library version can still change the results.
- Tolerances apply to floats only. Integers, strings, booleans, dates and hashes must match exactly.
