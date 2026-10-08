# notebook-to-pipeline report: iris_session.ipynb

**Verdict: REFERENCE INVALID: the notebook does not run top to bottom, so there is nothing trustworthy to compare against.**

Reason: the notebook does not run top to bottom (status failed, NameError in cell 4), so there are no reference outputs to compare against. Fix the notebook or the hidden state first.

Generated 2026-10-08T11:10:14+00:00 by notebook-to-pipeline 0.1.0.

## Notebook

- File: `iris_session.ipynb`
- sha256: `c60c06d3cf1addbd4beb96556acdd202e53a86e9a25f2bd2ae63dc1832c565ea`
- Cells: 7 total, 6 code

## Top-to-bottom run in a fresh kernel

- Command: `nb2p capture iris_session.ipynb --out evidence/reference`
- Python 3.12.13 (virtualenv found at ~/Downloads/notebook-to-pipeline/examples/.venv), ipykernel 7.4.0
- Packages: pandas 2.2.3, scikit-learn 1.9.1, numpy 2.1.2, ipykernel 7.4.0
- Result: **failed** at cell 4: `NameError: name 'features' is not defined`

  Failing cell starts with:

  ```python
  X = features[['petal length (cm)', 'petal width (cm)']]
  X.head()
  ```
- Static analysis predicted this failure before running anything: cell 4, In [8] reads `features`, which no cell in the notebook defines. It probably came from a deleted cell or from state outside the notebook. A top-to-bottom run will fail here with NameError.

### Saved outputs vs fresh run

Text outputs saved in the notebook were compared with the fresh run: 1 cell(s) same, 0 different, 1 with no saved text output. Differences can come from hidden state or from different library versions.

## Hidden-state findings

2 error(s), 4 warning(s), 0 info. Cell numbers count every cell from the top, markdown included; `In [n]` is the saved execution count.

| Severity | Kind | Finding |
|---|---|---|
| error | undefined_name | cell 4, In [8] reads `features`, which no cell in the notebook defines. It probably came from a deleted cell or from state outside the notebook. A top-to-bottom run will fail here with NameError. |
| error | use_before_def | cell 5, In [9] reads `X_scaled` but it is first defined later, in cell 6, In [5]. A top-to-bottom run will fail here with NameError. |
| warning | hidden_executions | Execution counts run up to 9 but only 6 executed cells are saved, so 3 executions are not visible in the notebook (re-runs or deleted cells). Kernel state from those runs may have fed the saved outputs. |
| warning | stale_output | The saved output of cell 5, In [9] used `X_scaled` from cell 6, In [5], but a top-to-bottom run takes it from nothing (NameError). Its saved output may not reproduce. |
| warning | out_of_order_execution | Saved execution counts are out of notebook order: cell 6, In [5] ran before cell 5, In [9]; cell 7, In [7] ran before cell 5, In [9]. The saved outputs reflect a different order than a top-to-bottom run. |
| warning | stale_output | The saved output of cell 6, In [5] used `X` from an execution that is no longer in the notebook (a deleted cell or an earlier run), but a top-to-bottom run takes it from cell 4, In [8]. Its saved output may not reproduce. |

## Pipeline verification

- Command: `nb2p verify --pipeline pipeline.py:run --reference evidence/reference --out evidence`
- Pipeline run: None in None s
- Tolerance: rtol=1e-07, atol=1e-10; ignore row order: False, ignore column order: False, ignore index: False, check dtype: True

## Proposed module split

| Stage | Cells | Inputs | Outputs |
|---|---|---|---|
| load | 2, 3, 4 | - | X, df |
| features | 6 | X | X_scaled |
| train | 5 | df | - |
| report | 7 | X_scaled | - |

## Limits

- Static analysis reads cell source only. It does not follow `exec`, `eval`, `%run`, imports of local modules or mutation through aliases (`b = a; b.append(1)`).
- Mutation through notebook-defined functions is tracked one level deep; mutation inside third-party code is only known for common method names (`fit`, `append`, `inplace=True`, ...).
- Only the variables listed in the capture are compared. Anything the notebook displayed but did not keep in a variable is not compared.
- Figure files are listed but not compared. Plots are not compared at all.
- Equivalence is checked on this data, in this environment. A different input file or library version can still change the results.
- Tolerances apply to floats only. Integers, strings, booleans, dates and hashes must match exactly.
