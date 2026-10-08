# notebook-to-pipeline report: snowiest_month.ipynb

**Verdict: EQUIVALENT: the pipeline reproduces every compared notebook output.**

Reason: all 6 compared outputs match; not counted: 5 not compared.

Generated 2026-10-08T19:44:55+00:00 by notebook-to-pipeline 0.1.1.

## Notebook

- File: `snowiest_month.ipynb`
- sha256: `fd88c88038973a6d07880113d8bd991f3d263357607f85200f5fb019ce003355`
- Cells: 29 total, 14 code

## Top-to-bottom run in a fresh kernel

- Command: `nb2p capture snowiest_month.ipynb --out evidence/reference --repeat 2`
- Python 3.12.13 (virtualenv found at ~/Downloads/notebook-to-pipeline/examples/.venv), ipykernel 7.4.0
- Packages: matplotlib 3.9.2, numpy 2.1.2, pandas 2.2.3, ipykernel 7.4.0
- Result: **ran to completion**, 13 of 13 code cells in 28.795 s

### Saved outputs vs fresh run

Text outputs saved in the notebook were compared with the fresh run: 4 cell(s) same, 6 different, 3 with no saved text output. Differences can come from hidden state or from different library versions.

| Cell | First differing line (saved) | Fresh run |
|---|---|---|
| cell 9 | `<matplotlib.axes._subplots.AxesSubplot at 0x...>` | `<Axes: xlabel='Date/Time'>` |
| cell 12 | `<matplotlib.axes._subplots.AxesSubplot at 0x...>` | `<Axes: xlabel='Date/Time'>` |
| cell 17 | `Freq: M, Name: Weather, dtype: float64` | `Freq: ME, Name: Weather, dtype: float64` |
| cell 18 | `<matplotlib.axes._subplots.AxesSubplot at 0x...>` | `<Axes: xlabel='Date/Time'>` |
| cell 25 | `<matplotlib.axes._subplots.AxesSubplot at 0x...>` | `<Axes: xlabel='Date/Time'>` |
| cell 27 | `array([<matplotlib.axes._subplots.AxesSubplot object at 0x...>,` | `array([<Axes: title={'center': 'Temperature'}, xlabel='Date/Time'>,` |

## Hidden-state findings

0 error(s), 3 warning(s), 8 info. Cell numbers count every cell from the top, markdown included; `In [n]` is the saved execution count.

| Severity | Kind | Finding |
|---|---|---|
| warning | hidden_executions | Execution counts run up to 21 but only 13 executed cells are saved, so 8 executions are not visible in the notebook (re-runs or deleted cells). Kernel state from those runs may have fed the saved outputs. |
| warning | out_of_order_execution | Saved execution counts are out of notebook order: cell 8, In [4] ran before cell 6, In [10]. The saved outputs reflect a different order than a top-to-bottom run. |
| warning | stale_output | The saved output of cell 8, In [4] used `is_snowing` from an execution that is no longer in the notebook (a deleted cell or an earlier run), but a top-to-bottom run takes it from cell 6, In [10]. Its saved output may not reproduce. |
| info | inspection_only | cell 8, In [4] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | rerun_hazard | cell 9, In [13] rebinds `is_snowing` from its own previous value. Running it twice gives a different state than running it once. |
| info | inspection_only | cell 12, In [14] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | inspection_only | cell 15, In [15] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | inspection_only | cell 17, In [16] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | inspection_only | cell 18, In [17] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | inspection_only | cell 25, In [20] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |
| info | inspection_only | cell 27, In [21] only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline. |

## Reference artifacts

| Name | Kind | Summary | sha256 (first 12) |
|---|---|---|---|
| `weather_2012` | dataframe | 8784 rows x 7 cols | `0d5f3e5dab83` |
| `weather_description` | series | length 8784, object | `8b136002db54` |
| `is_snowing` | series | length 8784, bool | `a9d56d0d548e` |
| `temperature` | series | length 12, float64 | `cb5e4c495f8d` |
| `snowiness` | series | length 12, float64 | `b8e9a4be8725` |
| `stats` | dataframe | 12 rows x 2 cols | `9cc2bed5d3ce` |

Figures recorded (matplotlib, rendered as PNG at 72 dpi): 1; 2; 3; 4; 5 (Temperature; Snowiness).

Determinism check: the notebook was run 2 times. Every artifact was reproduced.

## Pipeline verification

- Command: `nb2p verify --pipeline snowiest/pipeline.py:run --reference evidence/reference --out evidence`
- Pipeline run: ok in 9.978 s
- Packages: matplotlib 3.9.2, numpy 2.1.2, pandas 2.2.3
- Tolerance: rtol=1e-07, atol=1e-10; ignore row order: False, ignore column order: False, ignore index: False, check dtype: True

| Artifact | Kind | Result | Detail |
|---|---|---|---|
| `weather_2012` | dataframe | PASS (identical) | hash match |
| `weather_description` | series | PASS (identical) | hash match |
| `is_snowing` | series | PASS (identical) | hash match |
| `temperature` | series | PASS (identical) | hash match |
| `snowiness` | series | PASS (identical) | hash match |
| `stats` | dataframe | PASS (identical) | hash match |

| Figure | Result | Detail |
|---|---|---|
| figure 1 | n/a (not_compared) | the pipeline drew no matplotlib figures, so figures are not compared |
| figure 2 | n/a (not_compared) | the pipeline drew no matplotlib figures, so figures are not compared |
| figure 3 | n/a (not_compared) | the pipeline drew no matplotlib figures, so figures are not compared |
| figure 4 | n/a (not_compared) | the pipeline drew no matplotlib figures, so figures are not compared |
| figure 5 (Temperature; Snowiness) | n/a (not_compared) | the pipeline drew no matplotlib figures, so figures are not compared |

## Proposed module split

| Stage | Cells | Inputs | Outputs |
|---|---|---|---|
| load | 1, 3 | - | weather_2012 |
| clean | 6, 9 | weather_2012 | is_snowing |
| features | 22, 24 | weather_2012 | stats |
| report | 8, 12, 15, 17, 18, 25, 27 | is_snowing, stats, weather_2012 | - |

## Limits

- Static analysis reads cell source only. It does not follow `exec`, `eval`, `%run`, imports of local modules or mutation through aliases (`b = a; b.append(1)`).
- Mutation through notebook-defined functions is tracked one level deep; mutation inside third-party code is only known for common method names (`fit`, `append`, `inplace=True`, ...).
- Only the variables listed in the capture are compared. Values the notebook only printed are checked line by line against the pipeline's output, but that check is not counted in the verdict; anything displayed but neither printed nor kept in a variable (a DataFrame shown as a cell's last line) is not compared.
- matplotlib figures are re-rendered as PNG at 72 dpi and compared pixel by pixel, only if the pipeline draws figures too. Plotly, Bokeh and Altair charts are not compared. Saved figure files are compared pixel by pixel for PNG; SVG, PDF and JPEG files only by bytes.
- Equivalence is checked on this data, in this environment. A different input file or library version can still change the results.
- Tolerances apply to floats only. Integers, strings, booleans, dates and hashes must match exactly.
