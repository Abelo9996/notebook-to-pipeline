from __future__ import annotations

from pathlib import Path

import pytest

from notebook_to_pipeline.analysis import analyze
from notebook_to_pipeline.notebook import load_cells, read_notebook, script_to_notebook
from notebook_to_pipeline.pipeline import run_pipeline

pytestmark = pytest.mark.filterwarnings("ignore")


def test_script_to_notebook_splits_on_percent_markers():
    nb = script_to_notebook(
        '"""Title\n\nA docstring.\n"""\n\n'
        "# %%\n"
        "# Load\nimport numpy as np\nx = np.array([1, 2, 3])\n"
        "# %% [markdown]\n# Some prose about the next step\n"
        "# %%\ntotal = int(x.sum())\n"
    )
    kinds = [c.cell_type for c in nb.cells]
    assert kinds[0] == "markdown"  # leading docstring
    assert "markdown" in kinds and kinds.count("code") == 2
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert "import numpy as np" in code[0]
    assert "total = int(x.sum())" in code[1]
    md = [c.source for c in nb.cells if c.cell_type == "markdown"]
    assert any("prose about the next step" in m for m in md)
    assert all(not m.lstrip().startswith("#") for m in md)  # leading "# " stripped


def test_script_without_markers_is_one_code_cell():
    nb = script_to_notebook("a = 1\nb = a + 1\n")
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 1 and "b = a + 1" in code[0].source


def test_read_notebook_accepts_py(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("# %%\nimport numpy as np\n# %%\nx = np.arange(3)\n")
    nb = read_notebook(p)
    cells = load_cells(nb)
    assert len(cells) == 2
    r = analyze(p)  # analyze takes the .py path directly
    assert r["notebook"]["code_cells"] == 2


def test_pipeline_runs_on_a_py_script(tmp_path):
    p = tmp_path / "script.py"
    p.write_text(
        "# %%\n"
        "import numpy as np\n"
        "data = np.array([2.0, 4.0, 6.0])\n"
        "# %%\n"
        "mean = float(data.mean())\n"
        "total = float(data.sum())\n"
    )
    out = tmp_path / "proj"
    result = run_pipeline(p, out=out, emit=lambda *_: None)
    assert result["verdict"] == "equivalent", result.get("verify")
    assert result["locked"] >= 3
    assert Path(result["report_md"]).exists()
