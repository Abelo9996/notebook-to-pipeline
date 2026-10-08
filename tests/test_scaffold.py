from __future__ import annotations

import ast

from conftest import make_notebook

from notebook_to_pipeline.capture import capture
from notebook_to_pipeline.scaffold import _indent_code, scaffold
from notebook_to_pipeline.verify import verify


def test_indent_keeps_multiline_strings():
    code = 'x = """a\nb"""\ny = 1\n'
    out = _indent_code(code)
    assert out == '    x = """a\nb"""\n    y = 1\n'


def test_scaffold_generates_a_pipeline_that_verifies(tmp_path):
    nb = make_notebook(
        tmp_path / "Sales Analysis.ipynb",
        [
            "%precision 3\nimport numpy as np\nimport pandas as pd",
            "raw = pd.DataFrame({'region': ['n', 's', 'n', 'e'], 'units': [3, 5, 2, 7], 'price': [1.5, 2.0, 1.5, 3.0]})",
            "raw.head()",
            "clean = raw.dropna().copy()\nclean['revenue'] = clean['units'] * clean['price']",
            "by_region = np.round(clean.groupby('region')['revenue'].sum(), 1)",
            "top = by_region.idxmax()\nprint(top)",
        ],
    )
    cap = capture(nb)
    assert cap["execution"]["status"] == "ok"
    res = scaffold(nb, tmp_path, package="sales")
    assert res["package"] == "sales"
    assert res["reference_copied"]
    for rel in (
        "src/sales/pipeline.py",
        "tests/test_equivalence.py",
        "Makefile",
        "pyproject.toml",
        ".github/workflows/equivalence.yml",
        "tests/reference/capture.json",
    ):
        assert (tmp_path / rel).exists(), rel
    for py in (tmp_path / "src" / "sales").glob("*.py"):
        ast.parse(py.read_text())
    pyproject = (tmp_path / "pyproject.toml").read_text()
    assert '"pandas==' in pyproject
    v = verify(
        "src/sales/pipeline.py:run",
        tmp_path / "tests" / "reference",
        cwd=tmp_path,
        out=tmp_path / "v",
    )
    assert v["verdict"] == "equivalent", v

    again = scaffold(nb, tmp_path, package="sales")
    assert again["created"] == [] and "src/sales/pipeline.py" in again["skipped_existing"]
