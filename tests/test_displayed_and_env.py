"""Values only displayed as a cell's last expression, project detection, and the generated test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import make_notebook

from notebook_to_pipeline.capture import capture
from notebook_to_pipeline.scaffold import _assign_last_expression, detect_env, scaffold
from notebook_to_pipeline.verify import verify

CELLS = [
    "import pandas as pd\nfrom sklearn.linear_model import LinearRegression",
    "df = pd.DataFrame({'g': ['a', 'b', 'a', 'c'], 'v': [1.0, 2.0, 3.0, 4.0]})",
    "df.groupby('g')['v'].sum().to_frame()",
    "model = LinearRegression()\nmodel.fit(df[['v']], df['v'] * 2)",
    "df['v'].plot()\n",
    "import matplotlib.pyplot as plt\nplt.plot([1, 2])",
    "df.plot(subplots=True)",
    "total = df['v'].sum()\nprint(total)\ntotal * 10",
]


def _write_pipeline(root: Path, body: str) -> None:
    (root / "handmade.py").write_text(
        "import pandas as pd\n"
        "from sklearn.linear_model import LinearRegression\n\n\n"
        "def run():\n"
        "    df = pd.DataFrame({'g': ['a', 'b', 'a', 'c'], 'v': [1.0, 2.0, 3.0, 4.0]})\n"
        "    model = LinearRegression().fit(df[['v']], df['v'] * 2)\n"
        "    total = df['v'].sum()\n"
        f"{body}"
    )


def test_assign_last_expression():
    assert _assign_last_expression("a = 1\na + 1\n", "x") == "a = 1\nx = a + 1\n"
    assert _assign_last_expression("a = 1; (a\n  + 1)\n", "x") == "a = 1; x = (a\n  + 1)\n"
    assert _assign_last_expression("a = 1\n", "x") is None


def test_displayed_values_are_captured_and_compared(tmp_path):
    nb = make_notebook(tmp_path / "shown.ipynb", CELLS)
    cap = capture(nb)
    assert cap["execution"]["status"] == "ok"
    shown = {a["name"]: a for a in cap["artifacts"] if a["name"].startswith("displayed_cell_")}
    # cell 3 shows a new DataFrame, cell 8 a scalar. Cell 4 echoes `model` (already a variable),
    # cells 5 to 7 show an Axes, a list of lines and an array of Axes: none becomes an artifact.
    assert sorted(shown) == ["displayed_cell_3", "displayed_cell_8"], shown
    assert shown["displayed_cell_3"]["kind"] == "dataframe"
    assert shown["displayed_cell_3"]["displayed_in"] == "cell 3"
    ref = tmp_path / ".nb2p" / "shown" / "reference"

    # A pipeline that does not return the displayed values: listed as not compared, not failed.
    _write_pipeline(tmp_path, "    return {'df': df, 'model': model, 'total': total}\n")
    v = verify("handmade.py:run", ref, cwd=tmp_path, out=tmp_path / "v1")
    assert v["verdict"] == "equivalent", v["reason"]
    nc = {a["name"]: a for a in v["artifacts"] if a["status"] == "not_returned"}
    assert set(nc) == {"displayed_cell_3", "displayed_cell_8"}
    assert "2 not returned" in v["reason"]
    assert any("only displayed" in s for s in v["next_steps"])

    # Returning them compares them, and a changed displayed DataFrame is caught.
    _write_pipeline(
        tmp_path,
        "    shown = df.groupby('g')['v'].mean().to_frame()\n"
        "    return {'df': df, 'model': model, 'total': total,\n"
        "            'displayed_cell_3': shown, 'displayed_cell_8': total * 10}\n",
    )
    v = verify("handmade.py:run", ref, cwd=tmp_path, out=tmp_path / "v2")
    assert v["verdict"] == "differs"
    bad = [a["name"] for a in v["artifacts"] if a.get("passed") is False]
    assert bad == ["displayed_cell_3"]

    # The scaffold draft keeps them (`displayed_cell_3 = df.groupby(...)`) and verifies.
    res = scaffold(nb, tmp_path / "proj", package="shown", env="uv")
    assert res["displayed_values"] == ["displayed_cell_3", "displayed_cell_8"]
    stage_src = "".join(p.read_text() for p in (tmp_path / "proj/src/shown").glob("*.py"))
    assert "displayed_cell_3 = df.groupby('g')" in stage_src
    v = verify(
        "src/shown/pipeline.py:run",
        tmp_path / "proj/tests/reference",
        cwd=tmp_path / "proj",
        out=tmp_path / "v3",
    )
    assert v["verdict"] == "equivalent", v["reason"]
    assert "not returned" not in v["reason"]


def test_detect_env(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "uv.lock").write_text("")
    assert detect_env(tmp_path / "a")[0] == "uv"
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "requirements.txt").write_text("pandas\n")
    assert detect_env(tmp_path / "b")[0] == "pip"
    (tmp_path / "c" / ".venv").mkdir(parents=True)
    (tmp_path / "c" / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.12.1\n")
    assert detect_env(tmp_path / "c") == ("pip", ".venv/ was created without uv")
    (tmp_path / "c" / ".venv" / "pyvenv.cfg").write_text("home = /x\nuv = 0.9.0\n")
    assert detect_env(tmp_path / "c")[0] == "uv"


def _sales_notebook(tmp_path: Path) -> Path:
    return make_notebook(
        tmp_path / "sales.ipynb",
        [
            "import pandas as pd",
            "raw = pd.DataFrame({'region': ['n', 's', 'n'], 'units': [3, 5, 2]})",
            "by_region = raw.groupby('region')['units'].sum()",
            "by_region.sort_values()",
        ],
    )


def test_scaffold_files_for_uv_and_pip(tmp_path):
    nb = _sales_notebook(tmp_path)
    capture(nb)
    uv_root = tmp_path / "uvproj"
    uv_root.mkdir()
    (uv_root / "uv.lock").write_text("")
    res = scaffold(nb, uv_root, package="sales")
    assert res["env"]["kind"] == "uv"
    assert res["test_command"] == "uv run pytest -q"
    pyproject = (uv_root / "pyproject.toml").read_text()
    assert 'dev = ["pytest>=8"]' in pyproject
    assert "notebook-to-pipeline" not in pyproject and "ipykernel" not in pyproject
    ci = (uv_root / ".github/workflows/equivalence.yml").read_text()
    assert "uv sync" in ci and "uv run pytest -q" in ci
    assert not (uv_root / "requirements-dev.txt").exists()

    pip_root = tmp_path / "pipproj"
    res = scaffold(nb, pip_root, package="sales", env="pip")
    assert res["install_command"] == "python -m pip install -r requirements-dev.txt"
    dev = (pip_root / "requirements-dev.txt").read_text().split()
    assert dev[:2] == ["-r", "requirements.txt"] and "pytest>=8" in dev and "uv" in dev
    assert "pandas==" in (pip_root / "requirements.txt").read_text()
    assert "[dependency-groups]" not in (pip_root / "pyproject.toml").read_text()
    ci = (pip_root / ".github/workflows/equivalence.yml").read_text()
    assert "actions/setup-python@v5" in ci and "pip install -r requirements-dev.txt" in ci
    test_src = (pip_root / "tests/test_equivalence.py").read_text()
    assert "import notebook_to_pipeline" not in test_src
    assert "from notebook_to_pipeline" not in test_src


def _run_generated_test(root: Path, **env_changes: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("NB2P_PYTHON", None)
    env.pop("CI", None)
    for k, v in env_changes.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )


def test_generated_test_passes_and_catches_a_change(tmp_path):
    nb = _sales_notebook(tmp_path)
    capture(nb)
    root = tmp_path / "proj"
    scaffold(nb, root, package="sales", env="pip")
    # Offline stand-in for `uv tool run --from notebook-to-pipeline==X nb2p`: the same CLI.
    nb2p = f"{sys.executable} -m notebook_to_pipeline"
    ok = _run_generated_test(root, NB2P_COMMAND=nb2p)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "1 passed" in ok.stdout

    stage = next(p for p in (root / "src/sales").glob("*.py") if "groupby" in p.read_text())
    stage.write_text(stage.read_text().replace(".sum()", ".max()"))
    bad = _run_generated_test(root, NB2P_COMMAND=nb2p)
    assert bad.returncode == 1, bad.stdout
    assert "FAIL differs" in bad.stdout and "by_region" in bad.stdout


def test_generated_test_without_uv_skips_locally_and_fails_in_ci(tmp_path):
    nb = _sales_notebook(tmp_path)
    capture(nb)
    root = tmp_path / "proj"
    scaffold(nb, root, package="sales", env="pip")
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    local = _run_generated_test(root, NB2P_COMMAND=None, PATH=str(empty))
    assert local.returncode == 0 and "1 skipped" in local.stdout, local.stdout
    assert "uv was not found" in local.stdout
    ci = _run_generated_test(root, NB2P_COMMAND=None, PATH=str(empty), CI="true")
    assert ci.returncode == 1 and "uv was not found" in ci.stdout, ci.stdout
