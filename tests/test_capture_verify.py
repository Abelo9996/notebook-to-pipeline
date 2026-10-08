from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from notebook_to_pipeline.capture import capture
from notebook_to_pipeline.report import build_report
from notebook_to_pipeline.verify import verify

CELLS = [
    "import numpy as np\nimport pandas as pd",
    "df = pd.DataFrame({'city': ['a', 'b', 'c', 'a'], 'temp': [10.5, 20.25, 30.0, 12.5]})",
    "df = df[df['temp'] > 11]\nmean_temp = float(df['temp'].mean())",
    "by_city = df.groupby('city')['temp'].sum()",
    "arr = np.linspace(0, 1, 5)\nconfig = {'n': 3, 'tags': ['x', 'y']}",
    "df.to_csv('summary.csv', index=False)",
]

PIPELINE_SAME = """
import numpy as np
import pandas as pd


def run():
    df = pd.DataFrame({"city": ["a", "b", "c", "a"], "temp": [10.5, 20.25, 30.0, 12.5]})
    df = df[df["temp"] > 11]
    df.to_csv("summary.csv", index=False)
    return {
        "df": df,
        "mean_temp": float(df["temp"].mean()),
        "by_city": df.groupby("city")["temp"].sum(),
        "arr": np.linspace(0, 1, 5),
        "config": {"n": 3, "tags": ["x", "y"]},
    }
"""


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    import os
    import sys

    os.environ.setdefault("NB2P_PYTHON", sys.executable)
    root = tmp_path_factory.mktemp("proj")
    from conftest import make_notebook

    nb = make_notebook(root / "analysis.ipynb", CELLS)
    result = capture(nb)
    return root, nb, result


def write_pipeline(root: Path, body: str, name: str = "pipeline.py") -> Path:
    p = root / name
    p.write_text(textwrap.dedent(body))
    return p


def test_capture_records_artifacts_and_files(captured):
    _root, _nb, result = captured
    assert result["execution"]["status"] == "ok"
    assert result["execution"]["cells_run"] == len(CELLS)
    arts = {a["name"]: a for a in result["artifacts"]}
    assert set(arts) == {"df", "mean_temp", "by_city", "arr", "config"}
    assert arts["df"]["kind"] == "dataframe" and arts["df"]["summary"]["shape"] == [3, 2]
    assert arts["mean_temp"]["summary"]["value"] == repr((20.25 + 30.0 + 12.5) / 3)
    assert arts["arr"]["kind"] == "ndarray"
    assert len(arts["df"]["hash"]) == 64
    files = {f["path"]: f for f in result["files_written"]}
    assert files["summary.csv"]["kind"] == "data" and files["summary.csv"]["copied"]
    ref = Path(result["reference_dir"])
    assert (ref / "capture.json").exists() and (ref / "analysis.json").exists()
    assert (ref / "files" / "summary.csv").exists()
    assert "pandas" in result["env"]["packages"]


def test_verify_equivalent(captured, tmp_path):
    root, nb, _ = captured
    write_pipeline(root, PIPELINE_SAME)
    r = verify("pipeline.py:run", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "equivalent", r
    assert all(a["status"] == "identical" for a in r["artifacts"])
    assert r["files"][0]["status"] == "identical"
    assert json.loads((tmp_path / "verify.json").read_text())["verdict"] == "equivalent"


def test_verify_close_within_tolerance(captured, tmp_path):
    root, nb, _ = captured
    body = PIPELINE_SAME.replace(
        '"mean_temp": float(df["temp"].mean()),',
        '"mean_temp": float(df["temp"].sum() / len(df)) + 1e-12,',
    )
    write_pipeline(root, body, "pipe_close.py")
    r = verify("pipe_close.py:run", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "equivalent"
    mt = next(a for a in r["artifacts"] if a["name"] == "mean_temp")
    assert mt["status"] == "close" and "within tolerance" in mt["detail"]


def test_verify_differs_with_first_difference(captured, tmp_path):
    root, nb, _ = captured
    body = PIPELINE_SAME.replace('df = df[df["temp"] > 11]', 'df = df[df["temp"] > 10]')
    write_pipeline(root, body, "pipe_bad.py")
    r = verify("pipe_bad.py:run", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "differs"
    by = {a["name"]: a for a in r["artifacts"]}
    assert by["df"]["status"] == "differs" and "row count differs" in by["df"]["detail"]
    assert by["arr"]["status"] == "identical"
    csv = r["files"][0]
    assert csv["status"] == "differs"


def test_verify_missing_artifact_and_rename(captured, tmp_path):
    root, nb, _ = captured
    body = PIPELINE_SAME.replace('"mean_temp":', '"avg_temp":')
    write_pipeline(root, body, "pipe_renamed.py")
    r = verify("pipe_renamed.py:run", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "differs"
    assert next(a for a in r["artifacts"] if a["name"] == "mean_temp")["status"] == "missing"
    r = verify("pipe_renamed.py:run", nb, cwd=root, out=tmp_path, rename={"mean_temp": "avg_temp"})
    assert r["verdict"] == "equivalent", r


def test_verify_script_mode_uses_globals(captured, tmp_path):
    root, nb, _ = captured
    script = PIPELINE_SAME + "\nglobals().update(run())\n"
    write_pipeline(root, script, "pipe_script.py")
    r = verify("pipe_script.py", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "equivalent", r


def test_verify_pipeline_failure(captured, tmp_path):
    root, nb, _ = captured
    write_pipeline(root, "def run():\n    raise ValueError('boom')\n", "pipe_fail.py")
    r = verify("pipe_fail.py:run", nb, cwd=root, out=tmp_path)
    assert r["verdict"] == "pipeline_failed"
    assert "ValueError" in r["reason"] and "boom" in r["pipeline"]["run"]["traceback"]


def test_verify_artifact_subset(captured, tmp_path):
    root, nb, _ = captured
    write_pipeline(
        root,
        "import numpy as np\ndef run():\n    return {'arr': np.linspace(0, 1, 5)}\n",
        "pipe_sub.py",
    )
    r = verify(
        "pipe_sub.py:run", nb, cwd=root, out=tmp_path, artifacts=["arr"], compare_files=False
    )
    assert r["verdict"] == "equivalent"
    assert [a["name"] for a in r["artifacts"]] == ["arr"]


def test_report_after_verify(captured, tmp_path):
    root, nb, _ = captured
    write_pipeline(root, PIPELINE_SAME)
    verify("pipeline.py:run", nb, cwd=root)
    rep = build_report(nb, out=tmp_path)
    md = (tmp_path / "report.md").read_text()
    assert rep["verdict"] == "equivalent"
    assert "EQUIVALENT" in md and "| `df` |" in md and "## Limits" in md
    assert json.loads((tmp_path / "report.json").read_text())["verdict"] == "equivalent"


def test_capture_detects_hidden_state_failure(tmp_path):
    from conftest import make_notebook

    nb = make_notebook(
        tmp_path / "broken.ipynb",
        [
            ("import pandas as pd", 1),
            ("clean = raw.dropna()", 7),
            ("raw = pd.DataFrame({'a': [1, None, 3]})", 2),
        ],
    )
    r = capture(nb)
    ex = r["execution"]
    assert ex["status"] == "failed"
    assert ex["failed_cell"]["index"] == 1 and ex["failed_cell"]["ename"] == "NameError"
    assert r["prediction"]["predicted"] is True
    v = verify("whatever.py:run", nb, cwd=tmp_path)
    assert v["verdict"] == "reference_invalid"


def test_repeat_flags_unstable_outputs(tmp_path):
    from conftest import make_notebook

    nb = make_notebook(
        tmp_path / "rand.ipynb",
        [
            "import numpy as np",
            "stable = np.arange(3)\nnoisy = np.random.default_rng().random(3)",
        ],
    )
    r = capture(nb, repeat=2)
    stab = {a["name"]: a["stable"] for a in r["artifacts"]}
    assert stab == {"stable": True, "noisy": False}


def test_saved_outputs_and_runtime_shared_objects(tmp_path):
    from conftest import make_notebook

    nb = make_notebook(
        tmp_path / "shared.ipynb",
        [
            ("print('saved')", 1),
            ("print('changed')", 2),
            (
                "from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import LinearRegression\nsc = StandardScaler()\np1 = make_pipeline(sc, LinearRegression())\np2 = make_pipeline(sc, LinearRegression())\nalias = p1",
                None,
            ),
        ],
    )
    r = capture(nb)
    ex = r["execution"]
    assert ex["saved_outputs"] == {"same": 1, "different": 1, "none": 1}
    changed = next(c for c in ex["cells"] if c["index"] == 1)
    assert changed["first_difference"] == {"line": 1, "saved": "saved", "fresh": "changed"}
    shared = [f["names"] for f in r["runtime_findings"] if f["kind"] == "shared_object"]
    aliases = [set(f["names"]) for f in r["runtime_findings"] if f["kind"] == "alias"]
    assert shared == [["sc", "p1[0]", "p2[0]"]]
    assert aliases == [{"p1", "alias"}]


def test_verify_shell_command_compares_files(captured, tmp_path):
    import sys

    root, nb, _ = captured
    write_pipeline(root, PIPELINE_SAME + "\nrun()\n", "pipe_cmd.py")
    r = verify(None, nb, cmd=f'"{sys.executable}" pipe_cmd.py', cwd=root, out=tmp_path)
    assert r["verdict"] == "equivalent", r
    assert r["files"][0]["status"] == "identical"
    assert all(a["status"] == "not_compared" for a in r["artifacts"])

    body = PIPELINE_SAME.replace(
        'df.to_csv("summary.csv", index=False)',
        'df.assign(temp=df["temp"] + 1e-12).to_csv("summary.csv", index=False)',
    )
    write_pipeline(root, body + "\nrun()\n", "pipe_cmd2.py")
    r = verify(None, nb, cmd=f'"{sys.executable}" pipe_cmd2.py', cwd=root, out=tmp_path)
    assert r["verdict"] == "equivalent", r
    assert r["files"][0]["status"] == "close"
