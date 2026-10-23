from __future__ import annotations

from pathlib import Path

import pytest

from notebook_to_pipeline import refactor as R
from notebook_to_pipeline.capture import capture
from notebook_to_pipeline.scaffold import scaffold
from notebook_to_pipeline.verify import verify

pytestmark = pytest.mark.filterwarnings("ignore")


def _scaffolded(nb_factory):
    nb = nb_factory(
        [
            "import pandas as pd",
            "raw = pd.DataFrame({'region': ['n', 's', 'n'], 'units': [3, 5, 2]})",
            "by_region = raw.groupby('region')['units'].sum()",
            "total = int(by_region.sum())",
            "by_region.sort_values()",
        ]
    )
    cap = capture(nb)
    ref_dir = Path(cap["reference_dir"])
    out = Path(nb).parent / "proj"
    scaffold(nb, out, package="sales", reference=ref_dir)
    spec = "src/sales/pipeline.py:run"
    ver = verify(spec, ref_dir, cwd=out, out=out / "evidence")
    assert ver["verdict"] == "equivalent", ver.get("reason")
    return nb, out, ref_dir, spec, ver


def _stage_with(pkg_src: Path, needle: str) -> Path | None:
    for p in pkg_src.glob("*.py"):
        if needle in p.read_text() and "groupby" in p.read_text():
            return p
    return None


def _edit_break(pkg_src: Path) -> None:
    p = _stage_with(pkg_src, ".sum()")
    if p is not None:  # already broken on an earlier pass: leave it broken
        p.write_text(p.read_text().replace(".sum()", ".max()"))


def _edit_fix(pkg_src: Path) -> None:
    p = _stage_with(pkg_src, ".max()")
    if p is not None:
        p.write_text(p.read_text().replace(".max()", ".sum()"))


def _edit_noop(pkg_src: Path) -> None:
    p = pkg_src / "pipeline.py"
    p.write_text(p.read_text() + "\n# refactored by the agent\n")


def _fake_agent(edits):
    state = {"n": 0}

    def run(agent, prompt, cwd, model):
        edit = edits[min(state["n"], len(edits) - 1)]
        edit(Path(cwd) / "src" / "sales")
        state["n"] += 1
        return {
            "returncode": 0,
            "cost_usd": 0.02,
            "num_turns": 1,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    return run


@pytest.fixture(autouse=True)
def _agent_present(monkeypatch):
    monkeypatch.setattr(R, "_agent_available", lambda agent: "/usr/bin/fake-" + agent)


def test_refactor_needs_yes(nb_factory, monkeypatch):
    nb, out, ref_dir, spec, ver = _scaffolded(nb_factory)
    monkeypatch.setattr(R, "_run_agent", _fake_agent([_edit_break]))
    res = R.refactor_pipeline(
        Path(nb), out, ref_dir, spec, scaffold_verify=ver, yes=False, emit=lambda *_: None
    )
    assert res["ran"] is False and res["applied"] is False
    assert res["calls"] == 0
    # The agent never ran, so nothing was edited.
    assert ".max()" not in (out / "src/sales").joinpath("features.py").read_text()


def test_refactor_keeps_equivalent(nb_factory, monkeypatch):
    nb, out, ref_dir, spec, ver = _scaffolded(nb_factory)
    monkeypatch.setattr(R, "_run_agent", _fake_agent([_edit_noop]))
    res = R.refactor_pipeline(
        Path(nb), out, ref_dir, spec, scaffold_verify=ver, yes=True, emit=lambda *_: None
    )
    assert res["ran"] is True and res["applied"] is True
    assert res["verify"]["verdict"] == "equivalent"
    assert res["calls"] == 1 and res["total_cost_usd"] > 0
    assert "# refactored by the agent" in (out / "src/sales/pipeline.py").read_text()


def test_refactor_rejects_and_reverts_a_breaking_change(nb_factory, monkeypatch):
    nb, out, ref_dir, spec, ver = _scaffolded(nb_factory)
    monkeypatch.setattr(R, "_run_agent", _fake_agent([_edit_break]))
    res = R.refactor_pipeline(
        Path(nb),
        out,
        ref_dir,
        spec,
        scaffold_verify=ver,
        yes=True,
        max_rounds=3,
        emit=lambda *_: None,
    )
    assert res["ran"] is True and res["applied"] is False
    assert res["calls"] == 3  # tried every pass, none equivalent
    assert res["verify"]["verdict"] == "equivalent"  # the restored scaffold
    src_text = "\n".join(p.read_text() for p in (out / "src/sales").glob("*.py"))
    assert ".max()" not in src_text and ".sum()" in src_text  # reverted


def test_refactor_retries_then_succeeds(nb_factory, monkeypatch):
    nb, out, ref_dir, spec, ver = _scaffolded(nb_factory)
    monkeypatch.setattr(R, "_run_agent", _fake_agent([_edit_break, _edit_fix]))
    res = R.refactor_pipeline(
        Path(nb),
        out,
        ref_dir,
        spec,
        scaffold_verify=ver,
        yes=True,
        max_rounds=3,
        emit=lambda *_: None,
    )
    assert res["applied"] is True and res["calls"] == 2
    assert res["rounds"][0]["verdict"] == "differs"
    assert res["rounds"][1]["verdict"] == "equivalent"
