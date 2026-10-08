"""Run the agent-written pipeline and compare its artifacts with the reference capture."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import __version__
from .capture import default_work_dir, load_capture, now
from .execution import (
    collect_written,
    python_info,
    resolve_python,
    run_env,
    run_runtime_script,
    snapshot,
    strip_ansi,
)
from .runtime.nb2p_compare import DEFAULTS as COMPARE_DEFAULTS
from .util import write_json

VERDICTS = ("equivalent", "differs", "pipeline_failed", "reference_invalid", "inconclusive")


def _tail(text: str, n: int = 3000) -> str:
    text = strip_ansi(text or "")
    return text if len(text) <= n else "..." + text[-n:]


def verify(
    pipeline: str | None,
    reference: str | Path,
    *,
    cmd: str | None = None,
    python: str | None = None,
    cwd: str | Path | None = None,
    out: str | Path | None = None,
    artifacts: list[str] | None = None,
    rename: dict[str, str] | None = None,
    compare_files: bool = True,
    compare_figures: bool = True,
    timeout: int = 1800,
    rtol: float | None = None,
    atol: float | None = None,
    ignore_row_order: bool = False,
    ignore_column_order: bool = False,
    ignore_index: bool = False,
    check_dtype: bool = True,
    command: str | None = None,
) -> dict[str, Any]:
    if not pipeline and not cmd:
        raise ValueError("pass a pipeline (file.py, file.py:func, module or module:func) or --cmd")
    ref_dir, ref = load_capture(reference)
    work = Path(out).resolve() if out else default_work_dir(ref_dir)
    cand_dir = work / "candidate"
    run_cwd = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    py, how = resolve_python(python, run_cwd)
    rename = rename or {}
    options = {
        "rtol": rtol,
        "atol": atol,
        "ignore_row_order": ignore_row_order,
        "ignore_column_order": ignore_column_order,
        "ignore_index": ignore_index,
        "check_dtype": check_dtype,
    }
    effective = dict(COMPARE_DEFAULTS)
    effective.update({k: v for k, v in options.items() if v is not None})

    result: dict[str, Any] = {
        "tool": {"name": "notebook-to-pipeline", "version": __version__},
        "kind": "verify",
        "created_at": now(),
        "command": command
        or "nb2p verify "
        + (f"--pipeline {shlex.quote(pipeline)}" if pipeline else f"--cmd {shlex.quote(cmd or '')}")
        + f" --reference {ref_dir}",
        "reference": {
            "dir": str(ref_dir),
            "notebook": ref["notebook"],
            "execution_status": ref["execution"]["status"],
            "created_at": ref["created_at"],
        },
        "pipeline": {"spec": pipeline, "cmd": cmd, "cwd": str(run_cwd)},
        "python": {"path": py, "chosen_by": how, **python_info(py)},
        "options": effective,
        "artifacts": [],
        "files": [],
        "figures": [],
    }

    if ref["execution"]["status"] != "ok":
        failed = ref["execution"].get("failed_cell") or {}
        result["verdict"] = "reference_invalid"
        result["reason"] = (
            f"the notebook does not run top to bottom (status {ref['execution']['status']}"
            + (f", {failed.get('ename')} in {failed.get('label', '')}" if failed else "")
            + "), so there are no reference outputs to compare against. Fix the notebook or the hidden state first."
        )
        return _finish(result, work)

    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    cand_dir.mkdir(parents=True)

    ref_names = [a["name"] for a in ref["artifacts"] if a.get("status") == "captured"]
    names = list(artifacts) if artifacts else ref_names
    cand_names = [rename.get(n, n) for n in names]

    before = snapshot(run_cwd, exclude=[work, ref_dir])
    t0 = time.monotonic()
    stdout_full = ""
    if pipeline:
        spec = {
            "pipeline": shlex.split(pipeline)[0],
            "args": shlex.split(pipeline)[1:],
            "cwd": str(run_cwd),
            "names": cand_names,
            "outdir": str(cand_dir),
            "rename_back": {v: k for k, v in rename.items()},
            "figures": compare_figures,
            "compare": {
                "reference": str(ref_dir),
                "names": names,
                "options": options,
                "out": str(cand_dir / "compare_artifacts.json"),
            },
        }
        try:
            proc = run_runtime_script(py, "nb2p_runner.py", spec, cwd=run_cwd, timeout=timeout)
            run_info = (
                json.loads((cand_dir / "run.json").read_text())
                if (cand_dir / "run.json").exists()
                else {
                    "status": "failed",
                    "error": f"runner exited with {proc.returncode} before writing results",
                }
            )
            run_info["exit_code"] = proc.returncode
            stdout_full = proc.stdout or ""
            run_info["stdout_tail"] = _tail(proc.stdout, 2000)
            run_info["stderr_tail"] = _tail(proc.stderr, 3000)
        except subprocess.TimeoutExpired:
            run_info = {
                "status": "failed",
                "error": f"pipeline exceeded {timeout}s",
                "error_type": "Timeout",
            }
    else:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(run_cwd),
                env=run_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout_full = proc.stdout or ""
            run_info = {
                "status": "ok" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "stdout_tail": _tail(proc.stdout, 2000),
                "stderr_tail": _tail(proc.stderr, 3000),
            }
            if proc.returncode != 0:
                run_info["error"] = f"command exited with {proc.returncode}"
        except subprocess.TimeoutExpired:
            run_info = {
                "status": "failed",
                "error": f"command exceeded {timeout}s",
                "error_type": "Timeout",
            }
    run_info["duration_s"] = round(time.monotonic() - t0, 3)
    after = snapshot(run_cwd, exclude=[work, ref_dir])
    cand_written = collect_written(run_cwd, before, after, cand_dir / "files")
    result["pipeline"]["run"] = run_info
    result["pipeline"]["files_written"] = cand_written

    if run_info["status"] != "ok":
        result["verdict"] = "pipeline_failed"
        result["reason"] = (
            f"{run_info.get('error_type', 'error')}: {run_info.get('error', '')}".strip()
        )
        return _finish(result, work)

    cand_manifest_path = cand_dir / "artifacts.json"
    if pipeline:
        result["candidate_env"] = (
            json.loads(cand_manifest_path.read_text()).get("env", {})
            if cand_manifest_path.exists()
            else {}
        )

    # Files: data files the notebook wrote must be written by the pipeline too.
    file_specs = []
    file_results = []
    cand_by_path = {f["path"]: f for f in cand_written}
    if compare_files:
        for rf in ref.get("files_written", []):
            entry: dict[str, Any] = {"path": rf["path"], "kind": rf["kind"]}
            cf = cand_by_path.get(rf["path"])
            if cf is None:
                entry.update(
                    {
                        "status": "missing",
                        "passed": False,
                        "detail": "the pipeline did not write this file",
                    }
                )
            elif cf["sha256"] == rf["sha256"]:
                entry.update({"status": "identical", "passed": True, "detail": "sha256 match"})
            elif rf.get("copied") and cf.get("copied"):
                file_specs.append(
                    {
                        "path": rf["path"],
                        "reference_dir": str(ref_dir / "files"),
                        "candidate_dir": str(cand_dir / "files"),
                    }
                )
                entry.update({"status": "pending"})
            else:
                entry.update(
                    {
                        "status": "differs",
                        "passed": False,
                        "detail": "sha256 differs and the file is too large to compare in detail",
                    }
                )
            file_results.append(entry)

    if pipeline:
        comp_path = cand_dir / "compare_artifacts.json"
        if not comp_path.exists():
            result["verdict"] = "inconclusive"
            result["reason"] = "the comparison step failed: " + _tail(
                run_info.get("stderr_tail", ""), 2000
            )
            return _finish(result, work)
        comp_all = json.loads(comp_path.read_text())
        result["artifacts"] = comp_all["artifacts"]
        result["figures"] = comp_all.get("figures") or []
        _displayed_not_returned(result["artifacts"], ref, rename)
    if file_specs:
        spec = {
            "reference": str(ref_dir),
            "candidate": str(cand_dir),
            "names": [],
            "files": file_specs,
            "options": options,
            "out": str(cand_dir / "compare_files.json"),
        }
        proc = run_runtime_script(py, "nb2p_compare.py", spec, cwd=run_cwd, timeout=timeout)
        if proc.returncode != 0 or not (cand_dir / "compare_files.json").exists():
            result["verdict"] = "inconclusive"
            result["reason"] = "the file comparison step failed: " + _tail(proc.stderr, 2000)
            return _finish(result, work)
        comp = json.loads((cand_dir / "compare_files.json").read_text())
        by_path = {f["path"]: f for f in comp.get("files", [])}
        for entry in file_results:
            if entry["status"] == "pending":
                entry.update({k: v for k, v in by_path[entry["path"]].items() if k != "path"})
    if not pipeline:
        result["artifacts"] = [
            {
                "name": n,
                "status": "not_compared",
                "passed": None,
                "detail": "--cmd mode compares written files only",
            }
            for n in names
        ]
    result["files"] = file_results
    result.setdefault("figures", [])
    result["printed"] = compare_printed(ref, stdout_full)
    if not compare_figures:
        result["figures"] = []
    unstable_figs = {f["index"] for f in ref.get("figures", []) if f.get("stable") is False}
    for k, f in enumerate(result["figures"]):
        if f.get("passed") is False and (k + 1) in unstable_figs:
            f["status"] = "unstable"
            f["passed"] = None
            f["detail"] = (
                "the notebook draws this figure differently run to run, so it is not counted: "
                + f["detail"]
            )

    stability = {a["name"]: a.get("stable", True) for a in ref["artifacts"]}
    for a in result["artifacts"]:
        if a["name"] in rename:
            a["candidate_name"] = rename[a["name"]]
        if a.get("passed") is False and stability.get(a["name"]) is False:
            a["status"] = "unstable"
            a["passed"] = None
            a["detail"] = (
                "reference is not reproducible run to run (capture --repeat), so it is not counted: "
                + a["detail"]
            )

    compared = [
        x
        for x in result["artifacts"] + result["files"] + result["figures"]
        if x.get("passed") is not None
    ]
    if not compared:
        result["verdict"] = "inconclusive"
        result["reason"] = "nothing was compared"
    elif all(x["passed"] for x in compared):
        result["verdict"] = "equivalent"
        result["reason"] = f"all {len(compared)} compared outputs match"
    else:
        bad = [x for x in compared if not x["passed"]]
        result["verdict"] = "differs"
        result["reason"] = f"{len(bad)} of {len(compared)} compared outputs differ"
    skipped = [
        x
        for x in result["artifacts"] + result["files"] + result["figures"]
        if x.get("passed") is None and x.get("status") != "extra"
    ]
    if skipped and compared:
        kinds: dict[str, int] = {}
        for x in skipped:
            kinds[x["status"]] = kinds.get(x["status"], 0) + 1
        result["reason"] += "; not counted: " + ", ".join(
            f"{n} {k.replace('_', ' ')}" for k, n in kinds.items()
        )
    return _finish(result, work)


def _displayed_not_returned(
    artifacts: list[dict[str, Any]], ref: dict[str, Any], rename: dict[str, str]
) -> None:
    """A value the notebook only displayed is compared when the pipeline returns it under its
    artifact name. A pipeline that does not return it is not failed for that: the value is
    listed as not compared, so the verdict says what was left out."""
    shown = {a["name"]: a["displayed_in"] for a in ref["artifacts"] if a.get("displayed_in")}
    for a in artifacts:
        cell = shown.get(a["name"])
        if cell is None or a.get("status") != "missing":
            continue
        if not str(a.get("detail", "")).startswith("the candidate did not produce"):
            continue
        name = rename.get(a["name"], a["name"])
        a.update(
            {
                "status": "not_returned",
                "passed": None,
                "displayed_in": cell,
                "detail": f"only displayed in the notebook ({cell}); return it from the "
                f"pipeline as `{name}` (or map it with --rename {a['name']}=<your name>) to compare it",
            }
        )


def compare_printed(ref: dict[str, Any], stdout: str) -> dict[str, Any]:
    """Which lines the notebook printed also appear in the pipeline's stdout.

    Values that were only printed (an accuracy, a count) are not variables, so they are not
    artifacts. This check is informational: it is not counted in the verdict, because a
    pipeline may log differently on purpose."""
    ref_lines = [
        ln.strip()
        for c in ref.get("execution", {}).get("cells", [])
        for ln in (c.get("printed") or "").splitlines()
        if ln.strip()
    ]
    if not ref_lines:
        return {"lines": 0, "found": 0, "missing": [], "counted_in_verdict": False}
    cand = {ln.strip() for ln in strip_ansi(stdout).splitlines() if ln.strip()}
    missing = [ln for ln in ref_lines if ln not in cand]
    return {
        "lines": len(ref_lines),
        "found": len(ref_lines) - len(missing),
        "missing": missing[:10],
        "counted_in_verdict": False,
    }


def verify_next_steps(result: dict[str, Any]) -> list[str]:
    v = result.get("verdict")
    steps: list[str] = []
    if v == "equivalent":
        steps.append(
            "Report the verdict with the number of compared outputs, then write the report "
            "(MCP write_report, or `nb2p report --reference <reference dir>`). Keep verifying after any further change."
        )
    elif v == "differs":
        steps.append(
            "Read the first differences above and fix the pipeline. Do not widen tolerances or "
            "drop artifacts to get a pass; if a change is intended, tell the user what changed and why."
        )
    elif v == "pipeline_failed":
        steps.append("Fix the error in the pipeline (traceback above), then verify again.")
    elif v == "reference_invalid":
        steps.append(
            "The notebook itself does not run top to bottom. Tell the user, and capture again "
            "once it runs."
        )
    else:
        steps.append("Nothing usable was compared; check the reason above.")
    figs = result.get("figures", [])
    if figs and all(f.get("status") == "not_compared" for f in figs):
        steps.append(
            f"The notebook drew {len(figs)} figure(s) and the pipeline drew none, so figures were not "
            "compared. If the pipeline should make the plots, draw them with matplotlib in the "
            "verified run and they will be compared pixel by pixel."
        )
    shown = [
        a for a in result.get("artifacts", []) if a.get("displayed_in") and a.get("passed") is None
    ]
    if shown:
        steps.append(
            f"{len(shown)} value(s) the notebook only displayed as a cell's last expression were not "
            f"compared because the pipeline does not return them ({', '.join(a['name'] for a in shown[:5])}). "
            "If one matters, return it under that name."
        )
    pr = result.get("printed") or {}
    if pr.get("lines") and pr.get("found", 0) < pr["lines"]:
        steps.append(
            f"{pr['lines'] - pr['found']} of {pr['lines']} lines the notebook printed are not in the "
            "pipeline's output (not counted in the verdict). If a printed number matters, return it "
            "as an artifact or print it the same way."
        )
    return steps


def _finish(result: dict[str, Any], work: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for x in result.get("artifacts", []) + result.get("files", []) + result.get("figures", []):
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    result["counts"] = counts
    result["next_steps"] = verify_next_steps(result)
    work.mkdir(parents=True, exist_ok=True)
    path = work / "verify.json"
    write_json(path, result)
    result["verify_json"] = str(path)
    return result
