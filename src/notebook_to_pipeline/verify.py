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
from .capture import load_capture, now
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
    work = Path(out).resolve() if out else ref_dir.parent
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
    if pipeline:
        spec = {
            "pipeline": shlex.split(pipeline)[0],
            "args": shlex.split(pipeline)[1:],
            "cwd": str(run_cwd),
            "names": cand_names,
            "outdir": str(cand_dir),
            "rename_back": {v: k for k, v in rename.items()},
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
            if rf["kind"] == "figure":
                entry.update(
                    {
                        "status": "not_compared",
                        "passed": None,
                        "detail": "figure files are listed but not compared (renderers embed versions and timestamps)",
                    }
                )
            elif cf is None:
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
        result["artifacts"] = json.loads(comp_path.read_text())["artifacts"]
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

    compared = [x for x in result["artifacts"] + result["files"] if x.get("passed") is not None]
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
    return _finish(result, work)


def _finish(result: dict[str, Any], work: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for x in result.get("artifacts", []) + result.get("files", []):
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    result["counts"] = counts
    work.mkdir(parents=True, exist_ok=True)
    path = work / "verify.json"
    write_json(path, result)
    result["verify_json"] = str(path)
    return result
