"""Drive the user's installed agent CLI to refactor the scaffold, with the output-lock as guardrail.

The scaffold is the notebook pasted into one function per stage. This asks the agent (claude -p or
codex exec) to turn it into proper, readable stages while keeping every output identical, and it
verifies after every pass. A pass that changes any output is rejected; if no pass produces an
equivalent refactor, the last known-good version (the scaffold, which was verified EQUIVALENT) is
restored. Nothing is spent without --yes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .verify import verify

DEFAULT_MODEL = {"claude": "sonnet", "codex": None}


def _agent_available(agent: str) -> str | None:
    return shutil.which(agent)


def _snapshot(src: Path) -> Path:
    dest = Path(tempfile.mkdtemp(prefix="nb2p-refactor-"))
    shutil.copytree(src, dest / "src", dirs_exist_ok=True)
    return dest / "src"


def _restore(backup: Path, src: Path) -> None:
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(backup, src)


def _clear_pycache(src: Path) -> None:
    """Drop cached bytecode so verify always runs the just-edited source.

    An edit that keeps a file the same size within the same mtime second would otherwise let
    Python reuse the stale .pyc, and the loop would verify the old code."""
    for cache in src.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _diff_feedback(ver: dict[str, Any]) -> str:
    run = (ver.get("pipeline") or {}).get("run") or {}
    if ver.get("verdict") == "pipeline_failed":
        tb = run.get("traceback") or run.get("stderr_tail") or run.get("error") or ""
        return (
            "The pipeline failed to run. Fix this error and keep the outputs identical:\n"
            + str(tb)[-1500:]
        )
    lines = ["The refactor changed some outputs. It must reproduce them exactly. Differences:"]
    for a in ver.get("artifacts", []):
        if a.get("passed") is False:
            detail = a.get("detail", "")
            lines.append(f"- {a['name']}: {detail}")
            for d in (a.get("differences") or [])[:3]:
                lines.append(
                    f"    {d.get('path')}: reference {d.get('reference')}, candidate {d.get('candidate')}"
                )
    for f in ver.get("figures", []):
        if f.get("passed") is False:
            lines.append(f"- figure {f.get('name')}: {f.get('detail', '')}")
    for f in ver.get("files", []):
        if f.get("passed") is False:
            lines.append(f"- file {f.get('path')}: {f.get('detail', '')}")
    return "\n".join(lines)


def _build_prompt(pkg: str, artifacts: list[str], first: bool, feedback: str | None) -> str:
    names = ", ".join(artifacts) if artifacts else "the values it already returns"
    if first:
        return (
            f"This project is a mechanical first draft written by notebook-to-pipeline: the cells of "
            f"a Jupyter notebook pasted into one function per stage under src/{pkg}/, with "
            f"src/{pkg}/pipeline.py defining run() that calls them in order and returns a dict of "
            f"results.\n\n"
            f"Refactor it into a clean, readable pipeline: give the stages clear names and "
            f"responsibilities (loading, feature building, training, evaluation, reporting as they "
            f"apply), remove dead code and leftover notebook display lines, factor out repetition, "
            f"add short docstrings, and keep imports tidy. This is a real refactor, not cosmetic.\n\n"
            f"Hard constraint: the outputs must not change. run() must keep returning the same dict "
            f"keys with the same values ({names}). The results are locked by a test and re-checked "
            f"after you finish; any change to a value will be rejected.\n\n"
            f"Only edit files under src/{pkg}/. Do not touch tests/, the reference capture, or "
            f"pyproject. Do not run any commands; just edit the source files."
        )
    return (
        f"Your refactor of src/{pkg}/ did not reproduce the notebook's outputs exactly.\n\n"
        f"{feedback}\n\n"
        f"Fix the pipeline so run() returns the same values as before ({names}), editing only files "
        f"under src/{pkg}/. Do not change the tests or the reference. Do not run any commands."
    )


def _run_agent(agent: str, prompt: str, cwd: Path, model: str | None) -> dict[str, Any]:
    if agent == "claude":
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--add-dir",
            str(cwd),
        ]
        if model:
            cmd += ["--model", model]
    elif agent == "codex":
        cmd = ["codex", "exec", "-s", "workspace-write", "-C", str(cwd)]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt)
    else:
        raise ValueError(f"unknown agent {agent!r}; use claude or codex")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    info: dict[str, Any] = {
        "returncode": proc.returncode,
        "cost_usd": None,
        "num_turns": None,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }
    if agent == "claude" and proc.stdout:
        try:
            data = json.loads(proc.stdout)
            info["cost_usd"] = data.get("total_cost_usd")
            info["num_turns"] = data.get("num_turns")
            info["is_error"] = data.get("is_error")
        except (json.JSONDecodeError, AttributeError):
            pass
    return info


def refactor_pipeline(
    nb_path: Path,
    out_dir: Path,
    ref_dir: Path,
    pipeline_spec: str,
    *,
    scaffold_verify: dict[str, Any],
    agent: str = "claude",
    yes: bool = False,
    model: str | None = None,
    max_rounds: int = 3,
    python: str | None = None,
    emit=print,
) -> dict[str, Any]:
    pkg = Path(pipeline_spec.split(":")[0]).parent.name
    pkg_src = out_dir / "src" / pkg
    model = model if model is not None else DEFAULT_MODEL.get(agent)
    artifacts = [
        a["name"] for a in scaffold_verify.get("artifacts", []) if a.get("passed") is not None
    ]

    emit("")
    emit("Refactor plan:")
    emit(f"  agent: {agent}" + (f" (model {model})" if model else " (default model)"))
    emit(
        f"  up to {max_rounds} pass(es); the outputs are verified after each, and a pass that "
        "changes any output is rejected and reverted."
    )
    emit("  This spends on your agent account.")

    binpath = _agent_available(agent)
    if binpath is None:
        emit(f"  {agent} is not on PATH, so the refactor cannot run. Keeping the scaffold.")
        return {
            "ran": False,
            "reason": f"{agent} not found on PATH",
            "agent": agent,
            "rounds": [],
            "total_cost_usd": 0.0,
            "calls": 0,
            "applied": False,
            "summary": f"not run ({agent} not installed); scaffold kept",
        }
    if not yes:
        emit("  Re-run with --yes to spend and refactor.")
        return {
            "ran": False,
            "reason": "not authorized (pass --yes)",
            "agent": agent,
            "model": model,
            "max_rounds": max_rounds,
            "rounds": [],
            "total_cost_usd": 0.0,
            "calls": 0,
            "applied": False,
            "summary": "not run (needs --yes); scaffold kept",
        }

    scaffold_ok = scaffold_verify.get("verdict") == "equivalent"
    good_backup = _snapshot(pkg_src) if scaffold_ok else None
    good_ver = scaffold_verify if scaffold_ok else None

    rounds: list[dict[str, Any]] = []
    total_cost = 0.0
    feedback: str | None = None
    final_ver: dict[str, Any] = scaffold_verify
    applied = False

    try:
        for i in range(1, max_rounds + 1):
            emit(f"  pass {i}/{max_rounds}: asking {agent} to refactor...")
            prompt = _build_prompt(pkg, artifacts, first=(i == 1), feedback=feedback)
            run = _run_agent(agent, prompt, out_dir, model)
            if run.get("cost_usd"):
                total_cost += run["cost_usd"]
            emit(
                f"    {agent} finished"
                + (f" ({run['num_turns']} turns)" if run.get("num_turns") else "")
                + (f", cost ${run['cost_usd']:.4f}" if run.get("cost_usd") else "")
                + ". Verifying..."
            )
            _clear_pycache(pkg_src)
            ver = verify(
                pipeline_spec, ref_dir, cwd=out_dir, python=python, out=out_dir / "evidence"
            )
            round_rec = {
                "round": i,
                "agent_returncode": run["returncode"],
                "cost_usd": run.get("cost_usd"),
                "num_turns": run.get("num_turns"),
                "verdict": ver["verdict"],
                "reason": ver.get("reason"),
            }
            rounds.append(round_rec)
            final_ver = ver
            if ver["verdict"] == "equivalent":
                emit(f"    pass {i}: EQUIVALENT. Keeping this refactor.")
                if good_backup is not None:
                    shutil.rmtree(good_backup.parent, ignore_errors=True)
                good_backup = _snapshot(pkg_src)
                good_ver = ver
                applied = True
                break
            emit(f"    pass {i}: {ver['verdict'].upper()} ({ver.get('reason', '')}).")
            feedback = _diff_feedback(ver)

        if final_ver.get("verdict") != "equivalent" and good_backup is not None:
            emit("  No pass reproduced the outputs; restoring the last equivalent version.")
            _restore(good_backup, pkg_src)
            final_ver = good_ver  # type: ignore[assignment]
            applied = False
    finally:
        if good_backup is not None:
            shutil.rmtree(good_backup.parent, ignore_errors=True)

    if applied:
        summary = (
            f"refactored by {agent} in {len(rounds)} pass(es), EQUIVALENT, cost ${total_cost:.4f}"
        )
    elif final_ver.get("verdict") == "equivalent":
        summary = (
            f"{agent} did not keep the outputs in {len(rounds)} pass(es); reverted to the "
            f"scaffold (EQUIVALENT), cost ${total_cost:.4f}"
        )
    else:
        summary = (
            f"{agent} did not produce an equivalent refactor and there was no equivalent baseline "
            f"to restore, cost ${total_cost:.4f}"
        )

    return {
        "ran": True,
        "agent": agent,
        "model": model,
        "rounds": rounds,
        "calls": len(rounds),
        "total_cost_usd": round(total_cost, 6),
        "applied": applied,
        "verify": final_ver,
        "summary": summary,
    }
