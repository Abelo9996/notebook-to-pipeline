"""Capture reference outputs by running the notebook top to bottom in a fresh kernel."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze
from .execution import (
    DISPLAYED_PREFIX,
    collect_written,
    execute_notebook,
    python_info,
    resolve_python,
    run_runtime_script,
    snapshot,
)
from .notebook import read_notebook, sha256_file
from .util import write_json


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return s or "notebook"


def workspace_for(notebook: Path) -> Path:
    return notebook.resolve().parent / ".nb2p" / slug(notebook.stem)


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _match_prediction(execution: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any] | None:
    failed = execution.get("failed_cell")
    if not failed:
        return None
    errors = [f for f in analysis["findings"] if f["severity"] == "error"]
    for f in errors:
        if f["cells"] and f["cells"][0] == failed["index"]:
            return {"predicted": True, "finding": f}
    return {
        "predicted": False,
        "finding": None,
        "note": "static analysis did not flag this cell as a top-to-bottom failure",
    }


def _run_once(
    nb_path: Path,
    *,
    python: str,
    cwd: Path,
    timeout: int,
    names: list[str] | None,
    outdir: Path,
    max_bytes: int,
    display_values: bool = True,
) -> dict[str, Any]:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    nb = read_notebook(nb_path)
    before = snapshot(cwd, exclude=[outdir])
    execution = execute_notebook(
        nb,
        python=python,
        cwd=cwd,
        timeout=timeout,
        names=names,
        outdir=outdir,
        max_bytes=max_bytes,
        display_values=display_values,
    )
    after = snapshot(cwd, exclude=[outdir])
    written = collect_written(cwd, before, after, outdir / "files")
    manifest_path = outdir / "artifacts.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"artifacts": [], "env": {}}
    )
    if _keep_displayed_data(manifest, outdir):
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    figs_path = outdir / "figures" / "figures.json"
    figures = json.loads(figs_path.read_text()) if figs_path.exists() else None
    return {
        "execution": execution,
        "files_written": written,
        "manifest": manifest,
        "figures": figures,
    }


def _keep_displayed_data(manifest: dict[str, Any], outdir: Path) -> bool:
    """Keep a displayed value only when it is data the comparison understands.

    A cell's last expression is often a plot handle, a fitted model echoed by `.fit()`, or
    something without a stable value; those are dropped instead of listed as skipped artifacts.
    Returns True when the manifest changed."""
    kept = []
    changed = False
    for a in manifest.get("artifacts", []):
        name = a.get("name", "")
        if name.startswith(DISPLAYED_PREFIX):
            if a.get("status") != "captured" or a.get("kind") == "object":
                if a.get("payload"):
                    (outdir / a["payload"]).unlink(missing_ok=True)
                changed = True
                continue
            if "displayed_in" not in a:
                a["displayed_in"] = "cell " + name[len(DISPLAYED_PREFIX) :]
                changed = True
        kept.append(a)
    manifest["artifacts"] = kept
    shared = manifest.get("shared_objects")
    if shared:
        manifest["shared_objects"] = [
            sh
            for sh in shared
            if not any(str(p).startswith(DISPLAYED_PREFIX) for p in sh.get("paths", []))
        ]
        changed = changed or len(manifest["shared_objects"]) != len(shared)
    return changed


# Import names that differ from the package to install.
PACKAGE_FOR_MODULE = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "dateutil": "python-dateutil",
    "Bio": "biopython",
    "dotenv": "python-dotenv",
    "google.protobuf": "protobuf",
}


def failure_hint(execution: dict[str, Any], python: dict[str, Any]) -> str | None:
    """A one-line fix for the failures a newcomer hits first."""
    fc = execution.get("failed_cell") or {}
    if fc.get("ename") == "ModuleNotFoundError":
        m = re.search(r"No module named '([^']+)'", fc.get("evalue", ""))
        module = m.group(1) if m else None
        top = (module or "").split(".")[0]
        package = PACKAGE_FOR_MODULE.get(module or "", PACKAGE_FOR_MODULE.get(top, top))
        if python.get("chosen_by") == "the interpreter running nb2p":
            return (
                f"No project environment was found, so the notebook ran in nb2p's own Python, which "
                f"does not have {package or 'that package'}. Point nb2p at the environment the notebook "
                "normally uses (--python path/to/.venv/bin/python, or a .venv next to the notebook is "
                "picked up automatically), or add the packages for this run: "
                f"uvx --with {package or '<package>'} notebook-to-pipeline capture ..."
            )
        return (
            f"{python.get('path')} does not have {package or 'that package'}. Install it in that "
            "environment, or pass --python with the interpreter the notebook normally uses."
        )
    if fc.get("ename") == "FileNotFoundError":
        return (
            "The notebook reads a file that is not there when run from its own directory. Pass "
            "--cwd if it expects another working directory, or make the data file available."
        )
    if fc.get("ename") in ("NameError", "UnboundLocalError"):
        return (
            "A name is used before any cell defines it in top-to-bottom order: the notebook "
            "depended on hidden state from an earlier session. See the analyze findings."
        )
    return None


def runtime_findings(
    manifest: dict[str, Any], artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Findings that only a real run can show: shared objects and artifacts with identical content."""
    out: list[dict[str, Any]] = []
    for sh in manifest.get("shared_objects", []):
        names = ", ".join(f"`{p}`" for p in sh["paths"])
        if sh["kind"] == "shared_between_composites":
            out.append(
                {
                    "kind": "shared_object",
                    "severity": "warning",
                    "names": sh["paths"],
                    "message": f"After the run, {names} are one and the same {sh['type']} object. "
                    "Fitting or changing it through one name changed it for all of them.",
                }
            )
        elif sh["kind"] == "alias":
            out.append(
                {
                    "kind": "alias",
                    "severity": "info",
                    "names": sh["paths"],
                    "message": f"{names} refer to the same {sh['type']} object, so an in-place change to one is a change to the other.",
                }
            )
    by_hash: dict[str, list[str]] = {}
    for a in artifacts:
        if (
            a.get("status") == "captured"
            and a.get("kind") not in ("scalar", "container")
            and a.get("hash")
            and not a["name"].startswith(DISPLAYED_PREFIX)
        ):
            by_hash.setdefault(a["hash"], []).append(a["name"])
    for names in by_hash.values():
        if len(names) > 1:
            out.append(
                {
                    "kind": "identical_artifacts",
                    "severity": "info",
                    "names": names,
                    "message": f"{', '.join(f'`{n}`' for n in names)} have identical content after the run. If the notebook treats them as different results, check for shared objects or a step that was meant to differ.",
                }
            )
    return out


def capture(
    notebook: str | Path,
    *,
    out: str | Path | None = None,
    python: str | None = None,
    cwd: str | Path | None = None,
    timeout: int = 600,
    artifacts: list[str] | None = None,
    all_globals: bool = False,
    repeat: int = 1,
    max_bytes: int = 200 * 1024 * 1024,
    command: str | None = None,
) -> dict[str, Any]:
    nb_path = Path(notebook).resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"notebook not found: {nb_path}")
    outdir = Path(out).resolve() if out else workspace_for(nb_path) / "reference"
    run_cwd = Path(cwd).resolve() if cwd else nb_path.parent
    py, how = resolve_python(python, nb_path.parent)
    pinfo = python_info(py)
    analysis = analyze(nb_path)

    if artifacts:
        names: list[str] | None = list(artifacts)
        mode = "explicit"
    elif all_globals:
        names = None
        mode = "all data-like globals"
    else:
        names = [a["name"] for a in analysis["suggested_artifacts"]]
        mode = "suggested by analyze (top-level assignments)"

    result: dict[str, Any] = {
        "tool": {"name": "notebook-to-pipeline", "version": __version__},
        "kind": "capture",
        "created_at": now(),
        "command": command or f"nb2p capture {nb_path}",
        "notebook": {
            "path": str(nb_path),
            "sha256": sha256_file(nb_path),
            "cells_total": analysis["notebook"]["cells_total"],
            "code_cells": analysis["notebook"]["code_cells"],
        },
        "python": {"path": py, "chosen_by": how, **pinfo},
        "cwd": str(run_cwd),
        "timeout_s": timeout,
        "selection": {"mode": mode, "requested": names},
    }
    if "error" in pinfo or not pinfo.get("ipykernel"):
        result["execution"] = {
            "status": "kernel_error",
            "error": pinfo.get("error")
            or f"ipykernel is not installed for {py}. Install it in that environment "
            f"(uv pip install --python {py} ipykernel, or pip install ipykernel) or pass --python.",
            "cells": [],
            "cells_run": 0,
            "code_cells": analysis["notebook"]["code_cells"],
        }
        result.update(
            {
                "artifacts": [],
                "files_written": [],
                "env": {},
                "stability": None,
                "runtime_findings": [],
                "prediction": None,
                "figures": [],
                "printed": {"lines": 0},
            }
        )
        result["next_steps"] = [result["execution"]["error"]]
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(outdir / "capture.json", result)
        result["reference_dir"] = str(outdir)
        return result

    display_values = not artifacts
    first = _run_once(
        nb_path,
        python=py,
        cwd=run_cwd,
        timeout=timeout,
        names=names,
        outdir=outdir,
        max_bytes=max_bytes,
        display_values=display_values,
    )
    execution = first["execution"]
    for c in execution["cells"]:
        c["label"] = f"cell {c['index'] + 1}"
    if execution.get("failed_cell"):
        execution["failed_cell"]["label"] = f"cell {execution['failed_cell']['index'] + 1}"
    result.update(
        {
            "execution": execution,
            "artifacts": first["manifest"].get("artifacts", []),
            "env": first["manifest"].get("env", {}),
            "files_written": first["files_written"],
            "prediction": _match_prediction(execution, analysis),
            "stability": None,
            "figures": (first["figures"] or {}).get("figures", []),
            "printed": _printed_summary(execution),
        }
    )
    execution["inline_images"] = sum(c.get("inline_images", 0) for c in execution["cells"])
    hint = failure_hint(execution, result["python"])
    if hint:
        execution["hint"] = hint
    result["runtime_findings"] = runtime_findings(first["manifest"], result["artifacts"])

    if repeat > 1 and execution["status"] == "ok":
        result["stability"] = _check_stability(
            nb_path, py, run_cwd, timeout, names, outdir, max_bytes, repeat, display_values
        )
        unstable = {r["name"] for r in result["stability"]["artifacts"] if not r["passed"]}
        for a in result["artifacts"]:
            a["stable"] = a["name"] not in unstable
        unstable_figs = set(result["stability"].get("unstable_figures", []))
        for f in result["figures"]:
            f["stable"] = f["index"] not in unstable_figs

    result["next_steps"] = capture_next_steps(result, outdir)
    write_json(outdir / "capture.json", result)
    write_json(outdir / "analysis.json", analysis)
    result["reference_dir"] = str(outdir)
    return result


def _printed_summary(execution: dict[str, Any]) -> dict[str, Any]:
    lines = [
        ln.strip()
        for c in execution.get("cells", [])
        for ln in (c.get("printed") or "").splitlines()
        if ln.strip()
    ]
    return {"lines": len(lines)}


def capture_next_steps(result: dict[str, Any], outdir: Path) -> list[str]:
    ex = result["execution"]
    if ex["status"] != "ok":
        steps = []
        if ex.get("hint"):
            steps.append(ex["hint"])
        steps.append(
            "Tell the user the notebook does not run top to bottom (cell and error above). Do not "
            "fix it silently: if they want a fix, make the smallest change, say what it is, and capture again."
        )
        return steps
    steps = [
        "Write the pipeline as a function returning a dict {artifact name: value} (for example "
        "src/<package>/pipeline.py:run), or start from a mechanical draft (MCP scaffold_pipeline, "
        "or `nb2p scaffold`).",
        f"Verify after every change (MCP verify_pipeline, or `nb2p verify --pipeline <file.py:run> --reference {outdir}`).",
    ]
    if any(not a.get("stable", True) for a in result.get("artifacts", [])):
        steps.append(
            "Some artifacts are not stable run to run: they cannot be verified. Tell the user "
            "instead of loosening tolerances."
        )
    return steps


def _check_stability(
    nb_path, py, run_cwd, timeout, names, outdir, max_bytes, repeat, display_values=True
):
    runs = []
    tmp_root = Path(tempfile.mkdtemp(prefix="nb2p-repeat-"))
    try:
        worst: dict[str, dict[str, Any]] = {}
        unstable_figs: set[int] = set()
        for k in range(2, repeat + 1):
            rdir = tmp_root / f"run{k}"
            again = _run_once(
                nb_path,
                python=py,
                cwd=run_cwd,
                timeout=timeout,
                names=names,
                outdir=rdir,
                max_bytes=max_bytes,
                display_values=display_values,
            )
            runs.append({"run": k, "status": again["execution"]["status"]})
            if again["execution"]["status"] != "ok":
                continue
            spec = {
                "reference": str(outdir),
                "candidate": str(rdir),
                "names": None,
                "files": [],
                "figures": True,
                "options": {},
                "out": str(rdir / "compare.json"),
            }
            proc = run_runtime_script(py, "nb2p_compare.py", spec, cwd=run_cwd, timeout=None)
            if proc.returncode != 0:
                runs[-1]["compare_error"] = proc.stderr[-2000:]
                continue
            comp = json.loads((rdir / "compare.json").read_text())
            for r in comp["artifacts"]:
                prev = worst.get(r["name"])
                if prev is None or (prev["passed"] and not r["passed"]):
                    worst[r["name"]] = r
            for k, fr in enumerate(comp.get("figures") or []):
                if fr.get("kind") == "figure" and fr.get("passed") is False:
                    unstable_figs.add(k + 1)
        return {
            "repeats": repeat,
            "runs": runs,
            "artifacts": list(worst.values()),
            "unstable_figures": sorted(unstable_figs),
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def default_work_dir(ref_dir: Path) -> Path:
    """Where verify and report write by default: next to the reference inside .nb2p/, otherwise
    in a hidden .nb2p-verify/ beside it (so a reference kept in tests/ does not collect clutter)."""
    if ".nb2p" in ref_dir.parts:
        return ref_dir.parent
    return ref_dir.parent / ".nb2p-verify"


def load_capture(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Accept a capture dir, its capture.json, or a notebook path (uses the default workspace)."""
    p = Path(path).resolve()
    if p.suffix == ".ipynb":
        p = workspace_for(p) / "reference"
    if p.is_file() and p.name == "capture.json":
        p = p.parent
    cj = p / "capture.json"
    if not cj.exists():
        raise FileNotFoundError(f"no capture.json in {p}; run `nb2p capture` first")
    return p, json.loads(cj.read_text())
