"""`nb2p pipeline`: one command from a notebook to an output-locked pipeline.

Runs analyze, capture (fresh kernel), scaffold, verify and report in order and ends with a
one-screen Reproducibility Report plus a shareable card. With refactor=True it then drives the
user's agent CLI to refactor the scaffold into proper stages, re-verifying after every pass and
never accepting a change that alters the outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze
from .capture import capture, load_capture, slug
from .card import write_card
from .report import build_report
from .scaffold import scaffold
from .verify import verify

# Finding kinds grouped into the four categories the report ranks, most serious first.
CATEGORY_OF_KIND = {
    # hidden state
    "use_before_def": "hidden state",
    "undefined_name": "hidden state",
    "maybe_star_import": "hidden state",
    "function_reads_undefined": "hidden state",
    "star_import": "hidden state",
    "out_of_order_execution": "hidden state",
    "hidden_executions": "hidden state",
    "never_executed": "hidden state",
    "stale_output": "hidden state",
    "redefined_across_cells": "hidden state",
    "cross_cell_mutation": "hidden state",
    "rerun_hazard": "hidden state",
    "saved_output_changed": "hidden state",
    "parse_error": "hidden state",
    # shared-object mutation
    "shared_estimator": "shared-object mutation",
    "shared_object": "shared-object mutation",
    "alias": "shared-object mutation",
    "identical_artifacts": "shared-object mutation",
    # nondeterminism
    "unseeded_randomness": "nondeterminism",
    "network_access": "nondeterminism",
    "unstable_artifact": "nondeterminism",
    # unpinned deps
    "unpinned_deps": "unpinned deps",
}
CATEGORY_ORDER = ["hidden state", "shared-object mutation", "nondeterminism", "unpinned deps"]
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
# Issues the card counts as "hidden-state issues caught".
CARD_CATEGORIES = {"hidden state", "shared-object mutation"}


def _pipeline_out(nb_path: Path, out: str | Path | None) -> Path:
    if out:
        return Path(out).resolve()
    return nb_path.parent / f"{slug(nb_path.stem)}-pipeline"


def ranked_findings(
    analysis: dict[str, Any], capture_result: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """One ranked list of findings across static analysis, the fresh run and dependency pinning.

    Ordered by category (hidden state, shared-object mutation, nondeterminism, unpinned deps) and
    then by severity. Each entry has category, severity, kind, message and cells (0-based)."""
    items: list[dict[str, Any]] = []

    def add(kind: str, severity: str, message: str, cells: list[int] | None = None) -> None:
        items.append(
            {
                "category": CATEGORY_OF_KIND.get(kind, "hidden state"),
                "severity": severity,
                "kind": kind,
                "message": message,
                "cells": cells or [],
            }
        )

    for f in analysis.get("findings", []):
        add(f["kind"], f["severity"], f["message"], f.get("cells"))

    if capture_result:
        for f in capture_result.get("runtime_findings", []):
            add(f["kind"], f["severity"], f["message"], [])
        # Saved cell outputs that disagree with the fresh top-to-bottom run (no counts needed).
        ex = capture_result.get("execution", {})
        changed = [c for c in ex.get("cells", []) if c.get("saved_output") == "different"]
        if changed:
            labels = ", ".join(c.get("label", f"cell {c['index'] + 1}") for c in changed[:6])
            add(
                "saved_output_changed",
                "warning",
                f"{len(changed)} cell(s) have saved outputs that differ from a clean top-to-bottom "
                f"run ({labels}). The saved outputs do not all come from such a run, or library "
                "versions changed.",
                [c["index"] for c in changed],
            )
        # Artifacts the determinism check found unstable.
        st = capture_result.get("stability")
        if st:
            unstable = [r["name"] for r in st.get("artifacts", []) if not r.get("passed")]
            if unstable:
                add(
                    "unstable_artifact",
                    "warning",
                    f"{len(unstable)} artifact(s) change between runs and cannot be verified: "
                    + ", ".join(unstable),
                )
        # Dependencies imported but not pinned to a captured version.
        unpinned = _unpinned_deps(analysis, capture_result)
        if unpinned:
            add(
                "unpinned_deps",
                "info",
                f"{len(unpinned)} imported package(s) have no captured version to pin in the "
                "generated pyproject: " + ", ".join(unpinned) + ". Their transitive dependencies "
                "are never pinned.",
            )

    items.sort(
        key=lambda x: (
            CATEGORY_ORDER.index(x["category"]) if x["category"] in CATEGORY_ORDER else 99,
            SEVERITY_ORDER.get(x["severity"], 3),
            x["cells"][0] if x["cells"] else -1,
        )
    )
    return items


_STDLIB = {
    "os",
    "sys",
    "re",
    "json",
    "math",
    "random",
    "time",
    "datetime",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "warnings",
    "pickle",
    "csv",
    "glob",
    "shutil",
    "subprocess",
    "typing",
    "string",
    "statistics",
    "urllib",
    "zipfile",
    "io",
    "sqlite3",
    "copy",
    "logging",
    "decimal",
    "fractions",
    "operator",
    "textwrap",
}
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
}


def _unpinned_deps(analysis: dict[str, Any], capture_result: dict[str, Any]) -> list[str]:
    packages = capture_result.get("env", {}).get("packages", {})
    lower = {k.lower() for k in packages}
    out = []
    for imp in analysis.get("inputs", {}).get("imports", []):
        if imp in _STDLIB or imp == "ipykernel":
            continue
        dist = _IMPORT_TO_DIST.get(imp, imp)
        if dist.lower() not in lower:
            out.append(dist)
    return out


def _locked_summary(ver: dict[str, Any]) -> tuple[int, str]:
    """How many outputs the pipeline locked, and a short breakdown."""
    variables = sum(1 for a in ver.get("artifacts", []) if a.get("passed") is True)
    figures = sum(1 for f in ver.get("figures", []) if f.get("passed") is True)
    files = sum(1 for f in ver.get("files", []) if f.get("passed") is True)
    total = variables + figures + files
    parts = []
    if variables:
        parts.append(f"{variables} variable{'s' if variables != 1 else ''}")
    if figures:
        parts.append(f"{figures} figure{'s' if figures != 1 else ''}")
    if files:
        parts.append(f"{files} file{'s' if files != 1 else ''}")
    return total, ", ".join(parts)


def run_pipeline(
    notebook: str | Path,
    *,
    out: str | Path | None = None,
    refactor: bool = False,
    agent: str = "claude",
    yes: bool = False,
    model: str | None = None,
    max_rounds: int = 3,
    package: str | None = None,
    python: str | None = None,
    cwd: str | Path | None = None,
    timeout: int = 600,
    repeat: int = 1,
    emit=print,
    command: str | None = None,
) -> dict[str, Any]:
    """Run the whole flow and return a result dict. emit is called with human-readable lines."""
    nb_path = Path(notebook).resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"notebook not found: {nb_path}")
    out_dir = _pipeline_out(nb_path, out)

    result: dict[str, Any] = {
        "tool": {"name": "notebook-to-pipeline", "version": __version__},
        "kind": "pipeline",
        "notebook": str(nb_path),
        "out": str(out_dir),
        "command": command or f"nb2p pipeline {nb_path.name}",
    }

    analysis = analyze(nb_path)
    emit(f"Analyzing {nb_path.name}: {analysis['notebook']['code_cells']} code cells.")

    emit("Capturing the reference (running top to bottom in a fresh kernel)...")
    cap = capture(nb_path, python=python, cwd=cwd, timeout=timeout, repeat=repeat)
    ref_dir = Path(cap["reference_dir"])
    result["reference_dir"] = str(ref_dir)
    result["findings"] = ranked_findings(analysis, cap)

    if cap["execution"]["status"] != "ok":
        result["verdict"] = "reference_invalid"
        result["capture_status"] = cap["execution"]["status"]
        result["failed_cell"] = cap["execution"].get("failed_cell")
        result["hint"] = cap["execution"].get("hint")
        _write_card(result, nb_path, out_dir, locked=0, locked_detail="", refactored=False)
        emit(format_report(result))
        return result

    emit(f"Scaffolding the pipeline into {out_dir.name}/ ...")
    scaf = scaffold(nb_path, out_dir, package=package, reference=ref_dir)
    result["package"] = scaf["package"]
    result["stages"] = scaf["stages"]
    result["scaffold"] = {
        "test_command": scaf["test_command"],
        "install_command": scaf["install_command"],
        "env": scaf["env"],
    }
    result["pipeline_spec"] = f"src/{scaf['package']}/pipeline.py:run"
    result["test_file"] = "tests/test_equivalence.py"
    # Run the pipeline from the notebook's own directory so relative data reads
    # (./data/x.csv) resolve as they did in the notebook, with an absolute spec so the
    # package in out/ is still imported.
    abs_spec = f"{out_dir / 'src' / scaf['package'] / 'pipeline.py'}:run"
    run_cwd = Path(cwd).resolve() if cwd else nb_path.parent

    emit("Verifying the scaffold reproduces the reference...")
    ver = verify(abs_spec, ref_dir, cwd=run_cwd, python=python, out=out_dir / "evidence")
    result["verify"] = _verify_summary(ver)

    if refactor:
        from .refactor import refactor_pipeline

        ref_result = refactor_pipeline(
            nb_path,
            out_dir,
            ref_dir,
            abs_spec,
            scaffold_verify=ver,
            agent=agent,
            yes=yes,
            model=model,
            max_rounds=max_rounds,
            python=python,
            verify_cwd=run_cwd,
            emit=emit,
        )
        result["refactor"] = ref_result
        if ref_result.get("verify"):
            ver = ref_result["verify"]
            result["verify"] = _verify_summary(ver)

    report = build_report(ref_dir, ver.get("verify_json"), out=out_dir)
    result["report_md"] = report["report_md"]
    result["report_json"] = report["report_json"]

    result["verdict"] = ver["verdict"]
    locked, detail = _locked_summary(ver)
    result["locked"] = locked
    result["locked_detail"] = detail
    _write_card(
        result,
        nb_path,
        out_dir,
        locked=locked,
        locked_detail=detail,
        refactored=bool(refactor and result.get("refactor", {}).get("applied")),
    )

    emit("")
    emit(format_report(result))
    return result


def _verify_summary(ver: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": ver.get("verdict"),
        "reason": ver.get("reason"),
        "counts": ver.get("counts"),
        "verify_json": ver.get("verify_json"),
    }


def _write_card(
    result: dict[str, Any],
    nb_path: Path,
    out_dir: Path,
    *,
    locked: int,
    locked_detail: str,
    refactored: bool,
) -> None:
    verdict = result.get("verdict", "")
    findings_caught = sum(
        1
        for f in result.get("findings", [])
        if f["category"] in CARD_CATEGORIES and f["severity"] in ("error", "warning")
    )
    card = write_card(
        out_dir,
        {
            "notebook": nb_path.name,
            "verdict": verdict,
            "locked": locked,
            "locked_detail": locked_detail,
            "findings": findings_caught,
            "refactored": refactored,
            "command": f"uvx notebook-to-pipeline pipeline {nb_path.name}",
        },
    )
    result["card"] = card
    result["findings_caught"] = findings_caught


# --------------------------------------------------------------------------
# One-screen report
# --------------------------------------------------------------------------

BADGE = {
    "equivalent": "EQUIVALENT",
    "differs": "NOT EQUIVALENT",
    "pipeline_failed": "NOT EQUIVALENT (pipeline failed)",
    "reference_invalid": "NO REFERENCE (notebook does not run top to bottom)",
    "inconclusive": "INCONCLUSIVE",
}


def _rel(path: str | None, base: Path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def format_report(result: dict[str, Any], max_findings: int = 8) -> str:
    nb = Path(result["notebook"]).name
    out_dir = Path(result["out"])
    verdict = result.get("verdict", "inconclusive")
    L: list[str] = []
    L.append(f"Reproducibility Report: {nb}")
    L.append("=" * min(60, len(f"Reproducibility Report: {nb}")))

    badge = BADGE.get(verdict, verdict.upper())
    if verdict == "equivalent":
        locked = result.get("locked", 0)
        detail = result.get("locked_detail", "")
        tail = f" - {locked} output{'s' if locked != 1 else ''} locked"
        if detail:
            tail += f" ({detail})"
        L.append(f"{badge}{tail}")
    elif verdict == "reference_invalid":
        L.append(badge)
        fc = result.get("failed_cell") or {}
        if fc:
            L.append(
                f"  failed at cell {fc.get('index', 0) + 1}: {fc.get('ename')}: {fc.get('evalue')}"
            )
        if result.get("hint"):
            L.append(f"  {result['hint']}")
    else:
        L.append(f"{badge} - {result.get('verify', {}).get('reason', '')}")

    findings = result.get("findings", [])
    L.append("")
    if findings:
        shown = findings[:max_findings]
        L.append(f"Findings ({len(findings)} total, ranked):")
        for f in shown:
            cells = (
                "  (cell " + ", ".join(str(c + 1) for c in f["cells"][:4]) + ")"
                if f["cells"]
                else ""
            )
            msg = f["message"]
            if len(msg) > 100:
                msg = msg[:99] + "…"
            L.append(f"  [{f['severity']}] {f['category']}: {msg}{cells}")
        if len(findings) > max_findings:
            L.append(f"  ... and {len(findings) - max_findings} more (see report.md)")
    else:
        L.append("Findings: none.")
    L.append("")

    if verdict != "reference_invalid":
        L.append(f"Pipeline:  {out_dir}")
        test_cmd = result.get("scaffold", {}).get("test_command", "pytest -q")
        L.append(f"Lock test: cd {out_dir.name} && {test_cmd}")
        if result.get("test_file"):
            L.append(f"           ({result['test_file']})")
        if result.get("card", {}).get("png"):
            L.append(f"Card:      {_rel(result['card']['png'], out_dir.parent)}")
        if result.get("report_md"):
            L.append(f"Report:    {_rel(result['report_md'], out_dir.parent)}")
        ref = result.get("refactor")
        if ref:
            L.append("")
            L.append(f"Refactor:  {ref.get('summary', '')}")
    return "\n".join(L)


def load_pipeline_result(notebook: str | Path) -> dict[str, Any]:
    """Best-effort reload of the last capture for a notebook (used by tests)."""
    ref_dir, cap = load_capture(notebook)
    return {"reference_dir": str(ref_dir), "capture": cap}
