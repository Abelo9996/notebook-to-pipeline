"""Evidence report: one Markdown file and one JSON file from analysis, capture and verify results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze
from .capture import load_capture, now
from .util import expand, redact, write_json

LIMITS = [
    "Static analysis reads cell source only. It does not follow `exec`, `eval`, `%run`, imports of local modules or mutation through aliases (`b = a; b.append(1)`).",
    "Mutation through notebook-defined functions is tracked one level deep; mutation inside third-party code is only known for common method names (`fit`, `append`, `inplace=True`, ...).",
    "Only the variables listed in the capture are compared. Anything the notebook displayed but did not keep in a variable is not compared.",
    "Figure files are listed but not compared. Plots are not compared at all.",
    "Equivalence is checked on this data, in this environment. A different input file or library version can still change the results.",
    "Tolerances apply to floats only. Integers, strings, booleans, dates and hashes must match exactly.",
]


def _md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _key_packages(env: dict[str, Any], imports: list[str]) -> dict[str, str]:
    pk = env.get("packages", {}) if env else {}
    alias = {
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "cv2": "opencv-python",
        "yaml": "PyYAML",
        "skimage": "scikit-image",
        "bs4": "beautifulsoup4",
    }
    out = {}
    lower = {k.lower(): (k, v) for k, v in pk.items()}
    for imp in imports:
        dist = alias.get(imp, imp).lower()
        if dist in lower:
            k, v = lower[dist]
            out[k] = v
    for extra in ("numpy", "pandas", "ipykernel"):
        if extra in lower:
            k, v = lower[extra]
            out.setdefault(k, v)
    return out


def build_report(
    reference: str | Path, verify_json: str | Path | None = None, out: str | Path | None = None
) -> dict[str, Any]:
    ref_dir, cap = load_capture(reference)
    work = ref_dir.parent
    analysis_path = ref_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
    else:
        analysis = analyze(expand(cap["notebook"]["path"]))
    vpath = Path(verify_json) if verify_json else work / "verify.json"
    ver = json.loads(vpath.read_text()) if vpath.exists() else None
    out_dir = Path(out).resolve() if out else work
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_now = expand(cap["notebook"]["path"])
    notebook_changed = None
    if nb_now.exists():
        from .notebook import sha256_file

        notebook_changed = sha256_file(nb_now) != cap["notebook"]["sha256"]

    verdict = (
        ver["verdict"]
        if ver
        else ("not_verified" if cap["execution"]["status"] == "ok" else "reference_invalid")
    )
    data = {
        "tool": {"name": "notebook-to-pipeline", "version": __version__},
        "kind": "report",
        "created_at": now(),
        "verdict": verdict,
        "verdict_reason": ver.get("reason") if ver else None,
        "notebook": {**cap["notebook"], "changed_since_capture": notebook_changed},
        "reference": {
            "dir": str(ref_dir),
            "command": cap.get("command"),
            "python": cap.get("python"),
            "execution": {k: v for k, v in cap["execution"].items() if k != "cells"},
            "saved_output_differences": [
                {"cell": c.get("label"), **c.get("first_difference", {})}
                for c in cap["execution"].get("cells", [])
                if c.get("saved_output") == "different"
            ],
            "prediction": cap.get("prediction"),
            "key_packages": _key_packages(
                cap.get("env", {}), analysis.get("inputs", {}).get("imports", [])
            ),
            "artifacts": [
                {
                    k: a.get(k)
                    for k in (
                        "name",
                        "status",
                        "kind",
                        "type",
                        "hash",
                        "summary",
                        "reason",
                        "stable",
                    )
                }
                for a in cap.get("artifacts", [])
            ],
            "files_written": cap.get("files_written", []),
            "stability": cap.get("stability"),
        },
        "hidden_state": {
            "summary": analysis["summary"]["findings"],
            "findings": analysis["findings"],
            "runtime_findings": cap.get("runtime_findings", []),
        },
        "proposed_stages": [
            {k: s[k] for k in ("stage", "cells", "inputs", "outputs", "signature")}
            for s in analysis["stages"]
        ],
        "verification": None,
        "limits": LIMITS,
    }
    if ver:
        data["verification"] = {
            "command": ver.get("command"),
            "pipeline": ver.get("pipeline"),
            "python": ver.get("python"),
            "options": ver.get("options"),
            "artifacts": ver.get("artifacts"),
            "files": ver.get("files"),
            "counts": ver.get("counts"),
            "key_packages": _key_packages(
                ver.get("candidate_env", {}), analysis.get("inputs", {}).get("imports", [])
            ),
            "created_at": ver.get("created_at"),
        }
    md = render_markdown(data)
    write_json(out_dir / "report.json", data)
    (out_dir / "report.md").write_text(redact(md), encoding="utf-8")
    data["report_md"] = str(out_dir / "report.md")
    data["report_json"] = str(out_dir / "report.json")
    return data


VERDICT_LINE = {
    "equivalent": "EQUIVALENT: the pipeline reproduces every compared notebook output.",
    "differs": "DIFFERS: at least one compared output changed.",
    "pipeline_failed": "PIPELINE FAILED: the pipeline did not run to completion.",
    "reference_invalid": "REFERENCE INVALID: the notebook does not run top to bottom, so there is nothing trustworthy to compare against.",
    "inconclusive": "INCONCLUSIVE: nothing could be compared.",
    "not_verified": "NOT VERIFIED YET: reference captured, no pipeline has been verified against it.",
}


def _short_path(p: str, base: Path | None = None) -> str:
    if base is not None:
        try:
            return str(Path(p).resolve().relative_to(base))
        except ValueError:
            pass
    return p


def render_markdown(d: dict[str, Any]) -> str:
    L: list[str] = []
    nb = d["notebook"]
    base = Path(nb["path"]).parent
    L.append(f"# notebook-to-pipeline report: {Path(nb['path']).name}")
    L.append("")
    L.append(f"**Verdict: {VERDICT_LINE.get(d['verdict'], d['verdict'])}**")
    if d.get("verdict_reason"):
        L.append("")
        L.append(f"Reason: {d['verdict_reason'].rstrip('.')}.")
    L.append("")
    L.append(f"Generated {d['created_at']} by notebook-to-pipeline {d['tool']['version']}.")
    L.append("")

    L.append("## Notebook")
    L.append("")
    L.append(f"- File: `{Path(nb['path']).name}`")
    L.append(
        f"- sha256: `{nb['sha256']}`"
        + (" (the file has changed since this capture)" if nb.get("changed_since_capture") else "")
    )
    L.append(f"- Cells: {nb['cells_total']} total, {nb['code_cells']} code")
    L.append("")

    ref = d["reference"]
    ex = ref["execution"]
    L.append("## Top-to-bottom run in a fresh kernel")
    L.append("")
    py = ref.get("python") or {}
    L.append(f"- Command: `{ref.get('command')}`")
    L.append(
        f"- Python {py.get('version', '?')} ({py.get('chosen_by', '')}), ipykernel {py.get('ipykernel', '?')}"
    )
    if ref.get("key_packages"):
        L.append("- Packages: " + ", ".join(f"{k} {v}" for k, v in ref["key_packages"].items()))
    if ex["status"] == "ok":
        L.append(
            f"- Result: **ran to completion**, {ex.get('cells_run')} of {ex.get('code_cells')} code cells in {ex.get('duration_s')} s"
        )
    else:
        fc = ex.get("failed_cell") or {}
        L.append(
            f"- Result: **{ex['status']}**"
            + (
                f" at {fc.get('label', '')}: `{fc.get('ename')}: {_md_escape(fc.get('evalue'))}`"
                if fc
                else ""
            )
        )
        if fc.get("source_head"):
            L.append("")
            L.append("  Failing cell starts with:")
            L.append("")
            L.append("  ```python")
            for ln in fc["source_head"].splitlines():
                L.append(f"  {ln}")
            L.append("  ```")
        pred = ref.get("prediction")
        if pred and pred.get("predicted"):
            L.append(
                f"- Static analysis predicted this failure before running anything: {pred['finding']['message']}"
            )
        elif pred:
            L.append(f"- {pred.get('note', '')}")
        if ex.get("error"):
            L.append(f"- Error: {_md_escape(ex['error'])}")
    L.append("")

    so = ex.get("saved_outputs")
    if so and (so["same"] or so["different"]):
        L.append("### Saved outputs vs fresh run")
        L.append("")
        L.append(
            f"Text outputs saved in the notebook were compared with the fresh run: {so['same']} cell(s) same, "
            f"{so['different']} different, {so['none']} with no saved text output. Differences can come from "
            "hidden state or from different library versions."
        )
        L.append("")
        diffs = ref.get("saved_output_differences") or []
        if diffs:
            L.append("| Cell | First differing line (saved) | Fresh run |")
            L.append("|---|---|---|")
            for dd in diffs:
                L.append(
                    f"| {dd.get('cell')} | `{_md_escape(dd.get('saved', ''))}` | `{_md_escape(dd.get('fresh', ''))}` |"
                )
            L.append("")

    hs = d["hidden_state"]
    L.append("## Hidden-state findings")
    L.append("")
    s = hs["summary"]
    L.append(
        f"{s['error']} error(s), {s['warning']} warning(s), {s['info']} info. Cell numbers count every cell from the top, markdown included; `In [n]` is the saved execution count."
    )
    L.append("")
    if hs["findings"]:
        L.append("| Severity | Kind | Finding |")
        L.append("|---|---|---|")
        for f in hs["findings"]:
            L.append(f"| {f['severity']} | {f['kind']} | {_md_escape(f['message'])} |")
        L.append("")

    rf = hs.get("runtime_findings") or []
    if rf:
        L.append("Found during the fresh run (not visible in the source alone):")
        L.append("")
        L.append("| Severity | Kind | Finding |")
        L.append("|---|---|---|")
        for f in rf:
            L.append(f"| {f['severity']} | {f['kind']} | {_md_escape(f['message'])} |")
        L.append("")

    if ref["artifacts"]:
        L.append("## Reference artifacts")
        L.append("")
        L.append("| Name | Kind | Summary | sha256 (first 12) |")
        L.append("|---|---|---|---|")
        for a in ref["artifacts"]:
            if a.get("status") != "captured":
                L.append(
                    f"| `{a['name']}` | {a.get('status')} | {_md_escape(a.get('reason', ''))} | |"
                )
                continue
            summ = a.get("summary") or {}
            if a["kind"] == "dataframe":
                text = f"{summ.get('shape', ['?', '?'])[0]} rows x {summ.get('shape', ['?', '?'])[1]} cols"
            elif a["kind"] in ("series", "index"):
                text = f"length {summ.get('length')}, {summ.get('dtype')}"
            elif a["kind"] == "ndarray":
                text = f"shape {tuple(summ.get('shape', []))}, {summ.get('dtype')}"
            elif a["kind"] == "scalar":
                text = f"{summ.get('value')}"
            elif a["kind"] == "estimator":
                text = f"{summ.get('class', '').split('.')[-1]}, {len(summ.get('fitted_attributes', []))} fitted attrs"
            else:
                text = summ.get("preview") or summ.get("type", "")
            if a.get("stable") is False:
                text += " (NOT stable across runs)"
            L.append(
                f"| `{a['name']}` | {a['kind']} | {_md_escape(text)[:100]} | `{(a.get('hash') or '')[:12]}` |"
            )
        L.append("")
    if ref.get("stability"):
        st = ref["stability"]
        unstable = [r["name"] for r in st["artifacts"] if not r["passed"]]
        L.append(
            f"Determinism check: the notebook was run {st['repeats']} times. "
            + (
                f"Unstable artifacts: {', '.join(f'`{u}`' for u in unstable)}."
                if unstable
                else "Every artifact was reproduced."
            )
        )
        L.append("")

    v = d.get("verification")
    if v:
        L.append("## Pipeline verification")
        L.append("")
        L.append(f"- Command: `{v.get('command')}`")
        run = (v.get("pipeline") or {}).get("run", {})
        L.append(f"- Pipeline run: {run.get('status')} in {run.get('duration_s')} s")
        if run.get("status") != "ok" and run.get("traceback"):
            L.append("")
            L.append("```")
            L.append(run["traceback"][-1500:].rstrip())
            L.append("```")
        if v.get("key_packages"):
            L.append(
                "- Packages: " + ", ".join(f"{k} {val}" for k, val in v["key_packages"].items())
            )
        o = v.get("options") or {}
        L.append(
            f"- Tolerance: rtol={o.get('rtol')}, atol={o.get('atol')}; ignore row order: {o.get('ignore_row_order')}, ignore column order: {o.get('ignore_column_order')}, ignore index: {o.get('ignore_index')}, check dtype: {o.get('check_dtype')}"
        )
        L.append("")
        if v.get("artifacts"):
            L.append("| Artifact | Kind | Result | Detail |")
            L.append("|---|---|---|---|")
            for a in v["artifacts"]:
                mark = {True: "PASS", False: "FAIL", None: "n/a"}[a.get("passed")]
                L.append(
                    f"| `{a['name']}` | {a.get('kind') or ''} | {mark} ({a['status']}) | {_md_escape(a.get('detail', ''))} |"
                )
            L.append("")
            for a in v["artifacts"]:
                if a.get("passed") is False and a.get("differences"):
                    L.append(f"First differences in `{a['name']}`:")
                    L.append("")
                    for diff in a["differences"]:
                        why = f" ({diff['why']})" if diff.get("why") else ""
                        L.append(
                            f"- `{_md_escape(diff['path'])}`: reference `{_md_escape(diff['reference'])}`, candidate `{_md_escape(diff['candidate'])}`{why}"
                        )
                    L.append("")
        if v.get("files"):
            L.append("| File | Result | Detail |")
            L.append("|---|---|---|")
            for f in v["files"]:
                mark = {True: "PASS", False: "FAIL", None: "n/a"}[f.get("passed")]
                L.append(
                    f"| `{f['path']}` | {mark} ({f['status']}) | {_md_escape(f.get('detail', ''))} |"
                )
            L.append("")

    if ref.get("files_written"):
        L.append("## Files the notebook wrote")
        L.append("")
        for f in ref["files_written"]:
            L.append(
                f"- `{f['path']}` ({f['kind']}, {f['bytes']} bytes, sha256 `{f['sha256'][:12]}`)"
            )
        L.append("")

    if d.get("proposed_stages"):
        L.append("## Proposed module split")
        L.append("")
        L.append("| Stage | Cells | Inputs | Outputs |")
        L.append("|---|---|---|---|")
        for st in d["proposed_stages"]:
            cells = ", ".join(str(c + 1) for c in st["cells"])
            L.append(
                f"| {st['stage']} | {cells} | {', '.join(st['inputs']) or '-'} | {', '.join(st['outputs']) or '-'} |"
            )
        L.append("")

    L.append("## Limits")
    L.append("")
    for lim in d["limits"]:
        L.append(f"- {lim}")
    L.append("")
    del base
    return "\n".join(L)
