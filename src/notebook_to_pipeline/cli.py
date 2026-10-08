"""Command line interface: nb2p (alias notebook-to-pipeline)."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .util import redact

EXIT_OK = 0
EXIT_DIFFERS = 1
EXIT_USAGE = 2
EXIT_FAILED = 3
EXIT_REFERENCE_INVALID = 4


def _cmdline() -> str:
    prog = Path(sys.argv[0]).name if sys.argv else "nb2p"
    if prog not in ("nb2p", "notebook-to-pipeline"):
        prog = "nb2p"
    return shlex.join([prog, *sys.argv[1:]])


def _rel(text: Any) -> str:
    """Show paths under the current directory as relative paths, and the home directory as ~."""
    s = str(text)
    cwd = str(Path.cwd())
    if s == cwd:
        return "."
    if cwd != "/":
        s = s.replace(cwd + "/", "")
    return redact(s)


def _print_json(data: Any) -> None:
    print(json.dumps(redact(data), indent=2, default=str))


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze
    from .util import write_json

    result = analyze(args.notebook)
    if args.out:
        write_json(Path(args.out), result)
    if args.json:
        _print_json(result)
    else:
        nb = result["notebook"]
        s = result["summary"]["findings"]
        print(f"{Path(nb['path']).name}: {nb['code_cells']} code cells, sha256 {nb['sha256'][:12]}")
        print(f"Findings: {s['error']} error, {s['warning']} warning, {s['info']} info")
        for f in result["findings"]:
            print(f"  [{f['severity']}] {f['kind']}: {f['message']}")
        print(
            "Proposed stages (a starting point for modules; cell numbers count every cell, markdown included):"
        )
        for k, st in enumerate(result["stages"], 1):
            cells = ", ".join(str(c + 1) for c in st["cells"])
            print(f"  {k}. {st['stage']}.py  {st['signature']}  (cells {cells})")
            if st["outputs"]:
                print(f"       returns: {', '.join(st['outputs'])}")
            for c in st["cells"]:
                why = st.get("why", {}).get(c) or st.get("why", {}).get(str(c))
                if why and why != "empty":
                    print(f"       cell {c + 1}: {why}")
            notes = st.get("reorder_notes", {})
            for c, ns in notes.items():
                for n in ns:
                    print(f"       careful, cell {int(c) + 1} {n}")
        arts = [a["name"] for a in result["suggested_artifacts"]]
        print(f"Suggested artifacts ({len(arts)}): {', '.join(arts)}")
        if args.out:
            print(f"Wrote {args.out}")
        _print_next(
            [
                f"Run it in a fresh kernel and record the reference: `nb2p capture {_rel(nb['path'])}`",
            ]
        )
    if args.strict and result["summary"]["findings"]["error"]:
        return EXIT_DIFFERS
    return EXIT_OK


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> int:
    from .capture import capture

    result = capture(
        args.notebook,
        out=args.out,
        python=args.python,
        cwd=args.cwd,
        timeout=args.timeout,
        artifacts=_csv(args.artifacts),
        all_globals=args.all_globals,
        repeat=args.repeat,
        command=_cmdline(),
    )
    ex = result["execution"]
    if args.json:
        _print_json(result)
    else:
        py = result["python"]
        print(
            f"Python {py.get('version', '?')}: {_rel(py.get('path'))} ({_rel(py.get('chosen_by'))})"
        )
        if ex["status"] == "ok":
            print(
                f"Top-to-bottom run: OK, {ex['cells_run']}/{ex['code_cells']} code cells in {ex['duration_s']} s"
            )
        else:
            fc = ex.get("failed_cell") or {}
            where = f" at {fc.get('label')}: {fc.get('ename')}: {fc.get('evalue')}" if fc else ""
            print(f"Top-to-bottom run: {ex['status'].upper()}{where}")
            if ex.get("error"):
                print(f"  {ex['error']}")
            pred = result.get("prediction")
            if pred and pred.get("predicted"):
                print(f"  Predicted by static analysis: {pred['finding']['message']}")
            if ex.get("hint"):
                print(f"  How to fix: {ex['hint']}")
        so = ex.get("saved_outputs")
        if so and (so["same"] or so["different"]):
            print(
                f"Saved outputs vs fresh run: {so['same']} same, {so['different']} different, {so['none']} cells had no saved output"
            )
            for c in ex["cells"]:
                if c.get("saved_output") == "different" and c.get("first_difference"):
                    d = c["first_difference"]
                    print(f"  {c.get('label')}: saved {d['saved']!r}, fresh {d['fresh']!r}")
        captured = [a for a in result["artifacts"] if a.get("status") == "captured"]
        skipped = [a for a in result["artifacts"] if a.get("status") != "captured"]
        if captured:
            print(f"Captured {len(captured)} artifacts:")
            for a in captured:
                stable = "" if a.get("stable", True) else "  (NOT stable across runs)"
                print(f"  {a['name']:<28} {a['kind']:<10} {a['hash'][:12]}{stable}")
        if skipped:
            print(
                "Not captured: "
                + ", ".join(f"{a['name']} ({a.get('reason', a.get('status'))})" for a in skipped)
            )
        for f in result.get("runtime_findings", []):
            print(f"[{f['severity']}] {f['kind']}: {f['message']}")
        st = result.get("stability")
        if st:
            unstable = [r["name"] for r in st["artifacts"] if not r["passed"]]
            if unstable:
                print(
                    f"Determinism check ({st['repeats']} runs): NOT stable: {', '.join(unstable)}"
                )
            else:
                print(f"Determinism check ({st['repeats']} runs): every artifact reproduced")
        if result.get("files_written"):
            print("Files written: " + ", ".join(f["path"] for f in result["files_written"]))
        figs = result.get("figures") or []
        if figs:
            names = ", ".join(
                f"{f['index']}" + (f" ({f['label']})" if f.get("label") else "") for f in figs
            )
            print(f"Figures recorded: {len(figs)} (rendered as PNG for comparison): {names}")
        printed = (result.get("printed") or {}).get("lines")
        if printed:
            print(f"Printed output: {printed} non-empty lines recorded")
        print(f"Reference: {_rel(result['reference_dir'])}")
        _print_next(result.get("next_steps"))
    return EXIT_OK if ex["status"] == "ok" else EXIT_FAILED


def _print_next(steps: list[str] | None) -> None:
    if steps:
        prog = Path(sys.argv[0]).name if sys.argv else "nb2p"
        print("Next:")
        for st in steps:
            st = _rel(st)
            if prog == "notebook-to-pipeline":
                st = st.replace("`nb2p ", "`notebook-to-pipeline ")
            print(f"  - {st}")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _parse_rename(values: list[str] | None) -> dict[str, str]:
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--rename expects REFERENCE_NAME=PIPELINE_NAME, got {v!r}")
        a, b = v.split("=", 1)
        out[a.strip()] = b.strip()
    return out


def print_verify_table(result: dict[str, Any]) -> None:
    rows = []
    for a in result.get("artifacts", []):
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[a.get("passed")]
        rows.append((a["name"], a.get("kind") or "", f"{mark} {a['status']}", a.get("detail", "")))
    for f in result.get("files", []):
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[f.get("passed")]
        rows.append((f["path"], "file", f"{mark} {f['status']}", f.get("detail", "")))
    for f in result.get("figures", []):
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[f.get("passed")]
        rows.append((f["name"], "figure", f"{mark} {f['status']}", f.get("detail", "")))
    if rows:
        w0 = min(max(len(r[0]) for r in rows), 48)
        w1 = max(len(r[1]) for r in rows)
        w2 = max(len(r[2]) for r in rows)
        print(f"{'artifact':<{w0}}  {'kind':<{w1}}  {'result':<{w2}}  detail")
        for r in rows:
            detail = r[3] if len(r[3]) <= 160 else r[3][:157] + "..."
            print(f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:<{w2}}  {detail}")


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import verify

    result = verify(
        args.pipeline,
        args.reference,
        cmd=args.cmd,
        python=args.python,
        cwd=args.cwd,
        out=args.out,
        artifacts=_csv(args.artifacts),
        rename=_parse_rename(args.rename),
        compare_files=not args.no_files,
        compare_figures=not args.no_figures,
        timeout=args.timeout,
        rtol=args.rtol,
        atol=args.atol,
        ignore_row_order=args.ignore_row_order,
        ignore_column_order=args.ignore_column_order,
        ignore_index=args.ignore_index,
        check_dtype=not args.no_check_dtype,
        command=_cmdline(),
    )
    if args.json:
        _print_json(result)
    else:
        print_verify_table(result)
        run = result.get("pipeline", {}).get("run") or {}
        if result["verdict"] == "pipeline_failed" and run.get("traceback"):
            print(run["traceback"].rstrip()[-2500:])
        elif result["verdict"] == "pipeline_failed" and run.get("stderr_tail"):
            print(run["stderr_tail"].rstrip()[-2500:])
        for a in result.get("artifacts", []):
            if a.get("passed") is False and a.get("differences"):
                print(f"First differences in {a['name']}:")
                for d in a["differences"]:
                    why = f"  ({d['why']})" if d.get("why") else ""
                    print(
                        f"  {d['path']}: reference {d['reference']}  candidate {d['candidate']}{why}"
                    )
        pr = result.get("printed") or {}
        if pr.get("lines"):
            line = f"Printed lines: {pr['found']} of {pr['lines']} the notebook printed also appear in the pipeline's output (not counted in the verdict)"
            print(line)
            for m in pr.get("missing", [])[:3]:
                print(f"  not printed by the pipeline: {m!r}")
        print(f"Verdict: {result['verdict'].upper()} ({result.get('reason', '')})")
        print(f"Evidence: {_rel(result['verify_json'])}")
        _print_next(result.get("next_steps"))
    return {
        "equivalent": EXIT_OK,
        "differs": EXIT_DIFFERS,
        "pipeline_failed": EXIT_FAILED,
        "reference_invalid": EXIT_REFERENCE_INVALID,
        "inconclusive": EXIT_DIFFERS,
    }[result["verdict"]]


# --------------------------------------------------------------------------
# scaffold, report, setup, mcp
# --------------------------------------------------------------------------


def cmd_scaffold(args: argparse.Namespace) -> int:
    from .scaffold import scaffold

    out = args.out or str(Path(args.notebook).resolve().parent)
    result = scaffold(
        args.notebook, out, package=args.package, reference=args.reference, force=args.force
    )
    if args.json:
        _print_json(result)
        return EXIT_OK
    print(f"Package {result['package']} with stages: {', '.join(result['stages'])}")
    for f in result["created"]:
        print(f"  created {f}")
    for f in result["skipped_existing"]:
        print(f"  kept existing {f} (use --force to overwrite)")
    if not result["reference_copied"]:
        print(
            "  no reference capture found: run `nb2p capture` first, then scaffold again or `make capture`"
        )
    for n in result["notes"]:
        print(f"  note: {n}")
    _print_next(result["next_steps"])
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report

    result = build_report(args.reference, args.verify, args.out)
    if args.json:
        _print_json(result)
    else:
        print(f"Verdict: {result['verdict']}")
        print(f"Wrote {_rel(result['report_md'])}")
        print(f"Wrote {_rel(result['report_json'])}")
    return EXIT_OK


def cmd_setup(args: argparse.Namespace) -> int:
    from .setup_agents import run_setup

    command = shlex.split(args.command) if args.command else None
    project = Path(args.project).resolve() if args.project else None
    if args.json:
        result = run_setup(yes=args.yes, command=command, project=project, out=lambda *_: None)
        _print_json(result)
    else:
        result = run_setup(yes=args.yes, command=command, project=project)
    return EXIT_OK if not any("error" in a for a in result["applied"]) else EXIT_FAILED


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    mcp_main()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nb2p",
        description="Turn a messy Jupyter notebook into a tested, reproducible pipeline, and prove the outputs didn't change.",
    )
    p.add_argument("--version", action="version", version=f"notebook-to-pipeline {__version__}")
    sub = p.add_subparsers(dest="command_name", required=True, metavar="COMMAND")

    a = sub.add_parser(
        "analyze",
        help="static analysis: dependency graph, hidden-state risks, proposed module split",
    )
    a.add_argument("notebook")
    a.add_argument("--out", "-o", help="also write the analysis JSON here")
    a.add_argument("--json", action="store_true", help="print JSON instead of text")
    a.add_argument("--strict", action="store_true", help="exit 1 if any error-level finding exists")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser(
        "capture",
        help="run the notebook top to bottom in a fresh kernel and record reference outputs",
    )
    c.add_argument("notebook")
    c.add_argument(
        "--out", "-o", help="capture directory (default: <notebook dir>/.nb2p/<name>/reference)"
    )
    c.add_argument(
        "--python",
        help="interpreter for the kernel (default: NB2P_PYTHON, a nearby .venv, or this one)",
    )
    c.add_argument(
        "--cwd", help="working directory for the kernel (default: the notebook's directory)"
    )
    c.add_argument(
        "--timeout", type=int, default=600, help="per-cell timeout in seconds (default 600)"
    )
    c.add_argument(
        "--artifacts",
        help="comma-separated variable names to capture (default: suggested by analyze)",
    )
    c.add_argument(
        "--all-globals", action="store_true", help="capture every data-like global instead"
    )
    c.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run N times and flag outputs that change between runs",
    )
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_capture)

    v = sub.add_parser(
        "verify", help="run the pipeline and compare its artifacts with the reference"
    )
    v.add_argument(
        "--pipeline",
        "-p",
        help="file.py:func, package.module:func, file.py or package.module (extra args allowed)",
    )
    v.add_argument(
        "--cmd", help="shell command instead of a Python pipeline (only written files are compared)"
    )
    v.add_argument(
        "--reference", "-r", required=True, help="capture directory, or the notebook path"
    )
    v.add_argument("--python", help="interpreter for the pipeline")
    v.add_argument("--cwd", help="working directory for the pipeline (default: current directory)")
    v.add_argument(
        "--out",
        "-o",
        help="where to write verify.json and candidate/ (default: next to the reference)",
    )
    v.add_argument("--artifacts", help="comma-separated subset of reference artifacts to compare")
    v.add_argument(
        "--rename",
        action="append",
        metavar="REF=NEW",
        help="the pipeline names this artifact differently",
    )
    v.add_argument("--rtol", type=float, help="relative tolerance for floats (default 1e-7)")
    v.add_argument("--atol", type=float, help="absolute tolerance for floats (default 1e-10)")
    v.add_argument(
        "--ignore-row-order", action="store_true", help="sort DataFrame rows before comparing"
    )
    v.add_argument("--ignore-column-order", action="store_true")
    v.add_argument(
        "--ignore-index", action="store_true", help="do not compare DataFrame/Series index labels"
    )
    v.add_argument(
        "--no-check-dtype", action="store_true", help="allow dtype changes if values match"
    )
    v.add_argument(
        "--no-files", action="store_true", help="do not compare files the notebook wrote"
    )
    v.add_argument(
        "--no-figures",
        action="store_true",
        help="do not compare the matplotlib figures the notebook drew",
    )
    v.add_argument("--timeout", type=int, default=1800)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser(
        "scaffold", help="write a starting pipeline layout with an equivalence test and CI"
    )
    s.add_argument("notebook")
    s.add_argument(
        "--out",
        "-o",
        help="project directory to create or fill (default: the notebook's directory; existing files are kept)",
    )
    s.add_argument("--package", help="Python package name (default: from the notebook name)")
    s.add_argument("--reference", help="capture directory to copy into tests/reference")
    s.add_argument("--force", action="store_true", help="overwrite existing files")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_scaffold)

    r = sub.add_parser("report", help="write report.md and report.json with the evidence")
    r.add_argument(
        "--reference", "-r", required=True, help="capture directory, or the notebook path"
    )
    r.add_argument("--verify", help="verify.json (default: next to the reference)")
    r.add_argument("--out", "-o", help="output directory (default: next to the reference)")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_report)

    st = sub.add_parser(
        "setup",
        help="register the MCP server with Claude Code, Codex and Cursor and install the skill",
    )
    st.add_argument("--yes", "-y", action="store_true", help="apply the plan without asking")
    st.add_argument(
        "--project",
        help="write a project .mcp.json in this directory instead of the user-level Claude Code config",
    )
    st.add_argument(
        "--command", help='server command to register (default: "uvx notebook-to-pipeline mcp")'
    )
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_setup)

    m = sub.add_parser("mcp", help="run the MCP server over stdio")
    m.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command_name == "verify" and not args.pipeline and not args.cmd:
        parser.error("verify needs --pipeline or --cmd")
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
