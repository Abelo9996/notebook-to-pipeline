"""MCP server over stdio exposing the same operations as the CLI."""

from __future__ import annotations

from typing import Any

import anyio

from . import __version__
from .analysis import analyze
from .capture import capture
from .report import build_report
from .scaffold import scaffold
from .util import redact
from .verify import verify

INSTRUCTIONS = (
    "notebook-to-pipeline turns a Jupyter notebook into a tested pipeline and proves the outputs "
    "did not change. Workflow: analyze_notebook (static, nothing runs), then capture_reference "
    "(runs the notebook top to bottom in a fresh kernel and saves reference outputs), then write "
    "the pipeline as a function returning a dict of {artifact name: value} and call verify_pipeline "
    "after each change, then write_report. Every result has a next_steps list. Never change logic to "
    "make outputs match without telling the user. Paths may be absolute or relative to the server's "
    "working directory."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def _compact_analysis(r: dict[str, Any]) -> dict[str, Any]:
    """The parts an agent acts on. The full graph is in the JSON written by capture."""
    return redact(
        {
            "notebook": r.get("notebook"),
            "summary": r.get("summary"),
            "findings": r.get("findings"),
            "stages": [
                {
                    k: st.get(k)
                    for k in (
                        "stage",
                        "cells",
                        "signature",
                        "inputs",
                        "outputs",
                        "why",
                        "reorder_notes",
                    )
                }
                for st in r.get("stages", [])
            ],
            "suggested_artifacts": [a["name"] for a in r.get("suggested_artifacts", [])],
            "inputs": r.get("inputs"),
            "cell_numbers": "cells are numbered from 1 counting every cell, markdown included; "
            "the `cells` lists hold 0-based indexes",
            "next_steps": [
                "Call capture_reference on this notebook before changing anything. Pass artifacts=[...] "
                "to choose what to compare (final DataFrames, metrics, fitted models), or leave it "
                "empty to use suggested_artifacts."
            ],
        }
    )


def _compact_capture(r: dict[str, Any]) -> dict[str, Any]:
    ex = r.get("execution", {})
    return redact(
        {
            "reference_dir": r.get("reference_dir"),
            "status": ex.get("status"),
            "cells_run": ex.get("cells_run"),
            "code_cells": ex.get("code_cells"),
            "failed_cell": ex.get("failed_cell"),
            "error": ex.get("error"),
            "hint": ex.get("hint"),
            "prediction": r.get("prediction"),
            "artifacts": [
                {k: a.get(k) for k in ("name", "status", "kind", "summary", "reason", "stable")}
                for a in r.get("artifacts", [])
            ],
            "files_written": r.get("files_written"),
            "runtime_findings": r.get("runtime_findings"),
            "saved_outputs": ex.get("saved_outputs"),
            "stability": r.get("stability"),
            "figures": [
                {k: f.get(k) for k in ("index", "label", "status", "stable")}
                for f in r.get("figures", [])
            ],
            "printed_lines": (r.get("printed") or {}).get("lines", 0),
            "python": r.get("python"),
            "next_steps": r.get("next_steps"),
        }
    )


def _compact_verify(r: dict[str, Any]) -> dict[str, Any]:
    return redact(
        {
            "verdict": r.get("verdict"),
            "reason": r.get("reason"),
            "artifacts": [
                {k: a.get(k) for k in ("name", "status", "passed", "detail", "differences")}
                for a in r.get("artifacts", [])
            ],
            "files": r.get("files"),
            "figures": r.get("figures"),
            "printed": r.get("printed"),
            "pipeline_run": {
                k: v
                for k, v in (r.get("pipeline", {}).get("run") or {}).items()
                if k in ("status", "error", "error_type", "traceback", "duration_s")
            },
            "verify_json": r.get("verify_json"),
            "next_steps": r.get("next_steps"),
        }
    )


def build_server():
    Server = _server_class()
    try:
        server = Server(name="notebook-to-pipeline", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:  # older SDKs have no version argument
        server = Server(name="notebook-to-pipeline", instructions=INSTRUCTIONS)

    @server.tool()
    async def analyze_notebook(notebook: str) -> dict[str, Any]:
        """Step 1. Read a .ipynb without running it. Returns hidden-state findings (out-of-order
        execution, use before definition, names from deleted cells, cross-cell mutation, estimators
        shared between pipelines, unseeded randomness), a proposed split into
        load/clean/features/train/evaluate/report modules with each stage's inputs and outputs, and
        suggested artifacts to compare. Errors predict that a clean top-to-bottom run will fail.
        Next: capture_reference."""
        return _compact_analysis(await anyio.to_thread.run_sync(analyze, notebook))

    @server.tool()
    async def capture_reference(
        notebook: str,
        out: str | None = None,
        python: str | None = None,
        timeout: int = 600,
        artifacts: list[str] | None = None,
        all_globals: bool = False,
        repeat: int = 1,
    ) -> dict[str, Any]:
        """Step 2, before changing anything. Run the notebook top to bottom in a fresh kernel and save
        the reference: chosen variables (artifacts) with hashes and summaries, files it wrote, every
        matplotlib figure (rendered as PNG) and what it printed. Default artifacts are the notebook's
        top-level assignments; pass artifacts=[...] to choose. python defaults to NB2P_PYTHON or a
        .venv next to the notebook; pass the project's interpreter if the notebook needs packages.
        repeat=2 checks the outputs are reproducible run to run. If status is not "ok", read hint.
        Next: write the pipeline, then verify_pipeline with reference=reference_dir."""

        def run() -> dict[str, Any]:
            return capture(
                notebook,
                out=out,
                python=python,
                timeout=timeout,
                artifacts=artifacts,
                all_globals=all_globals,
                repeat=repeat,
                command=f"capture_reference(notebook={notebook!r})",
            )

        return _compact_capture(await anyio.to_thread.run_sync(run))

    @server.tool()
    async def verify_pipeline(
        reference: str,
        pipeline: str | None = None,
        cmd: str | None = None,
        cwd: str | None = None,
        python: str | None = None,
        artifacts: list[str] | None = None,
        rename: dict[str, str] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        ignore_row_order: bool = False,
        ignore_column_order: bool = False,
        ignore_index: bool = False,
        check_dtype: bool = True,
        compare_figures: bool = True,
        timeout: int = 1800,
    ) -> dict[str, Any]:
        """Step 3, after every change. Run the pipeline and compare it with the reference.
        pipeline is "path/to/file.py:run" (a function returning a dict {artifact name: value}),
        "package.module:run", or a script whose globals hold the results. reference is the
        reference_dir from capture_reference (or the notebook path). cwd is where the pipeline runs
        (default: the server's working directory); pass the project root. Compares artifacts (exact,
        floats within rtol/atol, DataFrames column by column), files the notebook wrote, and figures
        pixel by pixel when the pipeline draws them. Verdict: equivalent, differs (first differences
        listed), pipeline_failed (traceback), reference_invalid or inconclusive. rename maps a
        reference name to the pipeline's name. Do not loosen tolerances to get a pass."""

        def run() -> dict[str, Any]:
            return verify(
                pipeline,
                reference,
                cmd=cmd,
                cwd=cwd,
                python=python,
                artifacts=artifacts,
                rename=rename,
                rtol=rtol,
                atol=atol,
                ignore_row_order=ignore_row_order,
                ignore_column_order=ignore_column_order,
                ignore_index=ignore_index,
                check_dtype=check_dtype,
                compare_figures=compare_figures,
                timeout=timeout,
                command=f"verify_pipeline(pipeline={pipeline!r}, reference={reference!r})",
            )

        return _compact_verify(await anyio.to_thread.run_sync(run))

    @server.tool()
    async def scaffold_pipeline(
        notebook: str,
        out: str | None = None,
        package: str | None = None,
        reference: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Optional first draft. Writes src/<package>/ with one module per proposed stage (the
        notebook code pasted into functions, not a finished refactor), pipeline.py with run(),
        tests/test_equivalence.py, Makefile, pyproject.toml and a GitHub Actions workflow into out
        (default: the notebook's directory). Existing files are kept unless force=True. Run
        capture_reference first so the reference is copied into tests/reference.
        Next: verify_pipeline on the draft, then refactor one stage at a time."""

        def run() -> dict[str, Any]:
            from pathlib import Path

            target = out or str(Path(notebook).resolve().parent)
            return scaffold(notebook, target, package=package, reference=reference, force=force)

        return redact(await anyio.to_thread.run_sync(run))

    @server.tool()
    async def write_report(
        reference: str, verify_json: str | None = None, out: str | None = None
    ) -> dict[str, Any]:
        """Step 4. Write report.md and report.json with the evidence: notebook hash, versions,
        top-to-bottom result, hidden-state findings, artifact, file and figure comparisons, verdict
        and limits. reference is the reference_dir; verify_json defaults to the latest verify next
        to it. Give the user the verdict and the report path."""

        def run() -> dict[str, Any]:
            return build_report(reference, verify_json, out)

        r = await anyio.to_thread.run_sync(run)
        return redact(
            {"verdict": r["verdict"], "report_md": r["report_md"], "report_json": r["report_json"]}
        )

    return server


def main() -> None:
    build_server().run()
