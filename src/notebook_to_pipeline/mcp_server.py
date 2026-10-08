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
    "did not change. Workflow: analyze_notebook, then capture_reference (runs the notebook top to "
    "bottom in a fresh kernel and saves reference outputs), then write the pipeline in small steps "
    "and call verify_pipeline after each step, then write_report. Never change logic to make outputs "
    "match without saying so."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

        return FastMCP


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
            "prediction": r.get("prediction"),
            "artifacts": [
                {k: a.get(k) for k in ("name", "status", "kind", "summary", "reason", "stable")}
                for a in r.get("artifacts", [])
            ],
            "files_written": r.get("files_written"),
            "runtime_findings": r.get("runtime_findings"),
            "saved_outputs": ex.get("saved_outputs"),
            "stability": r.get("stability"),
            "python": r.get("python"),
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
            "pipeline_run": {
                k: v
                for k, v in (r.get("pipeline", {}).get("run") or {}).items()
                if k in ("status", "error", "error_type", "traceback", "duration_s")
            },
            "verify_json": r.get("verify_json"),
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
        """Static analysis of a .ipynb: def/use graph, hidden-state findings (out-of-order execution,
        use before definition, names from deleted cells, cross-cell mutation, unseeded randomness),
        a proposed split into load/clean/features/train/evaluate/report and suggested artifacts.
        Nothing is executed."""
        return redact(await anyio.to_thread.run_sync(analyze, notebook))

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
        """Run the notebook top to bottom in a fresh kernel and save reference outputs (variables with
        hashes and summaries, plus files written). Reports whether top-to-bottom execution works.
        repeat=2 also checks that outputs are reproducible run to run."""

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
        timeout: int = 1800,
    ) -> dict[str, Any]:
        """Run the pipeline (file.py:func, module:func, file.py or module) and compare its artifacts
        with the reference capture. Verdict: equivalent, differs (with first differences),
        pipeline_failed, reference_invalid or inconclusive. rename maps reference name to pipeline name."""

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
                timeout=timeout,
                command=f"verify_pipeline(pipeline={pipeline!r}, reference={reference!r})",
            )

        return _compact_verify(await anyio.to_thread.run_sync(run))

    @server.tool()
    async def scaffold_pipeline(
        notebook: str,
        out: str,
        package: str | None = None,
        reference: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Write a starting layout: src/<package>/ with one module per proposed stage (notebook code pasted
        into functions), pipeline.py with run(), tests/test_equivalence.py, Makefile, pyproject.toml and a
        GitHub Actions workflow. Existing files are kept unless force=True."""

        def run() -> dict[str, Any]:
            return scaffold(notebook, out, package=package, reference=reference, force=force)

        return redact(await anyio.to_thread.run_sync(run))

    @server.tool()
    async def write_report(
        reference: str, verify_json: str | None = None, out: str | None = None
    ) -> dict[str, Any]:
        """Write report.md and report.json with the evidence: notebook hash, versions, top-to-bottom result,
        hidden-state findings, artifact comparisons, verdict and limits."""

        def run() -> dict[str, Any]:
            return build_report(reference, verify_json, out)

        r = await anyio.to_thread.run_sync(run)
        return redact(
            {"verdict": r["verdict"], "report_md": r["report_md"], "report_json": r["report_json"]}
        )

    return server


def main() -> None:
    build_server().run()
