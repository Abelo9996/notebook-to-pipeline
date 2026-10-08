"""notebook-to-pipeline: turn a notebook into a tested pipeline and prove the outputs didn't change."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analyze_notebook",
    "build_report",
    "capture_reference",
    "scaffold_pipeline",
    "verify_pipeline",
]


def analyze_notebook(notebook: Any) -> dict[str, Any]:
    """Static analysis of a notebook. See `nb2p analyze`."""
    from .analysis import analyze

    return analyze(notebook)


def capture_reference(notebook: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the notebook top to bottom in a fresh kernel and save reference outputs."""
    from .capture import capture

    return capture(notebook, **kwargs)


def verify_pipeline(pipeline: str | None, reference: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a pipeline and compare its artifacts with a reference capture."""
    from .verify import verify

    return verify(pipeline, reference, **kwargs)


def scaffold_pipeline(notebook: Any, out: Any, **kwargs: Any) -> dict[str, Any]:
    """Write a starting pipeline layout with an equivalence test and CI."""
    from .scaffold import scaffold

    return scaffold(notebook, out, **kwargs)


def build_report(reference: Any, verify_json: Any = None, out: Any = None) -> dict[str, Any]:
    """Write report.md and report.json."""
    from .report import build_report as _build

    return _build(reference, verify_json, out)
