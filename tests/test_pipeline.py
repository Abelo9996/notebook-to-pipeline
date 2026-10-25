from __future__ import annotations

from pathlib import Path

import pytest

from notebook_to_pipeline.pipeline import format_report, ranked_findings, run_pipeline

pytestmark = pytest.mark.filterwarnings("ignore")


def _deterministic_notebook(nb_factory):
    return nb_factory(
        [
            "import pandas as pd",
            "raw = pd.DataFrame({'region': ['n', 's', 'n'], 'units': [3, 5, 2]})",
            "by_region = raw.groupby('region')['units'].sum()",
            "total = int(by_region.sum())",
        ]
    )


def test_pipeline_end_to_end_equivalent(nb_factory):
    nb = _deterministic_notebook(nb_factory)
    out = Path(nb).parent / "proj"
    result = run_pipeline(nb, out=out, emit=lambda *_: None)
    assert result["verdict"] == "equivalent"
    assert result["locked"] >= 3
    assert (out / "src" / result["package"] / "pipeline.py").exists()
    assert (out / "tests" / "test_equivalence.py").exists()
    assert (out / "card.svg").exists() and (out / "card.png").exists()
    assert Path(result["report_md"]).exists()
    # The cleared notebook still gets a real finding.
    kinds = {f["kind"] for f in result["findings"]}
    assert "no_execution_history" in kinds


def test_pipeline_equivalent_when_a_name_is_redefined(nb_factory):
    # A name redefined from scratch after earlier cells read it must not be hoisted before them,
    # or the generated pipeline feeds the wrong value and fails or differs.
    nb = nb_factory(
        [
            "import numpy as np\nfrom sklearn.linear_model import LinearRegression",
            "X = np.array([[1.0], [2.0], [3.0]])\ny = [1.0, 2.0, 3.0]",
            "model = LinearRegression().fit(X, y)",
            "score = float(model.score(X, y))",
            "X = np.array([[42.0]])\ndemo = LinearRegression().fit(X, [42.0])",
        ]
    )
    out = Path(nb).parent / "proj"
    result = run_pipeline(nb, out=out, emit=lambda *_: None)
    assert result["verdict"] == "equivalent", result.get("verify")


def test_pipeline_equivalent_when_a_displayed_name_is_updated_later(nb_factory):
    # The value a cell displayed must be the one it saw in notebook order, even when a later cell
    # with an earlier-stage signal updates that name (s = s.astype(float)).
    nb = nb_factory(
        [
            "import pandas as pd",
            "s = pd.Series(['Snow', 'Rain', 'Snow']).str.contains('Snow')",
            "s[:2]",
            "s = s.astype(float)",
            "total = float(s.sum())",
        ]
    )
    out = Path(nb).parent / "proj"
    result = run_pipeline(nb, out=out, emit=lambda *_: None)
    assert result["verdict"] == "equivalent", result.get("verify")


def test_pipeline_reference_invalid_skips_scaffold(nb_factory):
    nb = nb_factory(["y = x * 2", "x = 3"])  # top-to-bottom NameError
    out = Path(nb).parent / "proj"
    result = run_pipeline(nb, out=out, emit=lambda *_: None)
    assert result["verdict"] == "reference_invalid"
    assert "package" not in result  # scaffold never ran
    assert (out / "card.svg").exists()
    assert "use_before_def" in {f["kind"] for f in result["findings"]}
    report = format_report(result)
    assert "NO REFERENCE" in report


def test_ranked_findings_orders_categories():
    analysis = {
        "findings": [
            {"kind": "unseeded_randomness", "severity": "warning", "message": "rng", "cells": [2]},
            {"kind": "shared_estimator", "severity": "warning", "message": "shared", "cells": [1]},
            {"kind": "use_before_def", "severity": "error", "message": "ubd", "cells": [0]},
        ],
        "inputs": {"imports": []},
    }
    cap = {"runtime_findings": [], "execution": {"cells": []}, "env": {"packages": {}}}
    ranked = ranked_findings(analysis, cap)
    cats = [f["category"] for f in ranked]
    assert cats == ["hidden state", "shared-object mutation", "nondeterminism"]


def test_ranked_findings_flags_unpinned_and_saved_output(nb_factory):
    analysis = {"findings": [], "inputs": {"imports": ["pandas", "os"]}}
    cap = {
        "runtime_findings": [],
        "execution": {
            "cells": [
                {"index": 1, "label": "cell 2", "saved_output": "different"},
            ]
        },
        "env": {"packages": {}},  # nothing pinned
    }
    ranked = ranked_findings(analysis, cap)
    kinds = {f["kind"] for f in ranked}
    assert "saved_output_changed" in kinds
    assert "unpinned_deps" in kinds
    unpinned = next(f for f in ranked if f["kind"] == "unpinned_deps")
    assert "pandas" in unpinned["message"] and "os" not in unpinned["message"]
