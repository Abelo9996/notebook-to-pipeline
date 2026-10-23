from __future__ import annotations

from notebook_to_pipeline.card import build_svg, write_card


def test_build_svg_equivalent():
    svg = build_svg(
        {
            "notebook": "analysis.ipynb",
            "verdict": "equivalent",
            "locked": 12,
            "locked_detail": "9 variables, 3 figures",
            "findings": 3,
            "command": "uvx notebook-to-pipeline pipeline analysis.ipynb",
        }
    )
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "analysis.ipynb" in svg
    assert "EQUIVALENT" in svg and "NOT EQUIVALENT" not in svg
    assert "12 outputs locked" in svg
    assert "3 hidden-state issues caught" in svg


def test_build_svg_not_equivalent_and_singulars():
    svg = build_svg({"notebook": "x.ipynb", "verdict": "differs", "locked": 1, "findings": 1})
    assert "NOT EQUIVALENT" in svg
    assert "1 output locked" in svg
    assert "1 hidden-state issue caught" in svg


def test_build_svg_escapes_notebook_name():
    svg = build_svg({"notebook": "a&b<c>.ipynb", "verdict": "equivalent"})
    assert "a&amp;b&lt;c&gt;.ipynb" in svg
    assert "<c>" not in svg.replace("</svg>", "")


def test_write_card_writes_svg_and_png(tmp_path):
    r = write_card(
        tmp_path,
        {"notebook": "n.ipynb", "verdict": "equivalent", "locked": 5, "findings": 0},
    )
    assert (tmp_path / "card.svg").exists()
    assert "png_error" not in r, r.get("png_error")
    png = tmp_path / "card.png"
    assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
