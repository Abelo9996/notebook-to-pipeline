"""Figures drawn with matplotlib, PNG decoding, printed output and failure hints."""

from __future__ import annotations

import struct
import sys
import textwrap
import zlib
from pathlib import Path

import pytest

from notebook_to_pipeline.capture import capture
from notebook_to_pipeline.runtime import nb2p_compare
from notebook_to_pipeline.verify import verify

pytest.importorskip("matplotlib")

PLOT_CELLS = [
    "import matplotlib.pyplot as plt\nimport numpy as np",
    "x = np.linspace(0, 1, 20)\ny = x ** 2",
    "fig, ax = plt.subplots()\nax.plot(x, y)\nax.set_title('squares')\nplt.show()",
    "acc = float(y.mean())\nprint('Accuracy:', round(acc, 4))",
]


def _pipeline(body: str, root: Path, name: str) -> str:
    p = root / name
    p.write_text(textwrap.dedent(body))
    return f"{p}:run"


SAME = """
import matplotlib.pyplot as plt
import numpy as np


def run():
    x = np.linspace(0, 1, 20)
    y = x ** 2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("squares")
    fig.savefig("plot.png", dpi=150)  # a different dpi does not matter: figures are re-rendered
    plt.close(fig)
    acc = float(y.mean())
    print("Accuracy:", round(acc, 4))
    return {"x": x, "y": y, "acc": acc}
"""

CHANGED_PLOT = SAME.replace("ax.plot(x, y)", "ax.plot(x, y, color='red')")
NO_PLOT = """
import numpy as np


def run():
    x = np.linspace(0, 1, 20)
    y = x ** 2
    return {"x": x, "y": y, "acc": float(y.mean())}
"""


@pytest.fixture(scope="module")
def plotted(tmp_path_factory):
    import os

    os.environ.setdefault("NB2P_PYTHON", sys.executable)
    from conftest import make_notebook

    root = tmp_path_factory.mktemp("plots")
    nb = make_notebook(root / "plots.ipynb", PLOT_CELLS)
    return root, capture(nb)


def test_capture_records_figures_and_printed(plotted):
    _root, r = plotted
    assert r["execution"]["status"] == "ok"
    assert [f["label"] for f in r["figures"]] == ["squares"]
    assert r["figures"][0]["status"] == "saved"
    assert r["printed"] == {"lines": 1}
    ref = Path(r["reference_dir"])
    assert (ref / "figures" / "figure-01.png").exists()
    assert any("verify" in s for s in r["next_steps"])


def test_verify_same_figure_passes(plotted, tmp_path):
    root, r = plotted
    v = verify(_pipeline(SAME, root, "same.py"), r["reference_dir"], cwd=root, out=tmp_path)
    assert v["verdict"] == "equivalent", v["reason"]
    (fig,) = v["figures"]
    assert fig["status"] == "identical_pixels" and fig["passed"] is True
    assert v["printed"]["found"] == 1 and v["printed"]["lines"] == 1


def test_verify_changed_figure_differs(plotted, tmp_path):
    root, r = plotted
    v = verify(
        _pipeline(CHANGED_PLOT, root, "changed.py"), r["reference_dir"], cwd=root, out=tmp_path
    )
    assert v["verdict"] == "differs"
    (fig,) = v["figures"]
    assert fig["passed"] is False and "pixels differ" in fig["detail"]


def test_verify_without_plots_does_not_count_figures(plotted, tmp_path):
    root, r = plotted
    v = verify(_pipeline(NO_PLOT, root, "noplot.py"), r["reference_dir"], cwd=root, out=tmp_path)
    assert v["verdict"] == "equivalent"
    assert v["figures"][0]["status"] == "not_compared"
    assert "not counted" in v["reason"]
    assert v["printed"]["found"] == 0 and v["printed"]["missing"] == ["Accuracy: 0.3421"]
    assert any("drew none" in s for s in v["next_steps"])


def _png(path: Path, rows: list[bytes], width: int, filt: int = 0) -> None:
    raw = b"".join(bytes([filt]) + r for r in rows)

    def chunk(t: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body))

    ihdr = struct.pack(">IIBBBBB", width, len(rows), 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_png_compare_by_pixels(tmp_path):
    a, b, c, d = (tmp_path / f"{n}.png" for n in "abcd")
    rows = [bytes([10, 20, 30, 40, 50, 60]), bytes([70, 80, 90, 100, 110, 120])]
    _png(a, rows, 2)
    _png(b, rows, 2)
    b.write_bytes(b.read_bytes())  # same pixels
    _png(c, [rows[0], bytes([70, 80, 90, 100, 110, 121])], 2)
    _png(d, rows[:1], 2)
    assert nb2p_compare.compare_png(str(a), str(b))["passed"] is True
    diff = nb2p_compare.compare_png(str(a), str(c))
    assert diff["passed"] is False and "1 of 4 pixels differ" in diff["detail"]
    assert "x=1, y=1" in diff["detail"]
    assert "image size differs" in nb2p_compare.compare_png(str(a), str(d))["detail"]


def test_png_decoder_matches_matplotlib(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.scatter([1, 2, 3], [3, 1, 2], c=["r", "g", "b"])
    path = tmp_path / "m.png"
    fig.savefig(path, dpi=50)
    plt.close(fig)
    img = nb2p_compare.read_png(str(path))
    ours = np.frombuffer(b"".join(img["rows"]), dtype=np.uint8).reshape(
        img["height"], img["width"], -1
    )
    theirs = (plt.imread(path) * 255).round().astype(np.uint8)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_missing_module_hint(tmp_path):
    from conftest import make_notebook

    nb = make_notebook(tmp_path / "m.ipynb", ["import sklearn_not_here_xyz"])
    r = capture(nb)
    assert r["execution"]["status"] == "failed"
    assert "sklearn_not_here_xyz" in r["execution"]["hint"]
    assert r["next_steps"][0] == r["execution"]["hint"]
