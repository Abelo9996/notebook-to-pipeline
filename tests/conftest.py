from __future__ import annotations

import sys
from pathlib import Path

import nbformat
import pytest


def make_notebook(path: Path, cells: list, *, executed: bool = False) -> Path:
    """Build a notebook. Each cell is a source string or (source, execution_count)."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for item in cells:
        if isinstance(item, tuple):
            src, ec = item
        else:
            src, ec = item, None
        if src.startswith("# md:"):
            nb.cells.append(nbformat.v4.new_markdown_cell(src[5:]))
            continue
        cell = nbformat.v4.new_code_cell(src)
        if ec is not None:
            cell.execution_count = ec
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="saved\n")]
        nb.cells.append(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


@pytest.fixture
def nb_factory(tmp_path):
    def factory(cells, name="nb.ipynb"):
        return make_notebook(tmp_path / name, cells)

    return factory


@pytest.fixture(autouse=True)
def _pin_python(monkeypatch):
    # Kernels and pipelines in tests run on the interpreter running pytest.
    monkeypatch.setenv("NB2P_PYTHON", sys.executable)
