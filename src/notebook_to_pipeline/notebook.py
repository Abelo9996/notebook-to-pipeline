"""Load notebooks and turn IPython cell source into plain Python for static analysis."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat

# Cell magics whose body is still Python and runs in the user namespace.
PYTHON_BODY_CELL_MAGICS = {"time", "timeit", "capture", "prun", "debug"}


@dataclass
class Magic:
    kind: str  # "line", "cell" or "shell"
    name: str
    args: str


@dataclass
class Cell:
    index: int  # position in nb.cells (counts markdown cells too)
    ordinal: int  # position among code cells, 1-based, as a human would count
    source: str
    python_source: str  # source after IPython transforms, may be "" if not Python
    execution_count: int | None
    has_outputs: bool
    magics: list[Magic] = field(default_factory=list)
    parse_error: str | None = None
    tree: ast.Module | None = None

    @property
    def is_empty(self) -> bool:
        """True for blank cells and cells holding only comments."""
        if not self.source.strip():
            return True
        return self.tree is not None and not self.tree.body


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_notebook(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _transformer():
    from IPython.core.inputtransformer2 import TransformerManager

    return TransformerManager()


def _magic_from_call(node: ast.Call) -> Magic | None:
    """Recognise get_ipython().run_line_magic(...) style calls produced by IPython."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)):
        return None
    inner = func.value.func
    if not (isinstance(inner, ast.Name) and inner.id == "get_ipython"):
        return None
    args = [a.value if isinstance(a, ast.Constant) else "" for a in node.args]
    if func.attr == "run_line_magic":
        return Magic("line", str(args[0]) if args else "", str(args[1]) if len(args) > 1 else "")
    if func.attr == "run_cell_magic":
        return Magic("cell", str(args[0]) if args else "", str(args[1]) if len(args) > 1 else "")
    if func.attr in {"system", "getoutput"}:
        return Magic("shell", "!", str(args[0]) if args else "")
    return None


def to_python(source: str) -> tuple[str, list[Magic], str | None]:
    """Return (python_source, magics, parse_error) for one cell's source."""
    if not source.strip():
        return "", [], None
    stripped = source.lstrip()
    magics: list[Magic] = []
    if stripped.startswith("%%"):
        first, _, body = stripped.partition("\n")
        name, _, args = first[2:].partition(" ")
        magics.append(Magic("cell", name.strip(), args.strip()))
        if name.strip() not in PYTHON_BODY_CELL_MAGICS:
            return "", magics, None
        source = body
    try:
        py = _transformer().transform_cell(source)
    except Exception as exc:  # IPython raises a variety of errors on odd input
        return "", magics, f"IPython transform failed: {exc}"
    try:
        tree = ast.parse(py)
    except SyntaxError as exc:
        return py, magics, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            m = _magic_from_call(node)
            if m is not None:
                magics.append(m)
    return py, magics, None


def load_cells(nb: Any) -> list[Cell]:
    cells: list[Cell] = []
    ordinal = 0
    for idx, raw in enumerate(nb.cells):
        if raw.cell_type != "code":
            continue
        ordinal += 1
        source = raw.source if isinstance(raw.source, str) else "".join(raw.source)
        py, magics, err = to_python(source)
        tree = None
        if py and err is None:
            tree = ast.parse(py)
        cells.append(
            Cell(
                index=idx,
                ordinal=ordinal,
                source=source,
                python_source=py,
                execution_count=raw.get("execution_count"),
                has_outputs=bool(raw.get("outputs")),
                magics=magics,
                parse_error=err,
                tree=tree,
            )
        )
    return cells
