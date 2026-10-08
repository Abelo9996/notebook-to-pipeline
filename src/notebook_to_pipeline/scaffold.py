"""Starting layout for the refactored pipeline: package, entry point, equivalence test and CI."""

from __future__ import annotations

import ast
import io
import os
import re
import shutil
import textwrap
import tokenize
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze
from .capture import load_capture
from .notebook import PYTHON_BODY_CELL_MAGICS, load_cells, read_notebook

IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
}
STDLIB_HINT = {
    "os",
    "sys",
    "re",
    "json",
    "math",
    "random",
    "time",
    "datetime",
    "pathlib",
    "collections",
    "itertools",
    "functools",
    "warnings",
    "pickle",
    "csv",
    "glob",
    "shutil",
    "subprocess",
    "typing",
    "string",
    "statistics",
    "urllib",
    "zipfile",
    "io",
    "sqlite3",
    "copy",
    "logging",
    "decimal",
    "fractions",
    "operator",
    "textwrap",
}


def _indent_code(code: str, prefix: str = "    ") -> str:
    """Indent code without changing the contents of multi-line string literals."""
    lines = code.splitlines(keepends=True)
    protected: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
                protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    out = []
    for i, line in enumerate(lines, start=1):
        if i in protected or not line.strip():
            out.append(line)
        else:
            out.append(prefix + line)
    return "".join(out)


def _notebook_imports(cells) -> list[tuple[str, str]]:
    """(bound name, import statement) for every top-level import in the notebook."""
    out: list[tuple[str, str]] = []
    for c in cells:
        if c.tree is None:
            continue
        for node in c.tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound = a.asname or a.name.split(".")[0]
                    out.append((bound, ast.unparse(ast.Import(names=[a]))))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    if a.name == "*":
                        continue
                    stmt = ast.ImportFrom(module=node.module, names=[a], level=0)
                    out.append((a.asname or a.name, ast.unparse(stmt)))
    return out


def _strip_magics(source: str) -> str:
    """Comment out IPython magics and shell escapes so the code runs as plain Python."""
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        name = lines[0].lstrip()[2:].split(" ")[0]
        if name in PYTHON_BODY_CELL_MAGICS:
            return "\n".join([f"# (nb2p removed cell magic) {lines[0]}", *lines[1:]])
        return "\n".join(f"# (nb2p removed cell magic body) {ln}" for ln in lines)
    out = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith(("%", "!")):
            indent = ln[: len(ln) - len(stripped)]
            out.append(f"{indent}pass  # (nb2p removed magic) {stripped}")
        else:
            out.append(ln)
    return "\n".join(out)


def _package_name(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    if not s or s[0].isdigit():
        s = "pipeline_" + s
    return s


def _write(
    path: Path, content: str, force: bool, created: list[str], skipped: list[str], root: Path
) -> None:
    rel = str(path.relative_to(root))
    if path.exists() and not force:
        skipped.append(rel)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    created.append(rel)


def scaffold(
    notebook: str | Path,
    out: str | Path,
    *,
    package: str | None = None,
    reference: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    nb_path = Path(notebook).resolve()
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pkg = _package_name(package or nb_path.stem)
    analysis = analyze(nb_path)
    cells = {c.index: c for c in load_cells(read_notebook(nb_path))}
    created: list[str] = []
    skipped: list[str] = []

    versions: dict[str, str] = {}
    ref_dir = None
    if reference is not None or (nb_path.parent / ".nb2p").exists():
        try:
            ref_dir, cap = load_capture(reference or nb_path)
            versions = cap.get("env", {}).get("packages", {})
        except FileNotFoundError:
            ref_dir = None

    import_lines = _notebook_imports(cells.values())
    stage_funcs = []
    for st in analysis["stages"]:
        name = st["stage"]
        used: set[str] = set()
        for idx in st["cells"]:
            node = next(n for n in analysis["cells"] if n["index"] == idx)
            used |= set(node["reads"]) | set(node["deferred_reads"])
        header_imports = [line for bound, line in import_lines if bound in used]
        body_parts = []
        plots_here = False
        for idx in st["cells"]:
            c = cells[idx]
            src = _strip_magics(c.source)
            node = next(n for n in analysis["cells"] if n["index"] == idx)
            part = f"# --- from notebook cell {idx + 1} ---\n{src.rstrip()}\n"
            if node.get("plots"):
                # Jupyter's inline backend shows and closes a cell's figures when the cell ends,
                # so the next cell's pandas .plot() starts a new figure instead of drawing on this one.
                plots_here = True
                part += '_plt.close("all")  # end of the notebook cell: Jupyter closed its figures here\n'
            body_parts.append(part)
        body = "\n".join(body_parts)
        defined_here: set[str] = set()
        for idx in st["cells"]:
            node = next(n for n in analysis["cells"] if n["index"] == idx)
            defined_here |= set(node["defines"])
        finals = {
            a["name"] for a in analysis["suggested_artifacts"] if a["defined_in"] in st["cells"]
        }
        returns = sorted((set(st["outputs"]) | finals) & defined_here)
        ret = "{" + ", ".join(f'"{o}": {o}' for o in returns) + "}"
        word = "Cell" if len(st["cells"]) == 1 else "Cells"
        doc = (
            f'"""{word} {", ".join(str(i + 1) for i in st["cells"])} of {nb_path.name}. '
            f'Inputs: {", ".join(st["inputs"]) or "none"}."""'
        )
        code = (
            f"def {name}({', '.join(st['inputs'])}):\n"
            f"    {doc}\n"
            f"{_indent_code(body)}\n"
            f"    return {ret}\n"
        )
        if plots_here:
            header_imports.append("import matplotlib.pyplot as _plt")
        if header_imports:
            code = "\n".join(dict.fromkeys(header_imports)) + "\n\n\n" + code
        stage_funcs.append((name, st, code))

    pkg_dir = root / "src" / pkg
    _write(
        pkg_dir / "__init__.py",
        f'"""Pipeline generated from {nb_path.name} by notebook-to-pipeline."""\n',
        force,
        created,
        skipped,
        root,
    )
    for name, _st, code in stage_funcs:
        header = (
            f'"""Stage `{name}`: mechanical first draft extracted from the notebook.\n\n'
            "Edit freely, then run `make verify` after every change.\n"
            '"""\n\n'
        )
        _write(pkg_dir / f"{name}.py", header + code, force, created, skipped, root)

    lines = [
        '"""Entry point: runs every stage in order and returns the artifacts to verify."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for name, _st, _code in stage_funcs:
        lines.append(f"from .{name} import {name}")
    lines += ["", "", "def run() -> dict:", "    ns: dict = {}"]
    for name, st, _code in stage_funcs:
        args = ", ".join(f'{i}=ns["{i}"]' for i in st["inputs"])
        lines.append(f"    ns.update({name}({args}))")
    lines += [
        "    return ns",
        "",
        "",
        'if __name__ == "__main__":',
        "    results = run()",
        "    for key, value in results.items():",
        '        print(f"{key}: {type(value).__name__}")',
        "",
    ]
    _write(pkg_dir / "pipeline.py", "\n".join(lines), force, created, skipped, root)

    ref_rel = "tests/reference"
    if ref_dir is not None:
        target = root / ref_rel
        if not target.exists() or force:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(ref_dir, target, ignore=shutil.ignore_patterns("files"))
            if (ref_dir / "files").exists():
                shutil.copytree(ref_dir / "files", target / "files")
            created.append(ref_rel + "/")
        else:
            skipped.append(ref_rel + "/")

    test = textwrap.dedent(f'''\
        """Equivalence test: the pipeline must reproduce the notebook's captured outputs."""

        from pathlib import Path

        import pytest
        from notebook_to_pipeline import verify_pipeline

        ROOT = Path(__file__).resolve().parents[1]
        REFERENCE = ROOT / "{ref_rel}"


        @pytest.mark.skipif(not (REFERENCE / "capture.json").exists(), reason="run `make capture` first")
        def test_pipeline_matches_notebook(tmp_path):
            result = verify_pipeline(
                "src/{pkg}/pipeline.py:run",
                REFERENCE,
                cwd=ROOT,
                out=tmp_path,
            )
            failing = [
                a
                for a in result["artifacts"] + result["files"] + result["figures"]
                if a.get("passed") is False
            ]
            assert result["verdict"] == "equivalent", (result.get("reason"), failing[:5])
        ''')
    _write(root / "tests" / "test_equivalence.py", test, force, created, skipped, root)

    imports = analysis["inputs"]["imports"]
    deps = []
    for imp in imports:
        if imp in STDLIB_HINT or imp == pkg:
            continue
        dist = IMPORT_TO_DIST.get(imp, imp)
        ver = next((v for k, v in versions.items() if k.lower() == dist.lower()), None)
        deps.append(f'"{dist}=={ver}"' if ver else f'"{dist}"')
    # ipykernel is needed only to capture the notebook, so it goes in the dev group.
    deps = [d for d in deps if not d.startswith('"ipykernel')]
    ipk = versions.get("ipykernel")
    dev_deps = ", ".join(
        [
            '"pytest>=8"',
            f'"notebook-to-pipeline>={__version__}"',
            f'"ipykernel=={ipk}"' if ipk else '"ipykernel"',
        ]
    )
    dep_lines = "".join(f"    {d},\n" for d in deps)
    pyproject = (
        "[project]\n"
        f'name = "{pkg.replace("_", "-")}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        f"dependencies = [\n{dep_lines}]\n"
    ) + textwrap.dedent(f"""

        [dependency-groups]
        dev = [{dev_deps}]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{pkg}"]
        """)
    _write(root / "pyproject.toml", pyproject, force, created, skipped, root)

    try:
        nb_rel = os.path.relpath(nb_path, root)
    except ValueError:  # different drive on Windows
        nb_rel = str(nb_path)
    makefile = textwrap.dedent(f"""\
        NOTEBOOK ?= {nb_rel}

        .PHONY: capture verify test report

        capture:
        \tuv run nb2p capture "$(NOTEBOOK)" --out {ref_rel}

        verify:
        \tuv run nb2p verify --pipeline src/{pkg}/pipeline.py:run --reference {ref_rel} --out .nb2p-verify

        report: verify
        \tuv run nb2p report --reference {ref_rel} --verify .nb2p-verify/verify.json --out .nb2p-verify

        test:
        \tuv run pytest -q
        """)
    _write(root / "Makefile", makefile, force, created, skipped, root)

    ci = textwrap.dedent("""\
        name: equivalence

        on: [push, pull_request]

        jobs:
          verify:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: astral-sh/setup-uv@v5
              - run: uv sync
              - run: uv run pytest -q
        """)
    _write(root / ".github" / "workflows" / "equivalence.yml", ci, force, created, skipped, root)

    return {
        "out": str(root),
        "package": pkg,
        "stages": [s[0] for s in stage_funcs],
        "created": created,
        "skipped_existing": skipped,
        "reference_copied": ref_dir is not None,
        "next_steps": [
            "Read each generated stage: it is the notebook code pasted into functions, not a finished refactor.",
            f"Run `nb2p verify --pipeline src/{pkg}/pipeline.py:run --reference {ref_rel}` from {root} (or `make verify`).",
            "Refactor one stage at a time and verify after each change.",
        ],
        "notes": [
            n for s in analysis["stages"] for notes in s["reorder_notes"].values() for n in notes
        ],
        "analysis_findings": analysis["summary"]["findings"],
    }
