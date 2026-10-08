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
from .execution import DISPLAYED_PREFIX, displayed_name
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


def _assign_last_expression(source: str, name: str) -> str | None:
    """Turn a cell's trailing expression into `name = <expression>`, so the value Jupyter
    displayed is kept and can be returned. None when the cell does not end with an expression."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return None
    node = tree.body[-1]
    lines = source.splitlines(keepends=True)
    raw = lines[node.lineno - 1].encode("utf-8")
    col = node.col_offset  # a byte offset
    lines[node.lineno - 1] = (raw[:col] + f"{name} = ".encode() + raw[col:]).decode("utf-8")
    return "".join(lines)


def detect_env(root: Path) -> tuple[str, str]:
    """Whether the project is managed with uv or with pip, and why we think so."""
    if (root / "uv.lock").exists():
        return "uv", "uv.lock found"
    for name in ("requirements.txt", "requirements-dev.txt"):
        if (root / name).exists():
            return "pip", f"{name} found and no uv.lock"
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="replace")
        if "[tool.uv" in text or "[dependency-groups]" in text:
            return "uv", "pyproject.toml has uv settings or dependency groups"
    for venv in (".venv", "venv"):
        cfg = root / venv / "pyvenv.cfg"
        if cfg.exists():
            if re.search(r"^uv\s*=", cfg.read_text(errors="replace"), re.M):
                return "uv", f"{venv}/ was created by uv"
            return "pip", f"{venv}/ was created without uv"
    if shutil.which("uv"):
        return "uv", "new project and uv is installed"
    return "pip", "new project and uv is not installed"


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
    env: str = "auto",
) -> dict[str, Any]:
    """Write the draft package, equivalence test, Makefile and CI into out.

    env is "uv", "pip" or "auto" (detected from uv.lock, requirements files, pyproject.toml and
    the virtualenv in out). It decides how the test's single dev dependency (pytest) is declared
    and which commands the Makefile and the CI workflow use."""
    if env not in ("auto", "uv", "pip"):
        raise ValueError(f"env must be auto, uv or pip, got {env!r}")
    nb_path = Path(notebook).resolve()
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pkg = _package_name(package or nb_path.stem)
    analysis = analyze(nb_path)
    cells = {c.index: c for c in load_cells(read_notebook(nb_path))}
    created: list[str] = []
    skipped: list[str] = []

    if env == "auto":
        env_kind, env_why = detect_env(root)
    else:
        env_kind, env_why = env, "chosen with env"

    versions: dict[str, str] = {}
    ref_dir = None
    py_version = None
    displayed: set[str] = set()
    if reference is not None or (nb_path.parent / ".nb2p").exists():
        try:
            ref_dir, cap = load_capture(reference or nb_path)
            versions = cap.get("env", {}).get("packages", {})
            py_version = cap.get("env", {}).get("python") or cap.get("python", {}).get("version")
            displayed = {
                a["name"]
                for a in cap.get("artifacts", [])
                if a.get("status") == "captured" and a["name"].startswith(DISPLAYED_PREFIX)
            }
        except FileNotFoundError:
            ref_dir = None
    mm = ".".join(str(py_version).split(".")[:2]) if py_version else None
    python_series = mm if mm and re.fullmatch(r"3\.\d+", mm) else "3.12"

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
        shown_here: list[str] = []
        for idx in st["cells"]:
            c = cells[idx]
            src = _strip_magics(c.source)
            shown = displayed_name(idx)
            if shown in displayed:
                kept = _assign_last_expression(src, shown)
                if kept is not None:
                    src = kept
                    shown_here.append(shown)
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
        returns = sorted((set(st["outputs"]) | finals) & defined_here) + shown_here
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

    _write(
        root / "tests" / "test_equivalence.py",
        _equivalence_test(pkg, ref_rel),
        force,
        created,
        skipped,
        root,
    )

    imports = analysis["inputs"]["imports"]
    if any(n.get("plots") for n in analysis["cells"]) and "matplotlib" not in imports:
        imports = [*imports, "matplotlib"]  # pandas .plot() draws with matplotlib
    deps = []
    for imp in imports:
        if imp in STDLIB_HINT or imp == pkg:
            continue
        dist = IMPORT_TO_DIST.get(imp, imp)
        if dist == "ipykernel":  # needed only to capture the notebook, not to run the pipeline
            continue
        ver = next((v for k, v in versions.items() if k.lower() == dist.lower()), None)
        deps.append(f"{dist}=={ver}" if ver else dist)

    pyproject_existed = (root / "pyproject.toml").exists()
    dep_lines = "".join(f'    "{d}",\n' for d in deps)
    pyproject = (
        "[project]\n"
        f'name = "{pkg.replace("_", "-")}"\n'
        'version = "0.1.0"\n'
        f'requires-python = ">={python_series}"\n'
        f"dependencies = [\n{dep_lines}]\n"
    )
    if env_kind == "uv":
        pyproject += '\n[dependency-groups]\ndev = ["pytest>=8"]\n'
    pyproject += textwrap.dedent(f"""
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{pkg}"]
        """)
    _write(root / "pyproject.toml", pyproject, force, created, skipped, root)

    requirements_existed = (root / "requirements.txt").exists()
    if env_kind == "pip":
        _write(
            root / "requirements.txt",
            "# Packages the notebook imports, pinned to the versions of the reference run.\n"
            + "".join(f"{d}\n" for d in deps),
            force,
            created,
            skipped,
            root,
        )
        _write(
            root / "requirements-dev.txt",
            textwrap.dedent("""\
                -r requirements.txt
                pytest>=8
                # tests/test_equivalence.py runs the pinned notebook-to-pipeline through uv.
                uv
                """),
            force,
            created,
            skipped,
            root,
        )

    try:
        nb_rel = os.path.relpath(nb_path, root)
    except ValueError:  # different drive on Windows
        nb_rel = str(nb_path)
    _write(
        root / "Makefile",
        _makefile(env_kind, nb_rel, pkg, ref_rel),
        force,
        created,
        skipped,
        root,
    )
    _write(
        root / ".gitignore",
        textwrap.dedent(f"""\
            # {ref_rel}/ is the reference the equivalence test needs: commit it.
            .nb2p/
            .nb2p-verify/
            .venv/
            __pycache__/
            .pytest_cache/
            """),
        force,
        created,
        skipped,
        root,
    )
    _write(
        root / ".github" / "workflows" / "equivalence.yml",
        _ci_workflow(env_kind, python_series),
        force,
        created,
        skipped,
        root,
    )

    if env_kind == "uv":
        install = "uv sync"
        test_cmd = "uv run pytest -q"
    else:
        install = "python -m pip install -r requirements-dev.txt"
        test_cmd = "python -m pytest -q"
    next_steps = [
        "Read each generated stage: it is the notebook code pasted into functions, not a finished refactor.",
    ]
    if env_kind == "uv" and pyproject_existed:
        next_steps.append(
            "pyproject.toml already existed and was kept. The test needs pytest and the notebook's "
            "packages in this project: `uv add --dev pytest`"
            + (f" and `uv add {' '.join(deps)}`" if deps else "")
            + " if they are not there yet."
        )
    if env_kind == "pip" and requirements_existed:
        next_steps.append(
            "requirements.txt already existed and was kept; check that it has the notebook's "
            f"packages ({', '.join(deps) or 'none detected'}). requirements-dev.txt includes it."
        )
    next_steps += [
        f"Install and run the equivalence test from {root}: `{install}`, then `{test_cmd}`. It runs "
        f"notebook-to-pipeline {__version__} through uv, so nothing else needs to be installed in "
        "the project. It must pass before you change anything.",
        "Refactor one stage at a time and run the test (or `make verify`) after each change.",
    ]
    if displayed:
        next_steps.append(
            f"{len(displayed)} value(s) the notebook only displayed are kept as "
            f"`{DISPLAYED_PREFIX}<n>` and compared. Keep returning them under those names, or tell "
            "the user which ones you dropped."
        )

    return {
        "out": str(root),
        "package": pkg,
        "stages": [s[0] for s in stage_funcs],
        "created": created,
        "skipped_existing": skipped,
        "reference_copied": ref_dir is not None,
        "env": {"kind": env_kind, "why": env_why},
        "install_command": install,
        "test_command": test_cmd,
        "displayed_values": sorted(displayed),
        "next_steps": next_steps,
        "notes": [
            n for s in analysis["stages"] for notes in s["reorder_notes"].values() for n in notes
        ],
        "analysis_findings": analysis["summary"]["findings"],
    }


def _equivalence_test(pkg: str, ref_rel: str) -> str:
    return textwrap.dedent(f'''\
        """Equivalence test: the pipeline must reproduce the notebook's captured outputs.

        Written by notebook-to-pipeline {__version__}. The comparison runs through uv
        (`uv tool run --from notebook-to-pipeline=={__version__} nb2p verify`), so this project
        needs only pytest, not notebook-to-pipeline or ipykernel. The pipeline itself runs in the
        interpreter running pytest, with this project's packages. Set NB2P_COMMAND to run nb2p
        another way, for example NB2P_COMMAND=nb2p when it is installed.
        """

        import json
        import os
        import shlex
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        import pytest

        NB2P_VERSION = "{__version__}"
        ROOT = Path(__file__).resolve().parents[1]
        REFERENCE = ROOT / "{ref_rel}"
        PIPELINE = "src/{pkg}/pipeline.py:run"


        def _uv():
            found = shutil.which("uv")
            if found:
                return found
            try:
                from uv import find_uv_bin  # the `uv` package from PyPI (pip install uv)

                return find_uv_bin()
            except Exception:
                return None


        def nb2p_command():
            custom = os.environ.get("NB2P_COMMAND")
            if custom:
                return shlex.split(custom)
            uv = _uv()
            if uv is None:
                return None
            return [uv, "tool", "run", "--from", "notebook-to-pipeline==" + NB2P_VERSION, "nb2p"]


        def _skip_or_fail(why):
            # A skipped equivalence test in CI would look like a pass, so CI fails instead.
            if os.environ.get("CI"):
                pytest.fail(why)
            pytest.skip(why)


        def test_pipeline_matches_notebook(tmp_path):
            if not (REFERENCE / "capture.json").exists():
                _skip_or_fail("no reference capture in {ref_rel}: run `make capture` first")
            cmd = nb2p_command()
            if cmd is None:
                _skip_or_fail(
                    "this test runs notebook-to-pipeline through uv, and uv was not found. Install "
                    "it (https://docs.astral.sh/uv/, or `pip install uv`) or set NB2P_COMMAND."
                )
            env = dict(os.environ)
            env.pop("UV_PYTHON", None)  # a Python pin for this project is not meant for the tool
            args = [
                "verify",
                "--pipeline", PIPELINE,
                "--reference", str(REFERENCE),
                "--python", sys.executable,
                "--cwd", str(ROOT),
                "--out", str(tmp_path),
            ]
            proc = subprocess.run(
                cmd + args, cwd=str(ROOT), env=env, capture_output=True, text=True
            )
            verify_json = tmp_path / "verify.json"
            assert verify_json.exists(), "nb2p did not run (exit %s): %s\\n%s\\n%s" % (
                proc.returncode,
                " ".join(cmd),
                proc.stdout[-3000:],
                proc.stderr[-3000:],
            )
            result = json.loads(verify_json.read_text(encoding="utf-8"))
            assert result["verdict"] == "equivalent", proc.stdout[-6000:]
        ''')


def _makefile(env_kind: str, nb_rel: str, pkg: str, ref_rel: str) -> str:
    if env_kind == "uv":
        python = '$(shell uv run python -c "import sys; print(sys.executable)")'
        test = "uv run pytest -q"
        capture_note = (
            "# capture needs ipykernel in the project environment: uv add --dev ipykernel"
        )
    else:
        python = "$(if $(wildcard .venv/bin/python),.venv/bin/python,python)"
        test = "$(PYTHON) -m pytest -q"
        capture_note = "# capture needs ipykernel in the project environment: pip install ipykernel"
    return textwrap.dedent(f"""\
        # nb2p runs through uvx, pinned to the version that wrote this file; the notebook and the
        # pipeline run in this project's Python. Override with: make verify NB2P=nb2p
        NOTEBOOK ?= {nb_rel}
        NB2P ?= uvx notebook-to-pipeline=={__version__}
        PYTHON ?= {python}

        .PHONY: capture verify test report

        {capture_note}
        capture:
        \t$(NB2P) capture "$(NOTEBOOK)" --python "$(PYTHON)" --out {ref_rel}

        verify:
        \t$(NB2P) verify --pipeline src/{pkg}/pipeline.py:run --reference {ref_rel} --python "$(PYTHON)" --out .nb2p-verify

        report: verify
        \t$(NB2P) report --reference {ref_rel} --verify .nb2p-verify/verify.json --out .nb2p-verify

        test:
        \t{test}
        """)


def _ci_workflow(env_kind: str, python_series: str) -> str:
    if env_kind == "uv":
        steps = f"""\
              - uses: astral-sh/setup-uv@v6
                with:
                  python-version: "{python_series}"
              - run: uv sync
              - run: uv run pytest -q
        """
    else:
        steps = f"""\
              - uses: actions/setup-python@v5
                with:
                  python-version: "{python_series}"
              - run: python -m pip install -r requirements-dev.txt
              - run: python -m pytest -q
        """
    head = textwrap.dedent("""\
        name: equivalence

        on: [push, pull_request]

        jobs:
          verify:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
        """)
    return head + textwrap.indent(textwrap.dedent(steps), "      ")
