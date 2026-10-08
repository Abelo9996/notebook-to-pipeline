"""Interpreter resolution, fresh-kernel notebook execution and file-change tracking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).parent / "runtime"
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".nb2p",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}
FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".webp", ".eps"}
DATA_EXTS = {
    ".csv",
    ".tsv",
    ".json",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".txt",
    ".xlsx",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
}
MAX_COPY_BYTES = 50 * 1024 * 1024
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text or "")


def _venv_python(d: Path) -> Path | None:
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe", "venv/bin/python"):
        p = d / rel
        if p.exists():
            return p
    return None


def resolve_python(python: str | None, start: Path) -> tuple[str, str]:
    """Pick the interpreter that runs the notebook or pipeline, and say why."""
    if python:
        return python, "--python"
    env = os.environ.get("NB2P_PYTHON")
    if env:
        return env, "NB2P_PYTHON"
    start = start.resolve()
    for d in [start, *start.parents]:
        found = _venv_python(d)
        if found is not None:
            return str(found), f"virtualenv found at {found.parent.parent}"
        if (d / ".git").exists():
            break
    return sys.executable, "the interpreter running nb2p"


def python_info(python: str) -> dict[str, Any]:
    code = (
        "import sys, json\n"
        "info = {'version': sys.version.split()[0], 'executable': sys.executable}\n"
        "try:\n"
        "    import ipykernel; info['ipykernel'] = ipykernel.__version__\n"
        "except Exception as e:\n"
        "    info['ipykernel'] = None\n"
        "print(json.dumps(info))\n"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"cannot run {python}: {exc}"}
    if out.returncode != 0:
        return {"error": f"{python} exited with {out.returncode}: {out.stderr.strip()[-500:]}"}
    return json.loads(out.stdout.strip().splitlines()[-1])


def run_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    # matplotlib (and other tools that follow reproducible-builds conventions) then write a
    # fixed date into PDF, SVG and PS files instead of the current time.
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    env["MPLBACKEND"] = "Agg"
    env.pop("PYTHONSTARTUP", None)
    return env


def kernel_env() -> dict[str, str]:
    """Like run_env, but matplotlib keeps Jupyter's default inline backend, which shows and
    closes each cell's figures at the end of the cell. Forcing Agg here would let later cells
    draw onto earlier figures, which is not what the notebook does in Jupyter."""
    env = run_env()
    env.pop("MPLBACKEND", None)
    return env


# --------------------------------------------------------------------------
# File tracking
# --------------------------------------------------------------------------


def snapshot(
    root: Path, limit: int = 50000, exclude: list[Path] | None = None
) -> dict[str, tuple[int, int]]:
    """Size and mtime of every file under root, skipping VCS, virtualenv, cache and excluded dirs."""
    out: dict[str, tuple[int, int]] = {}
    root = root.resolve()
    excluded = {p.resolve() for p in exclude or []}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and (Path(dirpath) / d).resolve() not in excluded
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
            if len(out) >= limit:
                return out
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_kind(rel: str) -> str:
    ext = Path(rel).suffix.lower()
    if ext in FIGURE_EXTS:
        return "figure"
    if ext in DATA_EXTS:
        return "data"
    return "other"


def collect_written(root: Path, before: dict, after: dict, dest: Path) -> list[dict[str, Any]]:
    changed = sorted(rel for rel, meta in after.items() if before.get(rel) != meta)
    out = []
    for rel in changed:
        src = root / rel
        try:
            size = src.stat().st_size
            digest = _sha256(src)
        except OSError:
            continue
        copied = False
        if size <= MAX_COPY_BYTES:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied = True
        out.append(
            {
                "path": rel,
                "sha256": digest,
                "bytes": size,
                "new": rel not in before,
                "kind": file_kind(rel),
                "copied": copied,
            }
        )
    return out


# --------------------------------------------------------------------------
# Notebook execution in a fresh kernel
# --------------------------------------------------------------------------


_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def text_outputs(cell: Any) -> str | None:
    """Plain-text rendering of a cell's outputs (stream text and text/plain), or None if none."""
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            if out.get("name") == "stdout":  # stderr carries warnings that vary by version
                parts.append(out.get("text", ""))
        elif out.get("output_type") in ("execute_result", "display_data"):
            txt = out.get("data", {}).get("text/plain")
            if txt:
                parts.append(txt if isinstance(txt, str) else "".join(txt))
        elif out.get("output_type") == "error":
            parts.append(f"{out.get('ename')}: {out.get('evalue')}")
    if not parts:
        return None
    text = "\n".join(p.rstrip() for p in parts)
    return _ADDR.sub(" at 0x...", text).strip()


def compare_saved_output(saved: str | None, fresh: str | None) -> dict[str, Any]:
    if saved is None:
        return {"saved_output": "none"}
    if fresh == saved:
        return {"saved_output": "same"}
    s_lines = saved.splitlines()
    f_lines = (fresh or "").splitlines()
    for i in range(max(len(s_lines), len(f_lines))):
        a = s_lines[i] if i < len(s_lines) else "<no line>"
        b = f_lines[i] if i < len(f_lines) else "<no line>"
        if a != b:
            return {
                "saved_output": "different",
                "first_difference": {"line": i + 1, "saved": a[:200], "fresh": b[:200]},
            }
    return {"saved_output": "different"}


def printed_text(cell: Any) -> str | None:
    """What the cell printed to stdout (print() and friends), without reprs or displays."""
    parts = [
        out.get("text", "")
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    ]
    text = "".join(p if isinstance(p, str) else "".join(p) for p in parts)
    return text if text.strip() else None


def inline_images(cell: Any) -> int:
    return sum(
        1
        for out in cell.get("outputs", [])
        if out.get("output_type") in ("execute_result", "display_data")
        and any(k.startswith("image/") for k in out.get("data", {}))
    )


def _figures_hook_code(figdir: Path) -> str:
    src = (RUNTIME_DIR / "nb2p_figures.py").read_text(encoding="utf-8")
    return (
        "def __nb2p_figs():\n"
        "    import sys as _s, types as _t\n"
        "    _m = _t.ModuleType('nb2p_figures')\n"
        f"    exec(compile({src!r}, 'nb2p_figures.py', 'exec'), _m.__dict__)\n"
        "    _s.modules['nb2p_figures'] = _m\n"
        f"    _m.install({str(figdir)!r})\n"
        "__nb2p_figs(); del __nb2p_figs\n"
    )


_FIGURES_FINISH = (
    "def __nb2p_figs_done():\n"
    "    import sys as _s\n"
    "    _m = _s.modules.get('nb2p_figures')\n"
    "    if _m is not None:\n"
    "        _m.finish()\n"
    "__nb2p_figs_done(); del __nb2p_figs_done\n"
)


DISPLAYED_PREFIX = "displayed_cell_"


def displayed_name(cell_index: int) -> str:
    """Artifact name for the value a cell displayed as its last expression (1-based cell number)."""
    return f"{DISPLAYED_PREFIX}{cell_index + 1}"


def _probe_code(
    names: list[str] | None, outdir: Path, max_bytes: int, displayed: dict[str, int] | None = None
) -> str:
    """Code for the hidden cell that saves the artifacts.

    displayed maps an artifact name to the execution count of a cell whose last expression was
    displayed. Those values live only in IPython's Out history, so they are added to the namespace
    under that name, unless the value is the very object a variable holds (that variable is
    captured or not on its own terms) or holds plot objects (`plt.plot` returns a list of lines,
    `df.plot(subplots=True)` an array of Axes)."""
    src = (RUNTIME_DIR / "nb2p_probe.py").read_text(encoding="utf-8")
    return (
        "def __nb2p_run():\n"
        "    import types as _t\n"
        "    _m = _t.ModuleType('nb2p_probe')\n"
        f"    exec(compile({src!r}, 'nb2p_probe.py', 'exec'), _m.__dict__)\n"
        "    _g = globals()\n"
        f"    _names = {names!r}\n"
        f"    _shown = {displayed or {}!r}\n"
        "    _ns = _g\n"
        "    if _shown:\n"
        "        _out = _g.get('Out') or {}\n"
        "        _held = [v for k, v in _g.items() if not k.startswith('_') and k not in ('In', 'Out')]\n"
        "        _ns = dict(_g)\n"
        "        def _plotty(v, depth=0):\n"
        "            if _m.is_plot_object(v):\n"
        "                return True\n"
        "            if depth < 2 and isinstance(v, (list, tuple)):\n"
        "                return any(_plotty(x, depth + 1) for x in v[:100])\n"
        "            if depth < 2 and _m.is_ndarray(v) and v.dtype.kind == 'O':\n"
        "                return any(_plotty(x, depth + 1) for x in v.flat[:100])\n"
        "            return False\n"
        "        for _name, _count in _shown.items():\n"
        "            if _count not in _out:\n"
        "                continue\n"
        "            _val = _out[_count]\n"
        "            if any(_val is _v for _v in _held) or _plotty(_val):\n"
        "                continue\n"
        "            _ns[_name] = _val\n"
        "            if _names is not None:\n"
        "                _names = list(_names) + [_name]\n"
        f"    _m.dump(_ns, _names, {str(outdir)!r}, {max_bytes!r})\n"
        "__nb2p_run(); del __nb2p_run\n"
    )


def _cell_head(source: str) -> str:
    lines = [ln for ln in source.strip().splitlines() if ln.strip()]
    return "\n".join(lines[:3])[:300]


def execute_notebook(
    nb: Any,
    *,
    python: str,
    cwd: Path,
    timeout: int,
    names: list[str] | None,
    outdir: Path,
    max_bytes: int = 200 * 1024 * 1024,
    figures: bool = True,
    display_values: bool = True,
) -> dict[str, Any]:
    """Run every code cell top to bottom in a new kernel, then save artifacts with the probe.

    With display_values, a value a cell displayed as its last expression (a DataFrame shown
    with `df.describe()`, a score) is saved too, as `displayed_cell_<n>`."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    kdir = Path(tempfile.mkdtemp(prefix="nb2p-kernel-"))
    spec_dir = kdir / "nb2p"
    spec_dir.mkdir()
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "nb2p",
                "language": "python",
            }
        )
    )
    ksm = KernelSpecManager(kernel_dirs=[str(kdir)])
    km = KernelManager(kernel_name="nb2p", kernel_spec_manager=ksm)
    if os.name == "posix":
        # Unix domain sockets keep kernel traffic off TCP entirely.
        km.transport = "ipc"
        km.ip = str(kdir / "kernel-ipc")
    client = NotebookClient(
        nb,
        timeout=timeout,
        km=km,
        kernel_name="nb2p",
        allow_errors=False,
        record_timing=True,
        resources={"metadata": {"path": str(cwd)}},
    )
    record: dict[str, Any] = {"status": "ok", "cells": [], "failed_cell": None}
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.strip()]
    record["code_cells"] = len(code_cells)
    t_start = time.monotonic()
    try:
        client.reset_execution_trackers()
        with client.setup_kernel(env=kernel_env(), cleanup_kc=True):
            if figures:
                record["figures_hook"] = _run_hidden(
                    client, nb, _figures_hook_code(outdir / "figures")
                )
            for idx in code_cells:
                cell = nb.cells[idx]
                saved = text_outputs(cell)
                t0 = time.monotonic()
                try:
                    client.execute_cell(cell, idx)
                    entry = {
                        "index": idx,
                        "status": "ok",
                        "duration_s": round(time.monotonic() - t0, 3),
                    }
                    fresh = text_outputs(cell)
                    entry.update(compare_saved_output(saved, fresh))
                    if fresh:
                        entry["fresh_output"] = fresh[:2000]
                    printed = printed_text(cell)
                    if printed:
                        entry["printed"] = printed[:20000]
                    images = inline_images(cell)
                    if images:
                        entry["inline_images"] = images
                    if any(
                        o.get("output_type") == "execute_result" for o in cell.get("outputs", [])
                    ):
                        entry["displayed"] = True
                        entry["execution_count"] = cell.get("execution_count")
                    record["cells"].append(entry)
                except CellExecutionError as exc:
                    record["status"] = "failed"
                    record["failed_cell"] = {
                        "index": idx,
                        "ename": getattr(exc, "ename", type(exc).__name__),
                        "evalue": getattr(exc, "evalue", str(exc)),
                        "traceback": strip_ansi(getattr(exc, "traceback", "") or str(exc))[-4000:],
                        "source_head": _cell_head(cell.source),
                    }
                    record["cells"].append(
                        {
                            "index": idx,
                            "status": "error",
                            "duration_s": round(time.monotonic() - t0, 3),
                        }
                    )
                    break
                except CellTimeoutError as exc:
                    record["status"] = "timeout"
                    record["failed_cell"] = {
                        "index": idx,
                        "ename": "CellTimeoutError",
                        "evalue": f"cell exceeded {timeout}s",
                        "traceback": str(exc)[-2000:],
                        "source_head": _cell_head(cell.source),
                    }
                    break
            if record["status"] in ("ok", "failed"):
                # After a failure the probe saves no artifacts, only the environment versions.
                outdir.mkdir(parents=True, exist_ok=True)
                if figures and record["status"] == "ok":
                    _run_hidden(client, nb, _FIGURES_FINISH)
                probe_names = names if record["status"] == "ok" else []
                shown = {}
                if record["status"] == "ok" and display_values:
                    shown = {
                        displayed_name(c["index"]): c["execution_count"]
                        for c in record["cells"]
                        if c.get("displayed") and isinstance(c.get("execution_count"), int)
                    }
                probe = nbformat.v4.new_code_cell(
                    _probe_code(probe_names, outdir, max_bytes, shown)
                )
                nb.cells.append(probe)
                try:
                    client.execute_cell(probe, len(nb.cells) - 1, store_history=False)
                except CellExecutionError as exc:
                    if record["status"] == "ok":
                        record["status"] = "probe_failed"
                    record["probe_error"] = strip_ansi(str(exc))[-3000:]
                finally:
                    nb.cells.pop()
    except DeadKernelError as exc:
        record["status"] = "kernel_died"
        record["error"] = str(exc)[-2000:]
    except Exception as exc:  # kernel failed to start, missing ipykernel, etc.
        record["status"] = "kernel_error"
        record["error"] = f"{type(exc).__name__}: {exc}"[-2000:]
    finally:
        shutil.rmtree(kdir, ignore_errors=True)
    record["cells_run"] = sum(1 for c in record["cells"] if c["status"] == "ok")
    counts = {"same": 0, "different": 0, "none": 0}
    for c in record["cells"]:
        if "saved_output" in c:
            counts[c["saved_output"]] += 1
    record["saved_outputs"] = counts
    record["duration_s"] = round(time.monotonic() - t_start, 3)
    return record


def _run_hidden(client: Any, nb: Any, code: str) -> str:
    """Run tool code in the kernel without adding to the notebook's history or outputs."""
    import nbformat
    from nbclient.exceptions import CellExecutionError

    cell = nbformat.v4.new_code_cell(code)
    nb.cells.append(cell)
    try:
        client.execute_cell(cell, len(nb.cells) - 1, store_history=False)
        return "ok"
    except CellExecutionError as exc:
        return "failed: " + strip_ansi(str(exc))[-500:]
    finally:
        nb.cells.pop()


def run_runtime_script(
    python: str, script: str, spec: dict[str, Any], *, cwd: Path, timeout: int | None
) -> subprocess.CompletedProcess:
    """Run one of the runtime scripts in the target interpreter with a JSON spec file."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(spec, f)
        spec_path = f.name
    try:
        return subprocess.run(
            [python, str(RUNTIME_DIR / script), spec_path],
            cwd=str(cwd),
            env=run_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        os.unlink(spec_path)
