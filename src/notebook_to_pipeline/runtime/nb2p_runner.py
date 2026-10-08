"""Run an agent-written pipeline and save the requested artifacts.

Runs inside the user's interpreter: python nb2p_runner.py <spec.json>
spec: {"pipeline": str, "args": [...], "cwd": dir, "names": [...] | null,
       "outdir": dir, "max_bytes": int}

Pipeline forms:
  path/to/file.py:func   import the file, call func(), use the returned dict
  package.module:func    import the module, call func(), use the returned dict
  path/to/file.py        run the script as __main__, use its globals at the end
  package.module         run the module as __main__ (like python -m), use its globals
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback

import nb2p_probe


def _load_file_module(path):
    name = "nb2p_pipeline_" + os.path.splitext(os.path.basename(path))[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_path(path):
    """Import a .py file. Inside a package (with __init__.py files) import it by dotted name
    so relative imports work; otherwise load it as a standalone module."""
    directory, filename = os.path.split(path)
    parts = [os.path.splitext(filename)[0]]
    root = directory
    while os.path.exists(os.path.join(root, "__init__.py")):
        root, pkg = os.path.split(root)
        parts.insert(0, pkg)
    if len(parts) > 1:
        sys.path.insert(0, root)
        return importlib.import_module(".".join(parts))
    sys.path.insert(0, directory)
    return _load_file_module(path)


def run_pipeline(target, args):
    """Return the namespace (dict) holding the pipeline's results."""
    mod_part, sep, func_name = target.rpartition(":")
    if sep and func_name.isidentifier():
        if mod_part.endswith(".py") or os.sep in mod_part or "/" in mod_part:
            mod = _import_path(os.path.abspath(mod_part))
        else:
            mod = importlib.import_module(mod_part)
        func = getattr(mod, func_name)
        sys.argv = [mod_part] + list(args)
        result = func()
        if not isinstance(result, dict):
            raise TypeError(
                "%s returned %s; a pipeline function must return a dict of {artifact name: value}"
                % (target, type(result).__name__)
            )
        return result
    if target.endswith(".py") or os.path.exists(target):
        path = os.path.abspath(target)
        sys.path.insert(0, os.path.dirname(path))
        sys.argv = [path] + list(args)
        return runpy.run_path(path, run_name="__main__")
    sys.argv = [target] + list(args)
    return runpy.run_module(target, run_name="__main__", alter_sys=True)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    with open(argv[0], encoding="utf-8") as f:
        spec = json.load(f)
    outdir = spec["outdir"]
    os.makedirs(outdir, exist_ok=True)
    os.chdir(spec["cwd"])
    for extra in (os.path.join(spec["cwd"], "src"), spec["cwd"]):
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.insert(0, extra)
    status = {"status": "ok", "pipeline": spec["pipeline"], "args": spec.get("args", [])}
    t0 = time.time()
    try:
        ns = run_pipeline(spec["pipeline"], spec.get("args", []))
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status.update({"status": "failed", "error_type": "SystemExit", "error": repr(exc.code),
                           "traceback": traceback.format_exc()})
        ns = None
    except BaseException as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
        ns = None
    status["duration_s"] = round(time.time() - t0, 3)
    if ns is None and status["status"] == "ok":
        status.update({"status": "failed", "error_type": "SystemExit",
                       "error": "the script exited before finishing, so its globals are unavailable"})
    if status["status"] == "ok":
        nb2p_probe.dump(ns, spec.get("names"), outdir, spec.get("max_bytes", 200 * 1024 * 1024))
    with open(os.path.join(outdir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    return 0 if status["status"] == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
