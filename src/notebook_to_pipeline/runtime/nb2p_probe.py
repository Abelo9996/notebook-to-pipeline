"""Serialize chosen variables into a capture directory, with hashes and summaries.

This file runs inside the user's interpreter (the notebook kernel or the pipeline
process), not inside notebook-to-pipeline's own environment. It must import only
the standard library at module level and keep working on Python 3.8+.
pandas, numpy and scikit-learn are used only if the objects being saved need them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import sys
import types

SKIP_NAMES = {
    "In",
    "Out",
    "get_ipython",
    "exit",
    "quit",
    "_",
    "__",
    "___",
    "_i",
    "_ii",
    "_iii",
    "_oh",
    "_dh",
    "_ih",
}
PLOT_MODULES = ("matplotlib", "seaborn", "plotly", "bokeh", "altair", "mpl_toolkits")
TIMING_KEYS = ("fit_time", "score_time", "_time", "time_")
MAX_SUMMARY_COLUMNS = 40
PROTOCOL = 4


def _module_of(obj):
    return type(obj).__module__ or ""


def _type_name(obj):
    t = type(obj)
    return "%s.%s" % (t.__module__, t.__qualname__)


def is_dataframe(obj):
    return _module_of(obj).startswith("pandas") and type(obj).__name__ == "DataFrame"


def is_series(obj):
    return _module_of(obj).startswith("pandas") and type(obj).__name__ == "Series"


def is_index(obj):
    return _module_of(obj).startswith("pandas") and type(obj).__name__.endswith("Index")


def is_ndarray(obj):
    return type(obj).__name__ == "ndarray" and _module_of(obj) == "numpy"


def is_np_scalar(obj):
    return _module_of(obj) == "numpy" and getattr(obj, "shape", None) == () and hasattr(obj, "item")


def is_estimator(obj):
    return (
        not isinstance(obj, type)
        and hasattr(obj, "get_params")
        and hasattr(obj, "fit")
        and callable(getattr(obj, "get_params", None))
    )


def is_plot_object(obj):
    return _module_of(obj).startswith(PLOT_MODULES)


def is_scalar(obj):
    return obj is None or isinstance(obj, (bool, int, float, complex, str, bytes))


def classify(obj):
    """Return (kind, skip_reason). kind is None when the object is not data."""
    if isinstance(obj, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
        return None, "not data (%s)" % type(obj).__name__
    if isinstance(obj, (types.GeneratorType, types.MethodType)):
        return None, "not data (%s)" % type(obj).__name__
    if is_plot_object(obj):
        return None, "plot object (%s)" % _type_name(obj)
    if is_dataframe(obj):
        return "dataframe", None
    if is_series(obj):
        return "series", None
    if is_index(obj):
        return "index", None
    if is_ndarray(obj):
        return "ndarray", None
    if is_np_scalar(obj) or is_scalar(obj):
        return "scalar", None
    if isinstance(obj, range):
        return "container", None
    if isinstance(obj, (list, tuple, dict, set, frozenset)):
        return "container", None
    if is_estimator(obj):
        return "estimator", None
    if hasattr(obj, "read") and hasattr(obj, "close"):
        return None, "not data (file-like %s)" % _type_name(obj)
    return "object", None


# --------------------------------------------------------------------------
# Canonical hashing
# --------------------------------------------------------------------------


def _h(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8", "surrogatepass")
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()


def _pickle_hash(obj):
    try:
        return _h("pickle", pickle.dumps(obj, protocol=PROTOCOL))
    except Exception:
        return _h("repr", repr(obj))


def value_hash(obj):
    """Content hash that is equal for equal values (dict order ignored)."""
    if is_np_scalar(obj):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, int, str, bytes, complex)):
        return _h("scalar", type(obj).__name__, repr(obj))
    if isinstance(obj, float):
        if math.isnan(obj):
            return _h("scalar", "float", "nan")
        return _h("scalar", "float", repr(obj))
    if is_dataframe(obj):
        import pandas as pd

        cols = repr([str(c) for c in obj.columns])
        dtypes = repr([str(t) for t in obj.dtypes])
        try:
            body = pd.util.hash_pandas_object(obj, index=True).values.tobytes()
            idx_names = repr(list(obj.index.names))
        except Exception:
            return _h("dataframe", cols, dtypes, _pickle_hash(obj))
        return _h("dataframe", cols, dtypes, idx_names, body)
    if is_series(obj):
        import pandas as pd

        try:
            body = pd.util.hash_pandas_object(obj, index=True).values.tobytes()
        except Exception:
            return _h("series", repr(obj.name), str(obj.dtype), _pickle_hash(obj))
        return _h("series", repr(obj.name), str(obj.dtype), body)
    if is_index(obj):
        return _h("index", _pickle_hash(list(obj)), str(obj.dtype), repr(obj.name))
    if is_ndarray(obj):
        if obj.dtype.kind in "biufcmMSU?":
            import numpy as np

            arr = np.ascontiguousarray(obj)
            return _h("ndarray", obj.dtype.str, repr(obj.shape), arr.tobytes())
        return _h("ndarray", obj.dtype.str, repr(obj.shape), _pickle_hash(obj))
    if isinstance(obj, dict):
        items = sorted((repr(k), value_hash(v)) for k, v in obj.items())
        return _h("dict", repr(items))
    if isinstance(obj, (list, tuple)):
        return (
            _h(type(obj).__name__, *[value_hash(v) for v in obj]) if obj else _h(type(obj).__name__)
        )
    if isinstance(obj, (set, frozenset)):
        return _h("set", repr(sorted(value_hash(v) for v in obj)))
    if isinstance(obj, range):
        return _h("range", repr(obj))
    return _pickle_hash(obj)


# --------------------------------------------------------------------------
# Estimators: compare params and fitted state, not the object itself
# --------------------------------------------------------------------------


def _plain(value, depth=0):
    if depth > 6:
        return repr(value)
    if is_estimator(value):
        return estimator_state(value, depth + 1)
    if isinstance(value, dict):
        return {
            k: _plain(v, depth + 1)
            for k, v in value.items()
            if not (isinstance(k, str) and any(t in k for t in TIMING_KEYS))
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v, depth + 1) for v in value)
    if (
        is_scalar(value)
        or is_np_scalar(value)
        or is_ndarray(value)
        or is_dataframe(value)
        or is_series(value)
    ):
        return value
    return repr(value)


def estimator_state(est, depth=0):
    t = type(est)
    try:
        params = est.get_params(deep=False)
    except Exception:
        params = {}
    fitted = {}
    for attr, value in sorted(vars(est).items()):
        if attr.endswith("_") and not attr.startswith("_"):
            if any(k in attr for k in TIMING_KEYS):
                continue
            fitted[attr] = _plain(value, depth)
    steps = getattr(est, "steps", None)
    nested = {}
    if isinstance(steps, list):
        for item in steps:
            if isinstance(item, tuple) and len(item) == 2 and is_estimator(item[1]):
                nested[item[0]] = estimator_state(item[1], depth + 1)
    return {
        "__estimator__": "%s.%s" % (t.__module__, t.__qualname__),
        "params": {k: repr(v) for k, v in sorted(params.items())},
        "fitted": fitted,
        "steps": nested,
    }


# --------------------------------------------------------------------------
# Summaries for humans
# --------------------------------------------------------------------------


def _num(x):
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return repr(x)
    return x


def summarize(obj, kind):
    try:
        if kind == "dataframe":
            cols = list(obj.columns)
            out = {
                "shape": list(obj.shape),
                "columns": [str(c) for c in cols[:MAX_SUMMARY_COLUMNS]],
                "dtypes": {
                    str(c): str(t) for c, t in list(obj.dtypes.items())[:MAX_SUMMARY_COLUMNS]
                },
                "index": "%s (%s)" % (type(obj.index).__name__, obj.index.dtype),
                "nulls": int(obj.isna().sum().sum()),
            }
            stats = {}
            for c in cols[:MAX_SUMMARY_COLUMNS]:
                s = obj[c]
                if getattr(s, "ndim", 1) != 1:
                    continue
                if s.dtype.kind in "biuf":
                    stats[str(c)] = {
                        "mean": _num(s.mean()),
                        "std": _num(s.std()),
                        "min": _num(s.min()),
                        "max": _num(s.max()),
                    }
            out["numeric_stats"] = stats
            return out
        if kind == "series":
            out = {
                "length": int(len(obj)),
                "dtype": str(obj.dtype),
                "name": repr(obj.name),
                "nulls": int(obj.isna().sum()),
            }
            if obj.dtype.kind in "biuf" and len(obj):
                out.update(
                    {"mean": _num(obj.mean()), "min": _num(obj.min()), "max": _num(obj.max())}
                )
            return out
        if kind == "index":
            return {"length": int(len(obj)), "dtype": str(obj.dtype), "head": repr(list(obj[:5]))}
        if kind == "ndarray":
            out = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            if obj.dtype.kind in "biuf" and obj.size:
                import numpy as np

                out.update(
                    {
                        "min": _num(np.nanmin(obj)),
                        "max": _num(np.nanmax(obj)),
                        "mean": _num(np.nanmean(obj)),
                    }
                )
            return out
        if kind == "scalar":
            v = obj.item() if is_np_scalar(obj) else obj
            r = repr(v)
            return {"value": r if len(r) <= 200 else r[:197] + "...", "type": type(v).__name__}
        if kind == "container":
            return {"type": type(obj).__name__, "length": len(obj), "preview": repr(obj)[:200]}
        if kind == "estimator":
            st = estimator_state(obj)
            return {
                "class": st["__estimator__"],
                "params": len(st["params"]),
                "fitted_attributes": sorted(st["fitted"])[:30],
                "steps": sorted(st["steps"]),
            }
        return {"type": _type_name(obj), "repr": repr(obj)[:200]}
    except Exception as exc:  # summaries are best effort
        return {"error": "summary failed: %s" % exc}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def env_info():
    info = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    stdlib = getattr(sys, "stdlib_module_names", set())
    try:
        from importlib import metadata

        dists = metadata.packages_distributions()
    except Exception:
        dists = {}
        metadata = None
    seen = {}
    for name in sorted({m.split(".")[0] for m in list(sys.modules)}):
        if (
            not name
            or name.startswith("_")
            or name in stdlib
            or name in ("nb2p_probe", "nb2p_runner", "nb2p_compare")
        ):
            continue
        version = None
        for dist in dists.get(name, []):
            try:
                version = metadata.version(dist)
                seen[dist] = version
            except Exception:
                pass
        if version is None:
            mod = sys.modules.get(name)
            v = getattr(mod, "__version__", None)
            if isinstance(v, str):
                seen[name] = v
    info["packages"] = dict(sorted(seen.items(), key=lambda kv: kv[0].lower()))
    return info


# --------------------------------------------------------------------------
# Dump
# --------------------------------------------------------------------------


def _safe_filename(name, used):
    base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) or "artifact"
    cand = base
    i = 1
    while cand in used:
        i += 1
        cand = "%s_%d" % (base, i)
    used.add(cand)
    return cand


def auto_names(namespace):
    out = []
    for name, value in namespace.items():
        if name.startswith("_") or name in SKIP_NAMES:
            continue
        kind, _ = classify(value)
        if kind is not None and kind != "object":
            out.append(name)
    return out


def shared_objects(namespace, names):
    """Objects reachable under more than one name: aliases (b = a) and pipeline steps shared
    by two composites. Mutating one of these changes the others."""
    paths = {}
    kinds = {}

    def reg(obj, path, depth):
        kind, _ = classify(obj)
        if kind not in ("dataframe", "series", "ndarray", "estimator", "container", "object"):
            return
        if isinstance(obj, (tuple, range, frozenset, str, bytes)):
            return
        seen_before = id(obj) in paths
        paths.setdefault(id(obj), []).append(path)
        kinds[id(obj)] = _type_name(obj)
        if kind == "estimator" and depth < 4 and not seen_before:
            steps = getattr(obj, "steps", None) or getattr(obj, "transformer_list", None)
            if isinstance(steps, list):
                for i, item in enumerate(steps):
                    if isinstance(item, tuple) and len(item) >= 2:
                        reg(item[1], "%s[%d]" % (path, i), depth + 1)

    for n in names:
        if n in namespace:
            reg(namespace[n], n, 0)
    out = []
    for oid, ps in paths.items():
        if len(ps) < 2:
            continue
        top = [p for p in ps if "[" not in p]
        nested_roots = sorted({p.split("[")[0] for p in ps if "[" in p})
        if len(nested_roots) >= 2:
            out.append({"kind": "shared_between_composites", "paths": ps, "type": kinds[oid]})
        elif len(top) >= 2:
            out.append({"kind": "alias", "paths": ps, "type": kinds[oid]})
    return out


def dump(namespace, names, outdir, max_bytes=200 * 1024 * 1024):
    """Save each named variable from namespace into outdir. Returns the manifest dict."""
    art_dir = os.path.join(outdir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    if names is None:
        names = auto_names(namespace)
    used = set()
    entries = []
    for name in names:
        entry = {"name": name}
        if name not in namespace:
            entry.update({"status": "missing", "reason": "not defined at the end of the run"})
            entries.append(entry)
            continue
        obj = namespace[name]
        kind, reason = classify(obj)
        entry["type"] = _type_name(obj)
        if kind is None:
            entry.update({"status": "skipped", "reason": reason})
            entries.append(entry)
            continue
        entry["kind"] = kind
        payload_obj = estimator_state(obj) if kind == "estimator" else obj
        try:
            entry["hash"] = value_hash(payload_obj)
        except Exception as exc:
            entry["hash"] = _pickle_hash(payload_obj)
            entry["hash_note"] = "fallback hash: %s" % exc
        entry["summary"] = summarize(obj, kind)
        try:
            data = pickle.dumps(payload_obj, protocol=PROTOCOL)
        except Exception as exc:
            data = None
            entry["payload_note"] = "not picklable (%s); compared by hash only" % type(exc).__name__
        if data is not None and len(data) > max_bytes:
            entry["payload_note"] = (
                "payload is %d bytes, over the limit; compared by hash only" % len(data)
            )
            data = None
        if data is not None:
            fname = _safe_filename(name, used) + ".pkl"
            with open(os.path.join(art_dir, fname), "wb") as f:
                f.write(data)
            entry["payload"] = "artifacts/" + fname
            entry["bytes"] = len(data)
        entry["status"] = "captured"
        entries.append(entry)
    try:
        shared = shared_objects(
            namespace, [e["name"] for e in entries if e.get("status") == "captured"]
        )
    except Exception as exc:
        shared = [{"kind": "error", "paths": [], "type": "shared object check failed: %s" % exc}]
    manifest = {"artifacts": entries, "env": env_info(), "shared_objects": shared}
    with open(os.path.join(outdir, "artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
