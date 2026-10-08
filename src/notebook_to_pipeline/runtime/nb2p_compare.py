"""Compare two capture directories artifact by artifact, with tolerances.

Like nb2p_probe, this runs inside the user's interpreter so that pickled
DataFrames and arrays load with the same library versions that wrote them.
Standard library only at module level, Python 3.8+.

Usage: python nb2p_compare.py <spec.json>
spec: {"reference": dir, "candidate": dir, "names": [...] | null,
       "files": [...] | null, "options": {...}, "out": path}
"""

from __future__ import annotations

import cmath
import json
import math
import os
import pickle
import sys

DEFAULTS = {
    "rtol": 1e-7,
    "atol": 1e-10,
    "ignore_row_order": False,
    "ignore_column_order": False,
    "ignore_index": False,
    "check_dtype": True,
    "max_differences": 5,
}
PASSING = ("identical", "close")


def _opts(options):
    o = dict(DEFAULTS)
    o.update({k: v for k, v in (options or {}).items() if v is not None})
    return o


class Acc:
    """Accumulates differences and numeric error while walking two values."""

    def __init__(self, opts):
        self.opts = opts
        self.diffs = []
        self.mismatches = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.inexact = False
        self.notes = []

    def add(self, path, ref, cand, why=""):
        self.mismatches += 1
        if len(self.diffs) < self.opts["max_differences"]:
            d = {"path": path, "reference": _short(ref), "candidate": _short(cand)}
            if why:
                d["why"] = why
            self.diffs.append(d)

    def numeric(self, abs_diff, rel_diff):
        if abs_diff > 0:
            self.inexact = True
        self.max_abs = max(self.max_abs, abs_diff)
        self.max_rel = max(self.max_rel, rel_diff)


def _short(v):
    try:
        if _is_np_scalar(v):
            v = str(v) if v.dtype.kind in "mM" else v.item()
        r = repr(v)
    except Exception:
        r = "<unrepresentable %s>" % type(v).__name__
    return r if len(r) <= 120 else r[:117] + "..."


def _is(obj, module, name):
    m = type(obj).__module__ or ""
    return m.startswith(module) and type(obj).__name__ == name


def _is_np_scalar(obj):
    return (type(obj).__module__ or "") == "numpy" and getattr(obj, "shape", None) == () and hasattr(obj, "item")


def _isclose(a, b, opts):
    if isinstance(a, complex) or isinstance(b, complex):
        if cmath.isnan(a) and cmath.isnan(b):
            return True, 0.0, 0.0
        diff = abs(a - b)
        return diff <= opts["atol"] + opts["rtol"] * abs(a), diff, diff / abs(a) if a else diff
    if math.isnan(a) and math.isnan(b):
        return True, 0.0, 0.0
    if math.isinf(a) or math.isinf(b):
        return a == b, 0.0 if a == b else math.inf, 0.0 if a == b else math.inf
    diff = abs(a - b)
    rel = diff / abs(a) if a != 0 else (0.0 if diff == 0 else math.inf)
    return diff <= opts["atol"] + opts["rtol"] * abs(a), diff, rel


def compare_values(ref, cand, path, acc):
    """Recursively compare; returns True when equal within tolerance."""
    opts = acc.opts
    if _is_np_scalar(ref):
        ref = ref.item()
    if _is_np_scalar(cand):
        cand = cand.item()
    if _is(ref, "pandas", "DataFrame") and _is(cand, "pandas", "DataFrame"):
        return compare_frames(ref, cand, path, acc)
    if _is(ref, "pandas", "Series") and _is(cand, "pandas", "Series"):
        return compare_series(ref, cand, path, acc)
    if type(ref).__name__.endswith("Index") and type(cand).__name__.endswith("Index") and (type(ref).__module__ or "").startswith("pandas"):
        import numpy as np

        return compare_arrays(np.asarray(ref), np.asarray(cand), path, acc)
    if _is(ref, "numpy", "ndarray") and _is(cand, "numpy", "ndarray"):
        return compare_arrays(ref, cand, path, acc)
    if isinstance(ref, bool) or isinstance(cand, bool):
        if type(ref) is not type(cand) or ref != cand:
            acc.add(path, ref, cand)
            return False
        return True
    if isinstance(ref, float) and isinstance(cand, float) or isinstance(ref, complex) and isinstance(cand, complex):
        ok, d, r = _isclose(ref, cand, opts)
        acc.numeric(d, r)
        if not ok:
            acc.add(path, ref, cand, "abs diff %.3g" % d)
        return ok
    if isinstance(ref, (int, float)) and isinstance(cand, (int, float)) and type(ref) is not type(cand):
        acc.add(path, ref, cand, "type changed from %s to %s" % (type(ref).__name__, type(cand).__name__))
        return False
    if isinstance(ref, dict) and isinstance(cand, dict):
        ok = True
        missing = [k for k in ref if k not in cand]
        extra = [k for k in cand if k not in ref]
        for k in missing:
            acc.add("%s[%r]" % (path, k), ref[k], "<missing>")
            ok = False
        for k in extra:
            acc.add("%s[%r]" % (path, k), "<missing>", cand[k])
            ok = False
        for k in ref:
            if k in cand and not compare_values(ref[k], cand[k], "%s[%r]" % (path, k), acc):
                ok = False
        return ok
    if isinstance(ref, (list, tuple)) and isinstance(cand, (list, tuple)):
        if type(ref) is not type(cand):
            acc.add(path, type(ref).__name__, type(cand).__name__, "container type changed")
            return False
        if len(ref) != len(cand):
            acc.add(path, "len %d" % len(ref), "len %d" % len(cand), "length differs")
            return False
        ok = True
        for i, (a, b) in enumerate(zip(ref, cand)):
            if not compare_values(a, b, "%s[%d]" % (path, i), acc):
                ok = False
        return ok
    if type(ref) is not type(cand):
        acc.add(path, ref, cand, "type changed from %s to %s" % (type(ref).__name__, type(cand).__name__))
        return False
    try:
        eq = ref == cand
        if not isinstance(eq, bool):
            eq = bool(eq)
    except Exception:
        eq = repr(ref) == repr(cand)
    if not eq:
        acc.add(path, ref, cand)
    return eq


def _numeric_kind(k):
    return k in "fc"


def compare_arrays(a, b, path, acc):
    import numpy as np

    if a.shape != b.shape:
        acc.add(path + ".shape", a.shape, b.shape, "shape differs")
        return False
    if a.dtype != b.dtype:
        if acc.opts["check_dtype"]:
            acc.add(path + ".dtype", str(a.dtype), str(b.dtype), "dtype differs")
            return False
        acc.notes.append("%s: dtype %s vs %s (dtype check off)" % (path, a.dtype, b.dtype))
    if a.size == 0:
        return True
    ka, kb = a.dtype.kind, b.dtype.kind
    if (_numeric_kind(ka) or _numeric_kind(kb)) and ka in "biufc" and kb in "biufc":
        af = a.astype(np.complex128 if "c" in (ka, kb) else np.float64)
        bf = b.astype(af.dtype)
        close = np.isclose(af, bf, rtol=acc.opts["rtol"], atol=acc.opts["atol"], equal_nan=True)
        both_finite = np.isfinite(af) & np.isfinite(bf)
        if both_finite.any():
            diff = np.abs(af - bf)[both_finite]
            denom = np.abs(af)[both_finite]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(denom > 0, diff / np.where(denom > 0, denom, 1), np.where(diff > 0, np.inf, 0))
            acc.numeric(float(diff.max()), float(rel.max()))
    elif ka in "mM" or kb in "mM":
        close = (a == b) | (np.isnat(a) & np.isnat(b)) if ka == kb else np.zeros(a.shape, dtype=bool)
    elif ka in "biuSU?" and kb in "biuSU?":
        close = a == b
    else:
        flat_a, flat_b = a.ravel(), b.ravel()
        res = np.empty(flat_a.shape, dtype=bool)
        for i in range(flat_a.size):
            x, y = flat_a[i], flat_b[i]
            sub = Acc(dict(acc.opts, max_differences=0))
            res[i] = compare_values(x, y, "", sub) or (_isnan(x) and _isnan(y))
            if sub.max_abs:
                acc.numeric(sub.max_abs, sub.max_rel)
        close = res.reshape(a.shape)
    close = np.asarray(close)
    if close.all():
        return True
    bad = np.argwhere(~close)
    shown = bad[: acc.opts["max_differences"]]
    for idx in shown:
        t = tuple(int(i) for i in idx)
        acc.add("%s[%s]" % (path, ",".join(map(str, t))), a[t], b[t])
    acc.mismatches += int(bad.shape[0]) - len(shown)
    return False


def _isnan(x):
    try:
        return x != x
    except Exception:
        return False


def compare_series(a, b, path, acc):
    ok = True
    if a.name != b.name:
        acc.add(path + ".name", a.name, b.name, "series name differs")
        ok = False
    return compare_frames(a.to_frame(name="value"), b.to_frame(name="value"), path, acc, series=True) and ok


def _sorted_frame(df, keep_index):
    d = df.reset_index() if keep_index else df.reset_index(drop=True)
    cols = list(d.columns)
    try:
        return d.sort_values(cols, kind="mergesort", na_position="last").reset_index(drop=True)
    except TypeError:
        key = d.astype(str)
        order = key.sort_values(cols, kind="mergesort").index
        return d.loc[order].reset_index(drop=True)


def compare_frames(a, b, path, acc, series=False):
    import numpy as np

    opts = acc.opts
    ok = True
    ca, cb = list(a.columns), list(b.columns)
    if ca != cb:
        if set(map(str, ca)) == set(map(str, cb)) and len(ca) == len(cb):
            if opts["ignore_column_order"]:
                b = b[ca]
            else:
                acc.add(path + ".columns", [str(c) for c in ca], [str(c) for c in cb], "column order differs")
                return False
        else:
            missing = [str(c) for c in ca if c not in cb]
            extra = [str(c) for c in cb if c not in ca]
            acc.add(path + ".columns", "missing: %s" % missing, "extra: %s" % extra, "columns differ")
            return False
    if a.shape[0] != b.shape[0]:
        acc.add(path + ".rows", a.shape[0], b.shape[0], "row count differs")
        return False
    for c in ca:
        if getattr(a[c], "ndim", 1) != 1:
            continue
        if a[c].dtype != b[c].dtype:
            if opts["check_dtype"]:
                acc.add("%s[%r].dtype" % (path, c) if not series else path + ".dtype", str(a[c].dtype), str(b[c].dtype), "dtype differs")
                ok = False
            else:
                acc.notes.append("%s column %r: dtype %s vs %s (dtype check off)" % (path, c, a[c].dtype, b[c].dtype))
    if not ok:
        return False
    if opts["ignore_row_order"]:
        a = _sorted_frame(a, not opts["ignore_index"])
        b = _sorted_frame(b, not opts["ignore_index"])
        ca = list(a.columns)
    elif not opts["ignore_index"]:
        if not a.index.equals(b.index):
            ia, ib = np.asarray(a.index), np.asarray(b.index)
            first = next((i for i in range(len(ia)) if not (ia[i] == ib[i] or (_isnan(ia[i]) and _isnan(ib[i])))), None)
            if first is None and a.index.names != b.index.names:
                acc.add(path + ".index.names", list(a.index.names), list(b.index.names), "index names differ")
            else:
                acc.add("%s.index[%s]" % (path, first), ia[first] if first is not None else None,
                        ib[first] if first is not None else None, "index differs")
            ok = False
    labels = list(a.index)
    for c in ca:
        sa, sb = a[c], b[c]
        if getattr(sa, "ndim", 1) != 1:
            sa, sb = sa.iloc[:, 0], sb.iloc[:, 0]
        va, vb = sa.to_numpy(), sb.to_numpy()
        ka, kb = va.dtype.kind, vb.dtype.kind
        if va.dtype.kind == "O" or vb.dtype.kind == "O" or str(sa.dtype) in ("category", "string", "str") or ka not in "biufcmM" or kb not in "biufcmM":
            eq = np.array([_obj_equal(x, y, acc) for x, y in zip(va, vb)], dtype=bool)
        else:
            sub = Acc(dict(opts, max_differences=0))
            compare_arrays(va, vb, "", sub)
            acc.numeric(sub.max_abs, sub.max_rel)
            if ka in "biufc" and kb in "biufc":
                fa, fb = va.astype(np.float64 if "c" not in (ka, kb) else np.complex128), vb.astype(np.float64 if "c" not in (ka, kb) else np.complex128)
                eq = np.isclose(fa, fb, rtol=opts["rtol"], atol=opts["atol"], equal_nan=True)
            else:
                eq = (va == vb) | (np.isnat(va) & np.isnat(vb)) if ka == kb and ka in "mM" else (va == vb)
        if not eq.all():
            ok = False
            bad = np.flatnonzero(~eq)
            for k, pos in enumerate(bad):
                where = "%s[%s]" % (path, _short(labels[pos])) if series else "%s[row %s, %r]" % (path, _short(labels[pos]), c)
                if k < opts["max_differences"]:
                    acc.add(where, va[pos], vb[pos])
                else:
                    acc.mismatches += len(bad) - k
                    break
    return ok


def _obj_equal(x, y, acc):
    try:
        if _isnan(x) and _isnan(y):
            return True
    except Exception:
        pass
    sub = Acc(dict(acc.opts, max_differences=0))
    ok = compare_values(x, y, "", sub)
    acc.numeric(sub.max_abs, sub.max_rel)
    return ok


# --------------------------------------------------------------------------
# Directory level
# --------------------------------------------------------------------------


def _load(directory, entry):
    with open(os.path.join(directory, entry["payload"]), "rb") as f:
        return pickle.load(f)


def _result(name, kind, status, detail, acc=None):
    r = {"name": name, "kind": kind, "status": status, "passed": status in PASSING, "detail": detail}
    if acc is not None:
        r["differences"] = acc.diffs
        r["mismatch_count"] = acc.mismatches
        r["max_abs_diff"] = acc.max_abs
        r["max_rel_diff"] = acc.max_rel
        if acc.notes:
            r["notes"] = acc.notes[:10]
    return r


def _describe_failure(acc):
    if not acc.diffs:
        return "values differ"
    d = acc.diffs[0]
    why = (" (%s)" % d["why"]) if d.get("why") else ""
    more = (", %d differences in total" % acc.mismatches) if acc.mismatches > 1 else ""
    return "first difference at %s: reference %s, candidate %s%s%s" % (
        d["path"], d["reference"], d["candidate"], why, more)


def compare_artifact(ref_entry, cand_entry, ref_dir, cand_dir, opts):
    name = ref_entry["name"]
    kind = ref_entry.get("kind")
    if cand_entry is None or cand_entry.get("status") == "missing":
        return _result(name, kind, "missing", "the candidate did not produce `%s`" % name)
    if cand_entry.get("status") == "skipped":
        return _result(name, kind, "missing", "candidate value was not data: %s" % cand_entry.get("reason"))
    if ref_entry.get("hash") and ref_entry.get("hash") == cand_entry.get("hash"):
        return _result(name, kind, "identical", "hash match")
    if kind != cand_entry.get("kind"):
        return _result(name, kind, "type_mismatch", "reference is %s (%s), candidate is %s (%s)" % (
            kind, ref_entry.get("type"), cand_entry.get("kind"), cand_entry.get("type")))
    if not ref_entry.get("payload") or not cand_entry.get("payload"):
        return _result(name, kind, "differs", "hashes differ and no payload is stored to localize the difference")
    try:
        ref = _load(ref_dir, ref_entry)
        cand = _load(cand_dir, cand_entry)
    except Exception as exc:
        return _result(name, kind, "error", "could not load payloads: %s: %s" % (type(exc).__name__, exc))
    acc = Acc(opts)
    try:
        ok = compare_values(ref, cand, name, acc)
    except Exception as exc:
        return _result(name, kind, "error", "comparison raised %s: %s" % (type(exc).__name__, exc), acc)
    if ok:
        if acc.inexact:
            detail = "equal within tolerance (max abs diff %.3g, max rel diff %.3g; rtol=%g, atol=%g)" % (
                acc.max_abs, acc.max_rel, opts["rtol"], opts["atol"])
        else:
            detail = "equal values, different hash (representation differs, e.g. dtype or index metadata)"
            if acc.notes:
                detail = "equal values; " + "; ".join(acc.notes[:3])
        return _result(name, kind, "close", detail, acc)
    return _result(name, kind, "differs", _describe_failure(acc), acc)


def compare_manifests(ref_dir, cand_dir, names, options):
    opts = _opts(options)
    with open(os.path.join(ref_dir, "artifacts.json"), encoding="utf-8") as f:
        ref_m = json.load(f)
    cand_path = os.path.join(cand_dir, "artifacts.json")
    cand_m = {"artifacts": []}
    if os.path.exists(cand_path):
        with open(cand_path, encoding="utf-8") as f:
            cand_m = json.load(f)
    ref_by = {e["name"]: e for e in ref_m["artifacts"]}
    cand_by = {e["name"]: e for e in cand_m["artifacts"]}
    if names is None:
        names = [e["name"] for e in ref_m["artifacts"] if e.get("status") == "captured"]
    results = []
    for n in names:
        ref_e = ref_by.get(n)
        if ref_e is None or ref_e.get("status") != "captured":
            results.append(_result(n, None, "skipped", "not captured in the reference"))
            continue
        results.append(compare_artifact(ref_e, cand_by.get(n), ref_dir, cand_dir, opts))
    return results


def compare_file(rel, ref_dir, cand_dir, opts):
    """Compare one written file. Data formats get a tolerant comparison when hashes differ."""
    rp = os.path.join(ref_dir, rel)
    cp = os.path.join(cand_dir, rel)
    ext = os.path.splitext(rel)[1].lower()
    acc = Acc(opts)
    try:
        if ext in (".csv", ".tsv"):
            import pandas as pd

            sep = "\t" if ext == ".tsv" else ","
            a, b = pd.read_csv(rp, sep=sep), pd.read_csv(cp, sep=sep)
            ok = compare_frames(a, b, rel, acc)
        elif ext == ".json":
            with open(rp, encoding="utf-8") as f:
                a = json.load(f)
            with open(cp, encoding="utf-8") as f:
                b = json.load(f)
            ok = compare_values(a, b, rel, acc)
        elif ext in (".parquet", ".feather"):
            import pandas as pd

            reader = pd.read_parquet if ext == ".parquet" else pd.read_feather
            ok = compare_frames(reader(rp), reader(cp), rel, acc)
        elif ext == ".npy":
            import numpy as np

            ok = compare_arrays(np.load(rp, allow_pickle=False), np.load(cp, allow_pickle=False), rel, acc)
        else:
            return {"status": "differs", "passed": False, "detail": "bytes differ (no tolerant comparison for %s files)" % (ext or "extensionless")}
    except Exception as exc:
        return {"status": "differs", "passed": False, "detail": "bytes differ and parsing failed: %s: %s" % (type(exc).__name__, exc)}
    if ok:
        return {"status": "close", "passed": True, "detail": "parsed contents equal within tolerance (max abs diff %.3g)" % acc.max_abs}
    return {"status": "differs", "passed": False, "detail": _describe_failure(acc), "differences": acc.diffs}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    with open(argv[0], encoding="utf-8") as f:
        spec = json.load(f)
    opts = _opts(spec.get("options"))
    out = {"artifacts": compare_manifests(spec["reference"], spec["candidate"], spec.get("names"), opts)}
    files = []
    for item in spec.get("files") or []:
        r = compare_file(item["path"], item["reference_dir"], item["candidate_dir"], opts)
        r["path"] = item["path"]
        files.append(r)
    out["files"] = files
    out["options"] = opts
    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
