"""Record every matplotlib figure a notebook or pipeline draws, as a PNG.

This file runs inside the user's interpreter (the notebook kernel or the pipeline
process). It must import only the standard library at module level and keep working
on Python 3.8+. matplotlib is imported only if it is installed.

Each figure is rendered once, at a fixed 72 dpi, either when it is closed (plt.close,
or the inline backend closing it after a cell) or at the end of the run, whichever
comes first. So both sides are compared in their final drawn state, whatever backend
or dpi the code itself used.
"""

from __future__ import annotations

import json
import os

MAX_FIGURES = 50
DPI = 72

_state = {"outdir": None, "figures": [], "seen": {}, "installed": False, "skipped": 0}


def _label(fig):
    parts = []
    try:
        st = getattr(fig, "_suptitle", None)
        if st is not None and st.get_text():
            parts.append(st.get_text())
        for ax in fig.get_axes():
            t = ax.get_title()
            if t:
                parts.append(t)
    except Exception:
        pass
    text = "; ".join(parts)
    return text[:120]


def _render(fig):
    key = id(fig)
    if key in _state["seen"] or _state["outdir"] is None:
        return
    _state["seen"][key] = True
    if len(_state["figures"]) >= MAX_FIGURES:
        _state["skipped"] += 1
        return
    n = len(_state["figures"]) + 1
    name = "figure-%02d.png" % n
    path = os.path.join(_state["outdir"], name)
    entry = {"index": n, "file": name, "label": _label(fig)}
    try:
        w, h = fig.get_size_inches()
        entry["size_inches"] = [round(float(w), 3), round(float(h), 3)]
        entry["axes"] = len(fig.get_axes())
        fig.savefig(path, format="png", dpi=DPI, metadata={"Software": None})
        entry["status"] = "saved"
    except Exception as exc:
        entry["status"] = "render_failed"
        entry["error"] = "%s: %s" % (type(exc).__name__, exc)
    _state["figures"].append(entry)


def install(outdir):
    """Start recording figures into outdir. Does nothing if matplotlib is not installed."""
    import importlib.util

    _state["outdir"] = outdir
    if _state["installed"]:
        return True
    if importlib.util.find_spec("matplotlib") is None:
        return False
    os.makedirs(outdir, exist_ok=True)
    from matplotlib import _pylab_helpers
    from matplotlib.figure import Figure

    created = _state.setdefault("created", [])
    orig_init = Figure.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    Figure.__init__ = __init__
    Gcf = _pylab_helpers.Gcf
    orig_destroy = Gcf.destroy.__func__
    orig_destroy_all = Gcf.destroy_all.__func__

    def destroy(cls, num):
        try:
            manager = num if hasattr(num, "canvas") else cls.figs.get(num)
            if manager is not None:
                _render(manager.canvas.figure)
        except Exception:
            pass
        return orig_destroy(cls, num)

    def destroy_all(cls):
        try:
            for manager in list(cls.figs.values()):
                _render(manager.canvas.figure)
        except Exception:
            pass
        return orig_destroy_all(cls)

    Gcf.destroy = classmethod(destroy)
    Gcf.destroy_all = classmethod(destroy_all)
    _state["installed"] = True
    return True


def finish():
    """Render figures still open, write figures.json and return the manifest."""
    for fig in list(_state.get("created", [])):
        _render(fig)
    manifest = {"figures": _state["figures"], "not_recorded": _state["skipped"]}
    if _state["outdir"] is not None and (_state["figures"] or _state["installed"]):
        os.makedirs(_state["outdir"], exist_ok=True)
        with open(os.path.join(_state["outdir"], "figures.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest
