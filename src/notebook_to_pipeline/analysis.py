"""Static analysis of a notebook: def/use graph, hidden-state risks and a proposed module split.

Everything here is computed from the notebook file alone. Nothing is executed.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .notebook import Cell, load_cells, read_notebook, sha256_file

STAGES = ["load", "clean", "features", "train", "evaluate", "report"]
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}

IPYTHON_NAMES = {
    "get_ipython",
    "display",
    "In",
    "Out",
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
    "__builtins__",
    "__name__",
    "__doc__",
}
_IPY_NUMBERED = re.compile(r"^_i?\d+$")
BUILTIN_NAMES = set(dir(builtins))

MUTATING_METHODS = {
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "update",
    "sort",
    "reverse",
    "add",
    "discard",
    "setdefault",
    "popitem",
    "appendleft",
    "extendleft",
    "fill",
    "put",
    "resize",
    "itemset",
    "setflags",
    "fit",
    "partial_fit",
    "fit_transform",
    "fit_predict",
    "set_params",
    "set_output",
    "set_index_inplace",
    "__setitem__",
}
FILE_WRITE_METHODS = {
    "to_csv",
    "to_parquet",
    "to_excel",
    "to_json",
    "to_pickle",
    "to_feather",
    "to_hdf",
    "to_sql",
    "to_stata",
    "to_html",
    "to_latex",
    "to_markdown",
    "to_netcdf",
    "to_xml",
    "to_orc",
    "savefig",
    "tofile",
    "write_text",
    "write_bytes",
}
NUMPY_WRITERS = {"save", "savez", "savez_compressed", "savetxt"}
NUMPY_READERS = {"load", "loadtxt", "genfromtxt", "fromfile"}
FS_MUTATORS = {
    "remove",
    "unlink",
    "rmtree",
    "rename",
    "replace",
    "makedirs",
    "mkdir",
    "copy",
    "copyfile",
    "copytree",
    "move",
    "rmdir",
}
PLOT_BASES = {"plt", "sns", "px", "go", "alt", "pyplot", "seaborn"}
PLOT_METHODS = {
    "plot",
    "hist",
    "scatter",
    "bar",
    "barh",
    "imshow",
    "boxplot",
    "show",
    "savefig",
    "figure",
    "subplots",
    "heatmap",
    "lineplot",
    "scatterplot",
    "pairplot",
    "histplot",
    "countplot",
    "violinplot",
    "kdeplot",
    "displot",
    "catplot",
    "regplot",
    "tight_layout",
    "set_title",
    "set_xlabel",
    "set_ylabel",
    "legend",
    "pie",
    "area",
    "line",
    "box",
    "kde",
}
RANDOM_FNS = {
    "rand",
    "randn",
    "randint",
    "random",
    "choice",
    "shuffle",
    "permutation",
    "normal",
    "uniform",
    "binomial",
    "poisson",
    "sample",
    "random_sample",
    "standard_normal",
    "exponential",
    "beta",
    "gamma",
    "multivariate_normal",
    "integers",
    "gauss",
    "randrange",
}
RANDOM_ESTIMATORS = {
    "train_test_split",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "KMeans",
    "MiniBatchKMeans",
    "MLPClassifier",
    "MLPRegressor",
    "SGDClassifier",
    "SGDRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomizedSearchCV",
    "TSNE",
    "IsolationForest",
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "GaussianMixture",
    "RandomTreesEmbedding",
}
# scikit-learn composites that keep references to their steps instead of copying them.
COMPOSITE_CTORS = {"make_pipeline", "Pipeline", "make_union", "FeatureUnion"}
SEED_CALLS = {"seed", "manual_seed", "set_seed", "set_random_seed"}
CONFIG_CALLS = {
    "set_option",
    "set_printoptions",
    "filterwarnings",
    "simplefilter",
    "chdir",
    "use",
    "set_theme",
    "set_style",
    "set_context",
    "rc",
}
LOAD_SIGNALS = {
    "loadtxt",
    "genfromtxt",
    "connect",
    "load_dataset",
    "read_sql",
    "urlopen",
    "urlretrieve",
    "open_dataset",
}
CLEAN_SIGNALS = {
    "dropna",
    "fillna",
    "drop_duplicates",
    "drop",
    "astype",
    "rename",
    "replace",
    "to_datetime",
    "to_numeric",
    "strip",
    "lower",
    "upper",
    "str",
    "clip",
    "query",
    "isna",
    "notna",
    "isnull",
    "notnull",
    "interpolate",
    "ffill",
    "bfill",
    "reset_index",
    "set_index",
    "duplicated",
    "where",
    "mask",
    "copy",
    "contains",
    "split",
    "infer_objects",
}
FEATURE_SIGNALS = {
    "get_dummies",
    "train_test_split",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "PolynomialFeatures",
    "ColumnTransformer",
    "SimpleImputer",
    "PCA",
    "TfidfVectorizer",
    "CountVectorizer",
    "transform",
    "fit_transform",
    "resample",
    "groupby",
    "agg",
    "aggregate",
    "pivot",
    "pivot_table",
    "merge",
    "concat",
    "join",
    "apply",
    "map",
    "rolling",
    "cut",
    "qcut",
    "shift",
    "diff",
    "pct_change",
    "cumsum",
    "melt",
    "crosstab",
    "value_counts",
    "sum",
    "mean",
    "median",
    "weekday",
    "dayofweek",
    "month",
    "year",
}
TRAIN_SIGNALS = {
    "fit",
    "partial_fit",
    "GridSearchCV",
    "RandomizedSearchCV",
    "make_pipeline",
    "Pipeline",
    "cross_val_predict",
}
EVAL_SIGNALS = {
    "predict",
    "predict_proba",
    "score",
    "decision_function",
    "confusion_matrix",
    "classification_report",
    "cross_val_score",
    "cross_validate",
    "log_loss",
}
_ESTIMATOR_NAME = re.compile(r"(Classifier|Regressor|Regression|SVC|SVR|KMeans|CV|Model|Booster)$")
_METRIC_NAME = re.compile(r"(_score|_error|_loss)$")


def _is_ipython_name(name: str) -> bool:
    return name in IPYTHON_NAMES or bool(_IPY_NUMBERED.match(name))


def _dotted(node: ast.AST) -> str:
    """Best effort dotted name for a call target: pd.read_csv, df.to_csv, ?.dropna."""
    parts: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Name):
            parts.append(node.id)
            break
        elif isinstance(node, ast.Call):
            node = node.func
            parts.append("()")
        elif isinstance(node, ast.Subscript):
            node = node.value
            parts.append("[]")
        else:
            parts.append("?")
            break
    return ".".join(reversed(parts))


def _root_name(node: ast.AST) -> str | None:
    """Root variable of an attribute/subscript chain (df['a'].x -> df). Stops at calls."""
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _literal_path(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            else:
                out.append("{...}")
        return "".join(out)
    return None


def _kw(node: ast.Call, *names: str) -> ast.AST | None:
    for k in node.keywords:
        if k.arg in names:
            return k.value
    return None


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_target_names(elt))
        return out
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _bound_in(node: ast.AST) -> set[str]:
    """All names bound anywhere inside node (params, assignments, loop targets, defs)."""
    bound: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del):
            bound.add(sub.id)
        elif isinstance(sub, ast.arg):
            bound.add(sub.arg)
        elif isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(sub.name)
        elif isinstance(sub, ast.alias):
            bound.add((sub.asname or sub.name).split(".")[0])
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
    return bound


def _composite_call(node: ast.AST) -> tuple[ast.Call | None, bool]:
    """Find make_pipeline(...) in `make_pipeline(...)` or `make_pipeline(...).fit(...)`.
    Returns the constructor call and whether a mutating method was chained onto it."""
    mutated = False
    while isinstance(node, ast.Call):
        name = _dotted(node.func).split(".")[-1]
        if name in COMPOSITE_CTORS:
            return node, mutated
        if isinstance(node.func, ast.Attribute):
            mutated = mutated or node.func.attr in MUTATING_METHODS
            node = node.func.value
        else:
            break
    return None, False


def _direct_names(call: ast.Call) -> list[str]:
    """Names passed into a call directly or inside tuples/lists, not inside nested calls."""
    out: list[str] = []

    def walk(n: ast.AST) -> None:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in out:
                out.append(n.id)
        elif isinstance(n, ast.Tuple | ast.List):
            for e in n.elts:
                walk(e)
        elif isinstance(n, ast.keyword):
            walk(n.value)

    for a in call.args:
        walk(a)
    for k in call.keywords:
        walk(k)
    return out


def _loads_in(node: ast.AST) -> list[str]:
    out: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            out.append(sub.id)
    return out


@dataclass
class FunctionInfo:
    name: str
    params: list[str]
    mutated_params: set[str]
    global_mutations: set[str]
    free_names: set[str]
    cell_index: int


@dataclass
class CellFacts:
    cell: Cell
    composites: dict[str, list[str]] = field(default_factory=dict)
    defines: list[str] = field(default_factory=list)
    assigned: list[str] = field(default_factory=list)
    plot_objects: set[str] = field(default_factory=set)
    loop_vars: set[str] = field(default_factory=set)
    conditional: set[str] = field(default_factory=set)
    imports: dict[str, str] = field(default_factory=dict)
    functions: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    reads: list[str] = field(default_factory=list)
    deferred_reads: set[str] = field(default_factory=set)
    mutates: dict[str, list[str]] = field(default_factory=dict)
    self_rebinds: set[str] = field(default_factory=set)
    deletes: set[str] = field(default_factory=set)
    files_read: list[dict[str, str]] = field(default_factory=list)
    files_written: list[dict[str, str]] = field(default_factory=list)
    fs_changes: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    shell: list[str] = field(default_factory=list)
    plots: bool = False
    displays: bool = False
    prints: bool = False
    config: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    random_calls: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    user_calls: list[tuple[str, list[str | None], dict[str, str]]] = field(default_factory=list)
    star_imports: list[str] = field(default_factory=list)

    def define(self, name: str, *, conditional: bool = False) -> None:
        if name not in self.defines:
            self.defines.append(name)
        if conditional:
            self.conditional.add(name)

    def mutate(self, name: str, reason: str) -> None:
        self.mutates.setdefault(name, [])
        if reason not in self.mutates[name]:
            self.mutates[name].append(reason)


class _CellVisitor:
    """Walks one cell's module-level statements in execution order."""

    def __init__(self, facts: CellFacts, functions: dict[str, FunctionInfo]):
        self.f = facts
        self.bound: set[str] = set()
        self.functions = functions

    # reads -----------------------------------------------------------------
    def _read(self, name: str) -> None:
        if name not in self.bound and name not in self.f.reads:
            self.f.reads.append(name)

    def expr(self, node: ast.AST | None) -> None:
        if node is None:
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                self._read(node.id)
            return
        if isinstance(node, ast.Lambda):
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                self.expr(d)
            free = set(_loads_in(node.body)) - _bound_in(node)
            self.f.deferred_reads |= free - self.bound
            return
        if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            comp_bound = set()
            for gen in node.generators:
                comp_bound |= set(_target_names(gen.target))
            # walrus targets inside comprehensions bind in the enclosing scope
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    self.expr(sub.value)
                if isinstance(sub, ast.Call):
                    self._call(sub)
            for name in _loads_in(node):
                if name not in comp_bound:
                    self._read(name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    self._bind_target(sub.target)
            return
        if isinstance(node, ast.NamedExpr):
            self.expr(node.value)
            self._bind_target(node.target)
            return
        if isinstance(node, ast.Call):
            self.expr(node.func)
            for a in node.args:
                self.expr(a)
            for k in node.keywords:
                self.expr(k.value)
            self._call(node)
            return
        for child in ast.iter_child_nodes(node):
            self.expr(child)

    # binds -----------------------------------------------------------------
    def _bind_name(self, name: str, *, loop: bool = False, assigned: bool = True) -> None:
        self.bound.add(name)
        self.f.define(name, conditional=self.depth > 0)
        if loop:
            self.f.loop_vars.add(name)
        elif assigned and name not in self.f.assigned:
            self.f.assigned.append(name)

    def _bind_target(self, target: ast.AST, *, loop: bool = False) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, loop=loop)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._bind_target(elt, loop=loop)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, loop=loop)
        elif isinstance(target, ast.Attribute | ast.Subscript):
            # x.a = ... or x[k] = ... mutates x
            self.expr(target.value)
            if isinstance(target, ast.Subscript):
                self.expr(target.slice)
            root = _root_name(target)
            if root is not None:
                self._read(root)
                kind = "attribute set" if isinstance(target, ast.Attribute) else "item set"
                self.f.mutate(root, kind)
                if root in {"plt", "matplotlib", "mpl"} or _dotted(target).startswith(
                    ("pd.options", "os.environ")
                ):
                    self.f.config.append(_dotted(target))

    depth = 0

    # statements ------------------------------------------------------------
    def stmts(self, body: list[ast.stmt], *, top: bool = False) -> None:
        for i, stmt in enumerate(body):
            self.stmt(stmt)
            if top and i == len(body) - 1 and isinstance(stmt, ast.Expr):
                v = stmt.value
                is_magic = isinstance(v, ast.Call) and _dotted(v.func).startswith("get_ipython")
                if not is_magic:
                    self.f.displays = True

    def _block(self, body: list[ast.stmt]) -> None:
        self.depth += 1
        try:
            self.stmts(body)
        finally:
            self.depth -= 1

    def stmt(self, s: ast.stmt) -> None:
        f = self.f
        if isinstance(s, ast.Assign):
            self.expr(s.value)
            plotty = isinstance(s.value, ast.Call) and _is_plot_call(s.value)
            for t in s.targets:
                self._bind_target(t)
                if plotty:
                    f.plot_objects |= set(_target_names(t))
            ctor, fitted_now = _composite_call(s.value)
            if ctor is not None and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                members = _direct_names(ctor)
                if members:
                    f.composites[s.targets[0].id] = members
                    if fitted_now:
                        for m in members:
                            f.mutate(m, f"fitted through `{s.targets[0].id}`, which holds the same object")
            names_read_in_value = set(_loads_in(s.value))
            for t in s.targets:
                for n in _target_names(t):
                    if n in names_read_in_value:
                        f.self_rebinds.add(n)
        elif isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self.expr(s.value)
                self._bind_target(s.target)
        elif isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name):
                self._read(s.target.id)
                self.expr(s.value)
                self._bind_name(s.target.id)
                f.self_rebinds.add(s.target.id)
            else:
                self.expr(s.value)
                self._bind_target(s.target)
        elif isinstance(s, ast.For | ast.AsyncFor):
            self.expr(s.iter)
            self.depth += 1
            self._bind_target(s.target, loop=True)
            self.depth -= 1
            self._block(s.body)
            self._block(s.orelse)
        elif isinstance(s, ast.While | ast.If):
            self.expr(s.test)
            self._block(s.body)
            self._block(s.orelse)
        elif isinstance(s, ast.With | ast.AsyncWith):
            for item in s.items:
                self.expr(item.context_expr)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars)
            self.stmts(s.body)
        elif isinstance(s, ast.Try) or (hasattr(ast, "TryStar") and isinstance(s, ast.TryStar)):
            self._block(s.body)
            for h in s.handlers:
                self.expr(h.type)
                if h.name:
                    self._bind_name(h.name, assigned=False)
                self._block(h.body)
            self._block(s.orelse)
            self.stmts(s.finalbody)
        elif isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef):
            for d in s.decorator_list:
                self.expr(d)
            for d in s.args.defaults + [d for d in s.args.kw_defaults if d is not None]:
                self.expr(d)
            info = _function_info(s, f.cell.index)
            self.functions[s.name] = info
            f.deferred_reads |= info.free_names - self.bound
            for g in _global_assigns(s):
                self._bind_name(g)
            self._bind_name(s.name, assigned=False)
            f.functions.add(s.name)
        elif isinstance(s, ast.ClassDef):
            for d in s.decorator_list + s.bases + [k.value for k in s.keywords]:
                self.expr(d)
            local = _bound_in(s)
            for stmt in s.body:
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    f.deferred_reads |= (set(_loads_in(stmt)) - _bound_in(stmt)) - self.bound
                else:
                    for n in _loads_in(stmt):
                        if n not in local:
                            self._read(n)
            self._bind_name(s.name, assigned=False)
            f.classes.add(s.name)
        elif isinstance(s, ast.Import):
            for a in s.names:
                name = (a.asname or a.name).split(".")[0] if a.asname is None else a.asname
                f.imports[name] = a.name
                self._bind_name(name, assigned=False)
        elif isinstance(s, ast.ImportFrom):
            mod = ("." * s.level) + (s.module or "")
            for a in s.names:
                if a.name == "*":
                    f.star_imports.append(mod)
                    continue
                name = a.asname or a.name
                f.imports[name] = f"{mod}.{a.name}"
                self._bind_name(name, assigned=False)
        elif isinstance(s, ast.Expr):
            self.expr(s.value)
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    self._read(t.id)
                    f.deletes.add(t.id)
                    self.bound.discard(t.id)
                else:
                    self.expr(t.value if isinstance(t, ast.Attribute | ast.Subscript) else t)
                    root = _root_name(t)
                    if root:
                        f.mutate(root, "del item")
        elif isinstance(s, ast.Global | ast.Nonlocal | ast.Pass | ast.Break | ast.Continue):
            pass
        elif hasattr(ast, "Match") and isinstance(s, ast.Match):
            self.expr(s.subject)
            for case in s.cases:
                for n in _bound_in(case.pattern):
                    self._bind_name(n)
                self._block(case.body)
        else:
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.expr):
                    self.expr(child)

    # calls -----------------------------------------------------------------
    def _call(self, node: ast.Call) -> None:
        f = self.f
        dotted = _dotted(node.func)
        parts = dotted.split(".")
        last = parts[-1]
        base = parts[0]
        f.calls.append(last)
        first_arg = node.args[0] if node.args else None
        kwnames = {k.arg for k in node.keywords if k.arg}

        # magics
        if base == "get_ipython":
            return

        # user-defined function calls (resolved after all cells are seen)
        if isinstance(node.func, ast.Name):
            arg_names = [a.id if isinstance(a, ast.Name) else None for a in node.args]
            kw_names = {
                k.arg: k.value.id for k in node.keywords if k.arg and isinstance(k.value, ast.Name)
            }
            f.user_calls.append((node.func.id, arg_names, kw_names))

        # mutation through methods
        if isinstance(node.func, ast.Attribute):
            root = _root_name(node.func.value)
            inplace = any(
                k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in node.keywords
            )
            if root is not None and root not in f.imports and root not in {"np", "pd", "plt"}:
                if last in MUTATING_METHODS:
                    f.mutate(root, f".{last}()")
                elif inplace:
                    f.mutate(root, f".{last}(inplace=True)")
            if (
                base in {"np", "numpy", "random"}
                and last == "shuffle"
                and isinstance(first_arg, ast.Name)
            ):
                f.mutate(first_arg.id, f"{dotted}()")

        # file reads
        if last.startswith("read_") or (base in {"np", "numpy"} and last in NUMPY_READERS):
            path = _literal_path(first_arg) or _literal_path(
                _kw(node, "filepath_or_buffer", "path", "io", "fname", "file")
            )
            f.files_read.append({"path": path or "<dynamic>", "call": dotted})
            if path and path.startswith(("http://", "https://")):
                f.network.append(dotted)
        elif (
            dotted == "open"
            or last in {"open_dataset", "connect"}
            or dotted in {"joblib.load", "pickle.load", "torch.load", "Image.open"}
        ):
            path = _literal_path(first_arg) or _literal_path(_kw(node, "file", "filename", "path"))
            mode_node = node.args[1] if len(node.args) > 1 else _kw(node, "mode")
            mode = (
                mode_node.value
                if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str)
                else "r"
            )
            entry = {"path": path or "<dynamic>", "call": dotted}
            if any(c in mode for c in "wax+"):
                f.files_written.append(entry)
            else:
                f.files_read.append(entry)

        # file writes
        if last in FILE_WRITE_METHODS and (
            node.args or kwnames & {"path", "path_or_buf", "fname", "excel_writer", "name"}
        ):
            path = _literal_path(first_arg) or _literal_path(
                _kw(node, "path_or_buf", "path", "fname", "excel_writer")
            )
            f.files_written.append({"path": path or "<dynamic>", "call": dotted})
        elif base in {"np", "numpy"} and last in NUMPY_WRITERS:
            f.files_written.append(
                {"path": _literal_path(first_arg) or "<dynamic>", "call": dotted}
            )
        elif dotted in {"joblib.dump", "torch.save", "pickle.dump", "json.dump"}:
            target = node.args[1] if len(node.args) > 1 else None
            f.files_written.append({"path": _literal_path(target) or "<dynamic>", "call": dotted})
        elif base in {"os", "shutil"} and last in FS_MUTATORS:
            f.fs_changes.append(dotted)

        # network, shell
        if (
            last in {"urlopen", "urlretrieve", "load_dataset"}
            or base in {"requests", "httpx", "wget"}
            or last.startswith("fetch_")
        ):
            f.network.append(dotted)
        for a in node.args:
            p = _literal_path(a)
            if p and p.startswith(("http://", "https://")) and dotted not in f.network:
                f.network.append(dotted)
        if base == "subprocess" or dotted in {"os.system", "os.popen"}:
            f.shell.append(dotted)

        # output
        if dotted in {"print", "pprint", "pprint.pprint"}:
            f.prints = True
        if dotted == "display":
            f.displays = True
        if _is_plot_call(node):
            f.plots = True

        # config and seeds
        if last in SEED_CALLS and base in {
            "np",
            "numpy",
            "random",
            "torch",
            "tf",
            "keras",
            "tensorflow",
        }:
            f.seeds.append(dotted)
        elif (
            last in CONFIG_CALLS
            and base
            in {"pd", "pandas", "np", "numpy", "warnings", "os", "plt", "sns", "matplotlib", "mpl"}
        ) or (base == "sys" and "path" in parts):
            f.config.append(dotted)

        # randomness
        if (
            (
                base in {"np", "numpy"}
                and len(parts) >= 3
                and parts[1] == "random"
                and last in RANDOM_FNS
            )
            or (base == "random" and last in RANDOM_FNS)
            or (last == "default_rng" and not node.args and "seed" not in kwnames)
        ):
            f.random_calls.append(dotted)
        elif last in RANDOM_ESTIMATORS and "random_state" not in kwnames:
            f.random_calls.append(f"{last}() without random_state")
        elif (
            last == "sample"
            and isinstance(node.func, ast.Attribute)
            and base not in {"random", "np"}
            and "random_state" not in kwnames
        ):
            f.random_calls.append(f"{dotted}() without random_state")


def _is_plot_call(node: ast.Call) -> bool:
    dotted = _dotted(node.func)
    parts = dotted.split(".")
    if parts[0] in PLOT_BASES:
        return True
    if "plot" in parts[:-1]:  # df.plot.bar()
        return True
    return isinstance(node.func, ast.Attribute) and parts[-1] in PLOT_METHODS


def _function_info(fn: ast.FunctionDef | ast.AsyncFunctionDef, cell_index: int) -> FunctionInfo:
    a = fn.args
    params = [p.arg for p in a.posonlyargs + a.args] + ([a.vararg.arg] if a.vararg else [])
    params += [p.arg for p in a.kwonlyargs]
    globals_declared = _global_assigns(fn)
    local = (_bound_in(fn) - {fn.name} - globals_declared) | set(params)
    free: set[str] = set()
    for stmt in fn.body:
        free |= set(_loads_in(stmt))
    free -= local
    mutated_params: set[str] = set()
    global_mut: set[str] = set()
    for sub in ast.walk(fn):
        root = None
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            inplace = any(
                k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in sub.keywords
            )
            if sub.func.attr in MUTATING_METHODS or inplace:
                root = _root_name(sub.func.value)
        elif isinstance(sub, ast.Assign | ast.AugAssign):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for t in targets:
                if isinstance(t, ast.Attribute | ast.Subscript):
                    r = _root_name(t)
                    if r in params:
                        mutated_params.add(r)
                    elif r and r not in local:
                        global_mut.add(r)
        if root is None:
            continue
        if root in params:
            mutated_params.add(root)
        elif root not in local:
            global_mut.add(root)
    return FunctionInfo(fn.name, params, mutated_params, global_mut, free, cell_index)


def _global_assigns(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Global):
            names |= set(sub.names)
    return names


# --------------------------------------------------------------------------
# Whole-notebook analysis
# --------------------------------------------------------------------------


def _and_list(names: list[str]) -> str:
    q = [f"`{n}`" for n in names]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + " and " + q[-1]


def _label(c: Cell) -> str:
    ec = f", In [{c.execution_count}]" if c.execution_count is not None else ""
    return f"cell {c.index + 1}{ec}"


def _is_known_external(name: str) -> bool:
    return name in BUILTIN_NAMES or _is_ipython_name(name)


def analyze_cells(cells: list[Cell]) -> dict[str, Any]:
    functions: dict[str, FunctionInfo] = {}
    facts: list[CellFacts] = []
    for c in cells:
        cf = CellFacts(cell=c)
        if c.tree is not None:
            v = _CellVisitor(cf, functions)
            v.stmts(c.tree.body, top=True)
        for m in c.magics:
            if m.kind == "shell" or m.name in {"bash", "sh", "system", "script"}:
                cf.shell.append(f"{m.name} {m.args}".strip())
            if m.name in {
                "run",
                "load",
                "store",
                "autoreload",
                "load_ext",
                "writefile",
                "cd",
                "env",
            }:
                cf.config.append(f"%{m.name} {m.args}".strip())
            if m.name == "writefile":
                cf.files_written.append(
                    {"path": m.args.split()[-1] if m.args else "<dynamic>", "call": "%%writefile"}
                )
        facts.append(cf)

    # Resolve calls to notebook-defined functions: param mutation, global mutation, deferred reads.
    call_reads: dict[int, set[str]] = {}
    for cf in facts:
        for fname, args, kws in cf.user_calls:
            info = functions.get(fname)
            if info is None:
                continue
            for pos, argname in enumerate(args):
                if argname and pos < len(info.params) and info.params[pos] in info.mutated_params:
                    cf.mutate(argname, f"passed to {fname}() which mutates it")
            for k, argname in kws.items():
                if k in info.mutated_params:
                    cf.mutate(argname, f"passed to {fname}() which mutates it")
            for g in info.global_mutations:
                cf.mutate(g, f"{fname}() mutates global")
            call_reads.setdefault(cf.cell.index, set()).update(
                info.free_names - set(functions) - {fname}
            )

    # Fitting a pipeline fits the very objects it was built from (no copies are made).
    composites: dict[str, list[str]] = {}
    member_of: dict[str, list[tuple[str, int]]] = {}
    for i, cf in enumerate(facts):
        for comp, members in cf.composites.items():
            composites[comp] = members
            for m in members:
                member_of.setdefault(m, []).append((comp, i))
        for name in list(cf.mutates):
            for m in composites.get(name, []):
                cf.mutate(m, f"fitted through `{name}`, which holds the same object")
    shared_findings: list[dict[str, Any]] = []
    for m, comps in member_of.items():
        owners = sorted({c for c, _ in comps})
        if len(owners) > 1:
            shared_findings.append(
                {
                    "kind": "shared_estimator",
                    "severity": "warning",
                    "cells": sorted({facts[i].cell.index for _, i in comps}),
                    "names": [m, *owners],
                    "message": f"`{m}` is a step of {_and_list(owners)} at the same time. Pipelines keep the object itself, not a copy, so fitting one refits `{m}` inside the other{'s' if len(owners) > 2 else ''} too. Use separate instances (or sklearn.base.clone) unless sharing is intended.",
                }
            )

    pos = {cf.cell.index: i for i, cf in enumerate(facts)}
    defs: dict[str, list[int]] = {}
    muts: dict[str, list[int]] = {}
    for i, cf in enumerate(facts):
        for n in cf.defines:
            defs.setdefault(n, []).append(i)
        for n in cf.mutates:
            muts.setdefault(n, []).append(i)

    star = [i for i, cf in enumerate(facts) if cf.star_imports]
    findings: list[dict[str, Any]] = []
    edges: dict[tuple[int, int, str], set[str]] = {}

    def add_edge(src: int, dst: int, kind: str, name: str) -> None:
        edges.setdefault((src, dst, kind), set()).add(name)

    reaching: dict[int, dict[str, int | None]] = {}
    for i, cf in enumerate(facts):
        reaching[i] = {}
        effective_reads = list(cf.reads) + sorted(
            call_reads.get(cf.cell.index, set()) - set(cf.reads)
        )
        for n in effective_reads:
            if n in cf.functions:
                continue
            via_call = n not in cf.reads
            earlier = [d for d in defs.get(n, []) if d < i or (via_call and d == i)]
            if earlier:
                d = earlier[-1]
                reaching[i][n] = d
                if d != i:
                    add_edge(d, i, "def-use", n)
                for m in muts.get(n, []):
                    if d < m < i:
                        add_edge(m, i, "mutation", n)
                continue
            reaching[i][n] = None
            if _is_known_external(n):
                continue
            later = [d for d in defs.get(n, []) if d > i]
            how = " (read inside a function called here)" if via_call else ""
            if later:
                findings.append(
                    {
                        "kind": "use_before_def",
                        "severity": "error",
                        "cells": [cf.cell.index, facts[later[0]].cell.index],
                        "names": [n],
                        "message": f"{_label(cf.cell)} reads `{n}`{how} but it is first defined later, in {_label(facts[later[0]].cell)}. A top-to-bottom run will fail here with NameError.",
                    }
                )
            elif any(s < i for s in star):
                findings.append(
                    {
                        "kind": "maybe_star_import",
                        "severity": "warning",
                        "cells": [cf.cell.index],
                        "names": [n],
                        "message": f"{_label(cf.cell)} reads `{n}`{how}, which no cell defines. It may come from a `from ... import *`.",
                    }
                )
            else:
                findings.append(
                    {
                        "kind": "undefined_name",
                        "severity": "error",
                        "cells": [cf.cell.index],
                        "names": [n],
                        "message": f"{_label(cf.cell)} reads `{n}`{how}, which no cell in the notebook defines. It probably came from a deleted cell or from state outside the notebook. A top-to-bottom run will fail here with NameError.",
                    }
                )

    for cf in facts:
        missing = sorted(
            n for n in cf.deferred_reads if n not in defs and not _is_known_external(n) and not star
        )
        if missing:
            findings.append(
                {
                    "kind": "function_reads_undefined",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": missing,
                    "message": f"A function or lambda defined in {_label(cf.cell)} reads {', '.join(f'`{m}`' for m in missing)}, which no cell defines. Calling it will raise NameError.",
                }
            )

    findings.extend(shared_findings)
    findings.extend(_execution_order_findings(facts, defs))
    findings.extend(_other_findings(facts, defs, muts, edges))

    used_later: dict[int, bool] = {i: False for i in range(len(facts))}
    for src, dst, _kind in edges:
        if dst > src:
            used_later[src] = True

    stages = _propose_stages(facts, edges, used_later, reaching, defs)
    artifacts = _suggest_artifacts(facts, defs, edges)

    for i, cf in enumerate(facts):
        if used_later[i] or cf.cell.is_empty:
            continue
        if cf.files_written or cf.network or cf.shell or cf.config or cf.seeds or cf.mutates:
            continue
        if (
            not cf.defines
            or all(n in cf.plot_objects or n in cf.functions for n in cf.defines)
            or cf.displays
            or cf.plots
            or cf.prints
        ):
            if cf.defines and any(n in [a["name"] for a in artifacts] for n in cf.defines):
                continue
            findings.append(
                {
                    "kind": "inspection_only",
                    "severity": "info",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} only displays, prints or plots, and no later cell depends on it. Keep it in the report stage or drop it from the pipeline.",
                }
            )

    graph_nodes = []
    for i, cf in enumerate(facts):
        c = cf.cell
        graph_nodes.append(
            {
                "index": c.index,
                "label": _label(c),
                "execution_count": c.execution_count,
                "first_line": c.source.strip().splitlines()[0][:80] if c.source.strip() else "",
                "defines": cf.defines,
                "reads": [n for n in cf.reads if not _is_known_external(n) or n in defs],
                "deferred_reads": sorted(cf.deferred_reads),
                "mutates": cf.mutates,
                "imports": cf.imports,
                "functions": sorted(cf.functions),
                "files_read": cf.files_read,
                "files_written": cf.files_written,
                "network": cf.network,
                "shell": cf.shell,
                "magics": [{"kind": m.kind, "name": m.name, "args": m.args} for m in c.magics],
                "config": cf.config,
                "random_unseeded": cf.random_calls,
                "seeds": cf.seeds,
                "plots": cf.plots,
                "displays": cf.displays,
                "prints": cf.prints,
                "parse_error": c.parse_error,
                "used_by_later_cells": used_later[i],
            }
        )
    graph_edges = [
        {"from": facts[s].cell.index, "to": facts[d].cell.index, "kind": k, "names": sorted(ns)}
        for (s, d, k), ns in sorted(edges.items())
    ]
    sev_order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda x: (sev_order[x["severity"]], x["cells"][0] if x["cells"] else -1))
    del pos
    return {
        "cells": graph_nodes,
        "edges": graph_edges,
        "findings": findings,
        "stages": stages,
        "suggested_artifacts": artifacts,
        "inputs": _inputs(facts),
    }


def _execution_order_findings(
    facts: list[CellFacts], defs: dict[str, list[int]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    executed = [
        (i, cf.cell.execution_count)
        for i, cf in enumerate(facts)
        if cf.cell.execution_count is not None
    ]
    nonempty = [i for i, cf in enumerate(facts) if not cf.cell.is_empty]
    if not executed:
        out.append(
            {
                "kind": "no_execution_history",
                "severity": "info",
                "cells": [],
                "names": [],
                "message": "The notebook has no saved execution counts (outputs were cleared or it was never run), so execution-order checks were skipped.",
            }
        )
        return out
    running_max = -1
    max_cell = None
    late: list[str] = []
    late_cells: list[int] = []
    for i, ec in executed:
        if ec < running_max and max_cell is not None:
            late.append(f"{_label(facts[i].cell)} ran before {_label(facts[max_cell].cell)}")
            late_cells.append(facts[i].cell.index)
        if ec > running_max:
            running_max = ec
            max_cell = i
    if late:
        out.append(
            {
                "kind": "out_of_order_execution",
                "severity": "warning",
                "cells": late_cells,
                "names": [],
                "message": "Saved execution counts are out of notebook order: "
                + "; ".join(late)
                + ". The saved outputs reflect a different order than a top-to-bottom run.",
            }
        )
    counts = sorted(ec for _, ec in executed)
    hidden = counts[-1] - len(counts) - (counts[0] - 1 if counts[0] > 1 else 0)
    first_hidden = counts[0] - 1 if counts[0] > 1 else 0
    total_hidden = max(hidden, 0) + first_hidden
    if total_hidden > 0:
        out.append(
            {
                "kind": "hidden_executions",
                "severity": "warning",
                "cells": [],
                "names": [],
                "message": f"Execution counts run up to {counts[-1]} but only {len(counts)} executed cells are saved, so {total_hidden} executions are not visible in the notebook (re-runs or deleted cells). Kernel state from those runs may have fed the saved outputs.",
            }
        )
    never = [facts[i].cell.index for i in nonempty if facts[i].cell.execution_count is None]
    if never and len(executed) > 0:
        out.append(
            {
                "kind": "never_executed",
                "severity": "info",
                "cells": never,
                "names": [],
                "message": f"{len(never)} non-empty code cell(s) have no execution count in the saved session: "
                + ", ".join(f"cell {c + 1}" for c in never)
                + ".",
            }
        )

    # Saved outputs computed from a definition other than the one a clean run would use.
    for i, cf in enumerate(facts):
        ec = cf.cell.execution_count
        if ec is None:
            continue
        for n in cf.reads:
            if _is_known_external(n) or n in cf.functions:
                continue
            all_defs = defs.get(n, [])
            if not all_defs:
                continue
            clean = [d for d in all_defs if d < i]
            clean_def = clean[-1] if clean else None
            session = [
                d
                for d in all_defs
                if d != i
                and facts[d].cell.execution_count is not None
                and facts[d].cell.execution_count < ec
            ]
            session_def = (
                max(session, key=lambda d: facts[d].cell.execution_count) if session else None
            )
            if session_def == clean_def:
                continue
            if session_def is None:
                src = "an execution that is no longer in the notebook (a deleted cell or an earlier run)"
            else:
                src = f"{_label(facts[session_def].cell)}"
            clean_src = (
                _label(facts[clean_def].cell) if clean_def is not None else "nothing (NameError)"
            )
            out.append(
                {
                    "kind": "stale_output",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": [n],
                    "message": f"The saved output of {_label(cf.cell)} used `{n}` from {src}, but a top-to-bottom run takes it from {clean_src}. Its saved output may not reproduce.",
                }
            )
    return out


def _other_findings(facts, defs, muts, edges) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seeded_before = False
    for cf in facts:
        if cf.seeds:
            seeded_before = True
        rnd = [
            r
            for r in cf.random_calls
            if not (seeded_before and ("np.random" in r or r.startswith("random.")))
        ]
        if rnd:
            out.append(
                {
                    "kind": "unseeded_randomness",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} uses randomness without a fixed seed: {', '.join(sorted(set(rnd)))}. Outputs may change between runs; run `capture --repeat 2` to check.",
                }
            )
        if cf.star_imports:
            out.append(
                {
                    "kind": "star_import",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} uses `from {', '.join(cf.star_imports)} import *`, which hides which names it defines.",
                }
            )
        if cf.cell.parse_error:
            out.append(
                {
                    "kind": "parse_error",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} could not be analyzed: {cf.cell.parse_error}.",
                }
            )
        ext = [r["call"] for r in cf.files_read] + cf.network + cf.shell
        if cf.network:
            out.append(
                {
                    "kind": "network_access",
                    "severity": "warning",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} touches the network ({', '.join(sorted(set(cf.network)))}). The pipeline depends on remote data that can change.",
                }
            )
        del ext
        if cf.shell:
            out.append(
                {
                    "kind": "shell_command",
                    "severity": "info",
                    "cells": [cf.cell.index],
                    "names": [],
                    "message": f"{_label(cf.cell)} runs shell commands ({'; '.join(cf.shell)[:200]}). Their effects are not tracked.",
                }
            )
        for name in sorted(cf.self_rebinds):
            if name in cf.reads:  # value came from an earlier cell
                out.append(
                    {
                        "kind": "rerun_hazard",
                        "severity": "info",
                        "cells": [cf.cell.index],
                        "names": [name],
                        "message": f"{_label(cf.cell)} rebinds `{name}` from its own previous value. Running it twice gives a different state than running it once.",
                    }
                )
    # Objects mutated in a cell other than where they were defined, and read later.
    for name, mcells in muts.items():
        dcells = defs.get(name, [])
        for m in mcells:
            prior = [d for d in dcells if d < m]
            if not prior or prior[-1] == m:
                continue
            readers = sorted(
                {
                    dst
                    for (src, dst, kind), ns in edges.items()
                    if src == m and kind == "mutation" and name in ns
                }
            )
            if not readers:
                continue
            reasons = ", ".join(facts[m].mutates.get(name, []))
            out.append(
                {
                    "kind": "cross_cell_mutation",
                    "severity": "warning",
                    "cells": [facts[m].cell.index] + [facts[r].cell.index for r in readers],
                    "names": [name],
                    "message": f"`{name}` (defined in {_label(facts[prior[-1]].cell)}) is changed in place in {_label(facts[m].cell)} ({reasons}) and later read by {', '.join(_label(facts[r].cell) for r in readers)}. That dependency is invisible in a def/use view: keep the order or make the change explicit.",
                }
            )
    return out


def _stage_scores(cf: CellFacts) -> dict[str, int]:
    s = dict.fromkeys(STAGES, 0)
    for call in cf.calls:
        if call.startswith(("read_", "load_", "fetch_")) or call in LOAD_SIGNALS:
            s["load"] += 3
        if call in CLEAN_SIGNALS:
            s["clean"] += 1
        if call in FEATURE_SIGNALS:
            s["features"] += 1
        if call in TRAIN_SIGNALS or _ESTIMATOR_NAME.search(call):
            s["train"] += 2
        if call in EVAL_SIGNALS or _METRIC_NAME.search(call):
            s["evaluate"] += 2
    if cf.files_read:
        s["load"] += 3
    return s


def _propose_stages(facts, edges, used_later, reaching, defs) -> list[dict[str, Any]]:
    n = len(facts)
    deps: dict[int, set[int]] = {i: set() for i in range(n)}
    for src, dst, _k in edges:
        if src < dst:
            deps[dst].add(src)
    assigned: list[str] = []
    reasons: list[str] = []
    prev = "load"
    for i, cf in enumerate(facts):
        if cf.cell.is_empty:
            assigned.append(prev)
            reasons.append("empty")
            continue
        scores = _stage_scores(cf)
        only_setup = cf.defines and all(d in cf.imports for d in cf.defines)
        output_only = (
            cf.plots
            or cf.displays
            or cf.prints
            or any(w["call"].endswith("savefig") for w in cf.files_written)
        )
        if (
            not used_later[i]
            and output_only
            and not cf.files_written
            and not scores["load"]
            and not scores["train"]
            and not scores["evaluate"]
        ):
            stage, why = (
                "report",
                "only produces output (display, print or plot) and nothing later depends on it",
            )
        elif only_setup or (
            not cf.defines and (cf.config or cf.seeds) and not any(scores.values())
        ):
            stage, why = "load", "imports and configuration"
        elif any(scores.values()):
            stage = max(STAGES[:-1], key=lambda s: (scores[s], -STAGE_RANK[s]))
            why = f"signals: {', '.join(f'{k}={v}' for k, v in scores.items() if v)}"
        else:
            stage, why = prev, "no strong signal, kept with the previous cell"
        for d in deps[i]:
            if STAGE_RANK[assigned[d]] > STAGE_RANK[stage] and assigned[d] != "report":
                stage, why = (
                    assigned[d],
                    f"moved to {assigned[d]} because it depends on {_label(facts[d].cell)}",
                )
        assigned.append(stage)
        reasons.append(why)
        if stage != "report":
            prev = stage

    order = sorted(range(n), key=lambda i: (STAGE_RANK[assigned[i]], i))
    newpos = {i: k for k, i in enumerate(order)}

    # Does reordering change what any cell sees?
    reorder_notes: dict[int, list[str]] = {}
    for i in range(n):
        for name, d in reaching[i].items():
            cands = [x for x in defs.get(name, []) if newpos[x] < newpos[i]]
            d_new = max(cands, key=lambda x: newpos[x]) if cands else None
            if d_new != d and d is not None and d != i:
                where = _label(facts[d_new].cell) if d_new is not None else "nothing"
                reorder_notes.setdefault(i, []).append(
                    f"reads `{name}`: in notebook order it comes from {_label(facts[d].cell)}, in this split it would come from {where}"
                )

    out = []
    for stage in STAGES:
        members = [i for i in order if assigned[i] == stage and not facts[i].cell.is_empty]
        if not members:
            continue
        member_set = set(members)
        inputs: set[str] = set()
        outputs: set[str] = set()
        for (src, dst, _k), names in edges.items():
            if dst in member_set and src not in member_set:
                inputs |= names
            if (
                src in member_set
                and dst not in member_set
                and STAGE_RANK[assigned[dst]] > STAGE_RANK[stage]
            ):
                outputs |= names
        imports_needed = sorted({m for i in members for m in facts[i].imports.values()})
        imported = {k for i in range(n) for k in facts[i].imports}
        inputs -= imported
        outputs -= imported
        out.append(
            {
                "stage": stage,
                "module": f"{stage}.py",
                "function": stage,
                "cells": [facts[i].cell.index for i in members],
                "inputs": sorted(inputs),
                "outputs": sorted(outputs),
                "imports": imports_needed,
                "signature": f"def {stage}({', '.join(sorted(inputs))}) -> dict",
                "why": {facts[i].cell.index: reasons[i] for i in members},
                "reorder_notes": {
                    facts[i].cell.index: reorder_notes[i] for i in members if i in reorder_notes
                },
            }
        )
    return out


def _suggest_artifacts(facts, defs, edges) -> list[dict[str, Any]]:
    """Top-level variables worth comparing: assigned data, not loops, imports, functions or plots."""
    final_def: dict[str, int] = {}
    for i, cf in enumerate(facts):
        for name in cf.assigned:
            if name.startswith("_") or name in cf.loop_vars:
                continue
            if name in cf.plot_objects:
                final_def.pop(name, None)
                continue
            final_def[name] = i
        for name in cf.defines:
            if name in final_def and name not in cf.assigned:
                final_def.pop(name, None)  # rebound to an import/function later
    out = []
    for name, i in final_def.items():
        read_after = any(
            src == i and name in ns for (src, dst, _k), ns in edges.items() if dst > src
        )
        out.append(
            {
                "name": name,
                "defined_in": facts[i].cell.index,
                "terminal": not read_after,
            }
        )
    return out


def _inputs(facts) -> dict[str, Any]:
    written = {w["path"] for cf in facts for w in cf.files_written}
    reads = []
    for cf in facts:
        for r in cf.files_read:
            reads.append({**r, "cell": cf.cell.index, "produced_by_notebook": r["path"] in written})
    return {
        "files_read": reads,
        "files_written": [{**w, "cell": cf.cell.index} for cf in facts for w in cf.files_written],
        "network": sorted({n for cf in facts for n in cf.network}),
        "imports": sorted(
            {m.split(".")[0] for cf in facts for m in cf.imports.values() if not m.startswith(".")}
        ),
    }


def analyze(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    nb = read_notebook(path)
    cells = load_cells(nb)
    result = analyze_cells(cells)
    kernelspec = nb.metadata.get("kernelspec", {}) if hasattr(nb, "metadata") else {}
    lang = nb.metadata.get("language_info", {}) if hasattr(nb, "metadata") else {}
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in result["findings"]:
        counts[f["severity"]] += 1
    return {
        "tool": {"name": "notebook-to-pipeline", "version": __version__},
        "notebook": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "cells_total": len(nb.cells),
            "code_cells": len(cells),
            "kernelspec": dict(kernelspec),
            "saved_language_version": lang.get("version"),
        },
        "summary": {
            "findings": counts,
            "predicts_top_to_bottom_failure": counts["error"] > 0,
            "stages": [s["stage"] for s in result["stages"]],
        },
        **result,
    }
