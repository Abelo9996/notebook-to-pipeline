from __future__ import annotations

import numpy as np
import pandas as pd

from notebook_to_pipeline.runtime import nb2p_compare as cmp
from notebook_to_pipeline.runtime import nb2p_probe as probe


def check(a, b, **opts):
    acc = cmp.Acc(cmp._opts(opts))
    ok = cmp.compare_values(a, b, "x", acc)
    return ok, acc


def test_floats_within_tolerance_and_nan():
    ok, acc = check(1.0, 1.0 + 1e-12)
    assert ok and acc.inexact
    assert check(float("nan"), float("nan"))[0]
    assert not check(1.0, 1.001)[0]
    assert check(1.0, 1.001, rtol=1e-2)[0]


def test_exact_types_for_ints_strings_bools():
    assert not check(1, 2)[0]
    assert not check("a", "b")[0]
    ok, acc = check(1, 1.0)
    assert not ok and "type changed" in acc.diffs[0]["why"]
    assert not check(True, 1)[0]


def test_nested_containers_report_path():
    ok, acc = check({"a": [1, 2, {"b": 3.0}]}, {"a": [1, 2, {"b": 4.0}]})
    assert not ok
    assert acc.diffs[0]["path"] == "x['a'][2]['b']"


def test_arrays():
    a = np.arange(6, dtype=float).reshape(2, 3)
    assert check(a, a + 1e-13)[0]
    b = a.copy()
    b[1, 2] = 99
    ok, acc = check(a, b)
    assert not ok and acc.diffs[0]["path"] == "x[1,2]"
    assert not check(a, a.astype(np.float32))[0]
    assert check(a, a.astype(np.float32), check_dtype=False)[0]
    assert not check(a, a.reshape(3, 2))[0]


def test_dataframe_values_and_first_difference():
    a = pd.DataFrame({"k": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    b = a.copy()
    b.loc[1, "v"] = 2.5
    ok, acc = check(a, b)
    assert not ok
    assert acc.diffs[0]["path"] == "x[row 1, 'v']"
    assert acc.mismatches == 1


def test_dataframe_schema_checks():
    a = pd.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    ok, acc = check(a, a[["v", "k"]])
    assert not ok and acc.diffs[0]["why"] == "column order differs"
    assert check(a, a[["v", "k"]], ignore_column_order=True)[0]
    ok, acc = check(a, a.assign(k=a["k"].astype(float)))
    assert not ok and acc.diffs[0]["why"] == "dtype differs"
    ok, acc = check(a, a.iloc[:1])
    assert not ok and acc.diffs[0]["why"] == "row count differs"
    ok, acc = check(a, a.drop(columns="v"))
    assert not ok and acc.diffs[0]["why"] == "columns differ"


def test_dataframe_row_order_and_index():
    a = pd.DataFrame({"k": [3, 1, 2], "v": [0.3, 0.1, 0.2]})
    shuffled = a.sort_values("k")
    ok, _ = check(a, shuffled)
    assert not ok
    assert check(a, shuffled, ignore_row_order=True)[0]
    assert not check(a, a.reset_index(drop=True).set_axis([10, 11, 12]))[0]
    assert check(a, a.set_axis([10, 11, 12]), ignore_index=True)[0]


def test_series_and_name():
    s = pd.Series([1.0, 2.0], name="a")
    assert check(s, s + 1e-14)[0]
    ok, acc = check(s, s.rename("b"))
    assert not ok and acc.diffs[0]["why"] == "series name differs"


def test_estimator_state_is_compared_by_fitted_values():
    from sklearn.linear_model import LinearRegression

    X = np.arange(10, dtype=float).reshape(-1, 1)
    m1 = LinearRegression().fit(X, 2 * X.ravel())
    m2 = LinearRegression().fit(X, 2 * X.ravel())
    m3 = LinearRegression().fit(X, 3 * X.ravel())
    s1, s2, s3 = (probe.estimator_state(m) for m in (m1, m2, m3))
    assert probe.value_hash(s1) == probe.value_hash(s2)
    ok, acc = check(s1, s3)
    assert not ok and "coef_" in acc.diffs[0]["path"]


def test_value_hash_ignores_dict_order_but_not_values():
    assert probe.value_hash({"a": 1, "b": 2}) == probe.value_hash({"b": 2, "a": 1})
    assert probe.value_hash({"a": 1}) != probe.value_hash({"a": 2})
    df = pd.DataFrame({"a": [1, 2]})
    assert probe.value_hash(df) == probe.value_hash(df.copy())
    assert probe.value_hash(df) != probe.value_hash(df.rename(columns={"a": "b"}))


def test_classify_skips_non_data():
    import math

    assert probe.classify(math)[0] is None
    assert probe.classify(len)[0] is None
    assert probe.classify(lambda: 1)[0] is None
    assert probe.classify(pd.DataFrame())[0] == "dataframe"
    assert probe.classify(np.float64(1.0))[0] == "scalar"
