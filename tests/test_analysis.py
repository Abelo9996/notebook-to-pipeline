from __future__ import annotations

from notebook_to_pipeline.analysis import analyze


def kinds(result):
    return [f["kind"] for f in result["findings"]]


def find(result, kind):
    return [f for f in result["findings"] if f["kind"] == kind]


def test_def_use_edges(nb_factory):
    nb = nb_factory(["x = 1", "y = x + 1", "print(y)"])
    r = analyze(nb)
    edges = {(e["from"], e["to"], tuple(e["names"])) for e in r["edges"]}
    assert (0, 1, ("x",)) in edges
    assert (1, 2, ("y",)) in edges
    assert r["summary"]["findings"]["error"] == 0


def test_use_before_definition_is_an_error(nb_factory):
    nb = nb_factory(["y = x * 2", "x = 3"])
    r = analyze(nb)
    f = find(r, "use_before_def")
    assert f and f[0]["severity"] == "error" and f[0]["names"] == ["x"]
    assert r["summary"]["predicts_top_to_bottom_failure"]


def test_name_from_deleted_cell(nb_factory):
    nb = nb_factory(["import math", "z = deleted_var + math.pi"])
    r = analyze(nb)
    f = find(r, "undefined_name")
    assert f and f[0]["names"] == ["deleted_var"]


def test_builtins_and_ipython_names_are_not_flagged(nb_factory):
    nb = nb_factory(["n = len([1, 2])\ndisplay(n)\nprint(sum(range(3)))"])
    r = analyze(nb)
    assert r["summary"]["findings"]["error"] == 0


def test_function_free_variable_defined_after_call(nb_factory):
    nb = nb_factory(["def score():\n    return model_x * 2", "s = score()", "model_x = 5"])
    r = analyze(nb)
    f = find(r, "use_before_def")
    assert f and f[0]["names"] == ["model_x"] and f[0]["cells"][0] == 1


def test_out_of_order_and_stale_output(tmp_path):
    from conftest import make_notebook

    nb = make_notebook(tmp_path / "nb.ipynb", [
        ("import pandas as pd", 1),
        ("s = pd.Series([1, 2, 3])", 9),
        ("s.head()", 4),
    ])
    r = analyze(nb)
    assert "out_of_order_execution" in kinds(r)
    assert "hidden_executions" in kinds(r)
    stale = find(r, "stale_output")
    assert stale and stale[0]["names"] == ["s"] and stale[0]["cells"] == [2]


def test_no_execution_history_is_reported(nb_factory):
    r = analyze(nb_factory(["a = 1"]))
    assert "no_execution_history" in kinds(r)


def test_cross_cell_mutation(nb_factory):
    nb = nb_factory(["items = []", "items.append(1)", "total = len(items)"])
    r = analyze(nb)
    f = find(r, "cross_cell_mutation")
    assert f and f[0]["names"] == ["items"]
    assert any(e["kind"] == "mutation" and e["from"] == 1 and e["to"] == 2 for e in r["edges"])


def test_mutation_through_notebook_function(nb_factory):
    nb = nb_factory([
        "def add_one(target):\n    target.append(1)",
        "values = []",
        "add_one(values)",
        "n = len(values)",
    ])
    r = analyze(nb)
    cell = next(c for c in r["cells"] if c["index"] == 2)
    assert "values" in cell["mutates"]
    assert find(r, "cross_cell_mutation")


def test_magics_and_shell_are_parsed(nb_factory):
    nb = nb_factory(["%matplotlib inline\nimport os", "!ls -la", "%%bash\necho hi"])
    r = analyze(nb)
    c0, c1, c2 = r["cells"]
    assert c0["magics"][0]["name"] == "matplotlib"
    assert "os" in c0["defines"]
    assert c1["shell"]
    assert c2["magics"][0]["name"] == "bash"
    assert all(c["parse_error"] is None for c in r["cells"])


def test_file_io_detection(nb_factory):
    nb = nb_factory([
        "import pandas as pd\ndf = pd.read_csv('data/in.csv')",
        "df.to_csv('out/clean.csv', index=False)\nwith open('notes.txt', 'w') as f:\n    f.write('x')",
    ])
    r = analyze(nb)
    assert r["cells"][0]["files_read"][0]["path"] == "data/in.csv"
    written = {w["path"] for w in r["cells"][1]["files_written"]}
    assert written == {"out/clean.csv", "notes.txt"}


def test_unseeded_randomness(nb_factory):
    r = analyze(nb_factory(["import numpy as np\nx = np.random.rand(3)"]))
    assert find(r, "unseeded_randomness")
    r = analyze(nb_factory(["import numpy as np\nnp.random.seed(0)", "x = np.random.rand(3)"], name="b.ipynb"))
    assert not find(r, "unseeded_randomness")
    r = analyze(nb_factory(["from sklearn.ensemble import RandomForestClassifier\nm = RandomForestClassifier()"], name="c.ipynb"))
    assert find(r, "unseeded_randomness")


def test_stage_proposal_follows_ml_shape(nb_factory):
    nb = nb_factory([
        "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\nimport matplotlib.pyplot as plt",
        "df = pd.read_csv('data.csv')",
        "df = df.dropna().rename(columns=str.lower)",
        "X_train, X_test, y_train, y_test = train_test_split(df[['a']], df['y'], random_state=0)",
        "model = LogisticRegression().fit(X_train, y_train)",
        "acc = accuracy_score(y_test, model.predict(X_test))",
        "plt.plot([acc])",
    ])
    r = analyze(nb)
    by_stage = {s["stage"]: s for s in r["stages"]}
    assert list(by_stage) == ["load", "clean", "features", "train", "evaluate", "report"]
    assert by_stage["clean"]["inputs"] == ["df"]
    assert "model" in by_stage["train"]["outputs"]
    names = [a["name"] for a in r["suggested_artifacts"]]
    assert "acc" in names and "model" in names


def test_comment_only_cell_is_empty(nb_factory):
    r = analyze(nb_factory(["# just a license header", "a = 1"]))
    assert not [f for f in r["findings"] if f["kind"] == "inspection_only"]


def test_loop_variables_and_plots_are_not_suggested(nb_factory):
    nb = nb_factory([
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()",
        "total = 0\nfor i in range(3):\n    total += i",
    ])
    names = [a["name"] for a in analyze(nb)["suggested_artifacts"]]
    assert "total" in names
    assert "i" not in names and "fig" not in names and "ax" not in names
