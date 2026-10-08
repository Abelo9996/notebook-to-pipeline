"""Build iris_session.ipynb by replaying a messy interactive session in a real kernel.

The cells run in the order an analyst might run them while exploring, one helper cell is
deleted at the end, and the notebook is saved with the real outputs and execution counts.
Nothing in the saved file is hand-edited.

Usage: python make_broken_notebook.py <python-with-ipykernel-and-scikit-learn>
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

CELLS = {
    "imports": "import pandas as pd\nfrom sklearn.datasets import load_iris\nfrom sklearn.linear_model import LogisticRegression",
    "load": "df = load_iris(as_frame=True).frame\ndf.shape",
    "helper": "# quick helper while exploring\nfeatures = df.drop(columns='target')",
    "select": "X = features[['petal length (cm)', 'petal width (cm)']]\nX.head()",
    "fit": "model = LogisticRegression(max_iter=500).fit(X_scaled, df['target'])\ntrain_accuracy = model.score(X_scaled, df['target'])\ntrain_accuracy",
    "scale": "X_scaled = (X - X.mean()) / X.std()",
    "summary": "summary = X_scaled.describe().round(3)\nsummary",
}
NOTEBOOK_ORDER = ["imports", "load", "helper", "select", "fit", "scale", "summary"]
SESSION_ORDER = ["imports", "load", "helper", "select", "scale", "fit", "summary", "select", "fit"]
DELETED = "helper"


def main(python: str) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells.append(nbformat.v4.new_markdown_cell("# Iris: petal size vs species\n\nQuick exploration."))
    index = {}
    for key in NOTEBOOK_ORDER:
        index[key] = len(nb.cells)
        nb.cells.append(nbformat.v4.new_code_cell(CELLS[key]))

    kdir = Path(tempfile.mkdtemp())
    (kdir / "replay").mkdir()
    (kdir / "replay" / "kernel.json").write_text(json.dumps(
        {"argv": [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
         "display_name": "replay", "language": "python"}))
    km = KernelManager(kernel_name="replay", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kdir)]))
    km.transport, km.ip = "ipc", str(kdir / "ipc")
    client = NotebookClient(nb, km=km, timeout=120, allow_errors=False)
    with client.setup_kernel(cleanup_kc=True):
        for key in SESSION_ORDER:
            client.execute_cell(nb.cells[index[key]], index[key])
    del nb.cells[index[DELETED]]
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(nb, Path(__file__).with_name("iris_session.ipynb"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.executable)
