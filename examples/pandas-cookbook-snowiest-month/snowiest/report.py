"""Charts from the notebook, kept out of the data path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_stats(stats: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    ax = stats.plot(kind="bar", subplots=True, figsize=(15, 10))
    ax[0].figure.savefig(path)
