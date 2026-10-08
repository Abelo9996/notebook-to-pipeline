"""Load the hourly 2012 Montreal weather data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data") / "weather_2012.csv"


def load_weather(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Hourly observations indexed by timestamp."""
    return pd.read_csv(path, parse_dates=True, index_col="Date/Time")
