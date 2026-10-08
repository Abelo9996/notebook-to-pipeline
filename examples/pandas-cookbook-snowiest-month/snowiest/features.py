"""Monthly temperature and snowiness."""

from __future__ import annotations

import pandas as pd

MONTH_END = "ME"  # same bins as the notebook's deprecated "M" alias


def snow_flags(weather: pd.DataFrame) -> pd.Series:
    """True for every hour whose weather description mentions snow."""
    return weather["Weather"].str.contains("Snow")


def monthly_median_temperature(weather: pd.DataFrame) -> pd.Series:
    return weather["Temp (C)"].resample(MONTH_END).median().rename("Temperature")


def monthly_snowiness(is_snowing: pd.Series) -> pd.Series:
    """Fraction of hours with snow, per month."""
    return is_snowing.astype(float).resample(MONTH_END).mean().rename("Snowiness")


def monthly_stats(temperature: pd.Series, snowiness: pd.Series) -> pd.DataFrame:
    return pd.concat([temperature, snowiness], axis=1)
