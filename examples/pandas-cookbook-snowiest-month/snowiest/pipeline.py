"""Entry point. `run()` returns the artifacts that nb2p compares with the notebook."""

from __future__ import annotations

from .features import monthly_median_temperature, monthly_snowiness, monthly_stats, snow_flags
from .load import load_weather


def run() -> dict:
    weather_2012 = load_weather()
    is_snowing = snow_flags(weather_2012)
    temperature = monthly_median_temperature(weather_2012)
    snowiness = monthly_snowiness(is_snowing)
    return {
        "weather_2012": weather_2012,
        "weather_description": weather_2012["Weather"],
        "is_snowing": is_snowing,
        "temperature": temperature,
        "snowiness": snowiness,
        "stats": monthly_stats(temperature, snowiness),
    }


if __name__ == "__main__":
    print(run()["stats"])
