"""A plausible refactoring slip, kept here to show what a failed verification looks like.

Month-start bins ("MS") instead of month-end bins ("ME") give the same values with different
index labels, which is easy to miss by eye.
"""

from snowiest import features
from snowiest.pipeline import run


def run_month_start() -> dict:
    features.MONTH_END = "MS"
    try:
        return run()
    finally:
        features.MONTH_END = "ME"
