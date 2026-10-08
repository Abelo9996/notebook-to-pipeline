"""Load the wine data and split it exactly as the notebook does."""

from __future__ import annotations

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.30
SEED = 42


def load_and_split():
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return X, y, X_train, X_test, y_train, y_test
