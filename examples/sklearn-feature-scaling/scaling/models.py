"""The two PCA + logistic regression pipelines compared by the notebook."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = np.logspace(-5, 5, 20)


def standard_scaler() -> StandardScaler:
    return StandardScaler().set_output(transform="pandas")


def logistic_cv() -> LogisticRegressionCV:
    return LogisticRegressionCV(Cs=CS, use_legacy_attributes=False, l1_ratios=(0,), scoring="neg_log_loss")


def fit_pipelines(X_train, y_train, *, share_pca: bool = True):
    """Fit the unscaled and the standardized pipeline.

    share_pca=True reproduces the notebook: both pipelines hold the same PCA object, so fitting
    the standardized pipeline refits the PCA inside the unscaled one. share_pca=False gives the
    unscaled pipeline its own PCA.
    """
    pca = PCA(n_components=2).fit(X_train)
    unscaled_clf = make_pipeline(pca, logistic_cv()).fit(X_train, y_train)
    scaler = standard_scaler()
    scaled_pca_step = pca if share_pca else clone(pca)
    scaled_clf = make_pipeline(scaler, scaled_pca_step, logistic_cv()).fit(X_train, y_train)
    return pca, scaler, unscaled_clf, scaled_clf
