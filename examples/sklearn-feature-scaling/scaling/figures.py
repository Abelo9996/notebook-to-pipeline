"""Data behind the notebook's figures. Plot drawing itself stays out of the pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from .models import standard_scaler

FEATURES_2D = ["proline", "hue"]


def knn_boundary_inputs(X, y):
    """Two-feature data for the decision-boundary figure, and the KNN as last fitted there."""
    X_plot = X[FEATURES_2D]
    X_plot_scaled = standard_scaler().fit_transform(X_plot)
    clf = KNeighborsClassifier(n_neighbors=20)
    clf.fit(X_plot, y)
    clf.fit(X_plot_scaled, y)
    return X_plot, X_plot_scaled, clf


def pca_component_table(X, X_train, scaled_X_train):
    pca = PCA(n_components=2).fit(X_train)
    scaled_pca = PCA(n_components=2).fit(scaled_X_train)
    table = pd.DataFrame(pca.components_[0], index=X.columns, columns=["without scaling"])
    table["with scaling"] = scaled_pca.components_[0]
    return scaled_pca, table, pca.transform(X_train), scaled_pca.transform(scaled_X_train)


PLOT_STYLE = {
    "target_classes": range(0, 3),
    "colors": ("blue", "red", "green"),
    "markers": ("^", "s", "o"),
}
