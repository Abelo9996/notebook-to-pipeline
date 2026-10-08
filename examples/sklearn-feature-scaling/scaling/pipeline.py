"""Entry points. `run()` reproduces the notebook; `run_fixed()` gives each pipeline its own PCA."""

from __future__ import annotations

from .data import load_and_split
from .evaluate import metrics, predictions
from .figures import PLOT_STYLE, knn_boundary_inputs, pca_component_table
from .models import CS, fit_pipelines, standard_scaler


def run(share_pca: bool = True) -> dict:
    X, y, X_train, X_test, y_train, y_test = load_and_split()
    scaled_X_train = standard_scaler().fit_transform(X_train)
    X_plot, X_plot_scaled, clf = knn_boundary_inputs(X, y)
    scaled_pca, first_pca_component, X_train_transformed, X_train_std_transformed = pca_component_table(
        X, X_train, scaled_X_train
    )
    pca, scaler, unscaled_clf, scaled_clf = fit_pipelines(X_train, y_train, share_pca=share_pca)
    preds = predictions(unscaled_clf, scaled_clf, X_test)
    return {
        "X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "scaled_X_train": scaled_X_train, "scaler": scaler,
        "X_plot": X_plot, "X_plot_scaled": X_plot_scaled, "clf": clf,
        "pca": pca, "scaled_pca": scaled_pca, "first_pca_component": first_pca_component,
        "X_train_transformed": X_train_transformed, "X_train_std_transformed": X_train_std_transformed,
        **PLOT_STYLE, "Cs": CS,
        "unscaled_clf": unscaled_clf, "scaled_clf": scaled_clf,
        **preds,
        "metrics": metrics(y_test, preds),
    }


def run_fixed() -> dict:
    return run(share_pca=False)


if __name__ == "__main__":
    import sys

    fixed = "--fixed" in sys.argv[1:]
    print("separate PCA per pipeline" if fixed else "shared PCA, as in the notebook")
    for name, value in (run_fixed() if fixed else run())["metrics"].items():
        print(f"{name}: {value:.4f}")
