"""Test-set predictions and the two metrics the notebook prints."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, log_loss


def predictions(unscaled_clf, scaled_clf, X_test):
    return {
        "y_pred": unscaled_clf.predict(X_test),
        "y_pred_scaled": scaled_clf.predict(X_test),
        "y_proba": unscaled_clf.predict_proba(X_test),
        "y_proba_scaled": scaled_clf.predict_proba(X_test),
    }


def metrics(y_test, preds) -> dict:
    return {
        "accuracy_unscaled": accuracy_score(y_test, preds["y_pred"]),
        "accuracy_scaled": accuracy_score(y_test, preds["y_pred_scaled"]),
        "log_loss_unscaled": log_loss(y_test, preds["y_proba"]),
        "log_loss_scaled": log_loss(y_test, preds["y_proba_scaled"]),
    }
