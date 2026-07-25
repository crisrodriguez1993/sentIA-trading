"""Métricas de evaluación para el clasificador cuantitativo.

Calcula las métricas estándar de clasificación usadas en la memoria:
accuracy, precision, recall, F1 y ROC-AUC. Incluye una baseline trivial
(predecir siempre la clase mayoritaria) como punto de comparación honesto.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Devuelve un diccionario de métricas de clasificación."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def majority_baseline(y_true: np.ndarray) -> dict[str, float]:
    """Métricas de la baseline que siempre predice la clase mayoritaria."""
    y_true = np.asarray(y_true)
    majority = int(y_true.mean() >= 0.5)
    y_pred = np.full_like(y_true, majority)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_rate": float(y_true.mean()),
    }


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Promedia (media ± std) las métricas de todos los folds."""
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    agg: dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
    return agg
