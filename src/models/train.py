"""Orquestador de entrenamiento del modelo cuantitativo (Fase 4).

Flujo:
  1. Carga el dataset de features (Fase 3) y prepara X / y con orden temporal.
  2. Validación WALK-FORWARD (expansiva, con embargo) -> métricas por fold.
  3. Entrena en train y evalúa en un holdout final por fecha.
  4. Reentrena con todos los datos y guarda el modelo + importancia de features.

Ejecución desde la raíz del repositorio:
    python -m src.models.train                 # usa el algoritmo de config
    python -m src.models.train --algorithm xgboost
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from src import config
from src.models import metrics as metrics_mod
from src.models.classifier import QuantClassifier
from src.models.dataset import (
    final_holdout_split,
    prepare_dataset,
    time_ordered_folds,
)


def walk_forward_eval(data, algorithm: str | None, n_splits: int = 5) -> dict:
    """Validación temporal expansiva; devuelve métricas medias por fold."""
    folds = time_ordered_folds(data.dates, n_splits=n_splits)
    fold_metrics: list[dict] = []

    print("\n--- Validación walk-forward ---")
    for i, (idx_tr, idx_val) in enumerate(folds, start=1):
        X_tr, y_tr = data.X.iloc[idx_tr], data.y.iloc[idx_tr]
        X_val, y_val = data.X.iloc[idx_val], data.y.iloc[idx_val]

        clf = QuantClassifier(algorithm=algorithm).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)
        pred = (proba >= 0.5).astype(int)

        m = metrics_mod.classification_metrics(y_val.to_numpy(), pred, proba)
        fold_metrics.append(m)
        print(
            f"  Fold {i}: n_tr={len(idx_tr):5d} n_val={len(idx_val):5d} "
            f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}"
        )

    return metrics_mod.aggregate_folds(fold_metrics)


def holdout_eval(data, algorithm: str | None) -> dict:
    """Entrena en el pasado y evalúa en el último tramo temporal (holdout)."""
    idx_tr, idx_te, cutoff = final_holdout_split(data.dates, test_fraction=0.2)
    X_tr, y_tr = data.X.iloc[idx_tr], data.y.iloc[idx_tr]
    X_te, y_te = data.X.iloc[idx_te], data.y.iloc[idx_te]

    clf = QuantClassifier(algorithm=algorithm).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    pred = (proba >= 0.5).astype(int)

    m = metrics_mod.classification_metrics(y_te.to_numpy(), pred, proba)
    baseline = metrics_mod.majority_baseline(y_te.to_numpy())

    print(f"\n--- Holdout final (corte {cutoff.date()}) ---")
    print(f"  Train: {len(idx_tr)} filas | Test: {len(idx_te)} filas")
    print(
        f"  acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f} "
        f"| baseline acc={baseline['accuracy']:.3f} (clase+={baseline['positive_rate']:.3f})"
    )
    return {"metrics": m, "baseline": baseline, "cutoff": str(cutoff.date())}


def train_final(data, algorithm: str | None) -> QuantClassifier:
    """Reentrena con TODO el dataset y guarda el modelo e importancias."""
    clf = QuantClassifier(algorithm=algorithm).fit(data.X, data.y)

    model_path = config.models_dir() / f"quant_{clf.algorithm}.joblib"
    joblib.dump(clf, model_path)

    importance = clf.feature_importance()
    imp_path = config.reports_dir() / f"feature_importance_{clf.algorithm}.csv"
    importance.to_csv(imp_path, header=["importance"])

    print(f"\n--- Modelo final ---")
    print(f"  Guardado en: {model_path}")
    print(f"  Importancia de features -> {imp_path}")
    print("  Top 10 features:")
    for name, val in importance.head(10).items():
        print(f"    {name:22s} {val:.0f}")
    return clf


def run(algorithm: str | None = None, n_splits: int = 5) -> dict:
    np.random.seed(config.seed())
    data = prepare_dataset(dropna_features=True)

    print("=" * 60)
    print("Sentia-Trading — Entrenamiento modelo cuantitativo (Fase 4)")
    print(f"Filas: {len(data.X)} | Features: {len(data.feature_names)}")
    print(f"Distribución target (clase 1): {data.y.mean():.3f}")
    print("=" * 60)

    wf = walk_forward_eval(data, algorithm, n_splits=n_splits)
    print("\nResumen walk-forward (media ± std):")
    for key in ("accuracy", "f1", "roc_auc"):
        print(f"  {key:9s} {wf[f'{key}_mean']:.3f} ± {wf[f'{key}_std']:.3f}")

    ho = holdout_eval(data, algorithm)
    clf = train_final(data, algorithm)

    report = {
        "algorithm": clf.algorithm,
        "n_rows": int(len(data.X)),
        "n_features": len(data.feature_names),
        "positive_rate": float(data.y.mean()),
        "walk_forward": wf,
        "holdout": ho,
    }
    report_path = config.reports_dir() / f"quant_report_{clf.algorithm}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReporte guardado en: {report_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo cuantitativo")
    parser.add_argument(
        "--algorithm",
        choices=["lightgbm", "xgboost"],
        default=None,
        help="Algoritmo a usar (por defecto, el de config/params.yaml).",
    )
    parser.add_argument("--splits", type=int, default=5, help="Nº de folds walk-forward.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(algorithm=args.algorithm, n_splits=args.splits)
