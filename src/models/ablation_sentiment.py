"""Estudio de ablación: ¿aporta valor el sentimiento? (RQ2).

Compara el modelo cuantitativo en dos configuraciones, sobre el MISMO holdout
temporal y con el mismo algoritmo:
  - BASE:      features técnicos + fundamentales + macro.
  - + SENTIM.: lo anterior MÁS los features de sentimiento (FinBERT).

Reporta la diferencia en accuracy, F1 y ROC-AUC para cuantificar el aporte
marginal del análisis de sentimiento. Requiere que el dataset de features ya
incluya el bloque de sentimiento (ejecutar antes `python -m src.features.build`
tras generar `sentiment_daily.parquet`).

Ejecución desde la raíz del repositorio:
    python -m src.models.ablation_sentiment
"""

from __future__ import annotations

import json

import numpy as np

from src import config
from src.features.sentiment import sentiment_feature_columns
from src.models import metrics as metrics_mod
from src.models.classifier import QuantClassifier
from src.models.dataset import final_holdout_split, prepare_dataset


def _evaluate(exclude: list[str] | None, algorithm: str | None) -> dict:
    data = prepare_dataset(dropna_features=True, exclude=exclude)
    idx_tr, idx_te, cutoff = final_holdout_split(data.dates, test_fraction=0.2)
    X_tr, y_tr = data.X.iloc[idx_tr], data.y.iloc[idx_tr]
    X_te, y_te = data.X.iloc[idx_te], data.y.iloc[idx_te]

    clf = QuantClassifier(algorithm=algorithm).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    pred = (proba >= 0.5).astype(int)
    m = metrics_mod.classification_metrics(y_te.to_numpy(), pred, proba)
    return {"metrics": m, "n_features": len(data.feature_names), "cutoff": str(cutoff.date())}


def run(algorithm: str | None = None) -> dict:
    np.random.seed(config.seed())
    sentiment_cols = sentiment_feature_columns()

    print("=" * 60)
    print("Sentia-Trading — Ablación de sentimiento (RQ2)")
    print("=" * 60)

    base = _evaluate(exclude=sentiment_cols, algorithm=algorithm)
    full = _evaluate(exclude=None, algorithm=algorithm)

    bm, fm = base["metrics"], full["metrics"]
    print(f"\n{'Configuración':<16}{'Feats':>7}{'Accuracy':>10}{'F1':>8}{'ROC-AUC':>10}")
    print("-" * 51)
    print(f"{'BASE':<16}{base['n_features']:>7}{bm['accuracy']:>10.3f}{bm['f1']:>8.3f}{bm['roc_auc']:>10.3f}")
    print(f"{'+ Sentimiento':<16}{full['n_features']:>7}{fm['accuracy']:>10.3f}{fm['f1']:>8.3f}{fm['roc_auc']:>10.3f}")
    print("-" * 51)
    print(f"{'Δ (aporte)':<16}{full['n_features'] - base['n_features']:>7}"
          f"{fm['accuracy'] - bm['accuracy']:>+10.3f}{fm['f1'] - bm['f1']:>+8.3f}"
          f"{fm['roc_auc'] - bm['roc_auc']:>+10.3f}")

    delta_auc = fm["roc_auc"] - bm["roc_auc"]
    verdict = "el sentimiento APORTA" if delta_auc > 0 else "el sentimiento NO aporta (en AUC)"
    print(f"\nConclusión: {verdict} (ΔAUC = {delta_auc:+.3f})")

    report = {"base": base, "with_sentiment": full,
              "delta_auc": float(delta_auc), "algorithm": algorithm or config.model_quant_config().get("algorithm")}
    out_path = config.reports_dir() / "ablation_sentiment.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Reporte guardado en: {out_path}")
    return report


if __name__ == "__main__":
    run()
