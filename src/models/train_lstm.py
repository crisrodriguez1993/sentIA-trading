"""Comparación de modelos (Fase 4-bis): LSTM secuencial vs. boosting tabular.

Entrena ambos modelos sobre el MISMO holdout temporal (mismo `test_fraction`)
y compara sus métricas de clasificación en el periodo de test. Responde a la
pregunta metodológica: ¿aporta valor una red recurrente frente a un modelo de
boosting sobre datos financieros tabulares con historia limitada?

Ejecución desde la raíz del repositorio:
    python -m src.models.train_lstm
"""

from __future__ import annotations

import json

import numpy as np

from src import config
from src.models import metrics as metrics_mod
from src.models.classifier import QuantClassifier
from src.models.dataset import final_holdout_split, prepare_dataset
from src.models.lstm import LSTMTrainer
from src.models.sequence_dataset import build_sequence_split


def _eval_lstm() -> dict:
    cfg = config.model_lstm_config()
    split = build_sequence_split(
        lookback=cfg.get("lookback", 20),
        test_fraction=0.2,
    )
    print(
        f"[LSTM] secuencias train={len(split.X_train)} test={len(split.X_test)} "
        f"| lookback={cfg.get('lookback', 20)} features={len(split.feature_names)}"
    )
    print(f"[LSTM] entrenando en {LSTMTrainer(len(split.feature_names)).device} ...")

    trainer = LSTMTrainer(n_features=len(split.feature_names)).fit(split.X_train, split.y_train)
    proba = trainer.predict_proba(split.X_test)
    pred = (proba >= 0.5).astype(int)
    m = metrics_mod.classification_metrics(split.y_test.astype(int), pred, proba)
    return {"metrics": m, "cutoff": str(split.cutoff.date()),
            "n_train": int(len(split.X_train)), "n_test": int(len(split.X_test))}


def _eval_boosting() -> dict:
    data = prepare_dataset(dropna_features=True)
    idx_tr, idx_te, cutoff = final_holdout_split(data.dates, test_fraction=0.2)
    X_tr, y_tr = data.X.iloc[idx_tr], data.y.iloc[idx_tr]
    X_te, y_te = data.X.iloc[idx_te], data.y.iloc[idx_te]

    clf = QuantClassifier().fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    pred = (proba >= 0.5).astype(int)
    m = metrics_mod.classification_metrics(y_te.to_numpy(), pred, proba)
    return {"metrics": m, "algorithm": clf.algorithm, "cutoff": str(cutoff.date()),
            "n_train": int(len(idx_tr)), "n_test": int(len(idx_te))}


def run() -> dict:
    np.random.seed(config.seed())
    print("=" * 60)
    print("Sentia-Trading — Comparación LSTM vs. Boosting (Fase 4-bis)")
    print("=" * 60)

    print("\n--- Boosting (tabular) ---")
    boosting = _eval_boosting()
    bm = boosting["metrics"]
    print(f"  acc={bm['accuracy']:.3f} f1={bm['f1']:.3f} auc={bm['roc_auc']:.3f}")

    print("\n--- LSTM (secuencial) ---")
    lstm = _eval_lstm()
    lm = lstm["metrics"]
    print(f"  acc={lm['accuracy']:.3f} f1={lm['f1']:.3f} auc={lm['roc_auc']:.3f}")

    print("\n" + "=" * 60)
    print(f"{'Modelo':<12}{'Accuracy':>10}{'F1':>8}{'ROC-AUC':>10}")
    print("-" * 40)
    print(f"{boosting['algorithm']:<12}{bm['accuracy']:>10.3f}{bm['f1']:>8.3f}{bm['roc_auc']:>10.3f}")
    print(f"{'lstm':<12}{lm['accuracy']:>10.3f}{lm['f1']:>8.3f}{lm['roc_auc']:>10.3f}")
    print("=" * 60)

    winner = "lstm" if lm["roc_auc"] > bm["roc_auc"] else boosting["algorithm"]
    print(f"\nMejor ROC-AUC: {winner}")

    report = {"boosting": boosting, "lstm": lstm, "winner_by_auc": winner}
    out_path = config.reports_dir() / "model_comparison.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Reporte guardado en: {out_path}")
    return report


if __name__ == "__main__":
    run()
