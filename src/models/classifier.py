"""Clasificador cuantitativo (LightGBM / XGBoost).

Envuelve el modelo de gradient boosting elegido en `config` bajo una interfaz
uniforme, para que el resto del pipeline (walk-forward, evaluación, backtest)
no dependa de la librería concreta. Ambos son árboles con boosting: rápidos,
robustos en datos tabulares e interpretables vía importancia de features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config


class QuantClassifier:
    """Clasificador binario de dirección del retorno (boosting sobre tabular)."""

    def __init__(self, algorithm: str | None = None, **overrides: Any) -> None:
        cfg = dict(config.model_quant_config())
        self.algorithm = (algorithm or cfg.pop("algorithm", "lightgbm")).lower()
        cfg.pop("algorithm", None)
        cfg.update(overrides)
        self.params = cfg
        self.model: Any = None
        self.feature_names: list[str] | None = None

    def _build(self):
        seed = self.params.get("random_state", config.seed())
        if self.algorithm == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 500),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", -1),
                num_leaves=self.params.get("num_leaves", 31),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
        if self.algorithm == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 500),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 6) if self.params.get("max_depth", -1) > 0 else 6,
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            )
        raise ValueError(f"Algoritmo no soportado: {self.algorithm}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantClassifier":
        self.feature_names = list(X.columns)
        self.model = self._build()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad de la clase positiva (retorno > umbral)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        """Importancia de features ordenada de mayor a menor."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
