"""Construcción de secuencias temporales para modelos recurrentes (LSTM/TFT).

A diferencia del boosting (que ve cada día como una fila independiente), un
modelo secuencial necesita ventanas de `lookback` días consecutivos por ticker.

Puntos clave para evitar *data leakage*:
- Las secuencias no cruzan de un ticker a otro (se construyen por bloques).
- La imputación (mediana) y el escalado (estandarización) se ajustan ÚNICAMENTE
  con las filas de entrenamiento y luego se aplican al test.
- Una secuencia pertenece a train/test según la fecha de su ÚLTIMO día.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.dataset import (
    TARGET_COLUMN,
    feature_columns,
    load_features,
)


@dataclass
class SequenceSplit:
    """Tensores de secuencias para train y test, ya escalados."""

    X_train: np.ndarray  # (n_train, lookback, n_features)
    y_train: np.ndarray  # (n_train,)
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    cutoff: pd.Timestamp


def _build_sequences_for_ticker(
    frame: pd.DataFrame,
    features: list[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera ventanas (lookback, n_features) para un ticker ordenado por fecha.

    Returns:
        (X, y, end_dates) donde X[i] cubre los días [i, i+lookback-1] y
        y[i]/end_dates[i] corresponden al último día de la ventana.
    """
    frame = frame.sort_values("date").reset_index(drop=True)
    values = frame[features].to_numpy(dtype=np.float32)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    dates = frame["date"].to_numpy()

    n = len(frame)
    if n <= lookback:
        empty_x = np.empty((0, lookback, len(features)), dtype=np.float32)
        return empty_x, np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    xs, ys, ends = [], [], []
    for end in range(lookback - 1, n):
        start = end - lookback + 1
        xs.append(values[start : end + 1])
        ys.append(targets[end])
        ends.append(dates[end])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ends)


def build_sequence_split(
    lookback: int = 20,
    test_fraction: float = 0.2,
) -> SequenceSplit:
    """Construye el split temporal de secuencias para todo el universo.

    Args:
        lookback: nº de días de historia por secuencia.
        test_fraction: fracción final del eje temporal reservada a test.

    Returns:
        `SequenceSplit` con tensores escalados y sin leakage.
    """
    df = load_features()
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    features = feature_columns(df)

    # Fecha de corte global (último `test_fraction` del tiempo).
    unique_dates = np.sort(df["date"].unique())
    cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * (1 - test_fraction))])

    # Imputación de NaN (fundamentales): mediana calculada SOLO con train.
    train_rows = df[df["date"] < cutoff]
    medians = train_rows[features].median()
    df[features] = df[features].fillna(medians).fillna(0.0)

    # Estandarización con estadísticas SOLO de train.
    mean = train_rows[features].fillna(medians).mean()
    std = train_rows[features].fillna(medians).std().replace(0, 1.0)
    df[features] = (df[features] - mean) / std

    x_tr, y_tr, x_te, y_te = [], [], [], []
    for ticker, frame in df.groupby("ticker"):
        X, y, ends = _build_sequences_for_ticker(frame, features, lookback)
        if len(X) == 0:
            continue
        end_ts = pd.to_datetime(ends)
        train_mask = np.asarray(end_ts < cutoff)
        test_mask = ~train_mask

        x_tr.append(X[train_mask])
        y_tr.append(y[train_mask])
        x_te.append(X[test_mask])
        y_te.append(y[test_mask])

    return SequenceSplit(
        X_train=np.concatenate(x_tr),
        y_train=np.concatenate(y_tr),
        X_test=np.concatenate(x_te),
        y_test=np.concatenate(y_te),
        feature_names=features,
        cutoff=cutoff,
    )
