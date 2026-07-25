"""Preparación del dataset de modelado y particiones temporales.

Carga `data/processed/features.parquet`, separa la matriz de features `X`
del target `y`, y provee esquemas de validación TEMPORAL (sin `shuffle`)
para evitar *data leakage*: un modelo nunca se valida con datos anteriores
a los de su entrenamiento.

Se excluyen de `X`:
- Identificadores y fechas: `date`, `ticker`.
- Precios crudos (no estacionarios): `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- Columnas del target/futuro: `future_return`, `target`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.fundamental import fundamental_feature_columns

# Columnas que NUNCA entran como features.
_NON_FEATURE_COLUMNS = {
    "date", "ticker",
    "open", "high", "low", "close", "adj_close", "volume",
    "future_return", "target",
    # SMAs en nivel de precio (no estacionarias); usamos price_to_sma_* en su lugar.
    "sma_10", "sma_20", "sma_50", "sma_200",
}

TARGET_COLUMN = "target"


@dataclass
class Dataset:
    """Contenedor del dataset de modelado ya preparado."""

    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series
    tickers: pd.Series
    feature_names: list[str]


def load_features() -> pd.DataFrame:
    """Carga el parquet de features generado en la Fase 3."""
    path = config.processed_dir() / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m src.features.build"
        )
    return pd.read_parquet(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas numéricas utilizables como features."""
    return [
        c for c in df.columns
        if c not in _NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def prepare_dataset(dropna_features: bool = True, exclude: list[str] | None = None) -> Dataset:
    """Prepara el dataset: filtra filas con target válido y ordena por fecha.

    Los modelos de boosting (LightGBM/XGBoost) manejan NaN de forma nativa, por
    lo que NO descartamos filas por fundamentales ausentes (que solo cubren unos
    pocos años). Solo se exige la presencia de los features técnicos/macro, que
    quedan indefinidos únicamente durante el warmup de las medias móviles largas.

    Args:
        dropna_features: si True, elimina filas sin features técnicos/macro
            (warmup). Los features fundamentales pueden permanecer NaN.
        exclude: nombres de columnas de features a excluir (p. ej. para estudios
            de ablación con vs. sin sentimiento).

    Returns:
        Objeto `Dataset` con X, y, fechas, tickers y nombres de features.
    """
    df = load_features()

    # Solo filas con target observable (excluye las últimas h filas de cada ticker).
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    features = feature_columns(df)
    if exclude:
        features = [c for c in features if c not in set(exclude)]

    if dropna_features:
        fundamental_cols = set(fundamental_feature_columns())
        required = [c for c in features if c not in fundamental_cols]
        df = df.dropna(subset=required).reset_index(drop=True)

    X = df[features].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    return Dataset(
        X=X,
        y=y,
        dates=df["date"],
        tickers=df["ticker"],
        feature_names=features,
    )


def time_ordered_folds(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera folds de validación temporal expansiva (walk-forward) por fechas.

    Divide el eje temporal en `n_splits + 1` bloques por fechas únicas. Para
    cada fold i, entrena con todos los bloques hasta i y valida con el bloque
    i+1. Aplica un `embargo` (gap) entre train y validación para no filtrar
    información del horizonte del target.

    Args:
        dates: serie de fechas alineada con las filas del dataset.
        n_splits: número de folds de validación.
        embargo_days: días de gap entre el fin del train y el inicio del val.

    Returns:
        Lista de tuplas (idx_train, idx_val) con índices posicionales.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(dates.unique())
    blocks = np.array_split(unique_dates, n_splits + 1)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_dates = np.concatenate(blocks[: i + 1])
        val_dates = blocks[i + 1]

        train_end = pd.Timestamp(train_dates.max())
        val_start = pd.Timestamp(val_dates.min())
        embargo_cutoff = train_end + pd.Timedelta(days=embargo_days)

        # Embargo: descartar filas de train demasiado cercanas al inicio de val.
        train_mask = dates.isin(train_dates) & (dates <= (val_start - pd.Timedelta(days=embargo_days)))
        if not train_mask.any():  # respaldo si el embargo vacía el train
            train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates) & (dates >= embargo_cutoff)
        if not val_mask.any():
            val_mask = dates.isin(val_dates)

        idx_train = np.where(train_mask.to_numpy())[0]
        idx_val = np.where(val_mask.to_numpy())[0]
        folds.append((idx_train, idx_val))

    return folds


def final_holdout_split(
    dates: pd.Series,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Separa un holdout final por fecha (último `test_fraction` del tiempo).

    Returns:
        (idx_train, idx_test, fecha_de_corte).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(dates.unique())
    cut_pos = int(len(unique_dates) * (1 - test_fraction))
    cutoff = pd.Timestamp(unique_dates[cut_pos])

    idx_train = np.where((dates < cutoff).to_numpy())[0]
    idx_test = np.where((dates >= cutoff).to_numpy())[0]
    return idx_train, idx_test, cutoff
