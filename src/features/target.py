"""Construcción del target de clasificación.

Siguiendo la metodología, el problema se plantea como clasificación binaria:

    ¿El retorno acumulado a `horizon` días supera el umbral `threshold`?
    (Sí = 1, No = 0)

El retorno futuro se calcula mirando `horizon` días hacia adelante, por lo que
las últimas `horizon` filas de cada ticker quedan sin etiqueta (NaN) y deben
excluirse del entrenamiento. Este es el ÚNICO lugar donde se usa información
futura: para construir la etiqueta, nunca como feature.
"""

from __future__ import annotations

import pandas as pd

from src import config


def add_target(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    horizon: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Añade el retorno futuro y la etiqueta binaria de clasificación.

    Args:
        df: DataFrame diario de UN ticker, ordenado por fecha.
        price_col: columna de precio base.
        horizon: días hacia adelante; por defecto el de `config`.
        threshold: umbral de retorno; por defecto el de `config`.

    Returns:
        DataFrame con columnas 'future_return' y 'target'.
    """
    if df.empty:
        return df

    cfg_h, cfg_t = config.target_config()
    horizon = horizon if horizon is not None else cfg_h
    threshold = threshold if threshold is not None else cfg_t

    df = df.sort_values("date").reset_index(drop=True).copy()
    price = df[price_col]

    # Retorno acumulado a `horizon` días hacia adelante.
    df["future_return"] = price.shift(-horizon) / price - 1.0
    df["target"] = (df["future_return"] > threshold).astype("float")

    # Las últimas `horizon` filas no tienen futuro observable -> etiqueta inválida.
    df.loc[df["future_return"].isna(), "target"] = pd.NA

    return df
