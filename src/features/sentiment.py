"""Integración del sentimiento diario (FinBERT) como features del modelo.

Une el score de sentimiento diario por empresa (`sentiment_daily.parquet`,
generado en `src/nlp/sentiment.py`) a la serie de precios diaria. Los días sin
noticias se rellenan como neutrales (sentiment 0, news_count 0), y se añaden
versiones suavizadas (medias móviles) porque la cobertura de noticias es
intermitente y una sola jornada puede ser ruidosa.

Sin look-ahead: el sentimiento del día `t` refleja noticias publicadas hasta el
cierre de `t`; las medias móviles terminan (inclusive) en `t`.
"""

from __future__ import annotations

import pandas as pd

from src import config

_SENTIMENT_FEATURES = [
    "sentiment_mean",
    "news_count",
    "pos_ratio",
    "neg_ratio",
    "sentiment_5d",
    "sentiment_21d",
    "news_count_5d",
]


def _load_sentiment() -> pd.DataFrame:
    path = config.processed_dir() / "sentiment_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def add_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Añade features de sentimiento a la serie diaria de un ticker.

    Si no hay datos de sentimiento (aún no ejecutado), añade columnas neutras
    para que el esquema del dataset sea estable.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    sentiment = _load_sentiment()

    if sentiment.empty:
        for col in _SENTIMENT_FEATURES:
            df[col] = 0.0
        return df

    s = sentiment[sentiment["ticker"] == ticker][
        ["date", "sentiment_mean", "news_count", "pos_ratio", "neg_ratio"]
    ].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()

    df = df.merge(s, on="date", how="left")

    # Días sin noticias -> neutral / sin cobertura.
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0.0)
    df["pos_ratio"] = df["pos_ratio"].fillna(0.0)
    df["neg_ratio"] = df["neg_ratio"].fillna(0.0)

    # Suavizados (medias móviles que terminan en t, sin look-ahead).
    df["sentiment_5d"] = df["sentiment_mean"].rolling(5, min_periods=1).mean()
    df["sentiment_21d"] = df["sentiment_mean"].rolling(21, min_periods=1).mean()
    df["news_count_5d"] = df["news_count"].rolling(5, min_periods=1).sum()

    return df


def sentiment_feature_columns() -> list[str]:
    """Lista de columnas de features de sentimiento."""
    return list(_SENTIMENT_FEATURES)
