"""Agregación del sentimiento a un score diario por empresa (Fase 5).

Orquesta el pipeline de sentimiento:
  1. Descarga (o carga) noticias por empresa (GDELT).
  2. Puntúa cada titular con FinBERT.
  3. Agrega a un score diario por ticker.
  4. Guarda el resultado en `data/processed/sentiment_daily.parquet`.

El score diario de un ticker es la media del sentimiento de sus titulares del
día (sentiment = P(pos) - P(neg)), junto con el volumen de noticias, que es en
sí una señal (picos de cobertura suelen coincidir con eventos relevantes).

Ejecución desde la raíz del repositorio:
    python -m src.nlp.sentiment            # usa caché de noticias si existe
    python -m src.nlp.sentiment --refresh  # fuerza redescarga de noticias
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.data import news as news_data
from src.nlp.finbert import FinBERTScorer


def build_daily_sentiment(refresh: bool = False) -> pd.DataFrame:
    """Construye el score de sentimiento diario por empresa.

    Returns:
        DataFrame con columnas [date, ticker, sentiment_mean, sentiment_std,
        news_count, pos_ratio, neg_ratio].
    """
    scorer = FinBERTScorer()
    print(f"[sentiment] FinBERT en {scorer.device}")

    daily_frames: list[pd.DataFrame] = []
    for ticker in config.company_tickers():
        news = news_data.download_news_for_ticker(ticker, refresh=refresh)
        if news.empty:
            print(f"[sentiment] {ticker}: sin noticias, se omite")
            continue

        scored = scorer.score_texts(news["title"].tolist())
        scored.index = news.index
        merged = pd.concat([news[["date"]], scored], axis=1)

        grouped = merged.groupby("date")
        daily = pd.DataFrame({
            "sentiment_mean": grouped["sentiment"].mean(),
            "sentiment_std": grouped["sentiment"].std(),
            "news_count": grouped["sentiment"].size(),
            "pos_ratio": grouped["label"].apply(lambda s: (s == "positive").mean()),
            "neg_ratio": grouped["label"].apply(lambda s: (s == "negative").mean()),
        }).reset_index()
        daily["ticker"] = ticker
        daily_frames.append(daily)
        print(f"[sentiment] {ticker}: {len(daily)} días con noticias "
              f"(sentimiento medio {daily['sentiment_mean'].mean():+.3f})")

    if not daily_frames:
        return pd.DataFrame()

    result = pd.concat(daily_frames, ignore_index=True)
    result = result[["date", "ticker", "sentiment_mean", "sentiment_std",
                     "news_count", "pos_ratio", "neg_ratio"]]

    out_path = config.processed_dir() / "sentiment_daily.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\n[sentiment] Guardado en {out_path} ({len(result)} filas)")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de sentimiento diario (FinBERT)")
    parser.add_argument("--refresh", action="store_true", help="Fuerza redescarga de noticias.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_daily_sentiment(refresh=args.refresh)
