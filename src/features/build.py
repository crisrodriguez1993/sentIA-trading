"""Orquestador de features (Fase 3): ensambla el dataset de modelado.

Combina, por cada empresa del universo:
  1. Features técnicos (de su propia serie OHLCV).
  2. Features macro de mercado (VIX, oro, petróleo) unidos por fecha.
  3. Features fundamentales alineados a diario (sin look-ahead).
  4. Target de clasificación (retorno futuro > umbral).

El resultado es un único DataFrame en formato long guardado en
`data/processed/features.parquet`.

Ejecución desde la raíz del repositorio:
    python -m src.features.build
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import fundamentals as fund_data
from src.data import prices as price_data
from src.features import target as target_mod
from src.features.fundamental import align_fundamentals_to_daily
from src.features.sentiment import add_sentiment_features
from src.features.technical import add_technical_features


def _build_macro_features() -> pd.DataFrame:
    """Construye features macro de mercado indexados por fecha.

    - VIX: nivel de cierre (índice del miedo).
    - Oro y petróleo: retornos a 1 y 5 días (señales de refugio/tensión).
    """
    mapping = {"^VIX": "vix", "GC=F": "gold", "CL=F": "oil"}
    frames: list[pd.DataFrame] = []

    for ticker, name in mapping.items():
        df = price_data.download_prices(ticker)
        if df.empty:
            continue
        df = df.sort_values("date")[["date", "adj_close"]].copy()
        if name == "vix":
            df["vix_level"] = df["adj_close"]
            df = df[["date", "vix_level"]]
        else:
            df[f"{name}_ret_1d"] = df["adj_close"].pct_change()
            df[f"{name}_ret_5d"] = df["adj_close"].pct_change(5)
            df = df[["date", f"{name}_ret_1d", f"{name}_ret_5d"]]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date"])

    macro = frames[0]
    for extra in frames[1:]:
        macro = macro.merge(extra, on="date", how="outer")
    return macro.sort_values("date").reset_index(drop=True)


def build_dataset(refresh: bool = False) -> pd.DataFrame:
    """Ensambla el dataset completo de features + target para todas las empresas."""
    fundamentals = fund_data.download_all_fundamentals(refresh=refresh)
    macro = _build_macro_features()

    all_rows: list[pd.DataFrame] = []
    for ticker in config.company_tickers():
        prices = price_data.download_prices(ticker, refresh=refresh)
        if prices.empty:
            print(f"[features] AVISO: sin precios para {ticker}")
            continue

        df = add_technical_features(prices)
        if not macro.empty:
            df = df.merge(macro, on="date", how="left")
        df = align_fundamentals_to_daily(df, fundamentals, ticker)
        df = add_sentiment_features(df, ticker)
        df = target_mod.add_target(df)

        all_rows.append(df)
        print(f"[features] {ticker}: {len(df)} filas, {df.shape[1]} columnas")

    if not all_rows:
        return pd.DataFrame()

    dataset = pd.concat(all_rows, ignore_index=True)

    out_path = config.processed_dir() / "features.parquet"
    dataset.to_parquet(out_path, index=False)
    print(f"\n[features] Dataset guardado en {out_path}")
    print(f"[features] Total: {len(dataset)} filas, {dataset.shape[1]} columnas")
    return dataset


if __name__ == "__main__":
    build_dataset()
