"""Ingesta de precios OHLCV desde Yahoo Finance (yfinance).

Descarga series diarias para empresas, benchmark y activos macro/refugio,
las normaliza a un formato tabular homogéneo y las cachea en `data/raw/`
con fecha de descarga para garantizar reproducibilidad.

Uso rápido:
    from src.data.prices import download_prices
    df = download_prices("NVDA")            # una empresa
    data = download_all_prices()            # todo el universo
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import config

# Columnas OHLCV estándar que exponemos aguas abajo.
_OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def _cache_path(ticker: str) -> Path:
    """Ruta del CSV cacheado para un ticker (nombre saneado)."""
    safe = ticker.replace("^", "_idx_").replace("=", "_").replace("/", "_")
    return config.raw_dir() / "prices" / f"{safe}.csv"


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normaliza la salida de yfinance a un esquema homogéneo.

    - Índice de fecha -> columna `date` (tz-naive).
    - Columnas en minúsculas y renombradas a `_OHLCV_COLUMNS`.
    - Añade la columna `ticker`.
    """
    if df.empty:
        return df

    df = df.copy()

    # yfinance puede devolver columnas MultiIndex cuando se piden varios tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)

    # Si no hay 'adj_close' (auto_adjust=True), usar close como adj_close.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = ticker

    keep = ["date", "ticker"] + [c for c in _OHLCV_COLUMNS if c in df.columns]
    df = df[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


def download_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga (o carga desde caché) la serie OHLCV de un ticker.

    Args:
        ticker: símbolo (p. ej. 'NVDA', '^VIX', 'GC=F').
        start, end: rango de fechas ISO; por defecto los de `config`.
        interval: frecuencia; por defecto la de `config`.
        use_cache: si True, lee del CSV cacheado cuando existe.
        refresh: si True, ignora la caché y vuelve a descargar.

    Returns:
        DataFrame con columnas [date, ticker, open, high, low, close, adj_close, volume].
    """
    start = start or config.date_range()[0]
    end = end or config.date_range()[1]
    interval = interval or config.price_interval()

    cache = _cache_path(ticker)
    if use_cache and not refresh and cache.exists():
        cached = pd.read_csv(cache, parse_dates=["date"])
        return cached

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = _normalize(raw, ticker)

    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)

    return df


def download_all_prices(
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Descarga la serie OHLCV de todo el universo (empresas + benchmark + macro).

    Returns:
        dict {ticker: DataFrame}. Los tickers sin datos se omiten con aviso.
    """
    tickers = config.all_price_tickers()
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = download_prices(ticker, refresh=refresh)
        if df.empty:
            print(f"[prices] AVISO: sin datos para {ticker}")
            continue
        out[ticker] = df
        print(f"[prices] {ticker}: {len(df)} filas ({df['date'].min().date()} → {df['date'].max().date()})")
    return out


def combined_prices(refresh: bool = False) -> pd.DataFrame:
    """Devuelve un único DataFrame apilado (long format) con todo el universo."""
    frames = list(download_all_prices(refresh=refresh).values())
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *_OHLCV_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)

    # Guardar un consolidado con fecha de descarga (reproducibilidad).
    stamp = date.today().isoformat()
    out_path = config.raw_dir() / "prices" / f"_combined_{stamp}.csv"
    combined.to_csv(out_path, index=False)
    return combined
