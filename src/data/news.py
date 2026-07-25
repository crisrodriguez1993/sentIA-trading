"""Ingesta de noticias por empresa desde GDELT (gratuito).

GDELT Doc API permite buscar artículos por palabra clave y rango de fechas,
devolviendo hasta `max_records` por consulta. Para cubrir 2018→hoy sin superar
ese límite, se trocea el rango en ventanas mensuales y se consulta por empresa.

El resultado (titulares + fecha) se cachea por ticker en `data/raw/news/` para
no repetir descargas. La red se maneja de forma resiliente: si un chunk falla,
se registra un aviso y se continúa.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from gdeltdoc import Filters, GdeltDoc

from src import config


def _month_starts(start: str, end: str) -> list[tuple[str, str]]:
    """Genera pares (inicio, fin) de ventanas mensuales que cubren [start, end]."""
    rng = pd.date_range(start=start, end=end, freq="MS")  # inicios de mes
    bounds: list[tuple[str, str]] = []
    for i, month_start in enumerate(rng):
        next_start = rng[i + 1] if i + 1 < len(rng) else pd.Timestamp(end) + pd.Timedelta(days=1)
        bounds.append((month_start.strftime("%Y-%m-%d"), next_start.strftime("%Y-%m-%d")))
    return bounds


def _cache_path(ticker: str) -> Path:
    return config.news_dir() / f"{ticker}.csv"


def download_news_for_ticker(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    pause: float = 5.5,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Descarga (o carga de caché) los titulares de noticias de un ticker.

    GDELT limita a 1 consulta cada ~5 s; respetamos ese ritmo con `pause` y
    reintentamos con backoff ante fallos transitorios.

    Returns:
        DataFrame con columnas [ticker, date, title, url, domain].
    """
    cache = _cache_path(ticker)
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"])

    start = start or config.date_range()[0]
    end = end or config.date_range()[1]
    keyword = config.news_keyword(ticker)
    news_cfg = config.news_config()
    max_records = int(news_cfg.get("max_records", 250))

    gd = GdeltDoc()
    rows: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _month_starts(start, end):
        articles = None
        for attempt in range(1, max_retries + 1):
            try:
                filters = Filters(
                    start_date=chunk_start,
                    end_date=chunk_end,
                    num_records=max_records,
                    keyword=keyword,
                )
                articles = gd.article_search(filters)
                break
            except Exception as exc:  # noqa: BLE001 - la API puede fallar por red/límite
                wait = pause * attempt
                if attempt == max_retries:
                    print(f"[news] {ticker} {chunk_start}: aviso tras {attempt} intentos ({exc})")
                else:
                    time.sleep(wait)

        if articles is None or articles.empty:
            time.sleep(pause)
            continue

        df = articles.copy()
        # Filtrar a inglés si la columna existe (FinBERT es de dominio EN).
        if "language" in df.columns:
            df = df[df["language"].str.lower().eq("english")]
        if not df.empty:
            df["date"] = pd.to_datetime(df["seendate"], errors="coerce").dt.tz_localize(None).dt.normalize()
            df["ticker"] = ticker
            keep = [c for c in ["ticker", "date", "title", "url", "domain"] if c in df.columns]
            rows.append(df[keep].dropna(subset=["date", "title"]))
        time.sleep(pause)

    if not rows:
        print(f"[news] {ticker}: sin noticias descargadas")
        return pd.DataFrame(columns=["ticker", "date", "title", "url", "domain"])

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["title", "date"])
    out = out.sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"[news] {ticker}: {len(out)} titulares ({out['date'].min().date()} → {out['date'].max().date()})")
    return out


def download_all_news(refresh: bool = False) -> pd.DataFrame:
    """Descarga las noticias de todas las empresas del universo."""
    frames: list[pd.DataFrame] = []
    for ticker in config.company_tickers():
        df = download_news_for_ticker(ticker, refresh=refresh)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "title", "url", "domain"])
    return pd.concat(frames, ignore_index=True)
