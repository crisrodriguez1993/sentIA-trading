"""Ingesta de noticias por empresa desde GDELT (gratuito).

GDELT Doc API permite buscar artículos por palabra clave y rango de fechas,
devolviendo hasta `max_records` por consulta. Para cubrir 2018→hoy sin superar
ese límite, se trocea el rango en ventanas mensuales y se consulta por empresa.

El resultado (titulares + fecha) se cachea por ticker en `data/raw/news/`. La
descarga es **reanudable mes a mes**: cada mes descargado se persiste de forma
incremental en un archivo `<ticker>.partial.csv` junto con un registro de meses
completados (`<ticker>.progress.txt`). Si el proceso se interrumpe, al reanudar
se saltan los meses ya descargados. Cuando todos los meses están listos, se
consolida el CSV final `<ticker>.csv` y se limpian los temporales.
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


def _partial_path(ticker: str) -> Path:
    return config.news_dir() / f"{ticker}.partial.csv"


def _progress_path(ticker: str) -> Path:
    return config.news_dir() / f"{ticker}.progress.txt"


def _load_completed_months(ticker: str) -> set[str]:
    """Meses (por fecha de inicio) ya descargados para un ticker."""
    path = _progress_path(ticker)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _mark_month_done(ticker: str, chunk_start: str, rows: pd.DataFrame | None) -> None:
    """Persiste incrementalmente las filas del mes y lo marca como completado."""
    if rows is not None and not rows.empty:
        partial = _partial_path(ticker)
        header = not partial.exists()
        rows.to_csv(partial, mode="a", header=header, index=False)
    with _progress_path(ticker).open("a", encoding="utf-8") as fh:
        fh.write(f"{chunk_start}\n")


def download_news_for_ticker(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    pause: float = 7.0,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Descarga (o carga de caché) los titulares de noticias de un ticker.

    GDELT limita a 1 consulta cada ~5 s; respetamos ese ritmo con `pause` y
    reintentamos con backoff creciente ante fallos de rate limit. La descarga es
    reanudable: los meses ya completados (registrados en `<ticker>.progress.txt`)
    se saltan. El CSV final solo se consolida cuando TODOS los meses del rango
    están descargados; si quedan meses fallidos, se conservan los temporales para
    reintentarlos en una nueva ejecución (sin repetir los ya logrados).

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

    # Reanudación: si se pide refresh, se descartan los temporales previos.
    if refresh:
        _partial_path(ticker).unlink(missing_ok=True)
        _progress_path(ticker).unlink(missing_ok=True)

    completed = _load_completed_months(ticker)
    all_months = [m[0] for m in _month_starts(start, end)]
    pending = [m for m in all_months if m not in completed]
    gd = GdeltDoc()

    if pending:
        print(f"[news] {ticker}: {len(completed)}/{len(all_months)} meses ya listos; "
              f"faltan {len(pending)}")

    failed_this_run = 0
    for chunk_start, chunk_end in _month_starts(start, end):
        if chunk_start in completed:
            continue  # mes ya descargado en una ejecución anterior

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
                if attempt == max_retries:
                    print(f"[news] {ticker} {chunk_start}: falla tras {attempt} intentos (reintentable)")
                else:
                    # Backoff creciente: el rate limit de GDELT suele requerir esperas largas.
                    time.sleep(pause * (attempt + 1))

        month_rows: pd.DataFrame | None = None
        if articles is not None and not articles.empty:
            df = articles.copy()
            # Filtrar a inglés si la columna existe (FinBERT es de dominio EN).
            if "language" in df.columns:
                df = df[df["language"].str.lower().eq("english")]
            if not df.empty:
                df["date"] = pd.to_datetime(df["seendate"], errors="coerce").dt.tz_localize(None).dt.normalize()
                df["ticker"] = ticker
                keep = [c for c in ["ticker", "date", "title", "url", "domain"] if c in df.columns]
                month_rows = df[keep].dropna(subset=["date", "title"])

        # Solo marcar el mes como hecho si la consulta no falló por completo.
        if articles is not None:
            _mark_month_done(ticker, chunk_start, month_rows)
        else:
            failed_this_run += 1
        time.sleep(pause)

    partial = _partial_path(ticker)
    completed = _load_completed_months(ticker)
    missing = [m for m in all_months if m not in completed]

    # Si quedan meses por descargar, conservar temporales para reintentar luego.
    if missing:
        print(f"[news] {ticker}: cobertura incompleta, faltan {len(missing)} meses "
              f"(reintentar re-ejecutando). Fallidos en esta pasada: {failed_this_run}")
        if not partial.exists():
            return pd.DataFrame(columns=["ticker", "date", "title", "url", "domain"])
        return pd.read_csv(partial, parse_dates=["date"])

    # Cobertura completa: consolidar el parcial en el CSV final.
    if not partial.exists():
        print(f"[news] {ticker}: sin noticias descargadas")
        return pd.DataFrame(columns=["ticker", "date", "title", "url", "domain"])

    out = pd.read_csv(partial, parse_dates=["date"])
    out = out.drop_duplicates(subset=["title", "date"]).sort_values("date").reset_index(drop=True)
    out.to_csv(cache, index=False)

    # Limpieza de temporales tras consolidar con éxito.
    partial.unlink(missing_ok=True)
    _progress_path(ticker).unlink(missing_ok=True)

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
