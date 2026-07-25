"""Ingesta de datos fundamentales desde Yahoo Finance (yfinance).

Extrae partidas de los estados financieros (anuales) de cada empresa y
calcula las métricas clave usadas como features del modelo cuantitativo:

- ROIC (Return on Invested Capital) ≈ NOPAT / Capital Invertido
- OP   (Operating Margin) = Ingreso Operativo / Ingresos
- OG   (Organic/Revenue Growth) = variación interanual de ingresos

Los fundamentales gratuitos tienen profundidad histórica limitada (pocos
años); esta limitación se documenta en la metodología. Los activos macro y
el benchmark no tienen fundamentales y se omiten.

Uso rápido:
    from src.data.fundamentals import download_all_fundamentals
    df = download_all_fundamentals()
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import config

# Tasa impositiva efectiva de respaldo cuando no puede estimarse (para NOPAT).
_FALLBACK_TAX_RATE = 0.21


def _cache_path(ticker: str) -> Path:
    return config.raw_dir() / "fundamentals" / f"{ticker}.csv"


def _first_row(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    """Devuelve la primera fila cuyo índice coincida con alguno de `names`."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _compute_metrics(ticker: str, income: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Calcula ROIC, margen operativo y crecimiento de ingresos por período."""
    if income is None or income.empty:
        return pd.DataFrame()

    revenue = _first_row(income, ["Total Revenue", "TotalRevenue", "Operating Revenue"])
    op_income = _first_row(income, ["Operating Income", "OperatingIncome", "EBIT"])
    pretax = _first_row(income, ["Pretax Income", "PretaxIncome", "Income Before Tax"])
    tax = _first_row(income, ["Tax Provision", "TaxProvision", "Income Tax Expense"])

    total_debt = _first_row(balance, ["Total Debt", "TotalDebt"])
    equity = _first_row(
        balance,
        ["Stockholders Equity", "Total Stockholder Equity", "StockholdersEquity",
         "Common Stock Equity"],
    )
    cash = _first_row(
        balance,
        ["Cash And Cash Equivalents", "CashAndCashEquivalents",
         "Cash Cash Equivalents And Short Term Investments"],
    )

    if revenue is None:
        return pd.DataFrame()

    periods = sorted(revenue.index)  # columnas = fechas de cierre fiscal
    rows: list[dict] = []
    prev_rev: float | None = None

    for period in periods:
        rev = _safe(revenue, period)
        opi = _safe(op_income, period)
        ptx = _safe(pretax, period)
        txp = _safe(tax, period)
        debt = _safe(total_debt, period)
        eq = _safe(equity, period)
        csh = _safe(cash, period)

        # Margen operativo (OP).
        op_margin = opi / rev if (opi is not None and rev) else None

        # Tasa efectiva de impuestos -> NOPAT -> ROIC.
        tax_rate = (txp / ptx) if (txp is not None and ptx) else _FALLBACK_TAX_RATE
        tax_rate = min(max(tax_rate, 0.0), 0.6)  # acotar valores atípicos
        nopat = opi * (1 - tax_rate) if opi is not None else None

        invested_capital = None
        if eq is not None:
            invested_capital = eq + (debt or 0.0) - (csh or 0.0)
        roic = (
            nopat / invested_capital
            if (nopat is not None and invested_capital and invested_capital > 0)
            else None
        )

        # Crecimiento de ingresos (OG) interanual.
        rev_growth = (
            (rev - prev_rev) / prev_rev
            if (prev_rev is not None and prev_rev and rev is not None)
            else None
        )
        prev_rev = rev if rev is not None else prev_rev

        rows.append(
            {
                "ticker": ticker,
                "period": pd.to_datetime(period).normalize(),
                "revenue": rev,
                "operating_income": opi,
                "op_margin": op_margin,
                "roic": roic,
                "revenue_growth": rev_growth,
            }
        )

    return pd.DataFrame(rows)


def _safe(series: pd.Series | None, key) -> float | None:
    """Extrae un valor numérico de una serie, tolerando ausencias/NaN."""
    if series is None or key not in series.index:
        return None
    val = series[key]
    if pd.isna(val):
        return None
    return float(val)


def download_fundamentals(
    ticker: str,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Descarga (o carga de caché) los fundamentales anuales de un ticker."""
    cache = _cache_path(ticker)
    if use_cache and not refresh and cache.exists():
        return pd.read_csv(cache, parse_dates=["period"])

    tk = yf.Ticker(ticker)
    try:
        income = tk.financials          # estado de resultados anual
        balance = tk.balance_sheet      # balance general anual
    except Exception as exc:  # noqa: BLE001 - yfinance lanza errores variados
        print(f"[fundamentals] AVISO: fallo al descargar {ticker}: {exc}")
        return pd.DataFrame()

    df = _compute_metrics(ticker, income, balance)
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def download_all_fundamentals(refresh: bool = False) -> pd.DataFrame:
    """Descarga fundamentales de todas las empresas del portafolio."""
    frames: list[pd.DataFrame] = []
    for ticker in config.company_tickers():
        df = download_fundamentals(ticker, refresh=refresh)
        if df.empty:
            print(f"[fundamentals] AVISO: sin fundamentales para {ticker}")
            continue
        frames.append(df)
        print(f"[fundamentals] {ticker}: {len(df)} períodos")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    stamp = date.today().isoformat()
    out_path = config.raw_dir() / "fundamentals" / f"_combined_{stamp}.csv"
    combined.to_csv(out_path, index=False)
    return combined
