"""Motor de backtesting para Fase 6.

Compara la estrategia activa (señales del modelo) contra Buy & Hold en SPY.
Usa un motor vectorizado con pandas para simular operaciones con comisión.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.data import prices as price_data


def _pivot_prices(features: pd.DataFrame) -> pd.DataFrame:
    px = (
        features[["date", "ticker", "adj_close"]]
        .drop_duplicates(subset=["date", "ticker"])
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )
    return px


def _signals_to_weights(signals: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    s = (
        signals[["date", "ticker", "signal"]]
        .pivot(index="date", columns="ticker", values="signal")
        .reindex(columns=tickers)
        .fillna(0.0)
        .sort_index()
    )
    # Equal weight entre activos con señal=1; si ninguno activo -> 0% invertido.
    active_count = s.sum(axis=1).replace(0, np.nan)
    w = s.div(active_count, axis=0).fillna(0.0)
    return w


def _compute_metrics(equity: pd.Series, daily_returns: pd.Series) -> dict[str, float]:
    """Calcula métricas financieras estándar desde curva de capital y retornos."""
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100

    days = max(len(daily_returns), 1)
    ann_factor = 252.0
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (ann_factor / days) - 1.0) * 100

    mean = daily_returns.mean()
    std = daily_returns.std(ddof=0)
    downside = daily_returns[daily_returns < 0].std(ddof=0)

    sharpe = (mean / std) * np.sqrt(ann_factor) if std and std > 0 else np.nan
    sortino = (mean / downside) * np.sqrt(ann_factor) if downside and downside > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min() * 100
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

    return {
        "total_return_pct": float(total_return),
        "annualized_return_pct": float(cagr),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd),
        "calmar_ratio": float(calmar),
    }


def _simulate_portfolio(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float,
    fee: float,
) -> tuple[pd.Series, pd.Series]:
    """Simula un portafolio daily-rebalanced con costo por turnover.

    fee se aplica como fricción sobre el turnover diario:
    cost_t = fee * sum_i |w_t - w_{t-1}|.
    """
    close = close.sort_index()
    weights = weights.reindex(close.index).fillna(0.0)

    asset_ret = close.pct_change().fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)
    gross_ret = (w_prev * asset_ret).sum(axis=1)

    turnover = (weights - w_prev).abs().sum(axis=1)
    costs = fee * turnover
    net_ret = gross_ret - costs

    equity = (1.0 + net_ret).cumprod() * initial_capital
    return equity, net_ret


def run_backtest(signals: pd.DataFrame) -> dict:
    """Ejecuta backtest de estrategia activa vs SPY y guarda reporte.

    Returns:
        Dict con métricas de estrategia y benchmark.
    """
    params = config.load_params()
    features = pd.read_parquet(config.processed_dir() / "features.parquet")
    px = _pivot_prices(features)

    tickers = config.company_tickers()
    px_assets = px.reindex(columns=tickers).dropna(how="all")

    w = _signals_to_weights(signals, tickers).reindex(px_assets.index).fillna(0.0)

    initial_capital = float(params["backtest"]["initial_capital"])
    fee = float(params["backtest"]["commission"])

    benchmark = config.benchmark_ticker()
    spy_df = price_data.download_prices(benchmark)
    px_spy = (
        spy_df[["date", "adj_close"]]
        .dropna()
        .drop_duplicates(subset=["date"])
        .set_index("date")
        .rename(columns={"adj_close": benchmark})
        .sort_index()
    )

    # Estrategia activa multi-activo
    eq_active, ret_active = _simulate_portfolio(
        close=px_assets,
        weights=w,
        initial_capital=initial_capital,
        fee=fee,
    )

    # Benchmark buy & hold SPY (1 trade inicial, sin rebalanceos)
    spy_ret = px_spy[benchmark].pct_change().fillna(0.0)
    spy_ret.iloc[0] = spy_ret.iloc[0] - fee
    eq_spy = (1.0 + spy_ret).cumprod() * initial_capital

    report = {
        "strategy": _compute_metrics(eq_active, ret_active),
        "benchmark_spy": _compute_metrics(eq_spy, spy_ret),
        "meta": {
            "tickers": tickers,
            "benchmark": benchmark,
            "initial_capital": initial_capital,
            "commission": fee,
        },
    }

    out = config.reports_dir() / "backtest_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report
