"""Evaluación por regímenes (Fase 7).

Calcula el desempeño de la estrategia y benchmark por ventanas históricas
(covid, ai_boom, geopolitics) definidas en config/params.yaml.
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.backtest.engine import _compute_metrics, _pivot_prices, _signals_to_weights, _simulate_portfolio
from src.data import prices as price_data
from src.models.classifier import QuantClassifier
from src.models.dataset import prepare_dataset
from src.strategy.signals import build_trade_signals


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return df.loc[mask]


def _rolling_probability_frame(min_train_days: int = 504, retrain_every: int = 21) -> pd.DataFrame:
    """Genera probabilidades out-of-sample en toda la historia con walk-forward.

    - Entrena con datos pasados únicamente.
    - Reentrena cada `retrain_every` días de mercado.
    - Comienza cuando hay al menos `min_train_days` observaciones históricas.
    """
    ds = prepare_dataset(dropna_features=True)
    dates = pd.to_datetime(ds.dates).reset_index(drop=True)

    unique_dates = sorted(dates.unique())
    if len(unique_dates) <= min_train_days:
        raise ValueError("No hay suficientes fechas para rolling walk-forward")

    out_frames = []
    clf: QuantClassifier | None = None
    last_fit_idx = -10**9

    date_to_index = {d: i for i, d in enumerate(unique_dates)}

    for d in unique_dates[min_train_days:]:
        d_idx = date_to_index[d]
        train_mask = dates < d
        test_mask = dates == d
        if not train_mask.any() or not test_mask.any():
            continue

        if clf is None or (d_idx - last_fit_idx) >= retrain_every:
            clf = QuantClassifier().fit(ds.X.loc[train_mask], ds.y.loc[train_mask])
            last_fit_idx = d_idx

        proba = clf.predict_proba(ds.X.loc[test_mask])
        out_frames.append(
            pd.DataFrame(
                {
                    "date": dates.loc[test_mask].values,
                    "ticker": ds.tickers.loc[test_mask].values,
                    "proba_up": proba,
                }
            )
        )

    if not out_frames:
        return pd.DataFrame(columns=["date", "ticker", "proba_up"])
    return pd.concat(out_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def evaluate_regimes() -> dict:
    params = config.load_params()
    features = pd.read_parquet(config.processed_dir() / "features.parquet")
    px = _pivot_prices(features)
    tickers = config.company_tickers()

    proba = _rolling_probability_frame(min_train_days=504, retrain_every=21)
    signals = build_trade_signals(proba_df=proba, threshold=0.55)
    w = _signals_to_weights(signals, tickers)

    initial_capital = float(params["backtest"]["initial_capital"])
    fee = float(params["backtest"]["commission"])

    px_assets = px.reindex(columns=tickers).dropna(how="all")

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

    report: dict[str, dict] = {}
    for regime_name, win in params["regimes"].items():
        start = win["start"]
        end = win["end"]

        close_r = _slice(px_assets, start, end)
        w_r = _slice(w, start, end)
        if close_r.empty or w_r.empty:
            continue
        w_r = w_r.reindex(close_r.index).fillna(0.0)

        eq_a, ret_a = _simulate_portfolio(close_r, w_r, initial_capital, fee)

        spy_r = _slice(px_spy, start, end)
        spy_ret = spy_r[benchmark].pct_change().fillna(0.0)
        if not spy_ret.empty:
            spy_ret.iloc[0] = spy_ret.iloc[0] - fee
        eq_b = (1.0 + spy_ret).cumprod() * initial_capital if not spy_ret.empty else pd.Series(dtype=float)

        report[regime_name] = {
            "window": {"start": start, "end": end},
            "strategy": _compute_metrics(eq_a, ret_a),
            "benchmark_spy": _compute_metrics(eq_b, spy_ret) if not spy_ret.empty else {},
        }

    out = config.reports_dir() / "regime_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":
    result = evaluate_regimes()
    print("Regimes evaluados:", ", ".join(result.keys()))
