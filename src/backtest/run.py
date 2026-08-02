"""Runner de Fase 6: señales + backtest + resumen contra SPY."""

from __future__ import annotations

from src.backtest.engine import run_backtest
from src.strategy.signals import build_probability_frame, build_trade_signals


def run() -> dict:
    print("=" * 60)
    print("Sentia-Trading — Fase 6 Backtesting")
    print("=" * 60)

    proba = build_probability_frame()
    signals = build_trade_signals(proba_df=proba, threshold=0.55)

    report = run_backtest(signals)

    s = report["strategy"]
    b = report["benchmark_spy"]
    print("\nResumen (Estrategia vs SPY)")
    print("-" * 60)
    print(f"Return %      : {s['total_return_pct']:.2f} vs {b['total_return_pct']:.2f}")
    print(f"Sharpe        : {s['sharpe_ratio']:.3f} vs {b['sharpe_ratio']:.3f}")
    print(f"Max Drawdown %: {s['max_drawdown_pct']:.2f} vs {b['max_drawdown_pct']:.2f}")
    print("Reporte guardado en reports/backtest_report.json")

    return report


if __name__ == "__main__":
    run()
