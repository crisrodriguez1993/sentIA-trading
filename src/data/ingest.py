"""Orquestador del pipeline de ingesta de datos (Fase 2).

Descarga y cachea todo el universo de datos desde fuentes gratuitas:
precios OHLCV (empresas + benchmark + macro) y fundamentales (empresas).

Ejecución desde la raíz del repositorio:
    python -m src.data.ingest             # usa la caché si existe
    python -m src.data.ingest --refresh   # fuerza redescarga
"""

from __future__ import annotations

import argparse

from src import config
from src.data import fundamentals, prices


def run(refresh: bool = False) -> None:
    """Ejecuta el pipeline completo de ingesta."""
    start, end = config.date_range()
    print("=" * 60)
    print("Sentia-Trading — Ingesta de datos (Fase 2)")
    print(f"Rango: {start} → {end} | intervalo: {config.price_interval()}")
    print(f"Empresas: {', '.join(config.company_tickers())}")
    print(f"Benchmark: {config.benchmark_ticker()} | Macro: {', '.join(config.macro_tickers())}")
    print("=" * 60)

    print("\n[1/2] Precios OHLCV")
    price_df = prices.combined_prices(refresh=refresh)
    print(f"  -> {len(price_df)} filas totales, {price_df['ticker'].nunique()} tickers")

    print("\n[2/2] Fundamentales")
    fund_df = fundamentals.download_all_fundamentals(refresh=refresh)
    n_periods = 0 if fund_df.empty else len(fund_df)
    n_tickers = 0 if fund_df.empty else fund_df["ticker"].nunique()
    print(f"  -> {n_periods} períodos, {n_tickers} empresas")

    print("\nIngesta completada. Datos crudos en:", config.raw_dir())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de ingesta de datos Sentia-Trading")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora la caché y vuelve a descargar todo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(refresh=args.refresh)
