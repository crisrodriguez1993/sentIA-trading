"""Carga y acceso centralizado a la configuración del proyecto.

Lee `config/universe.yaml` y `config/params.yaml` para que ningún módulo
tenga tickers, fechas ni hiperparámetros "hardcodeados".
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Raíz del repositorio (dos niveles por encima de este archivo: src/config.py -> raíz).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_universe() -> dict[str, Any]:
    """Devuelve el universo de datos (empresas, benchmark, macro)."""
    return _load_yaml(CONFIG_DIR / "universe.yaml")


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    """Devuelve los parámetros del proyecto (fechas, target, backtest, etc.)."""
    return _load_yaml(CONFIG_DIR / "params.yaml")


def company_tickers() -> list[str]:
    """Lista de tickers de las empresas del portafolio."""
    return [c["ticker"] for c in load_universe()["companies"]]


def macro_tickers() -> list[str]:
    """Lista de tickers de activos macro/refugio."""
    return [m["ticker"] for m in load_universe()["macro"]]


def benchmark_ticker() -> str:
    """Ticker del benchmark pasivo (S&P 500)."""
    return load_universe()["benchmark"]["ticker"]


def all_price_tickers() -> list[str]:
    """Todos los tickers con serie de precios: empresas + benchmark + macro."""
    return company_tickers() + [benchmark_ticker()] + macro_tickers()


def date_range() -> tuple[str, str]:
    """Rango de fechas (start, end). `end` = hoy si está en null."""
    data_cfg = load_params()["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"] or date.today().isoformat()
    return start, end
