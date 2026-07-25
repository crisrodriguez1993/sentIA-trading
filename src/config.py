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


def price_interval() -> str:
    """Frecuencia de los datos de precio (p. ej. '1d')."""
    return load_params()["data"]["interval"]


def raw_dir() -> Path:
    """Directorio de datos crudos (creado si no existe)."""
    path = PROJECT_ROOT / load_params()["data"]["raw_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    """Directorio de datasets procesados (creado si no existe)."""
    path = PROJECT_ROOT / load_params()["data"]["processed_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def target_config() -> tuple[int, float]:
    """Configuración del target: (horizonte en días, umbral de retorno)."""
    tgt = load_params()["target"]
    return int(tgt["horizon_days"]), float(tgt["threshold"])


def model_quant_config() -> dict[str, Any]:
    """Hiperparámetros y algoritmo del modelo cuantitativo."""
    return load_params()["model_quant"]


def model_lstm_config() -> dict[str, Any]:
    """Hiperparámetros del modelo secuencial LSTM (Fase 4-bis)."""
    return load_params()["model_lstm"]


def seed() -> int:
    """Semilla global de reproducibilidad."""
    return int(load_params().get("seed", 42))


def models_dir() -> Path:
    """Directorio de artefactos de modelos entrenados (creado si no existe)."""
    path = PROJECT_ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    """Directorio de reportes/métricas de evaluación (creado si no existe)."""
    path = PROJECT_ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path
