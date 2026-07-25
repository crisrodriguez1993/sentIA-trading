"""Features técnicos derivados de la serie OHLCV.

Calcula indicadores clásicos de análisis técnico para cada ticker a partir
del precio ajustado y el volumen. Todos los features en la fila del día `t`
usan únicamente información disponible hasta el cierre de `t` (sin look-ahead).

Indicadores:
- Retornos: diario, y a 5 / 10 / 21 días.
- Tendencia: medias móviles (SMA) y su ratio respecto al precio; MACD.
- Momentum: RSI.
- Volatilidad: desviación de retornos (21d) y bandas de Bollinger (ancho).
- Volumen: volumen relativo respecto a su media móvil.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

# Ventanas (en días de trading) usadas por los indicadores.
_RETURN_WINDOWS = [5, 10, 21]
_SMA_WINDOWS = [10, 20, 50, 200]


def add_technical_features(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Añade columnas de features técnicos a un DataFrame OHLCV de un ticker.

    Args:
        df: DataFrame ordenado por fecha con columnas OHLCV de UN solo ticker.
        price_col: columna de precio base (por defecto 'adj_close').

    Returns:
        El DataFrame con las columnas de features técnicos añadidas.
    """
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True).copy()
    price = df[price_col]

    # --- Retornos ---
    df["ret_1d"] = price.pct_change()
    for w in _RETURN_WINDOWS:
        df[f"ret_{w}d"] = price.pct_change(w)

    # --- Medias móviles y su relación con el precio ---
    for w in _SMA_WINDOWS:
        sma = SMAIndicator(close=price, window=w, fillna=False).sma_indicator()
        df[f"sma_{w}"] = sma
        df[f"price_to_sma_{w}"] = price / sma - 1.0

    # --- MACD ---
    macd = MACD(close=price, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # --- RSI (momentum) ---
    df["rsi_14"] = RSIIndicator(close=price, window=14, fillna=False).rsi()

    # --- Volatilidad ---
    df["volatility_21d"] = df["ret_1d"].rolling(21).std()
    bb = BollingerBands(close=price, window=20, window_dev=2, fillna=False)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / price

    # --- Volumen relativo ---
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20).mean()
        df["volume_rel"] = df["volume"] / vol_ma

    return df


def technical_feature_columns() -> list[str]:
    """Lista de nombres de columnas de features técnicos generadas."""
    cols = ["ret_1d"]
    cols += [f"ret_{w}d" for w in _RETURN_WINDOWS]
    cols += [f"sma_{w}" for w in _SMA_WINDOWS]
    cols += [f"price_to_sma_{w}" for w in _SMA_WINDOWS]
    cols += ["macd", "macd_signal", "macd_diff", "rsi_14",
             "volatility_21d", "bb_width", "volume_rel"]
    return cols
