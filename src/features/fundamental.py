"""Alineación de features fundamentales a la serie diaria.

Los fundamentales (ROIC, margen operativo, crecimiento de ingresos) tienen
frecuencia anual y su `period` es la fecha de CIERRE fiscal. Sin embargo, los
reportes se publican semanas después de dicho cierre. Para evitar
*look-ahead bias*, aplicamos un desfase de publicación (`report_lag_days`)
antes de considerar un fundamental "disponible", y luego lo propagamos
(forward-fill) sobre la serie diaria hasta el siguiente reporte.
"""

from __future__ import annotations

import pandas as pd

# Desfase conservador entre cierre fiscal y publicación del reporte (10-K/10-Q).
REPORT_LAG_DAYS = 90

_FUNDAMENTAL_COLUMNS = ["roic", "op_margin", "revenue_growth"]


def align_fundamentals_to_daily(
    daily: pd.DataFrame,
    fundamentals: pd.DataFrame,
    ticker: str,
    report_lag_days: int = REPORT_LAG_DAYS,
) -> pd.DataFrame:
    """Une los fundamentales anuales a la serie diaria de un ticker.

    Args:
        daily: DataFrame diario (con columna 'date') de UN ticker.
        fundamentals: DataFrame de fundamentales (columnas 'ticker', 'period', métricas).
        ticker: símbolo cuyos fundamentales se alinean.
        report_lag_days: días de desfase entre cierre fiscal y disponibilidad.

    Returns:
        `daily` con las columnas fundamentales añadidas (forward-fill, sin look-ahead).
    """
    daily = daily.sort_values("date").reset_index(drop=True).copy()

    fund = fundamentals[fundamentals["ticker"] == ticker].copy()
    if fund.empty:
        for col in _FUNDAMENTAL_COLUMNS:
            daily[col] = pd.NA
        return daily

    # Fecha en la que el fundamental pasa a estar "disponible" al público.
    fund["available_date"] = pd.to_datetime(fund["period"]) + pd.Timedelta(days=report_lag_days)
    fund = fund.sort_values("available_date").reset_index(drop=True)

    keep = ["available_date", *[c for c in _FUNDAMENTAL_COLUMNS if c in fund.columns]]
    fund = fund[keep].rename(columns={"available_date": "date"})

    # merge_asof: para cada día, toma el último fundamental disponible ANTES de esa fecha.
    merged = pd.merge_asof(
        daily,
        fund,
        on="date",
        direction="backward",
    )
    return merged


def fundamental_feature_columns() -> list[str]:
    """Lista de columnas de features fundamentales."""
    return list(_FUNDAMENTAL_COLUMNS)
