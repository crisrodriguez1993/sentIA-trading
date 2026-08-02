"""Construcción de señales de trading para Fase 6.

Entrena el clasificador cuantitativo sobre el tramo histórico disponible y
emite una probabilidad diaria por activo. Luego aplica reglas de decisión:
- Comprar/mantener cuando la probabilidad supera un umbral.
- Reducir exposición cuando el riesgo macro se dispara (VIX alto).
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.models.classifier import QuantClassifier
from src.models.dataset import final_holdout_split, prepare_dataset


def build_probability_frame() -> pd.DataFrame:
    """Entrena el modelo y devuelve probabilidad diaria por ticker.

    Returns:
        DataFrame con columnas [date, ticker, proba_up].
    """
    ds = prepare_dataset(dropna_features=True)
    idx_train, idx_test, _ = final_holdout_split(ds.dates, test_fraction=0.2)

    # Entrenamiento out-of-sample: se entrena en pasado y se predice en futuro.
    clf = QuantClassifier().fit(ds.X.iloc[idx_train], ds.y.iloc[idx_train])
    proba = clf.predict_proba(ds.X.iloc[idx_test])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(ds.dates).iloc[idx_test].reset_index(drop=True),
            "ticker": ds.tickers.iloc[idx_test].reset_index(drop=True),
            "proba_up": proba,
        }
    )
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_trade_signals(
    proba_df: pd.DataFrame,
    threshold: float = 0.55,
    risk_off_vix: float | None = None,
) -> pd.DataFrame:
    """Genera señal binaria diaria de exposición por ticker.

    Args:
        proba_df: DataFrame de probabilidades [date, ticker, proba_up].
        threshold: umbral de compra/mantener.
        risk_off_vix: umbral VIX para modo defensivo; si None usa config.

    Returns:
        DataFrame con [date, ticker, signal, proba_up, risk_off].
    """
    if risk_off_vix is None:
        risk_off_vix = float(config.load_params()["risk"]["vix_threshold"])

    features = pd.read_parquet(config.processed_dir() / "features.parquet")
    vix = (
        features[["date", "vix_level"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    vix["risk_off"] = (vix["vix_level"] >= risk_off_vix).astype(int)

    sig = proba_df.copy()
    sig["signal"] = (sig["proba_up"] >= threshold).astype(int)
    sig = sig.merge(vix[["date", "risk_off"]], on="date", how="left")
    sig["risk_off"] = sig["risk_off"].fillna(0).astype(int)

    # Modo defensivo: cuando VIX está alto se desactiva señal larga.
    sig.loc[sig["risk_off"] == 1, "signal"] = 0

    return sig.sort_values(["date", "ticker"]).reset_index(drop=True)
