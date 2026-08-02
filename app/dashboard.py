from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _load_json(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def render_global(backtest: dict) -> None:
    section_header("Fase 6 · Backtesting Global", "Estrategia activa vs benchmark SPY (out-of-sample)")

    if not backtest:
        st.warning("No se encontró reports/backtest_report.json")
        return

    s = backtest.get("strategy", {})
    b = backtest.get("benchmark_spy", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Return % (Estrategia)", _fmt(s.get("total_return_pct")), _fmt(s.get("total_return_pct", 0) - b.get("total_return_pct", 0)))
    c2.metric("Sharpe (Estrategia)", _fmt(s.get("sharpe_ratio"), 3), _fmt(s.get("sharpe_ratio", 0) - b.get("sharpe_ratio", 0), 3))
    c3.metric("Max Drawdown % (Estrategia)", _fmt(s.get("max_drawdown_pct"), 2), _fmt(s.get("max_drawdown_pct", 0) - b.get("max_drawdown_pct", 0), 2))

    df = pd.DataFrame(
        [
            {
                "Modelo": "Estrategia",
                "Return %": s.get("total_return_pct"),
                "Sharpe": s.get("sharpe_ratio"),
                "Sortino": s.get("sortino_ratio"),
                "Max Drawdown %": s.get("max_drawdown_pct"),
                "Calmar": s.get("calmar_ratio"),
            },
            {
                "Modelo": "SPY",
                "Return %": b.get("total_return_pct"),
                "Sharpe": b.get("sharpe_ratio"),
                "Sortino": b.get("sortino_ratio"),
                "Max Drawdown %": b.get("max_drawdown_pct"),
                "Calmar": b.get("calmar_ratio"),
            },
        ]
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df,
        x="Modelo",
        y=["Return %", "Sharpe", "Max Drawdown %"],
        barmode="group",
        title="Comparación Global",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_regimes(regimes: dict) -> None:
    section_header("Fase 7 · Evaluación por Regímenes", "COVID, Boom IA y Geopolítica")

    if not regimes:
        st.warning("No se encontró reports/regime_report.json")
        return

    rows = []
    for name, payload in regimes.items():
        s = payload.get("strategy", {})
        b = payload.get("benchmark_spy", {})
        rows.append(
            {
                "Regimen": name,
                "Return % Estrategia": s.get("total_return_pct"),
                "Return % SPY": b.get("total_return_pct"),
                "Sharpe Estrategia": s.get("sharpe_ratio"),
                "Sharpe SPY": b.get("sharpe_ratio"),
                "MDD % Estrategia": s.get("max_drawdown_pct"),
                "MDD % SPY": b.get("max_drawdown_pct"),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig_ret = px.bar(
        df,
        x="Regimen",
        y=["Return % Estrategia", "Return % SPY"],
        barmode="group",
        title="Retorno por Régimen",
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    fig_dd = px.bar(
        df,
        x="Regimen",
        y=["MDD % Estrategia", "MDD % SPY"],
        barmode="group",
        title="Max Drawdown por Régimen",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


def render_ablation(abl: dict) -> None:
    section_header("Fase 5 · Ablación de Sentimiento (RQ2)", "¿Aporta FinBERT sobre el baseline cuantitativo?")

    if not abl:
        st.warning("No se encontró reports/ablation_sentiment.json")
        return

    b = abl.get("base", {}).get("metrics", {})
    w = abl.get("with_sentiment", {}).get("metrics", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC Base", _fmt(b.get("roc_auc"), 3))
    c2.metric("AUC + Sentimiento", _fmt(w.get("roc_auc"), 3))
    c3.metric("ΔAUC", _fmt(abl.get("delta_auc"), 3))

    df = pd.DataFrame(
        [
            {"Modelo": "Base", "Accuracy": b.get("accuracy"), "F1": b.get("f1"), "ROC-AUC": b.get("roc_auc")},
            {
                "Modelo": "Con Sentimiento",
                "Accuracy": w.get("accuracy"),
                "F1": w.get("f1"),
                "ROC-AUC": w.get("roc_auc"),
            },
        ]
    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = px.bar(df, x="Modelo", y=["Accuracy", "F1", "ROC-AUC"], barmode="group", title="Ablación")
    st.plotly_chart(fig, use_container_width=True)


def render_conclusions(backtest: dict, regimes: dict, abl: dict) -> None:
    section_header("Conclusiones Ejecutivas", "Lectura para defensa de titulación")

    if not backtest:
        return

    s = backtest.get("strategy", {})
    b = backtest.get("benchmark_spy", {})

    bullets = []
    if s and b:
        if s.get("max_drawdown_pct", 0) > b.get("max_drawdown_pct", 0):
            bullets.append("La estrategia no mejora drawdown frente a SPY en global.")
        else:
            bullets.append("La estrategia reduce drawdown global frente a SPY, pero sacrifica retorno y Sharpe.")
        if s.get("total_return_pct", 0) < b.get("total_return_pct", 0):
            bullets.append("No se observa alfa global: SPY supera a la estrategia en retorno acumulado.")

    if abl:
        dauc = abl.get("delta_auc")
        if dauc is not None:
            if dauc > 0:
                bullets.append(f"El sentimiento aporta señal incremental (ΔAUC = {dauc:.3f}).")
            else:
                bullets.append(f"Con la cobertura actual, el sentimiento no aporta en AUC (ΔAUC = {dauc:.3f}).")

    if regimes:
        covid = regimes.get("covid", {})
        if covid:
            sd = covid.get("strategy", {}).get("max_drawdown_pct", 0)
            bd = covid.get("benchmark_spy", {}).get("max_drawdown_pct", 0)
            if sd > bd:
                bullets.append("En COVID no hubo ventaja defensiva en drawdown.")
            else:
                bullets.append("En COVID sí se observó comportamiento defensivo en drawdown.")

    for btxt in bullets:
        st.markdown(f"- {btxt}")


def main() -> None:
    st.set_page_config(page_title="Sentia-Trading Dashboard", layout="wide")

    st.title("Sentia-Trading · Dashboard de Tesis")
    st.caption("Fases 5-7: Sentimiento FinBERT, Backtesting Global y Evaluación por Regímenes")

    backtest = _load_json("backtest_report.json")
    regimes = _load_json("regime_report.json")
    ablation = _load_json("ablation_sentiment.json")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen Global",
        "Regímenes",
        "Ablación Sentimiento",
        "Conclusiones",
    ])

    with tab1:
        render_global(backtest)
    with tab2:
        render_regimes(regimes)
    with tab3:
        render_ablation(ablation)
    with tab4:
        render_conclusions(backtest, regimes, ablation)


if __name__ == "__main__":
    main()
