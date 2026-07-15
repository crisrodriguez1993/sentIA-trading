# Sentia-Trading

**Sistema de trading algorítmico híbrido (fundamental + técnico + sentimiento)** que evalúa si la fusión de datos numéricos, fundamentales y de sentimiento genera **alfa** frente a una estrategia pasiva sobre el **S&P 500**, con énfasis en el desempeño bajo **regímenes de crisis** (pandemia, boom de IA, tensiones geopolíticas) y por **sectores** (IA disruptiva vs. tradicional/retail).

> Proyecto de titulación — Ingeniería en Inteligencia Artificial. **No constituye asesoría de inversión.**

---

## Idea en una frase
¿Puede un sistema de IA que combina la **salud fundamental** de una empresa, sus **señales técnicas** y el **sentimiento de las noticias** superar en retorno ajustado por riesgo a "comprar y mantener el S&P 500", especialmente cuando el mercado entra en caos?

## Universo inicial
| Grupo | Tickers |
|-------|---------|
| IA / Tecnología | `NVDA`, `GOOGL`, `MSFT`, `AMZN` |
| Retail / Tradicional | `NSRGY` (Nestlé ADR) |
| Benchmark | `SPY` |
| Macro / refugio | `^VIX`, oro (`GC=F`), petróleo (`CL=F`) |

El universo es parametrizable en [config/universe.yaml](config/universe.yaml) — añadir tickers no requiere tocar código.

## Estructura del repositorio
```
Sentia-Trading/
├── config/         # universe.yaml, params.yaml (sin hardcodear nada)
├── data/           # raw/ y processed/ (no versionado)
├── docs/           # documentación metodológica
├── notebooks/      # EDA y experimentos (consumen src/)
├── src/            # TODA la lógica del proyecto
│   ├── config.py   # carga centralizada de configuración
│   ├── data/       # ingesta y limpieza
│   ├── features/   # ingeniería de variables
│   ├── models/     # XGBoost / LightGBM
│   ├── nlp/        # FinBERT / sentimiento
│   ├── strategy/   # regla de decisión
│   └── backtest/   # motor y métricas
├── app/            # dashboard Streamlit
├── tests/          # pruebas unitarias
├── requirements.txt
└── README.md
```

## Puesta en marcha

### Opción A — Entorno local (venv)
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

### Opción B — Google Colab
Sube el repositorio o clónalo, y en la primera celda:
```python
!pip install -r requirements.txt
```

## Documentación
- **Metodología completa**: [docs/01-metodologia.md](docs/01-metodologia.md) — planteamiento del problema, hipótesis, objetivos, fuentes de datos, modelado, backtesting y plan de fases.

## Estado del proyecto (fases)
- [x] **Fase 0** — Documento metodológico
- [x] **Fase 1** — Scaffolding del repositorio
- [ ] **Fase 2** — Pipeline de datos (OHLCV + fundamentales + macro)
- [ ] **Fase 3** — Ingeniería de features
- [ ] **Fase 4** — Modelo cuantitativo (XGBoost/LightGBM)
- [ ] **Fase 5** — Módulo de sentimiento (FinBERT)
- [ ] **Fase 6** — Regla de decisión + backtesting vs. SPY
- [ ] **Fase 7** — Evaluación por escenarios y sectores
- [ ] **Fase 8** — Dashboard Streamlit
