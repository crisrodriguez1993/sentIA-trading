# Sentia-Trading — Documento Metodológico

**Proyecto de titulación — Máster en Inteligencia Artificial**
Sistema de trading algorítmico híbrido (fundamental + técnico + sentimiento) con evaluación frente al S&P 500 bajo regímenes de alta volatilidad.

> Estado: v0.1 (borrador metodológico). Este documento define el "qué" y el "por qué" antes de programar. Es la base de la memoria de titulación y guía todo el desarrollo del repositorio.

---

## 1. Planteamiento del problema

### 1.1 La historia completa
Las estrategias tradicionales de inversión pasiva (comprar y mantener el S&P 500) son sencillas y difíciles de superar de forma consistente, pero son **ciegas al contexto**: no reaccionan a la psicología del mercado (noticias), no distinguen la salud interna de cada empresa (métricas fundamentales) y sufren caídas profundas durante eventos de "cisne negro" (pandemias, guerras).

Por otro lado, los modelos que solo miran datos numéricos de precios (OHLC) ignoran dos fuentes de información valiosas:
1. **La salud fundamental de la empresa** (ROIC, márgenes, crecimiento de ingresos).
2. **La psicología del mercado** reflejada en noticias y eventos geopolíticos.

### 1.2 Definición formal del problema
> El problema no es "predecir el precio exacto de una acción", sino **optimizar la toma de decisiones de inversión y la gestión de riesgo** fusionando datos **fundamentales, técnicos y alternativos (sentimiento + geopolítica)**, y demostrar si dicha fusión **genera alfa** (retorno ajustado por riesgo superior) frente a una estrategia pasiva sobre el S&P 500, **especialmente durante regímenes de crisis**.

### 1.3 Diferenciador académico
La innovación no está en un único modelo, sino en **cómo se fusionan las fuentes heterogéneas** (tabular + texto + macro) y en la **evaluación honesta por regímenes de mercado** (pandemia, boom de IA, tensiones geopolíticas) y **por sectores** (IA disruptiva vs. sectores tradicionales/retail).

---

## 2. Preguntas de investigación e hipótesis

| # | Pregunta de investigación | Hipótesis |
|---|---------------------------|-----------|
| RQ1 | ¿Un sistema híbrido (fundamental + técnico + sentimiento) supera en retorno ajustado por riesgo a una estrategia Buy & Hold sobre el S&P 500? | H1: El sistema híbrido logra un Sharpe Ratio superior y un Max Drawdown menor que el benchmark. |
| RQ2 | ¿El aporte del análisis de sentimiento (FinBERT) mejora significativamente respecto a un modelo solo cuantitativo? | H2: Añadir el score de sentimiento mejora las métricas de clasificación y las financieras de forma estadísticamente significativa. |
| RQ3 | ¿El sistema mitiga mejor las pérdidas durante shocks (COVID-19, guerras) al incorporar variables geopolíticas (GPR, VIX, oro)? | H3: En ventanas de crisis, el sistema reduce el drawdown frente al benchmark. |
| RQ4 | ¿El comportamiento del modelo difiere entre sectores de IA disruptiva y sectores tradicionales/retail? | H4: Los sectores de IA presentan mayor sensibilidad al sentimiento; los tradicionales, a fundamentales. |

---

## 3. Objetivos

### 3.1 Objetivo general
Diseñar, implementar y evaluar un sistema de trading algorítmico híbrido basado en IA que integre información fundamental, técnica y de sentimiento, y comparar su desempeño frente a una estrategia pasiva sobre el S&P 500 bajo distintos regímenes de mercado.

### 3.2 Objetivos específicos
1. Construir un **pipeline de datos reproducible** que consolide precios OHLCV, métricas fundamentales, noticias y variables macro/geopolíticas desde fuentes gratuitas.
2. Desarrollar un **modelo cuantitativo de clasificación** (XGBoost/LightGBM) que prediga la dirección del retorno futuro a partir de features técnicos y fundamentales.
3. Implementar un **módulo de análisis de sentimiento financiero** (FinBERT) que genere un score diario por empresa.
4. Diseñar una **regla de decisión y un motor de backtesting** que combine ambos modelos y gestione el riesgo con señales macro (VIX, GPR, oro).
5. **Evaluar y comparar** el sistema frente al benchmark con métricas financieras (Retorno acumulado, Sharpe, Max Drawdown) en ventanas de estrés y por sectores.

---

## 4. Alcance y limitaciones

### 4.1 Alcance
- Frecuencia de datos: **diaria** (cierre). El sistema simula decisiones con rebalanceo semanal o mensual.
- Horizonte de backtesting: aproximadamente **2018–2026**, cubriendo pre-pandemia, COVID, boom de IA y tensiones geopolíticas.
- Universo inicial acotado (ver Sección 5).

### 4.2 Limitaciones conocidas (a documentar en la tesis)
- **Datos gratuitos**: `yfinance` y fuentes públicas pueden tener huecos, revisiones o *survivorship bias*. Se documentará el tratamiento.
- **Fundamentales gratuitos** tienen profundidad histórica limitada (pocos años de 10-K/10-Q). Se mitiga con las fuentes de la Sección 6.
- **No es asesoría financiera**: es un ejercicio académico; no considera costos completos de transacción de un bróker real salvo una comisión simulada.

---

## 5. Universo de datos (empresas)

Selección inicial de **5 empresas** balanceando IA disruptiva y sector tradicional/retail, para responder RQ4.

| Ticker | Empresa | Sector / Rol | Notas |
|--------|---------|--------------|-------|
| `NVDA` | NVIDIA | IA / Semiconductores | Núcleo del boom de IA. |
| `GOOGL` | Alphabet (Google) | IA / Tecnología | IA + publicidad. |
| `MSFT` | Microsoft | IA / Software | Inversor de OpenAI. |
| `AMZN` | Amazon | IA / Cloud + Retail | Cotiza en USD; exposición a IA (AWS, inversión en Anthropic) y a retail. Combina disrupción y consumo. |
| `NSRGY` | Nestlé (ADR) | Retail / Consumo defensivo | ADR OTC en USD (equivalente a `NESN.SW`). Sector tradicional de contraste. |

> **Escalabilidad a producción**: el universo está parametrizado en un archivo de configuración (`config/universe.yaml`). Añadir tickers no requiere cambiar código, solo la lista. Esto elimina el "limitante" de una lista fija cuando el sistema pase a producción.

**Benchmark**: `SPY` (ETF que replica el S&P 500).

---

## 6. Fuentes de datos (100% gratuitas)

| Tipo de dato | Fuente | Librería / Acceso | Notas |
|--------------|--------|-------------------|-------|
| Precios OHLCV | Yahoo Finance | `yfinance` | Diario e histórico, sin API key. |
| Fundamentales (ROIC, márgenes, crecimiento) | Yahoo Finance / SEC EDGAR | `yfinance`, `sec-edgar` (API pública) | EDGAR aporta 10-K/10-Q históricos oficiales sin costo. |
| Noticias | GDELT Project / Yahoo Finance news / Google News RSS | `gdeltdoc`, feeds RSS | GDELT es gratuito y excelente para eventos geopolíticos globales. |
| Sentimiento (modelo) | FinBERT | Hugging Face `transformers` | Modelo pre-entrenado en textos financieros; gratuito. |
| VIX (índice del miedo) | Yahoo Finance | `yfinance` (`^VIX`) | Volatilidad implícita del mercado. |
| Índice de riesgo geopolítico (GPR) | Caldara & Iacoviello (dataset académico) | Descarga CSV pública | Estándar académico, gratuito. |
| Commodities refugio | Yahoo Finance | `yfinance` (oro `GC=F`, petróleo `CL=F`) | Cobertura ante guerras/tensiones. |

Todas las fuentes son gratuitas y automatizables por API/descarga, alineado con la decisión de no depender de servicios de pago en esta fase.

---

## 7. Metodología de modelado (enfoque híbrido)

### 7.1 Modelo 1 — Cuantitativo (fundamental + técnico)
- **Planteamiento como clasificación** (más robusto que la regresión de precio exacto):
  > ¿El retorno acumulado a *h* días superará un umbral *τ*? (p. ej. +1.5% en 5 días → clase 1).
- **Algoritmos**: `XGBoost` / `LightGBM` (árboles con boosting: rápidos, robustos en datos tabulares, buen manejo de no linealidades).
- **Features**:
  - *Técnicos*: retornos históricos, medias móviles, RSI, MACD, volatilidad, volumen relativo.
  - *Fundamentales*: ROIC, margen operativo (OP), crecimiento de ingresos (OG), y ratios derivados.
- **Validación temporal**: *walk-forward / purged K-fold* para evitar *look-ahead bias*.

### 7.2 Modelo 2 — Sentimiento (NLP financiero)
- **Modelo**: `FinBERT` (BERT afinado en textos financieros).
- **Tarea**: clasificar cada noticia en Positivo / Negativo / Neutral.
- **Agregación**: *Score de Sentimiento Diario* por empresa, ponderado por relevancia/volumen de noticias.
- Se evita explícitamente el uso de diccionarios simples (VADER) para justificar el componente de IA.

### 7.3 Variables de contexto (shock geopolítico y macro)
- **GPR Index** (incertidumbre geopolítica), **VIX** (miedo), **oro** y **petróleo** (refugio).
- Alimentan tanto al modelo como a la **regla de gestión de riesgo** (migrar a caja/refugio cuando el riesgo se dispara).

### 7.4 Fusión
Los tres bloques convergen en una **regla de decisión** (Sección 8): señal cuantitativa + confirmación de sentimiento + filtro de riesgo macro.

---

## 8. Diseño experimental — Backtesting vs. S&P 500

### 8.1 Configuración de la simulación
- **Capital inicial**: $100,000 USD.
- **Benchmark (pasivo)**: comprar $100,000 de `SPY` el día 1 y mantener.
- **Estrategia activa (IA)**: en cada rebalanceo (semanal/mensual):
  - Si la predicción cuantitativa (XGBoost) es alcista **y** el sentimiento (FinBERT) es positivo por encima de un umbral → comprar/mantener.
  - Si el **VIX** o el **GPR** se disparan → migrar un porcentaje a caja o activos refugio (oro/bonos).
- **Motor**: `vectorbt` (rápido, vectorizado) o `backtrader`. Se simula una comisión por operación.

### 8.2 Escenarios de estrés (la prueba de fuego)

| Escenario | Período aprox. | Qué evalúa |
|-----------|----------------|------------|
| Pandemia COVID-19 | 2020–2021 | Reacción ante caídas sistémicas abruptas y recuperación por liquidez. |
| Boom de la IA | 2023–2025 | Rendimiento de empresas de IA (NVDA, MSFT, GOOGL) vs. tradicionales (NSRGY). |
| Tensiones geopolíticas | 2022 (Ucrania) / 2024–2026 (Medio Oriente) | Mitigación de pérdidas vía lectura de noticias y cobertura en commodities. |

### 8.3 Métricas de evaluación
- **Financieras**: Retorno acumulado, CAGR, **Sharpe Ratio**, Sortino, **Max Drawdown**, Calmar.
- **De clasificación (Modelo 1)**: Accuracy, Precision/Recall, F1, ROC-AUC.
- **De sentimiento (Modelo 2)**: Accuracy/F1 sobre etiquetas de referencia.
- **Significancia estadística**: comparación activa vs. benchmark con pruebas apropiadas (p. ej. bootstrap de la diferencia de Sharpe).

---

## 9. Arquitectura técnica (titulación + producción)

**Recomendación**: enfoque **mixto "research-to-production"**, que satisface tanto el rigor académico como la meta de llevarlo a producción.

```
                 [ Fuentes de datos gratuitas ]
        yfinance · SEC EDGAR · GDELT · GPR · VIX · commodities
                             │
                             ▼
   ┌───────────────────────────────────────────────────────┐
   │  CAPA DE DATOS (ingesta → limpieza → features)         │
   │  src/data/  ·  src/features/   (código .py versionado) │
   └───────────────────────────────┬───────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────┐
   │  CAPA DE MODELOS                                        │
   │  ┌─────────────────────────┐  ┌────────────────────┐   │
   │  │ Cuantitativo (LightGBM/ │  │ Sentimiento        │   │
   │  │ XGBoost) — src/models   │  │ (FinBERT) — src/nlp│   │
   │  └────────────┬────────────┘  └─────────┬──────────┘   │
   └───────────────┼─────────────────────────┼──────────────┘
                   └──────────────┬───────────┘
                                  ▼
              [ Regla de decisión + Backtesting (vectorbt) ]
                          src/strategy/  ·  src/backtest/
                                  │
                                  ▼
                 [ Dashboard Streamlit ]  ·  app/
```

### 9.1 Investigación / experimentación (para la tesis)
- **Jupyter Notebooks** en `notebooks/` para EDA, prototipado y figuras de la memoria (reproducibilidad y narrativa académica).
- **Google Colab (Pro)** opcional para entrenar FinBERT con GPU.

### 9.2 Producción (código mantenible)
- **Paquete Python** en `src/` con módulos desacoplados (datos, features, modelos, estrategia, backtest).
- **Configuración por archivos** (`config/*.yaml`) → sin *hardcodear* tickers, fechas ni hiperparámetros.
- **Git + estructura de repo** limpia; los notebooks *consumen* `src/`, no duplican lógica.
- **Dashboard Streamlit** (`app/`) para simular "tiempo real" con cierres diarios; desplegable gratis en Streamlit Community Cloud.
- **Automatización** (opcional producción): tarea programada (cron / GitHub Actions) que actualiza datos y recalcula señales al cierre de mercado.

> Regla de oro: **toda la lógica vive en `src/`**; notebooks, dashboard y jobs de producción solo la invocan. Esto permite que el mismo código sirva para la tesis y para producción sin reescribir nada.

---

## 10. Estructura de repositorio propuesta

```
Sentia-Trading/
├── config/
│   ├── universe.yaml          # tickers, benchmark, sectores
│   └── params.yaml            # fechas, umbrales, hiperparámetros
├── data/
│   ├── raw/                   # datos crudos descargados (no versionar)
│   └── processed/             # datasets de features (no versionar)
├── docs/
│   └── 01-metodologia.md      # este documento
├── notebooks/                 # EDA y experimentos (consumen src/)
├── src/
│   ├── data/                  # ingesta y limpieza
│   ├── features/              # ingeniería de variables
│   ├── models/                # XGBoost/LightGBM
│   ├── nlp/                   # FinBERT / sentimiento
│   ├── strategy/              # regla de decisión
│   └── backtest/              # motor y métricas
├── app/                       # dashboard Streamlit
├── tests/                     # pruebas unitarias
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 11. Plan de fases

| Fase | Entregable | Estado |
|------|-----------|--------|
| 0 | Documento metodológico (este) | En curso |
| 1 | Scaffolding del repo + config + `requirements.txt` | Pendiente |
| 2 | Pipeline de datos (OHLCV + fundamentales + macro) | Pendiente |
| 3 | Ingeniería de features técnicos y fundamentales | Pendiente |
| 4 | Modelo cuantitativo (XGBoost/LightGBM) + validación temporal | Pendiente |
| 5 | Módulo de sentimiento (FinBERT) + score diario | Pendiente |
| 6 | Regla de decisión + motor de backtesting vs. SPY | Pendiente |
| 7 | Evaluación por escenarios y sectores + métricas | Pendiente |
| 8 | Dashboard Streamlit + documentación final | Pendiente |

---

## 12. Consideraciones de reproducibilidad y ética
- **Reproducibilidad**: semillas fijas, versiones de librerías bloqueadas, datos crudos cacheados con fecha de descarga.
- **Sin look-ahead bias**: validación temporal estricta; los features de un día solo usan información disponible hasta ese cierre.
- **Transparencia**: se documentan supuestos, costos de transacción simulados y limitaciones de datos gratuitos.
- **Descargo**: proyecto académico; no constituye asesoría de inversión.

---

## 13. Referencias clave (a expandir en la memoria)
- Caldara, D. & Iacoviello, M. — *Geopolitical Risk (GPR) Index*.
- Araci, D. — *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
- Chen, T. & Guestrin, C. — *XGBoost: A Scalable Tree Boosting System*.
- López de Prado, M. — *Advances in Financial Machine Learning* (validación temporal, purged K-fold).
