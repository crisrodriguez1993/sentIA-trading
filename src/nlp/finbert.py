"""Análisis de sentimiento financiero con FinBERT.

FinBERT (`ProsusAI/finbert`) es un modelo BERT afinado en textos financieros
que clasifica cada texto en Positivo / Negativo / Neutral. A cada titular le
asignamos:
- `label`: clase predicha.
- `sentiment`: score continuo en [-1, 1] = P(positivo) - P(negativo).

Usa aceleración MPS (Apple Silicon) o CUDA si están disponibles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config

_LABELS = ["positive", "negative", "neutral"]  # orden de ProsusAI/finbert


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FinBERTScorer:
    """Envuelve FinBERT para puntuar el sentimiento de titulares financieros."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        cfg = config.model_sentiment_config()
        self.model_name = model_name or cfg.get("model_name", "ProsusAI/finbert")
        self.batch_size = batch_size or int(cfg.get("batch_size", 16))
        self.device = _get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        """Puntúa una lista de textos.

        Returns:
            DataFrame con columnas [p_positive, p_negative, p_neutral, label, sentiment].
        """
        if not texts:
            return pd.DataFrame(columns=["p_positive", "p_negative", "p_neutral", "label", "sentiment"])

        all_probs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [str(t) for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        probs = np.concatenate(all_probs, axis=0)
        df = pd.DataFrame(probs, columns=[f"p_{l}" for l in _LABELS])
        df["label"] = df[[f"p_{l}" for l in _LABELS]].to_numpy().argmax(axis=1)
        df["label"] = df["label"].map({i: l for i, l in enumerate(_LABELS)})
        df["sentiment"] = df["p_positive"] - df["p_negative"]
        return df


def score_news(news_df: pd.DataFrame, scorer: FinBERTScorer | None = None) -> pd.DataFrame:
    """Añade columnas de sentimiento a un DataFrame de noticias (columna 'title')."""
    if news_df.empty:
        return news_df.assign(sentiment=[], label=[])
    scorer = scorer or FinBERTScorer()
    scores = scorer.score_texts(news_df["title"].tolist())
    scores.index = news_df.index
    return pd.concat([news_df, scores], axis=1)
