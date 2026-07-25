"""Clasificador secuencial LSTM (Fase 4-bis) en PyTorch.

Modelo de contraste frente al boosting: una red recurrente que procesa la
ventana de `lookback` días y predice la probabilidad de que el retorno futuro
supere el umbral. Sirve para responder, con rigor, si la complejidad de una
red secuencial aporta valor sobre un modelo tabular con historia limitada.

Usa aceleración MPS (Apple Silicon) o CUDA si están disponibles.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


def get_device() -> torch.device:
    """Selecciona el mejor dispositivo disponible (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMClassifier(nn.Module):
    """LSTM apilada + cabeza lineal para clasificación binaria."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, lookback, hidden)
        last = out[:, -1, :]           # estado del último paso temporal
        return self.head(last).squeeze(-1)


class LSTMTrainer:
    """Entrena y evalúa el `LSTMClassifier` con la config del proyecto."""

    def __init__(self, n_features: int, **overrides) -> None:
        cfg = dict(config.model_lstm_config())
        cfg.update(overrides)
        self.cfg = cfg
        self.device = get_device()
        self.model = LSTMClassifier(
            n_features=n_features,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMTrainer":
        torch.manual_seed(config.seed())
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(ds, batch_size=self.cfg.get("batch_size", 128), shuffle=True)

        # Ponderación de clases para el desbalance (pos_weight en BCE).
        pos_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("learning_rate", 1e-3))

        epochs = self.cfg.get("epochs", 30)
        self.model.train()
        for epoch in range(1, epochs + 1):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"    época {epoch:3d}/{epochs}  loss={total / len(ds):.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xb = torch.from_numpy(X).float().to(self.device)
        logits = self.model(xb)
        return torch.sigmoid(logits).cpu().numpy()
