from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GateDataset:
    features: np.ndarray
    labels: np.ndarray
    keep_costs: np.ndarray
    apply_costs: np.ndarray


class DeploymentGateNet(nn.Module):
    """Binary Agent-D gate: keep current deployment or apply candidate."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeploymentGateAgent:
    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DeploymentGateNet(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, dataset: GateDataset, epochs: int = 300) -> dict[str, float]:
        x = torch.as_tensor(dataset.features, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(dataset.labels, dtype=torch.long, device=self.device)
        last = {"loss": 0.0, "accuracy": 0.0}
        for _ in range(epochs):
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()
            last = {"loss": float(loss.item()), "accuracy": float(acc.item())}
        return last

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        logits = self.model(x)
        return logits.argmax(dim=-1).cpu().numpy()
