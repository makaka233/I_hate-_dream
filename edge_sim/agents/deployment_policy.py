from __future__ import annotations

import torch
from torch import nn


class DeploymentCandidatePolicy(nn.Module):
    """Candidate-level slow-timescale policy for selecting a deployment option."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        reduced_dim = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
