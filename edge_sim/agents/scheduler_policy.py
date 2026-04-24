from __future__ import annotations

import torch
from torch import nn

from edge_sim.agents.scheduler_world_model import BatchedEdgeGATLayer


class SchedulerGNNPolicy(nn.Module):
    """Node-scoring student policy distilled from the WM-S teacher."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        req_dim: int = 5,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        request_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        gat_layers = []
        current_dim = node_dim
        for _ in range(layers):
            gat_layers.append(BatchedEdgeGATLayer(current_dim, edge_dim, hidden_dim, heads))
            current_dim = hidden_dim
        self.gnn = nn.ModuleList(gat_layers)
        self.request_encoder = nn.Sequential(
            nn.Linear(req_dim, request_dim),
            nn.SiLU(),
            nn.Linear(request_dim, request_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + request_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        candidate_edge_attr: torch.Tensor,
        request_feat: torch.Tensor,
        prev_node: torch.Tensor,
    ) -> torch.Tensor:
        h = node_feat
        for layer in self.gnn:
            h = layer(h, edge_index, edge_attr)

        batch_size, num_nodes, _ = h.shape
        batch_idx = torch.arange(batch_size, device=h.device)
        prev_h = h[batch_idx, prev_node.long()].unsqueeze(1).expand(batch_size, num_nodes, -1)
        req = self.request_encoder(request_feat).unsqueeze(1).expand(batch_size, num_nodes, -1)
        logits = self.policy_head(torch.cat([h, prev_h, candidate_edge_attr, req], dim=-1)).squeeze(-1)
        return logits
