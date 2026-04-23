from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from edge_sim.agents.gat_ppo import EdgeGATLayer


class SchedulerWorldModel(nn.Module):
    """Lightweight candidate-level WM-S regressor.

    The model predicts candidate costs for one request stage and one legal node.
    A planner can score all legal candidates and pick the node with the smallest
    predicted target cost.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class BatchedEdgeGATLayer(nn.Module):
    """Batch-friendly edge-aware GAT layer for fixed-size edge graphs."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.hidden_dim = hidden_dim

        self.node_lin = nn.Linear(node_dim, hidden_dim, bias=False)
        self.edge_lin = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(heads, self.head_dim))
        self.residual = nn.Linear(node_dim, hidden_dim, bias=False) if node_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_feat.shape
        src = edge_index[0].long()
        dst = edge_index[1].long()

        h = self.node_lin(node_feat).view(batch_size, num_nodes, self.heads, self.head_dim)
        e = self.edge_lin(edge_attr).view(batch_size, edge_attr.shape[1], self.heads, self.head_dim)
        h_src = h[:, src]
        h_dst = h[:, dst]
        logits = (
            (h_src * self.att_src.view(1, 1, self.heads, self.head_dim)).sum(dim=-1)
            + (h_dst * self.att_dst.view(1, 1, self.heads, self.head_dim)).sum(dim=-1)
            + (e * self.att_edge.view(1, 1, self.heads, self.head_dim)).sum(dim=-1)
        )
        logits = F.leaky_relu(logits, negative_slope=0.2)

        alpha = torch.zeros_like(logits)
        for node in range(num_nodes):
            mask = dst == node
            if torch.any(mask):
                alpha[:, mask, :] = torch.softmax(logits[:, mask, :], dim=1)

        messages = alpha.unsqueeze(-1) * (h_src + e)
        out = torch.zeros(
            batch_size,
            num_nodes,
            self.heads,
            self.head_dim,
            device=node_feat.device,
            dtype=node_feat.dtype,
        )
        for node in range(num_nodes):
            mask = dst == node
            if torch.any(mask):
                out[:, node] = messages[:, mask].sum(dim=1)

        out = out.reshape(batch_size, num_nodes, self.hidden_dim)
        out = self.norm(out + self.residual(node_feat))
        return F.elu(out)


class SchedulerGNNWorldModel(nn.Module):
    """Node-scoring GNN world model for one staged scheduling decision."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        req_dim: int = 5,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        request_dim: int = 32,
        output_dim: int = 3,
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
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + request_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
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
        return self.score_head(torch.cat([h, prev_h, candidate_edge_attr, req], dim=-1))


class SingleGraphSchedulerGNNWorldModel(nn.Module):
    """Single-sample wrapper used by lightweight planners when batching is not needed."""

    def __init__(
        self,
        node_dim: int = 8,
        edge_dim: int = 4,
        req_dim: int = 5,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        request_dim: int = 32,
        output_dim: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        gat_layers = []
        current_dim = node_dim
        for _ in range(layers):
            gat_layers.append(EdgeGATLayer(current_dim, edge_dim, hidden_dim, heads))
            current_dim = hidden_dim
        self.gnn = nn.ModuleList(gat_layers)
        self.request_encoder = nn.Sequential(
            nn.Linear(req_dim, request_dim),
            nn.SiLU(),
            nn.Linear(request_dim, request_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + request_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs_t: dict[str, torch.Tensor]) -> torch.Tensor:
        h = obs_t["node_feat"]
        for layer in self.gnn:
            h = layer(h, obs_t["edge_index"], obs_t["edge_attr"])
        req = self.request_encoder(obs_t["request_feat"])
        prev = int(obs_t["prev_node"].item())
        num_nodes = h.shape[0]
        prev_h = h[prev].unsqueeze(0).expand(num_nodes, -1)
        req_expand = req.unsqueeze(0).expand(num_nodes, -1)
        return self.score_head(torch.cat([h, prev_h, obs_t["candidate_edge_attr"], req_expand], dim=-1))
