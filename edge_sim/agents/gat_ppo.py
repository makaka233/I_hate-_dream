from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


@dataclass
class StageTransition:
    obs: dict
    action: int
    reward: float
    log_prob: float
    value: float


class EdgeGATLayer(nn.Module):
    """Small edge-aware GAT layer without torch-geometric dependency."""

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
        num_nodes = node_feat.shape[0]
        src = edge_index[0].long()
        dst = edge_index[1].long()

        h = self.node_lin(node_feat).view(num_nodes, self.heads, self.head_dim)
        e = self.edge_lin(edge_attr).view(edge_attr.shape[0], self.heads, self.head_dim)

        h_src = h[src]
        h_dst = h[dst]
        logits = (
            (h_src * self.att_src.unsqueeze(0)).sum(dim=-1)
            + (h_dst * self.att_dst.unsqueeze(0)).sum(dim=-1)
            + (e * self.att_edge.unsqueeze(0)).sum(dim=-1)
        )
        logits = F.leaky_relu(logits, negative_slope=0.2)

        alpha = torch.zeros_like(logits)
        for head in range(self.heads):
            for node in range(num_nodes):
                mask = dst == node
                if torch.any(mask):
                    alpha[mask, head] = torch.softmax(logits[mask, head], dim=0)

        messages = alpha.unsqueeze(-1) * (h_src + e)
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=node_feat.device, dtype=node_feat.dtype)
        for head in range(self.heads):
            out[:, head, :].index_add_(0, dst, messages[:, head, :])

        out = out.reshape(num_nodes, self.hidden_dim)
        out = self.norm(out + self.residual(node_feat))
        return F.elu(out)


class EdgeGATPolicy(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        req_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        request_dim: int = 32,
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
            nn.ReLU(),
            nn.Linear(request_dim, request_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + request_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + request_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, obs_t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = obs_t["node_feat"]
        for layer in self.gnn:
            h = layer(h, obs_t["edge_index"], obs_t["edge_attr"])
        req = self.request_encoder(obs_t["request_feat"])
        return h, req

    def forward(self, obs_t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, req = self.encode(obs_t)
        prev = int(obs_t["prev_node"].item())
        num_nodes = h.shape[0]
        prev_h = h[prev].unsqueeze(0).expand(num_nodes, -1)
        req_expand = req.unsqueeze(0).expand(num_nodes, -1)
        logits = self.action_head(torch.cat([h, prev_h, obs_t["candidate_edge_attr"], req_expand], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~obs_t["legal_mask"].bool(), -1e9)

        graph_h = h.mean(dim=0)
        value = self.value_head(torch.cat([graph_h, h[prev], req], dim=-1)).squeeze(-1)
        return logits, value


class MaskedPPOAgent:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        requested_device = device or cfg["training"].get("device", "cpu")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        self.policy = EdgeGATPolicy(
            node_dim=8,
            edge_dim=4,
            req_dim=5,
            hidden_dim=int(model_cfg["hidden_dim"]),
            heads=int(model_cfg["heads"]),
            layers=int(model_cfg["gat_layers"]),
            request_dim=int(model_cfg["request_dim"]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(train_cfg["learning_rate"]))
        self.clip_ratio = float(train_cfg["clip_ratio"])
        self.ppo_epochs = int(train_cfg["ppo_epochs"])
        self.minibatch_size = int(train_cfg["minibatch_size"])
        self.entropy_coef = float(train_cfg["entropy_coef"])
        self.value_coef = float(train_cfg["value_coef"])
        self.max_grad_norm = float(train_cfg["max_grad_norm"])
        self.max_update_samples = int(train_cfg.get("max_update_samples", 0))

    def _tensorize(self, obs: dict) -> dict[str, torch.Tensor]:
        return {
            "node_feat": torch.as_tensor(obs["node_feat"], dtype=torch.float32, device=self.device),
            "edge_index": torch.as_tensor(obs["edge_index"], dtype=torch.long, device=self.device),
            "edge_attr": torch.as_tensor(obs["edge_attr"], dtype=torch.float32, device=self.device),
            "candidate_edge_attr": torch.as_tensor(obs["candidate_edge_attr"], dtype=torch.float32, device=self.device),
            "request_feat": torch.as_tensor(obs["request_feat"], dtype=torch.float32, device=self.device),
            "prev_node": torch.as_tensor(obs["prev_node"], dtype=torch.long, device=self.device),
            "legal_mask": torch.as_tensor(obs["legal_mask"], dtype=torch.bool, device=self.device),
        }

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> tuple[int, float, float]:
        obs_t = self._tensorize(obs)
        logits, value = self.policy(obs_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.as_tensor(action, device=self.device)).item())
        return action, log_prob, float(value.item())

    def _evaluate_many(self, transitions: Sequence[StageTransition], indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, values, entropies = [], [], []
        for idx in indices:
            tr = transitions[int(idx)]
            logits, value = self.policy(self._tensorize(tr.obs))
            dist = Categorical(logits=logits)
            action = torch.as_tensor(tr.action, dtype=torch.long, device=self.device)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())
        return torch.stack(log_probs), torch.stack(values).squeeze(-1), torch.stack(entropies)

    def update(self, transitions: list[StageTransition]) -> dict[str, float]:
        if not transitions:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        if self.max_update_samples > 0 and len(transitions) > self.max_update_samples:
            chosen = np.random.choice(len(transitions), size=self.max_update_samples, replace=False)
            transitions = [transitions[int(i)] for i in chosen]

        rewards = torch.as_tensor([tr.reward for tr in transitions], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor([tr.value for tr in transitions], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor([tr.log_prob for tr in transitions], dtype=torch.float32, device=self.device)

        returns = rewards
        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_items = len(transitions)
        last_stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        all_indices = np.arange(num_items)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(all_indices)
            for start in range(0, num_items, self.minibatch_size):
                batch_idx = all_indices[start : start + self.minibatch_size]
                log_probs, values, entropies = self._evaluate_many(transitions, batch_idx)
                batch_old_log_probs = old_log_probs[torch.as_tensor(batch_idx, device=self.device)]
                batch_adv = advantages[torch.as_tensor(batch_idx, device=self.device)]
                batch_returns = returns[torch.as_tensor(batch_idx, device=self.device)]

                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.min(ratio * batch_adv, clipped * batch_adv).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy = entropies.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

        return last_stats
