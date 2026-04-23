from __future__ import annotations

import numpy as np

from edge_sim.env.edge_env import EdgeEnv
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load, kkt_load_cost


def _best_future_cost(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    request,
    next_stage: int,
    prev_node: int,
    gamma: np.ndarray,
    link_load: np.ndarray,
    depth: int,
) -> float:
    if depth <= 0 or next_stage >= request.num_stages:
        return 0.0

    legal = env.legal_nodes(deployment, request.service_id, next_stage, prev_node)
    legal_nodes = np.flatnonzero(legal)
    if legal_nodes.size == 0:
        return 1e9

    best = float("inf")
    for node in legal_nodes:
        gamma_after = gamma.copy()
        link_after = link_load.copy()
        add_stage_to_load(gamma_after, link_after, request, next_stage, prev_node, int(node))
        delta = allocator.incremental_cost(gamma, link_load, gamma_after, link_after)
        future = _best_future_cost(
            env,
            allocator,
            deployment,
            request,
            next_stage + 1,
            int(node),
            gamma_after,
            link_after,
            depth - 1,
        )
        best = min(best, float(delta + future))
    return best


def run_greedy_delta_slot(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    requests: list | None = None,
    slow_epoch: int | None = None,
) -> dict[str, float]:
    """Greedy baseline: choose the legal node with the smallest KKT load increment."""

    requests = requests if requests is not None else env.sample_requests(slow_epoch=slow_epoch)
    gamma = np.zeros(env.num_nodes, dtype=np.float32)
    link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
    schedules: dict[int, list[int]] = {}

    for req in requests:
        path: list[int] = []
        prev = req.source_node
        for stage_idx in range(req.num_stages):
            legal = env.legal_nodes(deployment, req.service_id, stage_idx, prev)
            legal_nodes = np.flatnonzero(legal)
            if legal_nodes.size == 0:
                raise RuntimeError(
                    f"No legal node for request={req.request_id}, service={req.service_id}, stage={stage_idx}."
                )

            best_node = int(legal_nodes[0])
            best_delta = float("inf")
            for node in legal_nodes:
                gamma_after = gamma.copy()
                link_after = link_load.copy()
                add_stage_to_load(gamma_after, link_after, req, stage_idx, prev, int(node))
                delta = allocator.incremental_cost(gamma, link_load, gamma_after, link_after)
                if delta < best_delta:
                    best_delta = delta
                    best_node = int(node)

            add_stage_to_load(gamma, link_load, req, stage_idx, prev, best_node)
            path.append(best_node)
            prev = best_node

        schedules[req.request_id] = path

    allocation = allocator.allocate(requests, schedules)
    metrics = {
        "requests": float(len(requests)),
        "stages": float(sum(len(path) for path in schedules.values())),
        "total_delay": allocation.total_delay,
        "compute_delay": allocation.compute_delay,
        "transmission_delay": allocation.transmission_delay,
        "kkt_virtual_cost": kkt_load_cost(gamma, link_load, env.compute_cap, env.effective_bandwidth),
        "infeasible": float(allocation.infeasible),
    }
    for i in range(env.num_services):
        metrics[f"service_{i}"] = float(sum(1 for req in requests if req.service_id == i))
    return metrics


def run_lookahead_delta_slot(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    requests: list | None = None,
    slow_epoch: int | None = None,
    lookahead_depth: int = 2,
    future_weight: float = 0.8,
) -> dict[str, float]:
    """Choose nodes by current KKT increment plus a short staged lookahead."""

    requests = requests if requests is not None else env.sample_requests(slow_epoch=slow_epoch)
    gamma = np.zeros(env.num_nodes, dtype=np.float32)
    link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
    schedules: dict[int, list[int]] = {}

    for req in requests:
        path: list[int] = []
        prev = req.source_node
        for stage_idx in range(req.num_stages):
            legal = env.legal_nodes(deployment, req.service_id, stage_idx, prev)
            legal_nodes = np.flatnonzero(legal)
            if legal_nodes.size == 0:
                raise RuntimeError(
                    f"No legal node for request={req.request_id}, service={req.service_id}, stage={stage_idx}."
                )

            best_node = int(legal_nodes[0])
            best_score = float("inf")
            for node in legal_nodes:
                gamma_after = gamma.copy()
                link_after = link_load.copy()
                add_stage_to_load(gamma_after, link_after, req, stage_idx, prev, int(node))
                delta = allocator.incremental_cost(gamma, link_load, gamma_after, link_after)
                future = _best_future_cost(
                    env,
                    allocator,
                    deployment,
                    req,
                    stage_idx + 1,
                    int(node),
                    gamma_after,
                    link_after,
                    lookahead_depth,
                )
                score = float(delta + future_weight * future)
                if score < best_score:
                    best_score = score
                    best_node = int(node)

            add_stage_to_load(gamma, link_load, req, stage_idx, prev, best_node)
            path.append(best_node)
            prev = best_node

        schedules[req.request_id] = path

    allocation = allocator.allocate(requests, schedules)
    metrics = {
        "requests": float(len(requests)),
        "stages": float(sum(len(path) for path in schedules.values())),
        "total_delay": allocation.total_delay,
        "compute_delay": allocation.compute_delay,
        "transmission_delay": allocation.transmission_delay,
        "kkt_virtual_cost": kkt_load_cost(gamma, link_load, env.compute_cap, env.effective_bandwidth),
        "infeasible": float(allocation.infeasible),
    }
    for i in range(env.num_services):
        metrics[f"service_{i}"] = float(sum(1 for req in requests if req.service_id == i))
    return metrics
