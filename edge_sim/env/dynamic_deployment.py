from __future__ import annotations

import numpy as np

from edge_sim.env.deployment import resource_usage
from edge_sim.env.edge_env import EdgeEnv


def deployment_change_count(old_x: np.ndarray | None, new_x: np.ndarray) -> float:
    if old_x is None:
        return float(new_x.sum())
    return float(np.abs(new_x - old_x).sum())


def compute_stage_weights(
    env: EdgeEnv,
    demand_probs: np.ndarray,
    alpha_compute: float = 1.0,
    alpha_data: float = 0.5,
) -> np.ndarray:
    """Demand-aware importance weight for every valid service stage."""

    compute_base = env.request_generator.compute_base
    input_data_base = env.request_generator.input_data_base
    stage_data_base = env.request_generator.stage_data_base

    valid_compute = []
    valid_data = []
    for i, stages in enumerate(env.service_stages):
        for j in range(stages):
            valid_compute.append(compute_base[i, j])
            valid_data.append(input_data_base[i] if j == 0 else stage_data_base[i, j - 1])
    c_scale = max(float(np.mean(valid_compute)), 1e-6)
    d_scale = max(float(np.mean(valid_data)), 1e-6)

    weights = np.zeros((env.num_services, env.max_stages), dtype=np.float32)
    for i, stages in enumerate(env.service_stages):
        for j in range(stages):
            d_in = input_data_base[i] if j == 0 else stage_data_base[i, j - 1]
            weights[i, j] = float(demand_probs[i]) * (
                alpha_compute * float(compute_base[i, j]) / c_scale + alpha_data * float(d_in) / d_scale
            )
    return weights


def target_replica_counts(
    env: EdgeEnv,
    stage_weights: np.ndarray,
    extra_budget: int,
    max_replicas_per_stage: int,
) -> np.ndarray:
    """Allocate extra replicas to the highest-weight stages."""

    counts = np.zeros((env.num_services, env.max_stages), dtype=np.int64)
    valid = [(i, j) for i, stages in enumerate(env.service_stages) for j in range(stages)]
    for i, j in valid:
        counts[i, j] = 1

    # Repeatedly assign one extra replica to the currently most valuable stage
    # that is still below its cap. Repetition allows very hot stages to receive
    # a third replica before lower-value stages receive a second one.
    for _ in range(extra_budget):
        candidates = [(stage_weights[i, j] / float(counts[i, j] + 1), i, j) for i, j in valid if counts[i, j] < max_replicas_per_stage]
        if not candidates:
            break
        _, i, j = max(candidates, key=lambda item: item[0])
        counts[i, j] += 1
    return counts


def _can_place(
    x: np.ndarray,
    env: EdgeEnv,
    i: int,
    j: int,
    m: int,
) -> bool:
    if x[i, j, m] > 0:
        return False
    storage_used, memory_used = resource_usage(x, env.service_storage, env.service_memory)
    return (
        storage_used[m] + env.service_storage[i, j] <= env.storage_cap[m]
        and memory_used[m] + env.service_memory[i, j] <= env.memory_cap[m]
    )


def _rank_nodes(
    env: EdgeEnv,
    x: np.ndarray,
    service_id: int | None = None,
    source_service_demand: np.ndarray | None = None,
    location_weight: float = 0.0,
) -> list[int]:
    storage_used, memory_used = resource_usage(x, env.service_storage, env.service_memory)
    deployed_count = x.sum(axis=(0, 1))
    storage_free = (env.storage_cap - storage_used) / np.maximum(env.storage_cap, 1e-8)
    memory_free = (env.memory_cap - memory_used) / np.maximum(env.memory_cap, 1e-8)
    score = storage_free + memory_free - 0.05 * deployed_count

    if source_service_demand is not None and service_id is not None and location_weight > 0:
        demand = np.asarray(source_service_demand[:, service_id], dtype=np.float32)
        if float(demand.sum()) > 0:
            demand = demand / demand.sum()
            proximity = np.eye(env.num_nodes, dtype=np.float32)
            bw = env.effective_bandwidth.copy()
            positive_bw = bw[bw > 0]
            bw_scale = max(float(positive_bw.mean()), 1e-6) if positive_bw.size else 1.0
            proximity += bw / bw_scale
            local_score = demand @ proximity
            local_score = local_score / max(float(local_score.max()), 1e-6)
            score += location_weight * local_score
    return list(np.argsort(-score))


def make_dynamic_deployment(
    env: EdgeEnv,
    demand_probs: np.ndarray,
    previous_x: np.ndarray | None,
    extra_budget: int,
    max_replicas_per_stage: int,
    alpha_compute: float = 1.0,
    alpha_data: float = 0.5,
    source_service_demand: np.ndarray | None = None,
    location_weight: float = 0.0,
    keep_previous: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Demand-aware slow-timescale deployment with capacity constraints.

    The total target replica count is base replicas plus ``extra_budget``.
    Existing replicas are kept first when possible, which gives the heuristic a
    simple inertia effect before V2-B introduces a learned deployment agent.
    """

    demand_probs = np.asarray(demand_probs, dtype=np.float32)
    demand_probs = demand_probs / max(float(demand_probs.sum()), 1e-8)
    weights = compute_stage_weights(env, demand_probs, alpha_compute, alpha_data)
    targets = target_replica_counts(env, weights, extra_budget, max_replicas_per_stage)
    x = np.zeros((env.num_services, env.max_stages, env.num_nodes), dtype=np.float32)

    valid_stages = [(i, j) for i, stages in enumerate(env.service_stages) for j in range(stages)]
    valid_stages.sort(key=lambda ij: float(weights[ij[0], ij[1]]), reverse=True)

    if keep_previous and previous_x is not None:
        for i, j in valid_stages:
            ranked = _rank_nodes(env, x, i, source_service_demand, location_weight)
            rank_pos = {node: pos for pos, node in enumerate(ranked)}
            previous_nodes = list(np.flatnonzero(previous_x[i, j] > 0.5))
            previous_nodes.sort(key=lambda node: rank_pos.get(int(node), env.num_nodes))
            for m in previous_nodes[: int(targets[i, j])]:
                if _can_place(x, env, i, j, int(m)):
                    x[i, j, int(m)] = 1.0

    for i, j in valid_stages:
        while int(x[i, j].sum()) < int(targets[i, j]):
            placed = False
            for m in _rank_nodes(env, x, i, source_service_demand, location_weight):
                if _can_place(x, env, i, j, int(m)):
                    x[i, j, int(m)] = 1.0
                    placed = True
                    break
            if not placed:
                break

    # Safety pass: every valid stage must have at least one replica.
    for i, j in valid_stages:
        if int(x[i, j].sum()) == 0:
            for m in _rank_nodes(env, x, i, source_service_demand, location_weight):
                if _can_place(x, env, i, j, int(m)):
                    x[i, j, int(m)] = 1.0
                    break
            if int(x[i, j].sum()) == 0:
                raise RuntimeError(f"Dynamic deployment cannot place required stage ({i},{j}).")

    info = {
        "replica_target_total": float(targets.sum()),
        "replica_actual_total": float(x.sum()),
        "change_count": deployment_change_count(previous_x, x),
    }
    for i in range(env.num_services):
        info[f"demand_prob_{i}"] = float(demand_probs[i])
    return x, info
