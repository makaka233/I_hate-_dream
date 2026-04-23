from __future__ import annotations

import numpy as np

from edge_sim.env.deployment import deployment_summary, resource_usage
from edge_sim.env.dynamic_deployment import (
    compute_stage_weights,
    deployment_change_count,
    make_dynamic_deployment,
)
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.training.simulate_v2 import global_share_from_source_demand, predict_trend_demand


CANDIDATE_SPECS = [
    {"name": "keep_previous", "mode": "keep"},
    {"name": "history_keep", "demand_mode": "history", "keep_previous": True, "extra_scale": 1.0, "location_scale": 1.0},
    {"name": "history_refresh", "demand_mode": "history", "keep_previous": False, "extra_scale": 1.0, "location_scale": 1.0},
    {"name": "trend_keep", "demand_mode": "trend", "keep_previous": True, "extra_scale": 1.0, "location_scale": 1.0},
    {"name": "trend_refresh", "demand_mode": "trend", "keep_previous": False, "extra_scale": 1.0, "location_scale": 1.0},
    {"name": "history_aggressive_refresh", "demand_mode": "history", "keep_previous": False, "extra_scale": 1.35, "location_scale": 1.3},
    {"name": "trend_aggressive_refresh", "demand_mode": "trend", "keep_previous": False, "extra_scale": 1.35, "location_scale": 1.3},
    {"name": "history_compact_keep", "demand_mode": "history", "keep_previous": True, "extra_scale": 0.65, "location_scale": 0.75},
    {"name": "trend_compact_keep", "demand_mode": "trend", "keep_previous": True, "extra_scale": 0.65, "location_scale": 0.75},
]
CANDIDATE_NAMES = [spec["name"] for spec in CANDIDATE_SPECS]


def valid_stage_pairs(env: EdgeEnv) -> list[tuple[int, int]]:
    return [(i, j) for i, stages in enumerate(env.service_stages) for j in range(stages)]


def default_previous_observed_matrix(env: EdgeEnv) -> np.ndarray:
    return np.full((env.num_nodes, env.num_services), 1.0 / env.num_services, dtype=np.float32)


def _proximity_matrix(env: EdgeEnv) -> np.ndarray:
    proximity = np.eye(env.num_nodes, dtype=np.float32)
    bw = env.effective_bandwidth.astype(np.float32)
    positive = bw[bw > 0]
    if positive.size:
        proximity += bw / max(float(positive.mean()), 1e-6)
    return proximity


def _stage_source_locality(
    source_service_demand: np.ndarray,
    deployed_nodes: np.ndarray,
    proximity: np.ndarray,
) -> float:
    if deployed_nodes.size == 0:
        return 0.0
    demand = np.asarray(source_service_demand, dtype=np.float32)
    total = float(demand.sum())
    if total <= 0:
        return 0.0
    demand = demand / total
    best = proximity[:, deployed_nodes].max(axis=1)
    return float((demand * best).sum() / max(float(best.max()), 1e-6))


def _stage_chain_score(nodes_a: np.ndarray, nodes_b: np.ndarray, proximity: np.ndarray) -> float:
    if nodes_a.size == 0 or nodes_b.size == 0:
        return 0.0
    best = proximity[np.ix_(nodes_a, nodes_b)].max(axis=1)
    return float(best.mean() / max(float(best.max()), 1e-6))


def _scaled_budget(base_budget: int, scale: float) -> int:
    return max(0, int(round(float(base_budget) * float(scale))))


def _build_dynamic_candidate(
    env: EdgeEnv,
    demand_probs: np.ndarray,
    source_service_demand: np.ndarray,
    previous_x: np.ndarray | None,
    dyn_cfg: dict,
    keep_previous: bool,
    extra_scale: float,
    location_scale: float,
) -> np.ndarray:
    return make_dynamic_deployment(
        env,
        demand_probs,
        previous_x if keep_previous else None,
        _scaled_budget(int(dyn_cfg.get("extra_replica_budget", 13)), extra_scale),
        int(dyn_cfg.get("max_replicas_per_stage", 3)),
        float(dyn_cfg.get("alpha_compute", 1.0)),
        float(dyn_cfg.get("alpha_data", 0.5)),
        source_service_demand,
        float(dyn_cfg.get("location_weight", 1.0)) * float(location_scale),
        keep_previous,
    )[0]


def candidate_pool(
    env: EdgeEnv,
    previous_x: np.ndarray | None,
    previous_observed_matrix: np.ndarray | None,
    older_observed_matrix: np.ndarray | None,
    dyn_cfg: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    previous_observed_matrix = (
        np.asarray(previous_observed_matrix, dtype=np.float32)
        if previous_observed_matrix is not None
        else default_previous_observed_matrix(env)
    )
    trend_source = predict_trend_demand(
        previous_observed_matrix,
        older_observed_matrix,
        float(dyn_cfg.get("history_trend_beta", 0.5)),
    ).astype(np.float32)

    history_demand = global_share_from_source_demand(previous_observed_matrix)
    trend_demand = global_share_from_source_demand(trend_source)
    proximity = _proximity_matrix(env)
    history_stage_weights = compute_stage_weights(
        env,
        history_demand,
        float(dyn_cfg.get("alpha_compute", 1.0)),
        float(dyn_cfg.get("alpha_data", 0.5)),
    )
    trend_stage_weights = compute_stage_weights(
        env,
        trend_demand,
        float(dyn_cfg.get("alpha_compute", 1.0)),
        float(dyn_cfg.get("alpha_data", 0.5)),
    )

    pool: dict[str, np.ndarray] = {}
    keep_previous = previous_x.copy() if previous_x is not None else env.make_deployment("heuristic")
    pool["keep_previous"] = keep_previous
    for spec in CANDIDATE_SPECS:
        if spec["name"] == "keep_previous":
            continue
        if spec["demand_mode"] == "history":
            demand_probs = history_demand
            source_service_demand = previous_observed_matrix
        else:
            demand_probs = trend_demand
            source_service_demand = trend_source
        pool[spec["name"]] = _build_dynamic_candidate(
            env,
            demand_probs,
            source_service_demand,
            previous_x,
            dyn_cfg,
            bool(spec["keep_previous"]),
            float(spec["extra_scale"]),
            float(spec["location_scale"]),
        )

    state = {
        "previous_observed_matrix": previous_observed_matrix.astype(np.float32),
        "trend_source_matrix": trend_source.astype(np.float32),
        "history_demand": history_demand.astype(np.float32),
        "trend_demand": trend_demand.astype(np.float32),
        "history_stage_weights": history_stage_weights.astype(np.float32),
        "trend_stage_weights": trend_stage_weights.astype(np.float32),
        "proximity": proximity.astype(np.float32),
    }
    return pool, state


def feature_names(env: EdgeEnv) -> list[str]:
    names: list[str] = []
    for i in range(env.num_services):
        names.append(f"history_demand_{i}")
    for i in range(env.num_services):
        names.append(f"trend_demand_{i}")
    names.extend(["history_entropy", "trend_entropy"])
    for name in CANDIDATE_NAMES:
        names.append(f"candidate_type_{name}")
    for i, j in valid_stage_pairs(env):
        names.extend(
            [
                f"stage_hist_weight_s{i}_t{j}",
                f"stage_trend_weight_s{i}_t{j}",
                f"stage_prev_repl_s{i}_t{j}",
                f"stage_cand_repl_s{i}_t{j}",
                f"stage_delta_repl_s{i}_t{j}",
                f"stage_change_s{i}_t{j}",
                f"stage_prev_hist_loc_s{i}_t{j}",
                f"stage_cand_hist_loc_s{i}_t{j}",
                f"stage_prev_trend_loc_s{i}_t{j}",
                f"stage_cand_trend_loc_s{i}_t{j}",
                f"stage_prev_next_chain_s{i}_t{j}",
                f"stage_cand_next_chain_s{i}_t{j}",
            ]
        )
    for m in range(env.num_nodes):
        names.extend(
            [
                f"node_prev_repl_{m}",
                f"node_cand_repl_{m}",
                f"node_delta_repl_{m}",
                f"node_storage_util_{m}",
                f"node_memory_util_{m}",
                f"node_compute_cap_{m}",
                f"node_history_demand_{m}",
                f"node_trend_demand_{m}",
            ]
        )
    for key in [
        "total_replicas",
        "min_stage_replicas",
        "max_stage_replicas",
        "avg_node_replicas",
        "max_node_replicas",
        "avg_storage_util",
        "max_storage_util",
        "avg_memory_util",
        "max_memory_util",
    ]:
        names.append(f"summary_{key}")
    for key in ["change_count", "add_count", "remove_count"]:
        names.append(key)
    return names


def encode_candidate_features(
    env: EdgeEnv,
    previous_x: np.ndarray | None,
    candidate_x: np.ndarray,
    candidate_name: str,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    previous_x = previous_x if previous_x is not None else np.zeros_like(candidate_x, dtype=np.float32)
    history_demand = state["history_demand"].astype(np.float32)
    trend_demand = state["trend_demand"].astype(np.float32)
    history_matrix = state["previous_observed_matrix"].astype(np.float32)
    trend_matrix = state["trend_source_matrix"].astype(np.float32)
    history_stage_weights = state["history_stage_weights"].astype(np.float32)
    trend_stage_weights = state["trend_stage_weights"].astype(np.float32)
    proximity = state["proximity"].astype(np.float32)
    candidate_type = np.zeros(len(CANDIDATE_NAMES), dtype=np.float32)
    candidate_type[CANDIDATE_NAMES.index(candidate_name)] = 1.0
    stage_features: list[float] = []
    for i, j in valid_stage_pairs(env):
        prev_nodes = np.flatnonzero(previous_x[i, j] > 0.5)
        cand_nodes = np.flatnonzero(candidate_x[i, j] > 0.5)
        prev_repl = float(prev_nodes.size) / float(env.num_nodes)
        cand_repl = float(cand_nodes.size) / float(env.num_nodes)
        stage_change = float(np.abs(candidate_x[i, j] - previous_x[i, j]).sum()) / float(env.num_nodes)
        next_prev_chain = 0.0
        next_cand_chain = 0.0
        if j + 1 < env.service_stages[i]:
            next_prev_nodes = np.flatnonzero(previous_x[i, j + 1] > 0.5)
            next_cand_nodes = np.flatnonzero(candidate_x[i, j + 1] > 0.5)
            next_prev_chain = _stage_chain_score(prev_nodes, next_prev_nodes, proximity)
            next_cand_chain = _stage_chain_score(cand_nodes, next_cand_nodes, proximity)
        stage_features.extend(
            [
                float(history_stage_weights[i, j]),
                float(trend_stage_weights[i, j]),
                prev_repl,
                cand_repl,
                cand_repl - prev_repl,
                stage_change,
                _stage_source_locality(history_matrix[:, i], prev_nodes, proximity),
                _stage_source_locality(history_matrix[:, i], cand_nodes, proximity),
                _stage_source_locality(trend_matrix[:, i], prev_nodes, proximity),
                _stage_source_locality(trend_matrix[:, i], cand_nodes, proximity),
                next_prev_chain,
                next_cand_chain,
            ]
        )

    storage_used, memory_used = resource_usage(candidate_x, env.service_storage, env.service_memory)
    previous_node_replica = previous_x.sum(axis=(0, 1)).astype(np.float32) / float(max(len(valid_stage_pairs(env)), 1))
    candidate_node_replica = candidate_x.sum(axis=(0, 1)).astype(np.float32) / float(max(len(valid_stage_pairs(env)), 1))
    history_node_demand = history_matrix.sum(axis=1).astype(np.float32)
    trend_node_demand = trend_matrix.sum(axis=1).astype(np.float32)
    history_node_demand = history_node_demand / max(float(history_node_demand.max()), 1e-6)
    trend_node_demand = trend_node_demand / max(float(trend_node_demand.max()), 1e-6)
    node_features: list[float] = []
    for m in range(env.num_nodes):
        node_features.extend(
            [
                float(previous_node_replica[m]),
                float(candidate_node_replica[m]),
                float(candidate_node_replica[m] - previous_node_replica[m]),
                float(storage_used[m] / max(float(env.storage_cap[m]), 1e-8)),
                float(memory_used[m] / max(float(env.memory_cap[m]), 1e-8)),
                float(env.compute_cap[m] / max(float(env.compute_cap.max()), 1e-8)),
                float(history_node_demand[m]),
                float(trend_node_demand[m]),
            ]
        )

    summary = deployment_summary(
        candidate_x,
        env.service_stages,
        env.service_storage,
        env.service_memory,
        env.storage_cap,
        env.memory_cap,
    )
    summary_vec = np.array(
        [
            summary["total_replicas"],
            summary["min_stage_replicas"],
            summary["max_stage_replicas"],
            summary["avg_node_replicas"],
            summary["max_node_replicas"],
            summary["avg_storage_util"],
            summary["max_storage_util"],
            summary["avg_memory_util"],
            summary["max_memory_util"],
        ],
        dtype=np.float32,
    )
    change_count = deployment_change_count(previous_x, candidate_x)
    add_count = float(np.maximum(candidate_x - previous_x, 0.0).sum())
    remove_count = float(np.maximum(previous_x - candidate_x, 0.0).sum())
    extras = np.array([change_count, add_count, remove_count], dtype=np.float32)
    entropies = np.array(
        [
            float(-(history_demand * np.log(history_demand + 1e-8)).sum()),
            float(-(trend_demand * np.log(trend_demand + 1e-8)).sum()),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            history_demand,
            trend_demand,
            entropies,
            candidate_type,
            np.asarray(stage_features, dtype=np.float32),
            np.asarray(node_features, dtype=np.float32),
            summary_vec,
            extras,
        ],
        axis=0,
    ).astype(np.float32)
