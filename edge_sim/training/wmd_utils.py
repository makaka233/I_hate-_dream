from __future__ import annotations

from pathlib import Path

import numpy as np

from edge_sim.env.deployment import deployment_summary
from edge_sim.env.dynamic_deployment import deployment_change_count, make_dynamic_deployment
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.training.simulate_v2 import global_share_from_source_demand, predict_trend_demand


CANDIDATE_NAMES = [
    "keep_previous",
    "history_keep",
    "history_refresh",
    "trend_keep",
    "trend_refresh",
]


def valid_stage_pairs(env: EdgeEnv) -> list[tuple[int, int]]:
    return [(i, j) for i, stages in enumerate(env.service_stages) for j in range(stages)]


def flatten_valid_deployment(env: EdgeEnv, x: np.ndarray) -> np.ndarray:
    parts = [x[i, j].astype(np.float32) for i, j in valid_stage_pairs(env)]
    return np.concatenate(parts, axis=0).astype(np.float32)


def default_previous_observed_matrix(env: EdgeEnv) -> np.ndarray:
    return np.full((env.num_nodes, env.num_services), 1.0 / env.num_services, dtype=np.float32)


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

    keep_previous = previous_x.copy() if previous_x is not None else env.make_deployment("heuristic")
    common_args = (
        int(dyn_cfg.get("extra_replica_budget", 13)),
        int(dyn_cfg.get("max_replicas_per_stage", 3)),
        float(dyn_cfg.get("alpha_compute", 1.0)),
        float(dyn_cfg.get("alpha_data", 0.5)),
        float(dyn_cfg.get("location_weight", 1.0)),
    )
    history_keep, _ = make_dynamic_deployment(
        env,
        history_demand,
        previous_x,
        common_args[0],
        common_args[1],
        common_args[2],
        common_args[3],
        previous_observed_matrix,
        common_args[4],
        True,
    )
    history_refresh, _ = make_dynamic_deployment(
        env,
        history_demand,
        None,
        common_args[0],
        common_args[1],
        common_args[2],
        common_args[3],
        previous_observed_matrix,
        common_args[4],
        False,
    )
    trend_keep, _ = make_dynamic_deployment(
        env,
        trend_demand,
        previous_x,
        common_args[0],
        common_args[1],
        common_args[2],
        common_args[3],
        trend_source,
        common_args[4],
        True,
    )
    trend_refresh, _ = make_dynamic_deployment(
        env,
        trend_demand,
        None,
        common_args[0],
        common_args[1],
        common_args[2],
        common_args[3],
        trend_source,
        common_args[4],
        False,
    )

    pool = {
        "keep_previous": keep_previous,
        "history_keep": history_keep,
        "history_refresh": history_refresh,
        "trend_keep": trend_keep,
        "trend_refresh": trend_refresh,
    }
    state = {
        "previous_observed_matrix": previous_observed_matrix.astype(np.float32),
        "trend_source_matrix": trend_source.astype(np.float32),
        "history_demand": history_demand.astype(np.float32),
        "trend_demand": trend_demand.astype(np.float32),
    }
    return pool, state


def feature_names(env: EdgeEnv) -> list[str]:
    names: list[str] = []
    for prefix in ["previous", "candidate", "delta"]:
        for i, j in valid_stage_pairs(env):
            for m in range(env.num_nodes):
                names.append(f"{prefix}_x_s{i}_t{j}_n{m}")
    for m in range(env.num_nodes):
        for i in range(env.num_services):
            names.append(f"history_src_n{m}_svc{i}")
    for m in range(env.num_nodes):
        for i in range(env.num_services):
            names.append(f"trend_src_n{m}_svc{i}")
    for i in range(env.num_services):
        names.append(f"history_demand_{i}")
    for i in range(env.num_services):
        names.append(f"trend_demand_{i}")
    for name in CANDIDATE_NAMES:
        names.append(f"candidate_type_{name}")
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
    for i, j in valid_stage_pairs(env):
        names.append(f"stage_replica_count_s{i}_t{j}")
    for m in range(env.num_nodes):
        names.append(f"node_replica_count_{m}")
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
    previous_flat = flatten_valid_deployment(env, previous_x)
    candidate_flat = flatten_valid_deployment(env, candidate_x)
    delta_flat = candidate_flat - previous_flat
    history_matrix = state["previous_observed_matrix"].astype(np.float32).reshape(-1)
    trend_matrix = state["trend_source_matrix"].astype(np.float32).reshape(-1)
    history_demand = state["history_demand"].astype(np.float32)
    trend_demand = state["trend_demand"].astype(np.float32)
    candidate_type = np.zeros(len(CANDIDATE_NAMES), dtype=np.float32)
    candidate_type[CANDIDATE_NAMES.index(candidate_name)] = 1.0
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
    stage_replica_counts = np.array([candidate_x[i, j].sum() for i, j in valid_stage_pairs(env)], dtype=np.float32)
    node_replica_counts = candidate_x.sum(axis=(0, 1)).astype(np.float32)
    change_count = deployment_change_count(previous_x, candidate_x)
    add_count = float(np.maximum(candidate_x - previous_x, 0.0).sum())
    remove_count = float(np.maximum(previous_x - candidate_x, 0.0).sum())
    extras = np.array([change_count, add_count, remove_count], dtype=np.float32)

    return np.concatenate(
        [
            previous_flat,
            candidate_flat,
            delta_flat,
            history_matrix,
            trend_matrix,
            history_demand,
            trend_demand,
            candidate_type,
            summary_vec,
            stage_replica_counts,
            node_replica_counts,
            extras,
        ],
        axis=0,
    ).astype(np.float32)
