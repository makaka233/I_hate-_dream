from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from edge_sim.env.deployment import resource_usage
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.policies import _best_future_cost
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load


META_COLUMNS = [
    "sample_id",
    "decision_id",
    "slow_epoch",
    "fast_slot",
    "request_id",
    "user_id",
    "source_node",
    "service_id",
    "stage_idx",
    "prev_node",
    "candidate_node",
    "candidate_count",
]

FEATURE_COLUMNS = [
    "service_id_norm",
    "stage_idx_norm",
    "source_node_norm",
    "prev_node_norm",
    "candidate_node_norm",
    "same_prev_candidate",
    "same_source_candidate",
    "req_compute_norm",
    "req_data_in_norm",
    "remaining_compute_norm",
    "remaining_data_norm",
    "remaining_stage_ratio",
    "candidate_compute_cap_norm",
    "candidate_gamma_util",
    "candidate_gamma_after_util",
    "mean_gamma_util",
    "max_gamma_util",
    "total_gamma_norm",
    "candidate_storage_util",
    "candidate_memory_util",
    "current_stage_replica_ratio",
    "next_stage_replica_ratio",
    "candidate_next_stage_deployed",
    "legal_candidate_ratio",
    "prev_candidate_bw_norm",
    "prev_candidate_link_load_norm",
    "prev_candidate_link_util",
    "prev_candidate_link_after_util",
    "source_candidate_bw_norm",
    "mean_link_util",
    "max_link_util",
    "total_link_load_norm",
    "service_global_prob",
    "source_arrival_rate_norm",
    "source_service_prob",
]

LABEL_COLUMNS = [
    "current_delta",
    "future_cost",
    "target_score",
    "is_best",
    "score_rank",
    "is_rollout_action",
]

SLOT_COLUMNS = [
    "slot_total_delay",
    "slot_compute_delay",
    "slot_transmission_delay",
    "slot_requests",
    "slot_stages",
    "slot_infeasible",
]


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stage_input_data(request, stage_idx: int) -> float:
    if stage_idx == 0:
        return float(request.input_data)
    return float(request.stage_data[stage_idx - 1])


def _safe_div(numer: float | np.ndarray, denom: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(numer) / np.maximum(np.asarray(denom), 1e-8)


def _positive_max(values: np.ndarray, default: float = 1.0) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return default
    return float(positive.max())


def _candidate_features(
    env: EdgeEnv,
    deployment: np.ndarray,
    gamma: np.ndarray,
    link_load: np.ndarray,
    request,
    stage_idx: int,
    prev_node: int,
    candidate_node: int,
    legal_count: int,
    slow_epoch: int,
) -> dict[str, float]:
    storage_used, memory_used = resource_usage(deployment, env.service_storage, env.service_memory)
    service_id = int(request.service_id)
    num_nodes = env.num_nodes
    max_bw = _positive_max(env.effective_bandwidth)
    compute_cap_max = max(float(env.compute_cap.max()), 1.0)
    arrival_rates = env.node_arrival_rates(slow_epoch)
    service_probs = env.service_probabilities(slow_epoch)
    node_service_probs = env.node_service_probabilities(slow_epoch)

    d_in = _stage_input_data(request, stage_idx)
    sqrt_compute = float(np.sqrt(max(float(request.compute[stage_idx]), 0.0)))
    sqrt_data = float(np.sqrt(max(d_in, 0.0)))
    remaining_compute = float(request.compute[stage_idx + 1 :].sum())
    if stage_idx + 1 < request.num_stages:
        remaining_data = float(request.stage_data[stage_idx:].sum())
    else:
        remaining_data = 0.0

    gamma_util = _safe_div(gamma, env.compute_cap).astype(np.float32)
    gamma_after = float(gamma[candidate_node] + sqrt_compute)
    candidate_gamma_after_util = gamma_after / max(float(env.compute_cap[candidate_node]), 1e-8)

    active_links = env.effective_bandwidth > 0
    link_util = np.zeros_like(link_load, dtype=np.float32)
    link_util[active_links] = link_load[active_links] / np.maximum(env.effective_bandwidth[active_links], 1e-8)

    if prev_node == candidate_node:
        prev_bw = max_bw
        prev_link_load = 0.0
        prev_link_after_util = 0.0
    else:
        prev_bw = float(env.effective_bandwidth[prev_node, candidate_node])
        prev_link_load = float(link_load[prev_node, candidate_node])
        prev_link_after_util = (prev_link_load + sqrt_data) / max(prev_bw, 1e-8)

    if request.source_node == candidate_node:
        source_bw = max_bw
    else:
        source_bw = float(env.effective_bandwidth[request.source_node, candidate_node])

    next_stage_replica_ratio = 0.0
    candidate_next_stage = 0.0
    if stage_idx + 1 < env.service_stages[service_id]:
        next_stage_replica_ratio = float(deployment[service_id, stage_idx + 1].sum()) / float(num_nodes)
        candidate_next_stage = float(deployment[service_id, stage_idx + 1, candidate_node] > 0.5)

    feature = {
        "service_id_norm": service_id / max(env.num_services - 1, 1),
        "stage_idx_norm": stage_idx / max(env.max_stages - 1, 1),
        "source_node_norm": request.source_node / max(num_nodes - 1, 1),
        "prev_node_norm": prev_node / max(num_nodes - 1, 1),
        "candidate_node_norm": candidate_node / max(num_nodes - 1, 1),
        "same_prev_candidate": float(prev_node == candidate_node),
        "same_source_candidate": float(request.source_node == candidate_node),
        "req_compute_norm": float(request.compute[stage_idx]) / 10.0,
        "req_data_in_norm": d_in / 10.0,
        "remaining_compute_norm": remaining_compute / max(10.0 * env.max_stages, 1.0),
        "remaining_data_norm": remaining_data / max(10.0 * env.max_stages, 1.0),
        "remaining_stage_ratio": (request.num_stages - stage_idx - 1) / max(env.max_stages - 1, 1),
        "candidate_compute_cap_norm": float(env.compute_cap[candidate_node]) / compute_cap_max,
        "candidate_gamma_util": float(gamma_util[candidate_node]),
        "candidate_gamma_after_util": candidate_gamma_after_util,
        "mean_gamma_util": float(gamma_util.mean()),
        "max_gamma_util": float(gamma_util.max()),
        "total_gamma_norm": float(gamma.sum() / max(float(env.compute_cap.sum()), 1e-8)),
        "candidate_storage_util": float(storage_used[candidate_node] / max(float(env.storage_cap[candidate_node]), 1e-8)),
        "candidate_memory_util": float(memory_used[candidate_node] / max(float(env.memory_cap[candidate_node]), 1e-8)),
        "current_stage_replica_ratio": float(deployment[service_id, stage_idx].sum()) / float(num_nodes),
        "next_stage_replica_ratio": next_stage_replica_ratio,
        "candidate_next_stage_deployed": candidate_next_stage,
        "legal_candidate_ratio": legal_count / float(num_nodes),
        "prev_candidate_bw_norm": prev_bw / max_bw,
        "prev_candidate_link_load_norm": prev_link_load / 10.0,
        "prev_candidate_link_util": 0.0 if prev_node == candidate_node else float(link_util[prev_node, candidate_node]),
        "prev_candidate_link_after_util": prev_link_after_util,
        "source_candidate_bw_norm": source_bw / max_bw,
        "mean_link_util": float(link_util[active_links].mean()) if np.any(active_links) else 0.0,
        "max_link_util": float(link_util[active_links].max()) if np.any(active_links) else 0.0,
        "total_link_load_norm": float(link_load.sum() / max(10.0 * num_nodes, 1.0)),
        "service_global_prob": float(service_probs[service_id]),
        "source_arrival_rate_norm": float(arrival_rates[request.source_node] / max(float(arrival_rates.max()), 1e-8)),
        "source_service_prob": float(node_service_probs[request.source_node, service_id]),
    }
    return feature


def _candidate_rows(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    gamma: np.ndarray,
    link_load: np.ndarray,
    request,
    stage_idx: int,
    prev_node: int,
    slow_epoch: int,
    fast_slot: int,
    decision_id: int,
    sample_start: int,
    lookahead_depth: int,
    future_weight: float,
) -> tuple[list[dict[str, float]], int]:
    legal = env.legal_nodes(deployment, request.service_id, stage_idx, prev_node)
    legal_nodes = np.flatnonzero(legal)
    if legal_nodes.size == 0:
        raise RuntimeError(
            f"No legal node for request={request.request_id}, service={request.service_id}, stage={stage_idx}."
        )

    rows: list[dict[str, float]] = []
    for offset, node in enumerate(legal_nodes):
        node = int(node)
        gamma_after = gamma.copy()
        link_after = link_load.copy()
        add_stage_to_load(gamma_after, link_after, request, stage_idx, prev_node, node)
        current_delta = float(allocator.incremental_cost(gamma, link_load, gamma_after, link_after))
        future_cost = float(
            _best_future_cost(
                env,
                allocator,
                deployment,
                request,
                stage_idx + 1,
                node,
                gamma_after,
                link_after,
                lookahead_depth,
            )
        )
        target_score = float(current_delta + future_weight * future_cost)

        row = {
            "sample_id": sample_start + offset,
            "decision_id": decision_id,
            "slow_epoch": slow_epoch,
            "fast_slot": fast_slot,
            "request_id": int(request.request_id),
            "user_id": int(request.user_id),
            "source_node": int(request.source_node),
            "service_id": int(request.service_id),
            "stage_idx": stage_idx,
            "prev_node": prev_node,
            "candidate_node": node,
            "candidate_count": int(legal_nodes.size),
            "current_delta": current_delta,
            "future_cost": future_cost,
            "target_score": target_score,
            "is_best": 0.0,
            "score_rank": 0.0,
            "is_rollout_action": 0.0,
        }
        row.update(
            _candidate_features(
                env,
                deployment,
                gamma,
                link_load,
                request,
                stage_idx,
                prev_node,
                node,
                int(legal_nodes.size),
                slow_epoch,
            )
        )
        rows.append(row)

    order = sorted(range(len(rows)), key=lambda idx: rows[idx]["target_score"])
    for rank, idx in enumerate(order):
        rows[idx]["score_rank"] = float(rank)
    rows[order[0]]["is_best"] = 1.0
    return rows, int(rows[order[0]]["candidate_node"])


def _choose_rollout_node(rows: list[dict[str, float]], policy: str, rng: np.random.Generator) -> int:
    if policy == "lookahead_delta":
        return int(min(rows, key=lambda row: row["target_score"])["candidate_node"])
    if policy == "greedy_delta":
        return int(min(rows, key=lambda row: row["current_delta"])["candidate_node"])
    if policy == "random_legal":
        return int(rng.choice([int(row["candidate_node"]) for row in rows]))
    raise ValueError(f"Unsupported rollout policy: {policy}")


def build_dataset(
    cfg: dict,
    deployment_mode: str,
    episodes: int,
    fast_slots: int,
    lookahead_depth: int,
    future_weight: float,
    rollout_policy: str,
    output_prefix: str | Path | None,
) -> tuple[Path, Path, dict[str, float]]:
    cfg = dict(cfg)
    cfg["deployment"] = dict(cfg["deployment"])
    cfg["deployment"]["mode"] = deployment_mode

    env = EdgeEnv(cfg)
    deployment = env.make_deployment(deployment_mode)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)

    rows: list[dict[str, float]] = []
    feature_matrix: list[list[float]] = []
    label_matrix: list[list[float]] = []
    decision_ids: list[int] = []
    candidate_nodes: list[int] = []
    slot_costs: list[float] = []
    slot_requests: list[int] = []

    decision_id = 0
    for slow_epoch in range(episodes):
        for fast_slot in range(fast_slots):
            requests = env.sample_requests(slow_epoch=slow_epoch)
            gamma = np.zeros(env.num_nodes, dtype=np.float32)
            link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
            schedules: dict[int, list[int]] = {}
            slot_row_start = len(rows)

            for request in requests:
                path: list[int] = []
                prev_node = int(request.source_node)
                for stage_idx in range(request.num_stages):
                    candidates, best_node = _candidate_rows(
                        env,
                        allocator,
                        deployment,
                        gamma,
                        link_load,
                        request,
                        stage_idx,
                        prev_node,
                        slow_epoch,
                        fast_slot,
                        decision_id,
                        len(rows),
                        lookahead_depth,
                        future_weight,
                    )
                    rollout_node = _choose_rollout_node(candidates, rollout_policy, env.rng)
                    for row in candidates:
                        row["is_rollout_action"] = float(int(row["candidate_node"]) == rollout_node)
                        rows.append(row)
                        feature_matrix.append([float(row[name]) for name in FEATURE_COLUMNS])
                        label_matrix.append([float(row[name]) for name in LABEL_COLUMNS])
                        decision_ids.append(decision_id)
                        candidate_nodes.append(int(row["candidate_node"]))

                    add_stage_to_load(gamma, link_load, request, stage_idx, prev_node, rollout_node)
                    path.append(rollout_node)
                    prev_node = rollout_node
                    decision_id += 1

                schedules[int(request.request_id)] = path

            allocation = allocator.allocate(requests, schedules)
            slot_costs.append(float(allocation.total_delay))
            slot_requests.append(len(requests))
            slot_stage_count = int(sum(len(path) for path in schedules.values()))
            for row in rows[slot_row_start:]:
                row["slot_total_delay"] = float(allocation.total_delay)
                row["slot_compute_delay"] = float(allocation.compute_delay)
                row["slot_transmission_delay"] = float(allocation.transmission_delay)
                row["slot_requests"] = float(len(requests))
                row["slot_stages"] = float(slot_stage_count)
                row["slot_infeasible"] = float(allocation.infeasible)

    run_name = cfg.get("run_name", "run")
    if output_prefix is None:
        output_prefix = Path("outputs") / "wms" / (
            f"{run_name}_{deployment_mode}_{rollout_policy}_d{lookahead_depth}"
        )
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    fieldnames = META_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS + SLOT_COLUMNS
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        npz_path,
        features=np.asarray(feature_matrix, dtype=np.float32),
        labels=np.asarray(label_matrix, dtype=np.float32),
        feature_columns=np.asarray(FEATURE_COLUMNS),
        label_columns=np.asarray(LABEL_COLUMNS),
        decision_ids=np.asarray(decision_ids, dtype=np.int64),
        candidate_nodes=np.asarray(candidate_nodes, dtype=np.int64),
    )

    summary = {
        "rows": float(len(rows)),
        "decisions": float(decision_id),
        "seed": float(cfg["seed"]),
        "episodes": float(episodes),
        "fast_slots": float(fast_slots),
        "avg_candidates_per_decision": float(len(rows) / max(decision_id, 1)),
        "avg_slot_delay": float(np.mean(slot_costs)) if slot_costs else 0.0,
        "avg_requests_per_slot": float(np.mean(slot_requests)) if slot_requests else 0.0,
    }
    return csv_path, npz_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deployment-mode", default=None, choices=["heuristic", "fixed", "random", "monolithic"])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--future-weight", type=float, default=0.8)
    parser.add_argument("--rollout-policy", default="lookahead_delta", choices=["lookahead_delta", "greedy_delta", "random_legal"])
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed_overridden = args.seed is not None
    if seed_overridden:
        cfg["seed"] = int(args.seed)
    deployment_mode = args.deployment_mode or cfg["deployment"]["mode"]
    fast_slots = int(args.fast_slots or cfg["training"]["fast_slots_per_episode"])
    output_prefix = args.output_prefix
    if output_prefix is None and seed_overridden:
        run_name = cfg.get("run_name", "run")
        output_prefix = Path("outputs") / "wms" / (
            f"{run_name}_{deployment_mode}_{args.rollout_policy}_d{args.lookahead_depth}_s{cfg['seed']}"
        )

    csv_path, npz_path, summary = build_dataset(
        cfg=cfg,
        deployment_mode=deployment_mode,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        lookahead_depth=int(args.lookahead_depth),
        future_weight=float(args.future_weight),
        rollout_policy=args.rollout_policy,
        output_prefix=output_prefix,
    )

    print("WM-S dataset built")
    for key, value in summary.items():
        print(f"{key}={value:.4f}")
    print(f"csv={csv_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
