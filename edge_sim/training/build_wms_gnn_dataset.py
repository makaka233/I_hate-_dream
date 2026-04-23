from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.policies import _best_future_cost
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load


TARGET_COLUMNS = ["current_delta", "future_cost", "target_score"]
BASE_ROLLOUT_POLICIES = ["lookahead_delta", "greedy_delta", "random_legal"]


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _candidate_targets(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    request,
    stage_idx: int,
    prev_node: int,
    gamma: np.ndarray,
    link_load: np.ndarray,
    lookahead_depth: int,
    future_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legal_mask = env.legal_nodes(deployment, request.service_id, stage_idx, prev_node)
    legal_nodes = np.flatnonzero(legal_mask)
    if legal_nodes.size == 0:
        raise RuntimeError(
            f"No legal node for request={request.request_id}, service={request.service_id}, stage={stage_idx}."
        )

    targets = np.zeros((env.num_nodes, len(TARGET_COLUMNS)), dtype=np.float32)
    for node in legal_nodes:
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
        targets[node] = [current_delta, future_cost, current_delta + future_weight * future_cost]

    return legal_nodes.astype(np.int64), legal_mask.astype(np.bool_), targets


def _choose_rollout_node(
    legal_nodes: np.ndarray,
    targets: np.ndarray,
    rollout_policy: str,
    rng: np.random.Generator,
) -> int:
    if rollout_policy == "lookahead_delta":
        scores = targets[legal_nodes, 2]
        return int(legal_nodes[int(np.argmin(scores))])
    if rollout_policy == "greedy_delta":
        scores = targets[legal_nodes, 0]
        return int(legal_nodes[int(np.argmin(scores))])
    if rollout_policy == "random_legal":
        return int(rng.choice(legal_nodes))
    raise ValueError(f"Unsupported rollout policy: {rollout_policy}")


def _choose_rollout(
    legal_nodes: np.ndarray,
    targets: np.ndarray,
    rollout_policy: str,
    rng: np.random.Generator,
    mixed_policy_names: list[str],
    mixed_policy_probs: np.ndarray,
) -> tuple[int, str]:
    if rollout_policy != "mixed":
        return _choose_rollout_node(legal_nodes, targets, rollout_policy, rng), rollout_policy

    chosen_policy = str(rng.choice(np.asarray(mixed_policy_names, dtype=object), p=mixed_policy_probs))
    return _choose_rollout_node(legal_nodes, targets, chosen_policy, rng), chosen_policy


def build_dataset(
    cfg: dict,
    deployment_mode: str,
    episodes: int,
    fast_slots: int,
    lookahead_depth: int,
    future_weight: float,
    rollout_policy: str,
    mixed_policy_names: list[str],
    mixed_policy_probs: np.ndarray,
    output_prefix: str | Path | None,
) -> tuple[Path, Path, dict[str, float]]:
    cfg = dict(cfg)
    cfg["deployment"] = dict(cfg["deployment"])
    cfg["deployment"]["mode"] = deployment_mode

    env = EdgeEnv(cfg)
    deployment = env.make_deployment(deployment_mode)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)

    node_feat_items: list[np.ndarray] = []
    edge_attr_items: list[np.ndarray] = []
    candidate_edge_attr_items: list[np.ndarray] = []
    request_feat_items: list[np.ndarray] = []
    prev_node_items: list[int] = []
    legal_mask_items: list[np.ndarray] = []
    target_items: list[np.ndarray] = []
    best_action_items: list[int] = []
    greedy_action_items: list[int] = []
    rollout_action_items: list[int] = []
    rollout_policy_id_items: list[int] = []
    hard_mask_items: list[bool] = []
    hard_gap_items: list[float] = []
    meta_rows: list[dict[str, float | int]] = []
    slot_costs: list[float] = []
    slot_requests: list[int] = []
    policy_to_id = {name: idx for idx, name in enumerate(mixed_policy_names)}

    decision_id = 0
    for slow_epoch in range(episodes):
        for fast_slot in range(fast_slots):
            requests = env.sample_requests(slow_epoch=slow_epoch)
            gamma = np.zeros(env.num_nodes, dtype=np.float32)
            link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
            schedules: dict[int, list[int]] = {}

            for request in requests:
                path: list[int] = []
                prev_node = int(request.source_node)
                for stage_idx in range(request.num_stages):
                    obs = env.graph_observation(deployment, gamma, link_load, request, stage_idx, prev_node)
                    legal_nodes, legal_mask, targets = _candidate_targets(
                        env,
                        allocator,
                        deployment,
                        request,
                        stage_idx,
                        prev_node,
                        gamma,
                        link_load,
                        lookahead_depth,
                        future_weight,
                    )
                    best_scores = targets[legal_nodes, 2]
                    greedy_scores = targets[legal_nodes, 0]
                    best_node = int(legal_nodes[int(np.argmin(best_scores))])
                    greedy_node = int(legal_nodes[int(np.argmin(greedy_scores))])
                    hard_mask = bool(greedy_node != best_node)
                    hard_gap = float(targets[greedy_node, 2] - targets[best_node, 2])
                    rollout_node, rollout_policy_name = _choose_rollout(
                        legal_nodes,
                        targets,
                        rollout_policy,
                        env.rng,
                        mixed_policy_names,
                        mixed_policy_probs,
                    )

                    node_feat_items.append(obs["node_feat"].astype(np.float32))
                    edge_attr_items.append(obs["edge_attr"].astype(np.float32))
                    candidate_edge_attr_items.append(obs["candidate_edge_attr"].astype(np.float32))
                    request_feat_items.append(obs["request_feat"].astype(np.float32))
                    prev_node_items.append(prev_node)
                    legal_mask_items.append(legal_mask)
                    target_items.append(targets)
                    best_action_items.append(best_node)
                    greedy_action_items.append(greedy_node)
                    rollout_action_items.append(rollout_node)
                    rollout_policy_id_items.append(policy_to_id[rollout_policy_name])
                    hard_mask_items.append(hard_mask)
                    hard_gap_items.append(hard_gap)
                    meta_rows.append(
                        {
                            "decision_id": decision_id,
                            "slow_epoch": slow_epoch,
                            "fast_slot": fast_slot,
                            "request_id": int(request.request_id),
                            "source_node": int(request.source_node),
                            "service_id": int(request.service_id),
                            "stage_idx": stage_idx,
                            "prev_node": prev_node,
                            "best_action": best_node,
                            "greedy_action": greedy_node,
                            "rollout_action": rollout_node,
                            "rollout_policy_id": policy_to_id[rollout_policy_name],
                            "candidate_count": int(legal_nodes.size),
                            "hard_flag": int(hard_mask),
                            "hard_gap": hard_gap,
                            "best_target_score": float(targets[best_node, 2]),
                        }
                    )

                    add_stage_to_load(gamma, link_load, request, stage_idx, prev_node, rollout_node)
                    path.append(rollout_node)
                    prev_node = rollout_node
                    decision_id += 1

                schedules[int(request.request_id)] = path

            allocation = allocator.allocate(requests, schedules)
            slot_costs.append(float(allocation.total_delay))
            slot_requests.append(len(requests))

    run_name = cfg.get("run_name", "run")
    if output_prefix is None:
        output_prefix = Path("outputs") / "wms" / (
            f"{run_name}_{deployment_mode}_{rollout_policy}_gnn_d{lookahead_depth}"
        )
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "decision_id",
            "slow_epoch",
            "fast_slot",
            "request_id",
            "source_node",
            "service_id",
            "stage_idx",
            "prev_node",
            "best_action",
            "greedy_action",
            "rollout_action",
            "rollout_policy_id",
            "candidate_count",
            "hard_flag",
            "hard_gap",
            "best_target_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    np.savez_compressed(
        npz_path,
        node_feat=np.asarray(node_feat_items, dtype=np.float32),
        edge_index=env.edge_index.astype(np.int64),
        edge_attr=np.asarray(edge_attr_items, dtype=np.float32),
        candidate_edge_attr=np.asarray(candidate_edge_attr_items, dtype=np.float32),
        request_feat=np.asarray(request_feat_items, dtype=np.float32),
        prev_node=np.asarray(prev_node_items, dtype=np.int64),
        legal_mask=np.asarray(legal_mask_items, dtype=np.bool_),
        targets=np.asarray(target_items, dtype=np.float32),
        target_columns=np.asarray(TARGET_COLUMNS),
        best_action=np.asarray(best_action_items, dtype=np.int64),
        greedy_action=np.asarray(greedy_action_items, dtype=np.int64),
        rollout_action=np.asarray(rollout_action_items, dtype=np.int64),
        rollout_policy_id=np.asarray(rollout_policy_id_items, dtype=np.int64),
        rollout_policy_names=np.asarray(mixed_policy_names),
        hard_mask=np.asarray(hard_mask_items, dtype=np.bool_),
        hard_gap=np.asarray(hard_gap_items, dtype=np.float32),
        decision_ids=np.arange(decision_id, dtype=np.int64),
    )

    summary = {
        "decisions": float(decision_id),
        "seed": float(cfg["seed"]),
        "episodes": float(episodes),
        "fast_slots": float(fast_slots),
        "avg_candidates_per_decision": float(np.mean([row["candidate_count"] for row in meta_rows])) if meta_rows else 0.0,
        "hard_ratio": float(np.mean(hard_mask_items)) if hard_mask_items else 0.0,
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
    parser.add_argument(
        "--rollout-policy",
        default="lookahead_delta",
        choices=["lookahead_delta", "greedy_delta", "random_legal", "mixed"],
    )
    parser.add_argument(
        "--mixed-policies",
        nargs="+",
        default=BASE_ROLLOUT_POLICIES,
        choices=BASE_ROLLOUT_POLICIES,
    )
    parser.add_argument("--mixed-policy-probs", type=float, nargs="+", default=None)
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
            f"{run_name}_{deployment_mode}_{args.rollout_policy}_gnn_d{args.lookahead_depth}_s{cfg['seed']}"
        )

    mixed_policy_names = list(dict.fromkeys(args.mixed_policies))
    if args.mixed_policy_probs is None:
        mixed_policy_probs = np.full(len(mixed_policy_names), 1.0 / len(mixed_policy_names), dtype=np.float32)
    else:
        if len(args.mixed_policy_probs) != len(mixed_policy_names):
            raise ValueError("mixed-policy-probs must match the number of mixed-policies.")
        mixed_policy_probs = np.asarray(args.mixed_policy_probs, dtype=np.float32)
        if np.any(mixed_policy_probs < 0):
            raise ValueError("mixed-policy-probs must be non-negative.")
        mixed_policy_probs = mixed_policy_probs / max(float(mixed_policy_probs.sum()), 1e-8)

    csv_path, npz_path, summary = build_dataset(
        cfg=cfg,
        deployment_mode=deployment_mode,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        lookahead_depth=int(args.lookahead_depth),
        future_weight=float(args.future_weight),
        rollout_policy=args.rollout_policy,
        mixed_policy_names=mixed_policy_names,
        mixed_policy_probs=mixed_policy_probs,
        output_prefix=output_prefix,
    )
    print("GNN WM-S dataset built")
    for key, value in summary.items():
        print(f"{key}={value:.4f}")
    print(f"csv={csv_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
