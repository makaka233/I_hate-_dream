from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from edge_sim.env.deployment import deployment_summary, format_deployment_summary
from edge_sim.env.dynamic_deployment import deployment_change_count, make_dynamic_deployment
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.env.request import Request
from edge_sim.evaluation.policies import run_greedy_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator
from edge_sim.training.train_v1 import set_seed


STRATEGIES = [
    "static_heuristic",
    "static_fixed",
    "static_random",
    "static_monolithic",
    "dynamic_oracle",
    "dynamic_history",
    "dynamic_trend",
]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_epoch_trace(cfg: dict, epochs: int, fast_slots: int) -> list[list[list[Request]]]:
    env = EdgeEnv(cfg)
    return [[env.sample_requests(slow_epoch=epoch) for _ in range(fast_slots)] for epoch in range(epochs)]


def observed_share(request_batches: list[list[Request]], num_services: int) -> np.ndarray:
    counts = np.zeros(num_services, dtype=np.float32)
    for batch in request_batches:
        for req in batch:
            counts[req.service_id] += 1.0
    counts += 1e-3
    return counts / counts.sum()


def observed_source_service_demand(
    request_batches: list[list[Request]],
    num_nodes: int,
    num_services: int,
) -> np.ndarray:
    counts = np.full((num_nodes, num_services), 1e-3, dtype=np.float32)
    for batch in request_batches:
        for req in batch:
            counts[req.source_node, req.service_id] += 1.0
    return counts / float(len(request_batches))


def global_share_from_source_demand(source_service_demand: np.ndarray) -> np.ndarray:
    service = np.asarray(source_service_demand, dtype=np.float32).sum(axis=0)
    service += 1e-6
    return service / service.sum()


def predict_trend_demand(
    previous: np.ndarray,
    older: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    if older is None:
        pred = previous.copy()
    else:
        pred = previous + beta * (previous - older)
    pred = np.maximum(pred, 1e-3)
    return pred


def maybe_keep_previous_deployment(
    env: EdgeEnv,
    allocator: KKTAllocator,
    previous_x: np.ndarray | None,
    candidate_x: np.ndarray,
    demand_trace: list[list[Request]],
    migration_changes: float,
    migration_weight: float,
    change_threshold: float,
) -> tuple[np.ndarray, bool, float, float]:
    """Keep previous deployment if candidate's estimated gain is too small."""

    if previous_x is None or change_threshold <= 0:
        return candidate_x, False, 0.0, 0.0

    prev_delay = 0.0
    cand_delay = 0.0
    for batch in demand_trace:
        prev_delay += run_greedy_delta_slot(env, allocator, previous_x, requests=batch)["total_delay"]
        cand_delay += run_greedy_delta_slot(env, allocator, candidate_x, requests=batch)["total_delay"]

    required_gain = migration_weight * migration_changes * (1.0 + change_threshold)
    gain = prev_delay - cand_delay
    if gain <= required_gain:
        return previous_x, True, prev_delay, cand_delay
    return candidate_x, False, prev_delay, cand_delay


def make_static_deployment(env: EdgeEnv, strategy: str) -> np.ndarray:
    if strategy == "static_heuristic":
        return env.make_deployment("heuristic")
    if strategy == "static_fixed":
        return env.make_deployment("fixed")
    if strategy == "static_random":
        return env.make_deployment("random")
    if strategy == "static_monolithic":
        return env.make_deployment("monolithic")
    raise ValueError(f"Unsupported static strategy: {strategy}")


def simulate_strategy(
    cfg: dict,
    strategy: str,
    trace: list[list[list[Request]]],
    fast_slots: int,
) -> list[dict[str, float | str]]:
    set_seed(int(cfg["seed"]))
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))

    previous_x = None
    static_x = None
    previous_observed_matrix = np.full(
        (env.num_nodes, env.num_services),
        1.0 / env.num_services,
        dtype=np.float32,
    )
    older_observed_matrix = None
    if strategy.startswith("static"):
        static_x = make_static_deployment(env, strategy)
        print(f"[{strategy}] deployment: {format_deployment_summary(deployment_summary(static_x, env.service_stages, env.service_storage, env.service_memory, env.storage_cap, env.memory_cap))}")

    rows: list[dict[str, float | str]] = []
    for epoch, batches in enumerate(trace):
        if strategy == "dynamic_oracle":
            source_demand = env.source_service_demand(epoch)
            demand = global_share_from_source_demand(source_demand)
            x, dep_info = make_dynamic_deployment(
                env,
                demand,
                previous_x,
                int(dyn_cfg.get("extra_replica_budget", 13)),
                int(dyn_cfg.get("max_replicas_per_stage", 3)),
                float(dyn_cfg.get("alpha_compute", 1.0)),
                float(dyn_cfg.get("alpha_data", 0.5)),
                source_demand,
                float(dyn_cfg.get("location_weight", 1.0)),
                bool(dyn_cfg.get("keep_previous", True)),
            )
        elif strategy in {"dynamic_history", "dynamic_trend"}:
            if strategy == "dynamic_trend":
                source_demand = predict_trend_demand(
                    previous_observed_matrix,
                    older_observed_matrix,
                    float(dyn_cfg.get("history_trend_beta", 0.5)),
                )
            else:
                source_demand = previous_observed_matrix
            demand = global_share_from_source_demand(source_demand)
            x, dep_info = make_dynamic_deployment(
                env,
                demand,
                previous_x,
                int(dyn_cfg.get("extra_replica_budget", 13)),
                int(dyn_cfg.get("max_replicas_per_stage", 3)),
                float(dyn_cfg.get("alpha_compute", 1.0)),
                float(dyn_cfg.get("alpha_data", 0.5)),
                source_demand,
                float(dyn_cfg.get("location_weight", 1.0)),
                bool(dyn_cfg.get("keep_previous", True)),
            )
        else:
            x = static_x
            dep_info = {
                "replica_target_total": float(x.sum()),
                "replica_actual_total": float(x.sum()),
                "change_count": deployment_change_count(previous_x, x) if previous_x is None else 0.0,
            }
            source_demand = env.source_service_demand(epoch)
            demand = global_share_from_source_demand(source_demand)

        if strategy.startswith("dynamic"):
            x, kept_previous, est_prev_delay, est_candidate_delay = maybe_keep_previous_deployment(
                env,
                allocator,
                previous_x,
                x,
                batches,
                float(dep_info["change_count"]),
                migration_weight,
                float(dyn_cfg.get("change_threshold", 0.0)),
            )
            if kept_previous:
                dep_info["change_count"] = 0.0
                dep_info["kept_previous"] = 1.0
            else:
                dep_info["kept_previous"] = 0.0
            dep_info["estimated_prev_delay"] = float(est_prev_delay)
            dep_info["estimated_candidate_delay"] = float(est_candidate_delay)

        totals, comp, tran, reqs = [], [], [], []
        for batch in batches:
            metrics = run_greedy_delta_slot(env, allocator, x, requests=batch, slow_epoch=epoch)
            totals.append(metrics["total_delay"])
            comp.append(metrics["compute_delay"])
            tran.append(metrics["transmission_delay"])
            reqs.append(metrics["requests"])

        observed = observed_share(batches, env.num_services)
        older_observed_matrix = previous_observed_matrix
        previous_observed_matrix = observed_source_service_demand(batches, env.num_nodes, env.num_services)
        delay_sum = float(np.sum(totals))
        migration_changes = float(dep_info["change_count"])
        migration_cost = migration_weight * migration_changes
        total_cost = delay_sum + migration_cost
        dep_summary = deployment_summary(x, env.service_stages, env.service_storage, env.service_memory, env.storage_cap, env.memory_cap)

        row: dict[str, float | str] = {
            "epoch": float(epoch),
            "strategy": strategy,
            "avg_total_delay": float(np.mean(totals)),
            "sum_total_delay": delay_sum,
            "avg_compute_delay": float(np.mean(comp)),
            "avg_transmission_delay": float(np.mean(tran)),
            "avg_requests_per_slot": float(np.mean(reqs)),
            "migration_changes": migration_changes,
            "migration_cost": migration_cost,
            "total_cost": total_cost,
            **dep_info,
            **dep_summary,
        }
        for i in range(env.num_services):
            row[f"demand_prob_{i}"] = float(demand[i])
            row[f"observed_share_{i}"] = float(observed[i])
        rows.append(row)
        previous_x = x

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--strategy", default="all", choices=STRATEGIES + ["all"])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]
    trace = build_epoch_trace(cfg, args.epochs, fast_slots)
    avg_reqs = np.mean([len(batch) for epoch in trace for batch in epoch])
    print(f"V2-A simulation | epochs={args.epochs} | fast_slots={fast_slots} | avg req/slot={avg_reqs:.2f}")

    all_rows: list[dict[str, float | str]] = []
    for strategy in strategies:
        rows = simulate_strategy(cfg, strategy, trace, fast_slots)
        all_rows.extend(rows)
        avg_cost = np.mean([float(row["total_cost"]) for row in rows])
        avg_delay = np.mean([float(row["avg_total_delay"]) for row in rows])
        avg_migration = np.mean([float(row["migration_changes"]) for row in rows])
        print(f"[{strategy}] avg_delay={avg_delay:.4f} avg_epoch_cost={avg_cost:.4f} avg_changes={avg_migration:.2f}")

    run_name = cfg.get("run_name", "v2_drift")
    out_path = Path(args.output) if args.output else Path("outputs") / "logs" / f"{run_name}_v2a_{args.strategy}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in all_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved V2-A simulation log to {out_path}")


if __name__ == "__main__":
    main()
