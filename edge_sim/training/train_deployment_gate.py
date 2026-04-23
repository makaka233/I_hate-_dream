from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.deployment_gate import DeploymentGateAgent, GateDataset
from edge_sim.env.dynamic_deployment import deployment_change_count, make_dynamic_deployment
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.env.request import Request
from edge_sim.evaluation.policies import run_greedy_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator
from edge_sim.training.simulate_v2 import (
    build_epoch_trace,
    global_share_from_source_demand,
    observed_source_service_demand,
)
from edge_sim.training.train_v1 import set_seed


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def epoch_delay(env: EdgeEnv, allocator: KKTAllocator, x: np.ndarray, batches: list[list[Request]], epoch: int) -> float:
    return float(sum(run_greedy_delta_slot(env, allocator, x, requests=batch, slow_epoch=epoch)["total_delay"] for batch in batches))


def make_features(
    demand: np.ndarray,
    candidate_x: np.ndarray,
    previous_x: np.ndarray,
    change_count: float,
    env: EdgeEnv,
) -> np.ndarray:
    service_replica_counts = candidate_x.sum(axis=(1, 2)) / np.maximum(np.array(env.service_stages, dtype=np.float32), 1.0)
    delta_replicas = np.abs(candidate_x - previous_x).sum(axis=(1, 2))
    return np.concatenate(
        [
            demand.astype(np.float32),
            service_replica_counts.astype(np.float32),
            delta_replicas.astype(np.float32) / max(env.num_nodes, 1),
            np.array([change_count / max(float(candidate_x.sum()), 1.0)], dtype=np.float32),
        ]
    )


def build_gate_dataset(cfg: dict, epochs: int, fast_slots: int) -> tuple[GateDataset, list[list[list[Request]]]]:
    set_seed(int(cfg["seed"]))
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))
    trace = build_epoch_trace(cfg, epochs, fast_slots)

    previous_x = env.make_deployment("fixed")
    previous_observed_matrix = np.full((env.num_nodes, env.num_services), 1.0 / env.num_services, dtype=np.float32)
    features, labels, keep_costs, apply_costs = [], [], [], []

    for epoch, batches in enumerate(trace):
        # History-demand candidate; this is intentionally realistic rather than oracle.
        source_demand = previous_observed_matrix
        demand = global_share_from_source_demand(source_demand)
        candidate_x, dep_info = make_dynamic_deployment(
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
        changes = deployment_change_count(previous_x, candidate_x)
        keep_cost = epoch_delay(env, allocator, previous_x, batches, epoch)
        apply_cost = epoch_delay(env, allocator, candidate_x, batches, epoch) + migration_weight * changes

        features.append(make_features(demand, candidate_x, previous_x, changes, env))
        labels.append(1 if apply_cost < keep_cost else 0)
        keep_costs.append(keep_cost)
        apply_costs.append(apply_cost)

        # Dataset rollout follows the oracle label so later states reflect the
        # best available keep/apply decision.
        previous_x = candidate_x if apply_cost < keep_cost else previous_x
        previous_observed_matrix = observed_source_service_demand(batches, env.num_nodes, env.num_services)

    return (
        GateDataset(
            features=np.stack(features).astype(np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            keep_costs=np.asarray(keep_costs, dtype=np.float32),
            apply_costs=np.asarray(apply_costs, dtype=np.float32),
        ),
        trace,
    )


def evaluate_gate_policy(cfg: dict, agent: DeploymentGateAgent, epochs: int, fast_slots: int, trace: list[list[list[Request]]]) -> list[dict[str, float]]:
    set_seed(int(cfg["seed"]))
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))
    previous_x = env.make_deployment("fixed")
    previous_observed_matrix = np.full((env.num_nodes, env.num_services), 1.0 / env.num_services, dtype=np.float32)
    rows = []

    for epoch, batches in enumerate(trace):
        source_demand = previous_observed_matrix
        demand = global_share_from_source_demand(source_demand)
        candidate_x, _ = make_dynamic_deployment(
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
        changes = deployment_change_count(previous_x, candidate_x)
        feat = make_features(demand, candidate_x, previous_x, changes, env)[None, :]
        action = int(agent.predict(feat)[0])
        x = candidate_x if action == 1 else previous_x
        delay = epoch_delay(env, allocator, x, batches, epoch)
        migration_cost = migration_weight * changes if action == 1 else 0.0
        rows.append(
            {
                "epoch": float(epoch),
                "action_apply": float(action),
                "delay_cost": delay,
                "migration_cost": migration_cost,
                "total_cost": delay + migration_cost,
                "change_count": changes if action == 1 else 0.0,
            }
        )
        previous_x = x
        previous_observed_matrix = observed_source_service_demand(batches, env.num_nodes, env.num_services)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    dataset, trace = build_gate_dataset(cfg, args.epochs, fast_slots)
    apply_rate = float(dataset.labels.mean())
    print(f"gate dataset: samples={len(dataset.labels)} apply_rate={apply_rate:.3f}")

    agent = DeploymentGateAgent(input_dim=dataset.features.shape[1], hidden_dim=64, lr=1e-3, device=cfg["training"].get("device", "cpu"))
    stats = agent.fit(dataset, epochs=args.train_epochs)
    print(f"gate training: loss={stats['loss']:.4f} accuracy={stats['accuracy']:.4f}")

    rows = evaluate_gate_policy(cfg, agent, args.epochs, fast_slots, trace)
    avg_cost = np.mean([row["total_cost"] for row in rows])
    avg_changes = np.mean([row["change_count"] for row in rows])
    apply_count = sum(row["action_apply"] for row in rows)
    print(f"gate policy: avg_epoch_cost={avg_cost:.4f} avg_changes={avg_changes:.2f} apply_count={apply_count:.0f}")

    run_name = cfg.get("run_name", "v2_drift")
    out_path = Path(args.output) if args.output else Path("outputs") / "logs" / f"{run_name}_deployment_gate.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    torch.save({"model": agent.model.state_dict(), "input_dim": dataset.features.shape[1], "config": cfg}, Path("outputs") / f"deployment_gate_{run_name}.pt")
    print(f"saved gate log to {out_path}")


if __name__ == "__main__":
    main()
