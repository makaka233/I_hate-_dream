from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.gat_ppo import MaskedPPOAgent, StageTransition
from edge_sim.env.deployment import deployment_summary, format_deployment_summary
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load, kkt_load_cost


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fast_slot(
    env: EdgeEnv,
    agent: MaskedPPOAgent,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    omega: float,
    deterministic: bool = False,
    requests: list | None = None,
    slow_epoch: int | None = None,
) -> tuple[list[StageTransition], dict[str, float]]:
    requests = requests if requests is not None else env.sample_requests(slow_epoch=slow_epoch)
    gamma = np.zeros(env.num_nodes, dtype=np.float32)
    link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
    schedules: dict[int, list[int]] = {}
    transitions: list[StageTransition] = []

    for req in requests:
        path: list[int] = []
        prev = req.source_node
        for stage_idx in range(req.num_stages):
            obs = env.graph_observation(deployment, gamma, link_load, req, stage_idx, prev)
            if not bool(obs["legal_mask"].any()):
                raise RuntimeError(
                    f"No legal node for request={req.request_id}, service={req.service_id}, stage={stage_idx}."
                )

            action, log_prob, value = agent.select_action(obs, deterministic=deterministic)
            gamma_before = gamma.copy()
            link_before = link_load.copy()

            add_stage_to_load(gamma, link_load, req, stage_idx, prev, action)
            delta_cost = allocator.incremental_cost(gamma_before, link_before, gamma, link_load)
            transitions.append(
                StageTransition(
                    obs=obs,
                    action=action,
                    reward=-float(delta_cost),
                    log_prob=log_prob,
                    value=value,
                )
            )

            path.append(action)
            prev = action

        schedules[req.request_id] = path

    allocation = allocator.allocate(requests, schedules)
    stage_count = max(len(transitions), 1)
    batch_share = -omega * allocation.total_delay / stage_count
    for tr in transitions:
        tr.reward += batch_share

    metrics = {
        "requests": float(len(requests)),
        "stages": float(len(transitions)),
        "total_delay": allocation.total_delay,
        "compute_delay": allocation.compute_delay,
        "transmission_delay": allocation.transmission_delay,
        "kkt_virtual_cost": kkt_load_cost(gamma, link_load, env.compute_cap, env.effective_bandwidth),
        "infeasible": float(allocation.infeasible),
    }
    for i in range(env.num_services):
        metrics[f"service_{i}"] = float(sum(1 for req in requests if req.service_id == i))
    return transitions, metrics


def normalize_transition_rewards(transitions: list[StageTransition]) -> None:
    if len(transitions) < 2:
        return
    rewards = np.array([tr.reward for tr in transitions], dtype=np.float32)
    mean = float(rewards.mean())
    std = float(rewards.std() + 1e-8)
    for tr in transitions:
        tr.reward = (tr.reward - mean) / std


def train(cfg: dict, deployment_mode: str, episodes_override: int | None = None) -> None:
    set_seed(int(cfg["seed"]))
    cfg = dict(cfg)
    cfg["deployment"] = dict(cfg["deployment"])
    cfg["deployment"]["mode"] = deployment_mode

    env = EdgeEnv(cfg)
    deployment = env.make_deployment(deployment_mode)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    agent = MaskedPPOAgent(cfg)

    episodes = int(episodes_override or cfg["training"]["episodes"])
    fast_slots = int(cfg["training"]["fast_slots_per_episode"])
    omega = float(cfg["training"]["omega_batch_reward"])

    print(f"V1 training | deployment={deployment_mode} | episodes={episodes} | device={agent.device}")
    print(f"nodes={env.num_nodes}, services={env.num_services}, stages={env.service_stages}")
    dep_summary = deployment_summary(
        deployment,
        env.service_stages,
        env.service_storage,
        env.service_memory,
        env.storage_cap,
        env.memory_cap,
    )
    print(f"deployment: {format_deployment_summary(dep_summary)}")

    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name", "v1")
    log_path = log_dir / f"{run_name}_{deployment_mode}_train.csv"
    fieldnames = [
        "episode",
        "avg_total_delay",
        "avg_compute_delay",
        "avg_transmission_delay",
        "avg_requests_per_slot",
        "stages",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    fieldnames.extend([f"service_{i}_share" for i in range(env.num_services)])

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            rollout: list[StageTransition] = []
            totals = []
            comp = []
            tran = []
            reqs = []
            service_counts = np.zeros(env.num_services, dtype=np.float32)

            for _ in range(fast_slots):
                transitions, metrics = run_fast_slot(env, agent, allocator, deployment, omega, slow_epoch=ep - 1)
                rollout.extend(transitions)
                totals.append(metrics["total_delay"])
                comp.append(metrics["compute_delay"])
                tran.append(metrics["transmission_delay"])
                reqs.append(metrics["requests"])
                for key, value in metrics.items():
                    if key.startswith("service_"):
                        service_counts[int(key.split("_")[1])] += float(value)

            normalize_transition_rewards(rollout)
            stats = agent.update(rollout)

            row = {
                "episode": ep,
                "avg_total_delay": float(np.mean(totals)),
                "avg_compute_delay": float(np.mean(comp)),
                "avg_transmission_delay": float(np.mean(tran)),
                "avg_requests_per_slot": float(np.mean(reqs)),
                "stages": len(rollout),
                "loss": stats["loss"],
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
            }
            total_services = max(float(service_counts.sum()), 1.0)
            for i in range(env.num_services):
                row[f"service_{i}_share"] = float(service_counts[i] / total_services)
            writer.writerow(row)
            f.flush()

            if ep == 1 or ep % 5 == 0:
                print(
                    "episode={:04d} delay={:.4f} comp={:.4f} tran={:.4f} req/slot={:.2f} "
                    "loss={:.4f} entropy={:.4f}".format(
                        ep,
                        row["avg_total_delay"],
                        row["avg_compute_delay"],
                        row["avg_transmission_delay"],
                        row["avg_requests_per_slot"],
                        stats["loss"],
                        stats["entropy"],
                    )
                )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / f"gat_ppo_{run_name}_{deployment_mode}.pt"
    torch.save({"model": agent.policy.state_dict(), "config": cfg, "deployment": deployment}, model_path)
    print(f"saved model to {model_path}")
    print(f"saved training log to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--deployment-mode", default=None, choices=["heuristic", "fixed", "random", "monolithic"])
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    deployment_mode = args.deployment_mode or cfg["deployment"]["mode"]
    train(cfg, deployment_mode, args.episodes)


if __name__ == "__main__":
    main()
