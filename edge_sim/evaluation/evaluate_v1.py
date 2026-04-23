from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.gat_ppo import MaskedPPOAgent
from edge_sim.env.deployment import deployment_summary, format_deployment_summary
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.env.request import Request
from edge_sim.evaluation.policies import run_greedy_delta_slot, run_lookahead_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator
from edge_sim.training.train_v1 import run_fast_slot, set_seed


DEPLOYMENT_MODES = ["heuristic", "fixed", "random", "monolithic"]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agent(cfg: dict, deployment_mode: str, model_path: str | None) -> tuple[MaskedPPOAgent, np.ndarray]:
    agent = MaskedPPOAgent(cfg)
    run_name = cfg.get("run_name", "v1")
    path = Path(model_path) if model_path else Path("outputs") / f"gat_ppo_{run_name}_{deployment_mode}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["model"])
    agent.policy.eval()
    return agent, checkpoint["deployment"]


def build_request_trace(cfg: dict, episodes: int, fast_slots: int) -> list[list[Request]]:
    """Generate one reusable request trace for fair cross-scheme evaluation."""

    env = EdgeEnv(cfg)
    trace: list[list[Request]] = []
    for episode in range(episodes):
        for _ in range(fast_slots):
            trace.append(env.sample_requests(slow_epoch=episode))
    return trace


def evaluate_one(
    cfg: dict,
    deployment_mode: str,
    policy: str,
    episodes: int,
    fast_slots: int,
    model_path: str | None = None,
    request_trace: list[list[Request]] | None = None,
) -> dict[str, float | str]:
    cfg = dict(cfg)
    cfg["deployment"] = dict(cfg["deployment"])
    cfg["deployment"]["mode"] = deployment_mode
    set_seed(int(cfg["seed"]))

    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    agent = None
    if policy == "ppo":
        agent, deployment = load_agent(cfg, deployment_mode, model_path)
    else:
        deployment = env.make_deployment(deployment_mode)

    dep_summary = deployment_summary(
        deployment,
        env.service_stages,
        env.service_storage,
        env.service_memory,
        env.storage_cap,
        env.memory_cap,
    )
    print(f"[{deployment_mode}/{policy}] deployment: {format_deployment_summary(dep_summary)}")

    totals, comp, tran, reqs, infeasible = [], [], [], [], []
    service_counts = np.zeros(env.num_services, dtype=np.float32)
    slot_idx = 0
    for _ in range(episodes):
        for _ in range(fast_slots):
            slow_epoch = slot_idx // fast_slots
            requests = request_trace[slot_idx] if request_trace is not None else None
            slot_idx += 1
            if policy == "ppo":
                _, metrics = run_fast_slot(
                    env,
                    agent,
                    allocator,
                    deployment,
                    omega=0.0,
                    deterministic=True,
                    requests=requests,
                    slow_epoch=slow_epoch,
                )
            elif policy == "greedy_delta":
                metrics = run_greedy_delta_slot(env, allocator, deployment, requests=requests, slow_epoch=slow_epoch)
            else:
                metrics = run_lookahead_delta_slot(env, allocator, deployment, requests=requests, slow_epoch=slow_epoch)
            totals.append(metrics["total_delay"])
            comp.append(metrics["compute_delay"])
            tran.append(metrics["transmission_delay"])
            reqs.append(metrics["requests"])
            infeasible.append(metrics["infeasible"])
            for key, value in metrics.items():
                if key.startswith("service_"):
                    service_counts[int(key.split("_")[1])] += float(value)

    result: dict[str, float | str] = {
        "deployment": deployment_mode,
        "policy": policy,
        "episodes": float(episodes),
        "fast_slots": float(fast_slots),
        "avg_total_delay": float(np.mean(totals)),
        "std_total_delay": float(np.std(totals)),
        "avg_compute_delay": float(np.mean(comp)),
        "avg_transmission_delay": float(np.mean(tran)),
        "avg_requests_per_slot": float(np.mean(reqs)),
        "infeasible_rate": float(np.mean(infeasible)),
        "fixed_trace": "yes" if request_trace is not None else "no",
        **dep_summary,
    }
    total_services = max(float(service_counts.sum()), 1.0)
    for i in range(service_counts.shape[0]):
        result[f"service_{i}_share"] = float(service_counts[i] / total_services)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--deployment-mode", default="all", choices=DEPLOYMENT_MODES + ["all"])
    parser.add_argument("--policy", default="ppo", choices=["ppo", "greedy_delta", "lookahead_delta"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-fixed-trace", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["training"]["fast_slots_per_episode"])
    modes = DEPLOYMENT_MODES if args.deployment_mode == "all" else [args.deployment_mode]
    request_trace = None if args.no_fixed_trace else build_request_trace(cfg, args.episodes, fast_slots)
    if request_trace is not None:
        avg_reqs = np.mean([len(slot) for slot in request_trace])
        print(f"using fixed request trace: slots={len(request_trace)}, avg requests/slot={avg_reqs:.2f}")

    rows = []
    for mode in modes:
        if args.model_path and len(modes) > 1:
            raise ValueError("--model-path can only be used with a single --deployment-mode")
        result = evaluate_one(cfg, mode, args.policy, args.episodes, fast_slots, args.model_path, request_trace)
        rows.append(result)
        print(
            "[{deployment}/{policy}] delay={avg_total_delay:.4f}±{std_total_delay:.4f} "
            "comp={avg_compute_delay:.4f} tran={avg_transmission_delay:.4f} req/slot={avg_requests_per_slot:.2f}".format(
                **result
            )
        )

    run_name = cfg.get("run_name", "v1")
    out_path = (
        Path(args.output)
        if args.output
        else Path("outputs") / "logs" / f"{run_name}_eval_{args.policy}_{args.deployment_mode}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
