from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.deployment_world_model import DeploymentWorldModel
from edge_sim.env.dynamic_deployment import deployment_change_count
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.evaluate_agent_s import load_agent_s, run_agent_s_policy_slot
from edge_sim.evaluation.evaluate_wms_gnn import load_gnn_wms, run_gnn_wms_planner_slot
from edge_sim.evaluation.policies import run_greedy_delta_slot, run_lookahead_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator
from edge_sim.training.simulate_v2 import observed_source_service_demand
from edge_sim.training.wmd_utils import CANDIDATE_NAMES, candidate_pool, encode_candidate_features


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wmd(checkpoint_path: str | Path, device: torch.device) -> tuple[DeploymentWorldModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeploymentWorldModel(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        output_dim=len(ckpt["target_columns"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_fast_evaluator(
    fast_policy: str,
    checkpoint_path: str | Path | None,
    device_name: str,
) -> tuple[object | None, torch.device]:
    device = torch.device(device_name)
    if fast_policy != "gnn_wms":
        if fast_policy != "agent_s":
            return None, device
        if checkpoint_path is None:
            raise ValueError("fast-policy checkpoint required for agent_s evaluation.")
        return load_agent_s(checkpoint_path, device), device
    if checkpoint_path is None:
        raise ValueError("fast-policy checkpoint required for gnn_wms evaluation.")
    return load_gnn_wms(checkpoint_path, device), device


def evaluate_epoch_cost(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    batches: list,
    epoch: int,
    migration_cost: float,
    fast_policy: str,
    loaded_fast_model: object | None,
    future_weight: float,
    wm_margin: float,
    device: torch.device,
    agent_s_min_prob: float = 0.0,
    agent_s_min_margin: float = 0.0,
    agent_s_fallback: str = "none",
) -> tuple[float, float, float]:
    delay_sum = 0.0
    if fast_policy == "greedy_delta":
        for batch in batches:
            delay_sum += run_greedy_delta_slot(env, allocator, deployment, requests=batch, slow_epoch=epoch)["total_delay"]
    elif fast_policy == "lookahead_delta":
        for batch in batches:
            delay_sum += run_lookahead_delta_slot(
                env,
                allocator,
                deployment,
                requests=batch,
                slow_epoch=epoch,
                lookahead_depth=2,
                future_weight=future_weight,
            )["total_delay"]
    elif fast_policy == "gnn_wms":
        if loaded_fast_model is None:
            raise RuntimeError("GNN fast evaluator requested but no model was loaded.")
        model, ckpt = loaded_fast_model
        for batch in batches:
            delay_sum += run_gnn_wms_planner_slot(
                env,
                allocator,
                deployment,
                model,
                ckpt,
                requests=batch,
                slow_epoch=epoch,
                future_weight=future_weight,
                score_mode="exact_delta_pred_future",
                wm_margin=wm_margin,
                device=device,
            )["total_delay"]
    elif fast_policy == "agent_s":
        if loaded_fast_model is None:
            raise RuntimeError("Agent-S evaluator requested but no model was loaded.")
        model, ckpt = loaded_fast_model
        for batch in batches:
            delay_sum += run_agent_s_policy_slot(
                env,
                allocator,
                deployment,
                model,
                ckpt,
                requests=batch,
                slow_epoch=epoch,
                device=device,
                min_prob=agent_s_min_prob,
                min_margin=agent_s_min_margin,
                fallback_mode=agent_s_fallback,
            )["total_delay"]
    else:
        raise ValueError(f"Unsupported fast_policy={fast_policy!r}")
    return float(delay_sum + migration_cost), float(delay_sum), float(migration_cost)


def evaluate(
    cfg: dict,
    checkpoint_path: str | Path,
    fast_policy: str,
    fast_checkpoint_path: str | Path | None,
    episodes: int,
    fast_slots: int,
    future_weight: float,
    wm_margin: float,
    eval_seed: int,
    device_name: str,
    output_path: str | Path | None,
    agent_s_min_prob: float = 0.0,
    agent_s_min_margin: float = 0.0,
    agent_s_fallback: str = "none",
) -> tuple[Path, dict[str, dict[str, float]]]:
    cfg = dict(cfg)
    cfg["seed"] = int(eval_seed)
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))
    device = torch.device(device_name)
    wmd_model, wmd_ckpt = load_wmd(checkpoint_path, device)
    loaded_fast_model, fast_device = load_fast_evaluator(fast_policy, fast_checkpoint_path, device_name)

    trace = [[env.sample_requests(slow_epoch=epoch) for _ in range(fast_slots)] for epoch in range(episodes)]
    bootstrap_keep_previous_x = env.make_deployment("heuristic")
    previous_x = None
    previous_observed_matrix = None
    older_observed_matrix = None

    if output_path is None:
        output_path = Path("outputs") / "logs" / f"{cfg.get('run_name', 'run')}_wmd_eval.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies = ["wmd_agent", "keep_previous", "history_keep", "trend_keep"]
    totals: dict[str, list[float]] = {name: [] for name in policies}
    policy_state = {
        name: {
            "previous_x": None,
            "previous_observed_matrix": None,
            "older_observed_matrix": None,
        }
        for name in policies
    }
    fieldnames = ["epoch", "policy", "total_cost", "delay_sum", "migration_cost", "chosen_candidate", "pred_total_cost"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch, batches in enumerate(trace):
            for policy_name in policies:
                state_for_policy = policy_state[policy_name]
                previous_x = state_for_policy["previous_x"]
                previous_observed_matrix = state_for_policy["previous_observed_matrix"]
                older_observed_matrix = state_for_policy["older_observed_matrix"]
                pool, state = candidate_pool(env, previous_x, previous_observed_matrix, older_observed_matrix, dyn_cfg)
                if previous_x is None:
                    pool["keep_previous"] = bootstrap_keep_previous_x.copy()

                feature_rows = np.stack(
                    [encode_candidate_features(env, previous_x, pool[name], name, state) for name in CANDIDATE_NAMES],
                    axis=0,
                ).astype(np.float32)
                feature_mean = np.asarray(wmd_ckpt["feature_mean"], dtype=np.float32).reshape(1, -1)
                feature_std = np.asarray(wmd_ckpt["feature_std"], dtype=np.float32).reshape(1, -1)
                target_mean = np.asarray(wmd_ckpt["target_mean"], dtype=np.float32).reshape(1, -1)
                target_std = np.asarray(wmd_ckpt["target_std"], dtype=np.float32).reshape(1, -1)
                x_norm = (feature_rows - feature_mean) / feature_std
                with torch.no_grad():
                    pred_norm = wmd_model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
                pred = pred_norm * target_std + target_mean
                pred_total_cost = pred[:, 0]

                if policy_name == "wmd_agent":
                    chosen_candidate = CANDIDATE_NAMES[int(np.argmin(pred_total_cost))]
                elif policy_name == "keep_previous":
                    chosen_candidate = "keep_previous"
                elif policy_name == "history_keep":
                    chosen_candidate = "history_keep"
                elif policy_name == "trend_keep":
                    chosen_candidate = "trend_keep"
                else:
                    raise ValueError(f"Unsupported policy_name={policy_name!r}")

                x = pool[chosen_candidate]
                migration_changes = deployment_change_count(previous_x, x)
                migration_cost = migration_weight * migration_changes
                total_cost, delay_sum, migration_cost = evaluate_epoch_cost(
                    env,
                    allocator,
                    x,
                    batches,
                    epoch,
                    migration_cost,
                    fast_policy,
                    loaded_fast_model,
                    future_weight,
                    wm_margin,
                    fast_device,
                    agent_s_min_prob=agent_s_min_prob,
                    agent_s_min_margin=agent_s_min_margin,
                    agent_s_fallback=agent_s_fallback,
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "policy": policy_name,
                        "total_cost": total_cost,
                        "delay_sum": delay_sum,
                        "migration_cost": migration_cost,
                        "chosen_candidate": chosen_candidate,
                        "pred_total_cost": float(pred_total_cost[CANDIDATE_NAMES.index(chosen_candidate)]),
                    }
                )
                totals[policy_name].append(total_cost)

                state_for_policy["previous_x"] = x
                state_for_policy["older_observed_matrix"] = previous_observed_matrix
                state_for_policy["previous_observed_matrix"] = observed_source_service_demand(
                    batches, env.num_nodes, env.num_services
                )

    summary = {
        policy: {
            "avg_total_cost": float(np.mean(values)) if values else 0.0,
            "std_total_cost": float(np.std(values)) if values else 0.0,
        }
        for policy, values in totals.items()
    }
    return output_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--checkpoint", default="outputs/wmd/v2_drift_gnn_wms_dataset_model.pt")
    parser.add_argument("--fast-policy", default="gnn_wms", choices=["greedy_delta", "lookahead_delta", "gnn_wms", "agent_s"])
    parser.add_argument("--fast-checkpoint", default="outputs/wms/v2_drift_gnn_wms_mixed_hard_model.pt")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--agent-s-min-prob", type=float, default=0.0)
    parser.add_argument("--agent-s-min-margin", type=float, default=0.05)
    parser.add_argument("--agent-s-fallback", default="greedy", choices=["none", "greedy"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seed = int(args.eval_seed if args.eval_seed is not None else int(cfg["seed"]) + 2000)
    fast_checkpoint = None if args.fast_policy not in {"gnn_wms", "agent_s"} else args.fast_checkpoint
    output_path, summary = evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        fast_policy=args.fast_policy,
        fast_checkpoint_path=fast_checkpoint,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        future_weight=float(args.future_weight),
        wm_margin=float(args.wm_margin),
        eval_seed=eval_seed,
        device_name=args.device,
        output_path=args.output,
        agent_s_min_prob=float(args.agent_s_min_prob),
        agent_s_min_margin=float(args.agent_s_min_margin),
        agent_s_fallback=args.agent_s_fallback,
    )
    print("WM-D evaluation")
    for policy, metrics in summary.items():
        print(
            "{} avg_total_cost={:.6f} std_total_cost={:.6f}".format(
                policy,
                metrics["avg_total_cost"],
                metrics["std_total_cost"],
            )
        )
    print(f"log={output_path}")


if __name__ == "__main__":
    main()
