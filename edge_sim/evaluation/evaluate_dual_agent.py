from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from edge_sim.agents.deployment_policy import DeploymentCandidatePolicy
from edge_sim.evaluation.evaluate_wmd import evaluate_epoch_cost, load_cfg, load_fast_evaluator, load_wmd
from edge_sim.training.simulate_v2 import observed_source_service_demand
from edge_sim.training.wmd_utils import CANDIDATE_NAMES, candidate_pool, encode_candidate_features
from edge_sim.env.dynamic_deployment import deployment_change_count
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.optim.kkt_allocator import KKTAllocator


def load_agent_d(checkpoint_path: str | Path, device: torch.device) -> tuple[DeploymentCandidatePolicy, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeploymentCandidatePolicy(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _score_wmd_candidates(
    model,
    ckpt: dict,
    feature_rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1)
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32).reshape(1, -1)
    x_norm = (feature_rows - feature_mean) / feature_std
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    pred = pred_norm * target_std + target_mean
    return pred[:, 0]


def _score_agentd_candidates(
    model,
    ckpt: dict,
    feature_rows: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32).reshape(1, -1)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32).reshape(1, -1)
    x_norm = (feature_rows - feature_mean) / feature_std
    with torch.no_grad():
        logits = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    return logits.astype(np.float32), probs.astype(np.float32)


def evaluate(
    cfg: dict,
    wmd_checkpoint_path: str | Path,
    agentd_checkpoint_path: str | Path,
    fast_policy: str,
    fast_checkpoint_path: str | Path | None,
    episodes: int,
    fast_slots: int,
    future_weight: float,
    wm_margin: float,
    eval_seed: int,
    device_name: str,
    output_path: str | Path | None,
) -> tuple[Path, dict[str, dict[str, float]]]:
    cfg = dict(cfg)
    cfg["seed"] = int(eval_seed)
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))
    device = torch.device(device_name)

    wmd_model, wmd_ckpt = load_wmd(wmd_checkpoint_path, device)
    agentd_model, agentd_ckpt = load_agent_d(agentd_checkpoint_path, device)
    loaded_fast_model, fast_device = load_fast_evaluator(fast_policy, fast_checkpoint_path, device_name)

    trace = [[env.sample_requests(slow_epoch=epoch) for _ in range(fast_slots)] for epoch in range(episodes)]

    if output_path is None:
        output_path = Path("outputs") / "logs" / f"{cfg.get('run_name', 'run')}_dual_agent_eval.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies = ["dual_wmd_wms", "dual_agentd_wms", "keep_previous_wms", "history_keep_wms", "trend_keep_wms"]
    totals: dict[str, list[float]] = {name: [] for name in policies}
    policy_state = {
        name: {
            "previous_x": None,
            "previous_observed_matrix": None,
            "older_observed_matrix": None,
        }
        for name in policies
    }
    fieldnames = [
        "epoch",
        "policy",
        "total_cost",
        "delay_sum",
        "migration_cost",
        "chosen_candidate",
        "wmd_pred_total_cost",
        "agentd_logit",
        "agentd_prob",
    ]

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

                feature_rows = np.stack(
                    [encode_candidate_features(env, previous_x, pool[name], name, state) for name in CANDIDATE_NAMES],
                    axis=0,
                ).astype(np.float32)
                pred_total_cost = _score_wmd_candidates(wmd_model, wmd_ckpt, feature_rows, device)
                agentd_logits, agentd_probs = _score_agentd_candidates(agentd_model, agentd_ckpt, feature_rows, device)

                if policy_name == "dual_wmd_wms":
                    chosen_idx = int(np.argmin(pred_total_cost))
                elif policy_name == "dual_agentd_wms":
                    chosen_idx = int(np.argmax(agentd_logits))
                elif policy_name == "keep_previous_wms":
                    chosen_idx = CANDIDATE_NAMES.index("keep_previous")
                elif policy_name == "history_keep_wms":
                    chosen_idx = CANDIDATE_NAMES.index("history_keep")
                elif policy_name == "trend_keep_wms":
                    chosen_idx = CANDIDATE_NAMES.index("trend_keep")
                else:
                    raise ValueError(f"Unsupported policy_name={policy_name!r}")

                chosen_candidate = CANDIDATE_NAMES[chosen_idx]
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
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "policy": policy_name,
                        "total_cost": total_cost,
                        "delay_sum": delay_sum,
                        "migration_cost": migration_cost,
                        "chosen_candidate": chosen_candidate,
                        "wmd_pred_total_cost": float(pred_total_cost[chosen_idx]),
                        "agentd_logit": float(agentd_logits[chosen_idx]),
                        "agentd_prob": float(agentd_probs[chosen_idx]),
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
    parser.add_argument("--wmd-checkpoint", default="outputs/wmd/v2_drift_wmd_v2_multiseed_model.pt")
    parser.add_argument("--agentd-checkpoint", default="outputs/agent_d/v2_drift_agent_d_v2_multiseed_model.pt")
    parser.add_argument("--fast-policy", default="gnn_wms", choices=["greedy_delta", "lookahead_delta", "gnn_wms"])
    parser.add_argument("--fast-checkpoint", default="outputs/wms/v2_drift_gnn_wms_mixed_hard_model.pt")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seed = int(args.eval_seed if args.eval_seed is not None else int(cfg["seed"]) + 3000)
    fast_checkpoint = None if args.fast_policy != "gnn_wms" else args.fast_checkpoint
    output_path, summary = evaluate(
        cfg=cfg,
        wmd_checkpoint_path=args.wmd_checkpoint,
        agentd_checkpoint_path=args.agentd_checkpoint,
        fast_policy=args.fast_policy,
        fast_checkpoint_path=fast_checkpoint,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        future_weight=float(args.future_weight),
        wm_margin=float(args.wm_margin),
        eval_seed=eval_seed,
        device_name=args.device,
        output_path=args.output,
    )
    print("Dual-agent evaluation")
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
