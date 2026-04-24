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


KEEP_PREVIOUS_ID = CANDIDATE_NAMES.index("keep_previous")


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


def _required_pred_gain(
    candidate_name: str,
    base_gain: float,
    refresh_extra_gain: float,
    aggressive_extra_gain: float,
) -> float:
    threshold = float(base_gain)
    if candidate_name != "keep_previous" and "refresh" in candidate_name:
        threshold += float(refresh_extra_gain)
    if "aggressive" in candidate_name:
        threshold += float(aggressive_extra_gain)
    return threshold


def _top_margin(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 1.0
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])


def _select_guarded_agentd_candidate(
    pred_total_cost: np.ndarray,
    agentd_logits: np.ndarray,
    agentd_probs: np.ndarray,
    min_prob: float,
    min_margin: float,
    base_gain: float,
    refresh_extra_gain: float,
    aggressive_extra_gain: float,
    max_wmd_gap: float,
    fallback_mode: str,
    fallback_gain: float,
) -> tuple[int, dict[str, float | str]]:
    keep_idx = KEEP_PREVIOUS_ID
    raw_idx = int(np.argmax(agentd_logits))
    wmd_idx = int(np.argmin(pred_total_cost))
    raw_name = CANDIDATE_NAMES[raw_idx]
    raw_prob = float(agentd_probs[raw_idx])
    prob_margin = _top_margin(agentd_probs)
    raw_pred_gain = float(pred_total_cost[keep_idx] - pred_total_cost[raw_idx])
    wmd_pred_gain = float(pred_total_cost[keep_idx] - pred_total_cost[wmd_idx])
    required_gain = _required_pred_gain(raw_name, base_gain, refresh_extra_gain, aggressive_extra_gain)
    wmd_gap = float(pred_total_cost[raw_idx] - pred_total_cost[wmd_idx])

    accept = True
    reason = "accepted"
    if raw_idx == keep_idx:
        reason = "raw_keep_previous"
    else:
        if raw_prob < float(min_prob):
            accept = False
            reason = "low_prob"
        elif prob_margin < float(min_margin):
            accept = False
            reason = "low_margin"
        elif raw_pred_gain < required_gain:
            accept = False
            reason = "low_pred_gain"
        elif wmd_gap > float(max_wmd_gap):
            accept = False
            reason = "far_from_wmd"

    chosen_idx = raw_idx
    decision_source = "agentd_raw"
    if not accept:
        if fallback_mode == "wmd_if_gain" and wmd_idx != keep_idx and wmd_pred_gain >= float(fallback_gain):
            chosen_idx = wmd_idx
            decision_source = "wmd_fallback"
        else:
            chosen_idx = keep_idx
            decision_source = "keep_previous_fallback"

    return chosen_idx, {
        "agentd_raw_candidate": raw_name,
        "agentd_raw_prob": raw_prob,
        "agentd_prob_margin": prob_margin,
        "agentd_raw_pred_gain": raw_pred_gain,
        "agentd_required_gain": required_gain,
        "wmd_best_candidate": CANDIDATE_NAMES[wmd_idx],
        "wmd_best_pred_gain": wmd_pred_gain,
        "agentd_wmd_gap": wmd_gap,
        "guard_reason": reason,
        "guard_decision_source": decision_source,
        "guard_triggered": float(chosen_idx != raw_idx),
    }


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
    agentd_min_prob: float,
    agentd_min_margin: float,
    agentd_base_gain: float,
    agentd_refresh_extra_gain: float,
    agentd_aggressive_extra_gain: float,
    agentd_max_wmd_gap: float,
    agentd_fallback_mode: str,
    agentd_fallback_gain: float,
    agent_s_min_prob: float,
    agent_s_min_margin: float,
    agent_s_fallback: str,
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
    bootstrap_keep_previous_x = env.make_deployment("heuristic")

    if output_path is None:
        output_path = Path("outputs") / "logs" / f"{cfg.get('run_name', 'run')}_dual_agent_eval.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policies = [
        "dual_wmd_wms",
        "dual_agentd_wms",
        "dual_agentd_guarded_wms",
        "keep_previous_wms",
        "history_keep_wms",
        "trend_keep_wms",
    ]
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
        "agentd_raw_candidate",
        "agentd_raw_prob",
        "agentd_prob_margin",
        "agentd_raw_pred_gain",
        "agentd_required_gain",
        "wmd_best_candidate",
        "wmd_best_pred_gain",
        "agentd_wmd_gap",
        "guard_reason",
        "guard_decision_source",
        "guard_triggered",
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
                if previous_x is None:
                    pool["keep_previous"] = bootstrap_keep_previous_x.copy()

                feature_rows = np.stack(
                    [encode_candidate_features(env, previous_x, pool[name], name, state) for name in CANDIDATE_NAMES],
                    axis=0,
                ).astype(np.float32)
                pred_total_cost = _score_wmd_candidates(wmd_model, wmd_ckpt, feature_rows, device)
                agentd_logits, agentd_probs = _score_agentd_candidates(agentd_model, agentd_ckpt, feature_rows, device)

                if policy_name == "dual_wmd_wms":
                    chosen_idx = int(np.argmin(pred_total_cost))
                    guard_meta = {
                        "agentd_raw_candidate": CANDIDATE_NAMES[int(np.argmax(agentd_logits))],
                        "agentd_raw_prob": float(agentd_probs[int(np.argmax(agentd_logits))]),
                        "agentd_prob_margin": _top_margin(agentd_probs),
                        "agentd_raw_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmax(agentd_logits))]),
                        "agentd_required_gain": 0.0,
                        "wmd_best_candidate": CANDIDATE_NAMES[chosen_idx],
                        "wmd_best_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[chosen_idx]),
                        "agentd_wmd_gap": float(pred_total_cost[int(np.argmax(agentd_logits))] - pred_total_cost[chosen_idx]),
                        "guard_reason": "wmd_planner",
                        "guard_decision_source": "wmd_planner",
                        "guard_triggered": 0.0,
                    }
                elif policy_name == "dual_agentd_wms":
                    chosen_idx = int(np.argmax(agentd_logits))
                    guard_meta = {
                        "agentd_raw_candidate": CANDIDATE_NAMES[chosen_idx],
                        "agentd_raw_prob": float(agentd_probs[chosen_idx]),
                        "agentd_prob_margin": _top_margin(agentd_probs),
                        "agentd_raw_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[chosen_idx]),
                        "agentd_required_gain": 0.0,
                        "wmd_best_candidate": CANDIDATE_NAMES[int(np.argmin(pred_total_cost))],
                        "wmd_best_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "agentd_wmd_gap": float(pred_total_cost[chosen_idx] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "guard_reason": "agentd_raw",
                        "guard_decision_source": "agentd_raw",
                        "guard_triggered": 0.0,
                    }
                elif policy_name == "dual_agentd_guarded_wms":
                    chosen_idx, guard_meta = _select_guarded_agentd_candidate(
                        pred_total_cost,
                        agentd_logits,
                        agentd_probs,
                        agentd_min_prob,
                        agentd_min_margin,
                        agentd_base_gain,
                        agentd_refresh_extra_gain,
                        agentd_aggressive_extra_gain,
                        agentd_max_wmd_gap,
                        agentd_fallback_mode,
                        agentd_fallback_gain,
                    )
                elif policy_name == "keep_previous_wms":
                    chosen_idx = CANDIDATE_NAMES.index("keep_previous")
                    guard_meta = {
                        "agentd_raw_candidate": CANDIDATE_NAMES[int(np.argmax(agentd_logits))],
                        "agentd_raw_prob": float(agentd_probs[int(np.argmax(agentd_logits))]),
                        "agentd_prob_margin": _top_margin(agentd_probs),
                        "agentd_raw_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmax(agentd_logits))]),
                        "agentd_required_gain": 0.0,
                        "wmd_best_candidate": CANDIDATE_NAMES[int(np.argmin(pred_total_cost))],
                        "wmd_best_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "agentd_wmd_gap": float(pred_total_cost[int(np.argmax(agentd_logits))] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "guard_reason": "forced_keep_previous",
                        "guard_decision_source": "forced_keep_previous",
                        "guard_triggered": 0.0,
                    }
                elif policy_name == "history_keep_wms":
                    chosen_idx = CANDIDATE_NAMES.index("history_keep")
                    guard_meta = {
                        "agentd_raw_candidate": CANDIDATE_NAMES[int(np.argmax(agentd_logits))],
                        "agentd_raw_prob": float(agentd_probs[int(np.argmax(agentd_logits))]),
                        "agentd_prob_margin": _top_margin(agentd_probs),
                        "agentd_raw_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmax(agentd_logits))]),
                        "agentd_required_gain": 0.0,
                        "wmd_best_candidate": CANDIDATE_NAMES[int(np.argmin(pred_total_cost))],
                        "wmd_best_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "agentd_wmd_gap": float(pred_total_cost[int(np.argmax(agentd_logits))] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "guard_reason": "forced_history_keep",
                        "guard_decision_source": "forced_history_keep",
                        "guard_triggered": 0.0,
                    }
                elif policy_name == "trend_keep_wms":
                    chosen_idx = CANDIDATE_NAMES.index("trend_keep")
                    guard_meta = {
                        "agentd_raw_candidate": CANDIDATE_NAMES[int(np.argmax(agentd_logits))],
                        "agentd_raw_prob": float(agentd_probs[int(np.argmax(agentd_logits))]),
                        "agentd_prob_margin": _top_margin(agentd_probs),
                        "agentd_raw_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmax(agentd_logits))]),
                        "agentd_required_gain": 0.0,
                        "wmd_best_candidate": CANDIDATE_NAMES[int(np.argmin(pred_total_cost))],
                        "wmd_best_pred_gain": float(pred_total_cost[KEEP_PREVIOUS_ID] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "agentd_wmd_gap": float(pred_total_cost[int(np.argmax(agentd_logits))] - pred_total_cost[int(np.argmin(pred_total_cost))]),
                        "guard_reason": "forced_trend_keep",
                        "guard_decision_source": "forced_trend_keep",
                        "guard_triggered": 0.0,
                    }
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
                        "wmd_pred_total_cost": float(pred_total_cost[chosen_idx]),
                        "agentd_logit": float(agentd_logits[chosen_idx]),
                        "agentd_prob": float(agentd_probs[chosen_idx]),
                        **guard_meta,
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
    parser.add_argument("--fast-policy", default="gnn_wms", choices=["greedy_delta", "lookahead_delta", "gnn_wms", "agent_s"])
    parser.add_argument("--fast-checkpoint", default="outputs/wms/v2_drift_gnn_wms_mixed_hard_model.pt")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--agentd-min-prob", type=float, default=0.22)
    parser.add_argument("--agentd-min-margin", type=float, default=0.05)
    parser.add_argument("--agentd-base-gain", type=float, default=0.6)
    parser.add_argument("--agentd-refresh-extra-gain", type=float, default=0.25)
    parser.add_argument("--agentd-aggressive-extra-gain", type=float, default=0.6)
    parser.add_argument("--agentd-max-wmd-gap", type=float, default=0.75)
    parser.add_argument("--agentd-fallback-mode", default="keep_previous", choices=["keep_previous", "wmd_if_gain"])
    parser.add_argument("--agentd-fallback-gain", type=float, default=0.0)
    parser.add_argument("--agent-s-min-prob", type=float, default=0.0)
    parser.add_argument("--agent-s-min-margin", type=float, default=0.05)
    parser.add_argument("--agent-s-fallback", default="greedy", choices=["none", "greedy"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seed = int(args.eval_seed if args.eval_seed is not None else int(cfg["seed"]) + 3000)
    fast_checkpoint = None if args.fast_policy not in {"gnn_wms", "agent_s"} else args.fast_checkpoint
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
        agentd_min_prob=float(args.agentd_min_prob),
        agentd_min_margin=float(args.agentd_min_margin),
        agentd_base_gain=float(args.agentd_base_gain),
        agentd_refresh_extra_gain=float(args.agentd_refresh_extra_gain),
        agentd_aggressive_extra_gain=float(args.agentd_aggressive_extra_gain),
        agentd_max_wmd_gap=float(args.agentd_max_wmd_gap),
        agentd_fallback_mode=args.agentd_fallback_mode,
        agentd_fallback_gain=float(args.agentd_fallback_gain),
        agent_s_min_prob=float(args.agent_s_min_prob),
        agent_s_min_margin=float(args.agent_s_min_margin),
        agent_s_fallback=args.agent_s_fallback,
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
