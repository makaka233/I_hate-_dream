from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.scheduler_world_model import SchedulerWorldModel
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.policies import run_greedy_delta_slot, run_lookahead_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load, kkt_load_cost
from edge_sim.training.build_wms_dataset import FEATURE_COLUMNS, _candidate_features


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wms(checkpoint_path: str | Path, device: torch.device) -> tuple[SchedulerWorldModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SchedulerWorldModel(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        output_dim=len(ckpt["target_columns"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_wms_planner_slot(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    model: SchedulerWorldModel,
    ckpt: dict,
    requests: list,
    slow_epoch: int,
    future_weight: float,
    score_mode: str,
    wm_margin: float,
    device: torch.device,
) -> dict[str, float]:
    gamma = np.zeros(env.num_nodes, dtype=np.float32)
    link_load = np.zeros((env.num_nodes, env.num_nodes), dtype=np.float32)
    schedules: dict[int, list[int]] = {}

    feature_mean = ckpt["feature_mean"].astype(np.float32)
    feature_std = ckpt["feature_std"].astype(np.float32)
    target_mean = ckpt["target_mean"].astype(np.float32)
    target_std = ckpt["target_std"].astype(np.float32)
    target_columns = [str(x) for x in ckpt["target_columns"]]
    target_score_idx = target_columns.index("target_score")
    future_cost_idx = target_columns.index("future_cost")
    checkpoint_features = [str(x) for x in ckpt["feature_columns"]]
    if checkpoint_features != FEATURE_COLUMNS:
        raise RuntimeError("WM-S checkpoint feature columns do not match the current feature builder.")

    for req in requests:
        path: list[int] = []
        prev = int(req.source_node)
        for stage_idx in range(req.num_stages):
            legal = env.legal_nodes(deployment, req.service_id, stage_idx, prev)
            legal_nodes = np.flatnonzero(legal)
            if legal_nodes.size == 0:
                raise RuntimeError(
                    f"No legal node for request={req.request_id}, service={req.service_id}, stage={stage_idx}."
                )

            feature_rows = []
            current_deltas = []
            for node in legal_nodes:
                gamma_after = gamma.copy()
                link_after = link_load.copy()
                add_stage_to_load(gamma_after, link_after, req, stage_idx, prev, int(node))
                current_deltas.append(float(allocator.incremental_cost(gamma, link_load, gamma_after, link_after)))
                features = _candidate_features(
                    env,
                    deployment,
                    gamma,
                    link_load,
                    req,
                    stage_idx,
                    prev,
                    int(node),
                    int(legal_nodes.size),
                    slow_epoch,
                )
                feature_rows.append([float(features[name]) for name in FEATURE_COLUMNS])

            x = np.asarray(feature_rows, dtype=np.float32)
            x_norm = (x - feature_mean) / feature_std
            with torch.no_grad():
                pred_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
            pred = pred_norm * target_std + target_mean
            if score_mode == "pred_target_score":
                score = pred[:, target_score_idx]
            elif score_mode == "exact_delta_pred_future":
                score = np.asarray(current_deltas, dtype=np.float32) + future_weight * pred[:, future_cost_idx]
            else:
                raise ValueError(f"Unsupported score_mode={score_mode!r}")
            chosen_idx = int(np.argmin(score))
            if wm_margin > 0.0:
                greedy_idx = int(np.argmin(np.asarray(current_deltas, dtype=np.float32)))
                if score[chosen_idx] + wm_margin >= score[greedy_idx]:
                    chosen_idx = greedy_idx
            chosen = int(legal_nodes[chosen_idx])

            add_stage_to_load(gamma, link_load, req, stage_idx, prev, chosen)
            path.append(chosen)
            prev = chosen
        schedules[int(req.request_id)] = path

    allocation = allocator.allocate(requests, schedules)
    metrics = {
        "requests": float(len(requests)),
        "stages": float(sum(len(path) for path in schedules.values())),
        "total_delay": allocation.total_delay,
        "compute_delay": allocation.compute_delay,
        "transmission_delay": allocation.transmission_delay,
        "kkt_virtual_cost": kkt_load_cost(gamma, link_load, env.compute_cap, env.effective_bandwidth),
        "infeasible": float(allocation.infeasible),
    }
    for i in range(env.num_services):
        metrics[f"service_{i}"] = float(sum(1 for req in requests if req.service_id == i))
    return metrics


def evaluate(
    cfg: dict,
    deployment_mode: str,
    checkpoint_path: str | Path,
    episodes: int,
    fast_slots: int,
    lookahead_depth: int,
    future_weight: float,
    score_mode: str,
    wm_margin: float,
    eval_seed: int,
    device_name: str,
    output_path: str | Path | None,
) -> tuple[Path, dict[str, dict[str, float]]]:
    cfg = dict(cfg)
    cfg["seed"] = int(eval_seed)
    cfg["deployment"] = dict(cfg["deployment"])
    cfg["deployment"]["mode"] = deployment_mode
    env = EdgeEnv(cfg)
    deployment = env.make_deployment(deployment_mode)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    device = torch.device(device_name)
    model, ckpt = load_wms(checkpoint_path, device)

    run_name = cfg.get("run_name", "run")
    if output_path is None:
        output_path = Path("outputs") / "logs" / f"{run_name}_wms_compare_{deployment_mode}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "episode",
        "fast_slot",
        "policy",
        "requests",
        "stages",
        "total_delay",
        "compute_delay",
        "transmission_delay",
        "kkt_virtual_cost",
        "infeasible",
    ]
    fieldnames.extend([f"service_{i}" for i in range(env.num_services)])
    totals: dict[str, list[float]] = {"greedy_delta": [], "lookahead_delta": [], "wms_planner": []}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(episodes):
            for slot in range(fast_slots):
                requests = env.sample_requests(slow_epoch=ep)
                policy_metrics = {
                    "greedy_delta": run_greedy_delta_slot(env, allocator, deployment, requests=requests, slow_epoch=ep),
                    "lookahead_delta": run_lookahead_delta_slot(
                        env,
                        allocator,
                        deployment,
                        requests=requests,
                        slow_epoch=ep,
                        lookahead_depth=lookahead_depth,
                        future_weight=future_weight,
                    ),
                    "wms_planner": run_wms_planner_slot(
                        env,
                        allocator,
                        deployment,
                        model,
                        ckpt,
                        requests=requests,
                        slow_epoch=ep,
                        future_weight=future_weight,
                        score_mode=score_mode,
                        wm_margin=wm_margin,
                        device=device,
                    ),
                }
                for policy, metrics in policy_metrics.items():
                    row = {"episode": ep, "fast_slot": slot, "policy": policy}
                    row.update(metrics)
                    writer.writerow(row)
                    totals[policy].append(float(metrics["total_delay"]))

    summary = {
        policy: {
            "avg_total_delay": float(np.mean(values)) if values else 0.0,
            "std_total_delay": float(np.std(values)) if values else 0.0,
        }
        for policy, values in totals.items()
    }
    return output_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--checkpoint", default="outputs/wms/v2_drift_heuristic_lookahead_delta_d2_model.pt")
    parser.add_argument("--deployment-mode", default=None, choices=["heuristic", "fixed", "random", "monolithic"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fast-slots", type=int, default=20)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--future-weight", type=float, default=0.8)
    parser.add_argument(
        "--score-mode",
        default="exact_delta_pred_future",
        choices=["exact_delta_pred_future", "pred_target_score"],
    )
    parser.add_argument("--wm-margin", type=float, default=0.0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    deployment_mode = args.deployment_mode or cfg["deployment"]["mode"]
    eval_seed = int(args.eval_seed if args.eval_seed is not None else int(cfg["seed"]) + 1000)
    output_path, summary = evaluate(
        cfg=cfg,
        deployment_mode=deployment_mode,
        checkpoint_path=args.checkpoint,
        episodes=int(args.episodes),
        fast_slots=int(args.fast_slots),
        lookahead_depth=int(args.lookahead_depth),
        future_weight=float(args.future_weight),
        score_mode=args.score_mode,
        wm_margin=float(args.wm_margin),
        eval_seed=eval_seed,
        device_name=args.device,
        output_path=args.output,
    )
    print("WM-S planner evaluation")
    for policy, metrics in summary.items():
        print(
            "{} avg_total_delay={:.6f} std_total_delay={:.6f}".format(
                policy,
                metrics["avg_total_delay"],
                metrics["std_total_delay"],
            )
        )
    print(f"log={output_path}")


if __name__ == "__main__":
    main()
