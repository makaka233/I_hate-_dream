from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.agents.scheduler_world_model import SchedulerGNNWorldModel
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.policies import run_greedy_delta_slot, run_lookahead_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator, add_stage_to_load, kkt_load_cost


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gnn_wms(checkpoint_path: str | Path, device: torch.device) -> tuple[SchedulerGNNWorldModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SchedulerGNNWorldModel(
        hidden_dim=int(ckpt["hidden_dim"]),
        heads=int(ckpt["heads"]),
        layers=int(ckpt["layers"]),
        request_dim=int(ckpt["request_dim"]),
        output_dim=len(ckpt["target_columns"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _normalise_obs(obs: dict, normalizers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "node_feat": ((obs["node_feat"][None, :, :] - normalizers["node_mean"]) / normalizers["node_std"]).astype(
            np.float32
        ),
        "edge_attr": ((obs["edge_attr"][None, :, :] - normalizers["edge_mean"]) / normalizers["edge_std"]).astype(
            np.float32
        ),
        "candidate_edge_attr": (
            (obs["candidate_edge_attr"][None, :, :] - normalizers["candidate_edge_mean"])
            / normalizers["candidate_edge_std"]
        ).astype(np.float32),
        "request_feat": ((obs["request_feat"][None, :] - normalizers["request_mean"]) / normalizers["request_std"]).astype(
            np.float32
        ),
        "prev_node": np.asarray([int(obs["prev_node"])], dtype=np.int64),
    }


def run_gnn_wms_planner_slot(
    env: EdgeEnv,
    allocator: KKTAllocator,
    deployment: np.ndarray,
    model: SchedulerGNNWorldModel,
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
    edge_index = torch.from_numpy(env.edge_index.astype(np.int64)).to(device)
    target_columns = [str(x) for x in ckpt["target_columns"]]
    future_cost_idx = target_columns.index("future_cost")
    target_score_idx = target_columns.index("target_score")
    normalizers = ckpt["normalizers"]
    target_mean = normalizers["target_mean"].reshape(1, 1, -1)
    target_std = normalizers["target_std"].reshape(1, 1, -1)

    for req in requests:
        path: list[int] = []
        prev = int(req.source_node)
        for stage_idx in range(req.num_stages):
            obs = env.graph_observation(deployment, gamma, link_load, req, stage_idx, prev)
            legal_nodes = np.flatnonzero(obs["legal_mask"])
            if legal_nodes.size == 0:
                raise RuntimeError(
                    f"No legal node for request={req.request_id}, service={req.service_id}, stage={stage_idx}."
                )

            norm_obs = _normalise_obs(obs, normalizers)
            with torch.no_grad():
                pred_norm = model(
                    torch.from_numpy(norm_obs["node_feat"]).to(device),
                    edge_index,
                    torch.from_numpy(norm_obs["edge_attr"]).to(device),
                    torch.from_numpy(norm_obs["candidate_edge_attr"]).to(device),
                    torch.from_numpy(norm_obs["request_feat"]).to(device),
                    torch.from_numpy(norm_obs["prev_node"]).to(device),
                ).cpu().numpy()
            pred = pred_norm * target_std + target_mean
            pred = pred[0]

            current_deltas = []
            for node in legal_nodes:
                gamma_after = gamma.copy()
                link_after = link_load.copy()
                add_stage_to_load(gamma_after, link_after, req, stage_idx, prev, int(node))
                current_deltas.append(float(allocator.incremental_cost(gamma, link_load, gamma_after, link_after)))
            current_deltas = np.asarray(current_deltas, dtype=np.float32)

            if score_mode == "exact_delta_pred_future":
                score = current_deltas + future_weight * pred[legal_nodes, future_cost_idx]
            elif score_mode == "pred_target_score":
                score = pred[legal_nodes, target_score_idx]
            elif score_mode == "pred_future_only":
                score = pred[legal_nodes, future_cost_idx]
            else:
                raise ValueError(f"Unsupported score_mode={score_mode!r}")

            chosen_idx = int(np.argmin(score))
            if wm_margin > 0.0:
                greedy_idx = int(np.argmin(current_deltas))
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
    model, ckpt = load_gnn_wms(checkpoint_path, device)

    run_name = cfg.get("run_name", "run")
    if output_path is None:
        output_path = Path("outputs") / "logs" / f"{run_name}_gnn_wms_compare_{deployment_mode}.csv"
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
    totals: dict[str, list[float]] = {"greedy_delta": [], "lookahead_delta": [], "gnn_wms_planner": []}

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
                    "gnn_wms_planner": run_gnn_wms_planner_slot(
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
    parser.add_argument("--checkpoint", default="outputs/wms/v2_drift_heuristic_lookahead_delta_d2_gnn_model.pt")
    parser.add_argument("--deployment-mode", default=None, choices=["heuristic", "fixed", "random", "monolithic"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fast-slots", type=int, default=20)
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--future-weight", type=float, default=0.8)
    parser.add_argument(
        "--score-mode",
        default="exact_delta_pred_future",
        choices=["exact_delta_pred_future", "pred_target_score", "pred_future_only"],
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
    print("GNN WM-S planner evaluation")
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
