from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.evaluate_agent_s import load_agent_s, run_agent_s_policy_slot
from edge_sim.evaluation.evaluate_wms_gnn import load_cfg, load_gnn_wms, run_gnn_wms_planner_slot
from edge_sim.evaluation.policies import run_greedy_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator


def _parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _safe_gap_closing(greedy: float, agent: float, teacher: float) -> float:
    denom = greedy - teacher
    if abs(denom) < 1e-8:
        return 0.0
    return float((greedy - agent) / denom)


def evaluate_combo(
    cfg: dict,
    agent_s_checkpoint: str | Path,
    gnn_wms_checkpoint: str | Path,
    deployment_mode: str,
    eval_seeds: list[int],
    episodes: int,
    fast_slots: int,
    future_weight: float,
    wm_margin: float,
    min_prob: float,
    min_margin: float,
    fallback_mode: str,
    device_name: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | str]]:
    device = torch.device(device_name)
    teacher_model, teacher_ckpt = load_gnn_wms(gnn_wms_checkpoint, device)
    student_model, student_ckpt = load_agent_s(agent_s_checkpoint, device)

    detail_rows: list[dict[str, float | int | str]] = []
    for seed in eval_seeds:
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = int(seed)
        seed_cfg["deployment"] = dict(cfg["deployment"])
        seed_cfg["deployment"]["mode"] = deployment_mode
        env = EdgeEnv(seed_cfg)
        deployment = env.make_deployment(deployment_mode)
        allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)

        greedy_delays: list[float] = []
        teacher_delays: list[float] = []
        agent_delays: list[float] = []
        guard_ratios: list[float] = []
        infeasible_flags: list[float] = []

        for ep in range(episodes):
            for slot in range(fast_slots):
                requests = env.sample_requests(slow_epoch=ep)
                greedy_metrics = run_greedy_delta_slot(env, allocator, deployment, requests=requests, slow_epoch=ep)
                teacher_metrics = run_gnn_wms_planner_slot(
                    env,
                    allocator,
                    deployment,
                    teacher_model,
                    teacher_ckpt,
                    requests=requests,
                    slow_epoch=ep,
                    future_weight=future_weight,
                    score_mode="exact_delta_pred_future",
                    wm_margin=wm_margin,
                    device=device,
                )
                agent_metrics = run_agent_s_policy_slot(
                    env,
                    allocator,
                    deployment,
                    student_model,
                    student_ckpt,
                    requests=requests,
                    slow_epoch=ep,
                    device=device,
                    min_prob=min_prob,
                    min_margin=min_margin,
                    fallback_mode=fallback_mode,
                )
                greedy_delays.append(float(greedy_metrics["total_delay"]))
                teacher_delays.append(float(teacher_metrics["total_delay"]))
                agent_delays.append(float(agent_metrics["total_delay"]))
                guard_ratios.append(float(agent_metrics["guard_ratio"]))
                infeasible_flags.append(float(agent_metrics["infeasible"]))

        greedy_mean = float(np.mean(greedy_delays))
        teacher_mean = float(np.mean(teacher_delays))
        agent_mean = float(np.mean(agent_delays))
        detail_rows.append(
            {
                "seed": int(seed),
                "min_prob": float(min_prob),
                "min_margin": float(min_margin),
                "fallback_mode": fallback_mode,
                "greedy_avg_total_delay": greedy_mean,
                "teacher_avg_total_delay": teacher_mean,
                "agent_avg_total_delay": agent_mean,
                "agent_guard_ratio": float(np.mean(guard_ratios)),
                "agent_infeasible_rate": float(np.mean(infeasible_flags)),
                "improvement_vs_greedy": float((greedy_mean - agent_mean) / max(greedy_mean, 1e-8)),
                "delta_vs_teacher": float(agent_mean - teacher_mean),
                "gap_closing_ratio": _safe_gap_closing(greedy_mean, agent_mean, teacher_mean),
            }
        )

    agent_means = np.asarray([float(row["agent_avg_total_delay"]) for row in detail_rows], dtype=np.float64)
    greedy_means = np.asarray([float(row["greedy_avg_total_delay"]) for row in detail_rows], dtype=np.float64)
    teacher_means = np.asarray([float(row["teacher_avg_total_delay"]) for row in detail_rows], dtype=np.float64)
    gap_rows = np.asarray([float(row["gap_closing_ratio"]) for row in detail_rows], dtype=np.float64)
    summary = {
        "min_prob": float(min_prob),
        "min_margin": float(min_margin),
        "fallback_mode": fallback_mode,
        "num_seeds": int(len(detail_rows)),
        "agent_mean_total_delay": float(agent_means.mean()),
        "agent_std_across_seeds": float(agent_means.std()),
        "teacher_mean_total_delay": float(teacher_means.mean()),
        "greedy_mean_total_delay": float(greedy_means.mean()),
        "win_rate_vs_greedy": float(np.mean(agent_means <= greedy_means)),
        "win_rate_vs_teacher": float(np.mean(agent_means <= teacher_means)),
        "mean_delta_vs_teacher": float((agent_means - teacher_means).mean()),
        "worst_delta_vs_greedy": float((agent_means - greedy_means).max()),
        "mean_gap_closing_ratio": float(gap_rows.mean()),
        "mean_guard_ratio": float(np.mean([float(row["agent_guard_ratio"]) for row in detail_rows])),
        "mean_infeasible_rate": float(np.mean([float(row["agent_infeasible_rate"]) for row in detail_rows])),
    }
    return detail_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--agent-s-checkpoint", default="outputs/agent_s/v2_drift_agent_s_multiseed_model.pt")
    parser.add_argument("--gnn-wms-checkpoint", default="outputs/wms/v2_drift_gnn_wms_mixed_hard_model.pt")
    parser.add_argument("--deployment-mode", default="heuristic", choices=["heuristic", "fixed", "random", "monolithic"])
    parser.add_argument("--eval-seeds", default="1007,1107,1207,1307,1407")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--prob-grid", default="0.0,0.30,0.38,0.46")
    parser.add_argument("--margin-grid", default="0.0,0.03,0.05,0.07")
    parser.add_argument("--fallback-mode", default="greedy", choices=["none", "greedy"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detail-output", default="outputs/logs/agent_s_guard_calibration_detail.csv")
    parser.add_argument("--summary-output", default="outputs/logs/agent_s_guard_calibration_summary.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seeds = _parse_list(args.eval_seeds, int)
    prob_grid = _parse_list(args.prob_grid, float)
    margin_grid = _parse_list(args.margin_grid, float)

    detail_output = Path(args.detail_output)
    summary_output = Path(args.summary_output)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    detail_rows_all: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for min_prob in prob_grid:
        for min_margin in margin_grid:
            detail_rows, summary = evaluate_combo(
                cfg=cfg,
                agent_s_checkpoint=args.agent_s_checkpoint,
                gnn_wms_checkpoint=args.gnn_wms_checkpoint,
                deployment_mode=args.deployment_mode,
                eval_seeds=eval_seeds,
                episodes=int(args.episodes),
                fast_slots=fast_slots,
                future_weight=float(args.future_weight),
                wm_margin=float(args.wm_margin),
                min_prob=float(min_prob),
                min_margin=float(min_margin),
                fallback_mode=args.fallback_mode,
                device_name=args.device,
            )
            detail_rows_all.extend(detail_rows)
            summary_rows.append(summary)
            print(
                "prob={:.2f} margin={:.2f} agent={:.6f} teacher={:.6f} greedy={:.6f} gap_close={:.4f} win_vs_greedy={:.2f}".format(
                    float(min_prob),
                    float(min_margin),
                    float(summary["agent_mean_total_delay"]),
                    float(summary["teacher_mean_total_delay"]),
                    float(summary["greedy_mean_total_delay"]),
                    float(summary["mean_gap_closing_ratio"]),
                    float(summary["win_rate_vs_greedy"]),
                )
            )

    detail_fieldnames = [
        "seed",
        "min_prob",
        "min_margin",
        "fallback_mode",
        "greedy_avg_total_delay",
        "teacher_avg_total_delay",
        "agent_avg_total_delay",
        "agent_guard_ratio",
        "agent_infeasible_rate",
        "improvement_vs_greedy",
        "delta_vs_teacher",
        "gap_closing_ratio",
    ]
    with open(detail_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows_all)

    summary_rows.sort(key=lambda row: (float(row["agent_mean_total_delay"]), float(row["worst_delta_vs_greedy"])))
    summary_fieldnames = [
        "min_prob",
        "min_margin",
        "fallback_mode",
        "num_seeds",
        "agent_mean_total_delay",
        "agent_std_across_seeds",
        "teacher_mean_total_delay",
        "greedy_mean_total_delay",
        "win_rate_vs_greedy",
        "win_rate_vs_teacher",
        "mean_delta_vs_teacher",
        "worst_delta_vs_greedy",
        "mean_gap_closing_ratio",
        "mean_guard_ratio",
        "mean_infeasible_rate",
    ]
    with open(summary_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    best = summary_rows[0]
    print("Best Agent-S guard setting")
    print(
        "min_prob={:.2f} min_margin={:.2f} agent={:.6f} teacher={:.6f} greedy={:.6f} gap_close={:.4f} win_vs_greedy={:.2f} mean_guard_ratio={:.4f}".format(
            float(best["min_prob"]),
            float(best["min_margin"]),
            float(best["agent_mean_total_delay"]),
            float(best["teacher_mean_total_delay"]),
            float(best["greedy_mean_total_delay"]),
            float(best["mean_gap_closing_ratio"]),
            float(best["win_rate_vs_greedy"]),
            float(best["mean_guard_ratio"]),
        )
    )
    print(f"detail_log={detail_output}")
    print(f"summary_log={summary_output}")


if __name__ == "__main__":
    main()
