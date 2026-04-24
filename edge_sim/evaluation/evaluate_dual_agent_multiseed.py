from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from edge_sim.evaluation.evaluate_dual_agent import evaluate
from edge_sim.evaluation.evaluate_wmd import load_cfg


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _extract_guard_stats(log_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(log_path, "r", encoding="utf-8")))
    guarded_rows = [row for row in rows if row["policy"] == "dual_agentd_guarded_wms"]
    if not guarded_rows:
        return {
            "guard_trigger_rate": 0.0,
            "guard_keep_previous_rate": 0.0,
            "guard_wmd_fallback_rate": 0.0,
            "guard_agentd_raw_rate": 0.0,
        }
    count = float(len(guarded_rows))
    return {
        "guard_trigger_rate": float(np.mean([float(row["guard_triggered"]) for row in guarded_rows])),
        "guard_keep_previous_rate": float(
            sum(1 for row in guarded_rows if row["guard_decision_source"] == "keep_previous_fallback") / count
        ),
        "guard_wmd_fallback_rate": float(sum(1 for row in guarded_rows if row["guard_decision_source"] == "wmd_fallback") / count),
        "guard_agentd_raw_rate": float(sum(1 for row in guarded_rows if row["guard_decision_source"] == "agentd_raw") / count),
    }


def run_multiseed_evaluation(
    cfg: dict,
    wmd_checkpoint: str | Path,
    agentd_checkpoint: str | Path,
    fast_policy: str,
    fast_checkpoint: str | Path | None,
    eval_seeds: list[int],
    episodes: int,
    fast_slots: int,
    future_weight: float,
    wm_margin: float,
    device: str,
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
    detail_output: str | Path,
    summary_output: str | Path,
) -> tuple[Path, Path, list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    detail_output = Path(detail_output)
    summary_output = Path(summary_output)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | int | str]] = []
    for seed in eval_seeds:
        per_seed_log = detail_output.parent / f"{detail_output.stem}_seed{seed}.csv"
        _, summary = evaluate(
            cfg=cfg,
            wmd_checkpoint_path=wmd_checkpoint,
            agentd_checkpoint_path=agentd_checkpoint,
            fast_policy=fast_policy,
            fast_checkpoint_path=fast_checkpoint if fast_policy in {"gnn_wms", "agent_s"} else None,
            episodes=int(episodes),
            fast_slots=fast_slots,
            future_weight=float(future_weight),
            wm_margin=float(wm_margin),
            eval_seed=int(seed),
            device_name=device,
            output_path=per_seed_log,
            agentd_min_prob=float(agentd_min_prob),
            agentd_min_margin=float(agentd_min_margin),
            agentd_base_gain=float(agentd_base_gain),
            agentd_refresh_extra_gain=float(agentd_refresh_extra_gain),
            agentd_aggressive_extra_gain=float(agentd_aggressive_extra_gain),
            agentd_max_wmd_gap=float(agentd_max_wmd_gap),
            agentd_fallback_mode=agentd_fallback_mode,
            agentd_fallback_gain=float(agentd_fallback_gain),
            agent_s_min_prob=float(agent_s_min_prob),
            agent_s_min_margin=float(agent_s_min_margin),
            agent_s_fallback=agent_s_fallback,
        )
        guard_stats = _extract_guard_stats(per_seed_log)
        keep_previous_cost = float(summary["keep_previous_wms"]["avg_total_cost"])
        wmd_cost = float(summary["dual_wmd_wms"]["avg_total_cost"])
        for policy, metrics in summary.items():
            row = {
                "seed": int(seed),
                "policy": policy,
                "avg_total_cost": float(metrics["avg_total_cost"]),
                "std_total_cost": float(metrics["std_total_cost"]),
                "delta_vs_keep_previous": float(metrics["avg_total_cost"] - keep_previous_cost),
                "delta_vs_dual_wmd": float(metrics["avg_total_cost"] - wmd_cost),
            }
            if policy == "dual_agentd_guarded_wms":
                row.update(guard_stats)
            else:
                row.update(
                    {
                        "guard_trigger_rate": 0.0,
                        "guard_keep_previous_rate": 0.0,
                        "guard_wmd_fallback_rate": 0.0,
                        "guard_agentd_raw_rate": 0.0,
                    }
                )
            detail_rows.append(row)
        print(
            "seed={} guarded={:.6f} keep_previous={:.6f} dual_wmd={:.6f}".format(
                int(seed),
                float(summary["dual_agentd_guarded_wms"]["avg_total_cost"]),
                keep_previous_cost,
                wmd_cost,
            )
        )

    detail_fieldnames = [
        "seed",
        "policy",
        "avg_total_cost",
        "std_total_cost",
        "delta_vs_keep_previous",
        "delta_vs_dual_wmd",
        "guard_trigger_rate",
        "guard_keep_previous_rate",
        "guard_wmd_fallback_rate",
        "guard_agentd_raw_rate",
    ]
    with open(detail_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows: list[dict[str, float | int | str]] = []
    policies = sorted({str(row["policy"]) for row in detail_rows})
    for policy in policies:
        rows = [row for row in detail_rows if row["policy"] == policy]
        avg_costs = np.asarray([float(row["avg_total_cost"]) for row in rows], dtype=np.float64)
        delta_keep = np.asarray([float(row["delta_vs_keep_previous"]) for row in rows], dtype=np.float64)
        delta_wmd = np.asarray([float(row["delta_vs_dual_wmd"]) for row in rows], dtype=np.float64)
        summary_rows.append(
            {
                "policy": policy,
                "num_seeds": int(len(rows)),
                "mean_avg_total_cost": float(avg_costs.mean()),
                "std_across_seeds": float(avg_costs.std()),
                "best_seed_cost": float(avg_costs.min()),
                "worst_seed_cost": float(avg_costs.max()),
                "win_rate_vs_keep_previous": float(np.mean(delta_keep <= 0.0)),
                "mean_delta_vs_keep_previous": float(delta_keep.mean()),
                "worst_delta_vs_keep_previous": float(delta_keep.max()),
                "win_rate_vs_dual_wmd": float(np.mean(delta_wmd <= 0.0)),
                "mean_delta_vs_dual_wmd": float(delta_wmd.mean()),
                "mean_guard_trigger_rate": float(np.mean([float(row["guard_trigger_rate"]) for row in rows])),
                "mean_guard_keep_previous_rate": float(np.mean([float(row["guard_keep_previous_rate"]) for row in rows])),
                "mean_guard_wmd_fallback_rate": float(np.mean([float(row["guard_wmd_fallback_rate"]) for row in rows])),
                "mean_guard_agentd_raw_rate": float(np.mean([float(row["guard_agentd_raw_rate"]) for row in rows])),
            }
        )

    summary_rows.sort(key=lambda row: float(row["mean_avg_total_cost"]))
    summary_fieldnames = [
        "policy",
        "num_seeds",
        "mean_avg_total_cost",
        "std_across_seeds",
        "best_seed_cost",
        "worst_seed_cost",
        "win_rate_vs_keep_previous",
        "mean_delta_vs_keep_previous",
        "worst_delta_vs_keep_previous",
        "win_rate_vs_dual_wmd",
        "mean_delta_vs_dual_wmd",
        "mean_guard_trigger_rate",
        "mean_guard_keep_previous_rate",
        "mean_guard_wmd_fallback_rate",
        "mean_guard_agentd_raw_rate",
    ]
    with open(summary_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return detail_output, summary_output, detail_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--wmd-checkpoint", default="outputs/wmd/v2_drift_wmd_v2_multiseed_model.pt")
    parser.add_argument("--agentd-checkpoint", default="outputs/agent_d/v2_drift_agent_d_v2_multiseed_model.pt")
    parser.add_argument("--fast-policy", default="agent_s", choices=["greedy_delta", "lookahead_delta", "gnn_wms", "agent_s"])
    parser.add_argument("--fast-checkpoint", default="outputs/agent_s/v2_drift_agent_s_multiseed_model.pt")
    parser.add_argument("--eval-seeds", default="1007,1107,1207,1307,1407")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--detail-output", default="outputs/logs/dual_agent_multiseed_detail.csv")
    parser.add_argument("--summary-output", default="outputs/logs/dual_agent_multiseed_summary.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seeds = _parse_int_list(args.eval_seeds)
    _, summary_output, _, summary_rows = run_multiseed_evaluation(
        cfg=cfg,
        wmd_checkpoint=args.wmd_checkpoint,
        agentd_checkpoint=args.agentd_checkpoint,
        fast_policy=args.fast_policy,
        fast_checkpoint=args.fast_checkpoint,
        eval_seeds=eval_seeds,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        future_weight=float(args.future_weight),
        wm_margin=float(args.wm_margin),
        device=args.device,
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
        detail_output=args.detail_output,
        summary_output=args.summary_output,
    )

    print("Multi-seed dual-agent summary")
    for row in summary_rows:
        print(
            "{} mean={:.6f} std={:.6f} win_vs_keep={:.2f} mean_delta_keep={:.6f}".format(
                str(row["policy"]),
                float(row["mean_avg_total_cost"]),
                float(row["std_across_seeds"]),
                float(row["win_rate_vs_keep_previous"]),
                float(row["mean_delta_vs_keep_previous"]),
            )
        )
    print(f"detail_log={args.detail_output}")
    print(f"summary_log={summary_output}")


if __name__ == "__main__":
    main()
