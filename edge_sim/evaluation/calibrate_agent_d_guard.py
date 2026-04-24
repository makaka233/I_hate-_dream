from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from edge_sim.evaluation.evaluate_dual_agent import evaluate
from edge_sim.evaluation.evaluate_wmd import load_cfg


def _parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _extract_guard_stats(log_path: Path) -> dict[str, float]:
    with open(log_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
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
        "guard_wmd_fallback_rate": float(
            sum(1 for row in guarded_rows if row["guard_decision_source"] == "wmd_fallback") / count
        ),
        "guard_agentd_raw_rate": float(sum(1 for row in guarded_rows if row["guard_decision_source"] == "agentd_raw") / count),
    }


def _candidate_rows(
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
    per_seed_dir: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    detail_rows: list[dict[str, float | int | str]] = []
    for seed in eval_seeds:
        per_seed_log = per_seed_dir / (
            f"agentd_guard_seed{seed}_fp{agentd_min_prob:.2f}_fm{agentd_min_margin:.2f}"
            f"_gap{agentd_max_wmd_gap:.2f}_{agentd_fallback_mode}_fg{agentd_fallback_gain:.2f}.csv"
        )
        _, summary = evaluate(
            cfg=cfg,
            wmd_checkpoint_path=wmd_checkpoint,
            agentd_checkpoint_path=agentd_checkpoint,
            fast_policy=fast_policy,
            fast_checkpoint_path=fast_checkpoint if fast_policy in {"gnn_wms", "agent_s"} else None,
            episodes=episodes,
            fast_slots=fast_slots,
            future_weight=future_weight,
            wm_margin=wm_margin,
            eval_seed=seed,
            device_name=device,
            output_path=per_seed_log,
            agentd_min_prob=agentd_min_prob,
            agentd_min_margin=agentd_min_margin,
            agentd_base_gain=agentd_base_gain,
            agentd_refresh_extra_gain=agentd_refresh_extra_gain,
            agentd_aggressive_extra_gain=agentd_aggressive_extra_gain,
            agentd_max_wmd_gap=agentd_max_wmd_gap,
            agentd_fallback_mode=agentd_fallback_mode,
            agentd_fallback_gain=agentd_fallback_gain,
            agent_s_min_prob=agent_s_min_prob,
            agent_s_min_margin=agent_s_min_margin,
            agent_s_fallback=agent_s_fallback,
        )
        guard_stats = _extract_guard_stats(per_seed_log)
        guarded = float(summary["dual_agentd_guarded_wms"]["avg_total_cost"])
        keep_previous = float(summary["keep_previous_wms"]["avg_total_cost"])
        dual_wmd = float(summary["dual_wmd_wms"]["avg_total_cost"])
        raw_agentd = float(summary["dual_agentd_wms"]["avg_total_cost"])
        detail_rows.append(
            {
                "seed": int(seed),
                "agentd_min_prob": float(agentd_min_prob),
                "agentd_min_margin": float(agentd_min_margin),
                "agentd_base_gain": float(agentd_base_gain),
                "agentd_refresh_extra_gain": float(agentd_refresh_extra_gain),
                "agentd_aggressive_extra_gain": float(agentd_aggressive_extra_gain),
                "agentd_max_wmd_gap": float(agentd_max_wmd_gap),
                "agentd_fallback_mode": agentd_fallback_mode,
                "agentd_fallback_gain": float(agentd_fallback_gain),
                "guarded_avg_total_cost": guarded,
                "keep_previous_avg_total_cost": keep_previous,
                "dual_wmd_avg_total_cost": dual_wmd,
                "raw_agentd_avg_total_cost": raw_agentd,
                "delta_vs_keep_previous": float(guarded - keep_previous),
                "delta_vs_dual_wmd": float(guarded - dual_wmd),
                "delta_vs_raw_agentd": float(guarded - raw_agentd),
                **guard_stats,
            }
        )

    guarded_costs = np.asarray([float(row["guarded_avg_total_cost"]) for row in detail_rows], dtype=np.float64)
    delta_keep = np.asarray([float(row["delta_vs_keep_previous"]) for row in detail_rows], dtype=np.float64)
    delta_wmd = np.asarray([float(row["delta_vs_dual_wmd"]) for row in detail_rows], dtype=np.float64)
    delta_raw = np.asarray([float(row["delta_vs_raw_agentd"]) for row in detail_rows], dtype=np.float64)
    summary_row = {
        "agentd_min_prob": float(agentd_min_prob),
        "agentd_min_margin": float(agentd_min_margin),
        "agentd_base_gain": float(agentd_base_gain),
        "agentd_refresh_extra_gain": float(agentd_refresh_extra_gain),
        "agentd_aggressive_extra_gain": float(agentd_aggressive_extra_gain),
        "agentd_max_wmd_gap": float(agentd_max_wmd_gap),
        "agentd_fallback_mode": agentd_fallback_mode,
        "agentd_fallback_gain": float(agentd_fallback_gain),
        "num_seeds": int(len(detail_rows)),
        "mean_guarded_total_cost": float(guarded_costs.mean()),
        "std_guarded_total_cost": float(guarded_costs.std()),
        "win_rate_vs_keep_previous": float(np.mean(delta_keep <= 0.0)),
        "mean_delta_vs_keep_previous": float(delta_keep.mean()),
        "worst_delta_vs_keep_previous": float(delta_keep.max()),
        "win_rate_vs_dual_wmd": float(np.mean(delta_wmd <= 0.0)),
        "mean_delta_vs_dual_wmd": float(delta_wmd.mean()),
        "mean_delta_vs_raw_agentd": float(delta_raw.mean()),
        "mean_guard_trigger_rate": float(np.mean([float(row["guard_trigger_rate"]) for row in detail_rows])),
        "mean_guard_keep_previous_rate": float(np.mean([float(row["guard_keep_previous_rate"]) for row in detail_rows])),
        "mean_guard_wmd_fallback_rate": float(np.mean([float(row["guard_wmd_fallback_rate"]) for row in detail_rows])),
        "mean_guard_agentd_raw_rate": float(np.mean([float(row["guard_agentd_raw_rate"]) for row in detail_rows])),
    }
    return detail_rows, summary_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--wmd-checkpoint", default="outputs/wmd/v2_drift_wmd_v2_multiseed_model.pt")
    parser.add_argument("--agentd-checkpoint", default="outputs/agent_d/v2_drift_agent_d_v2_multiseed_model.pt")
    parser.add_argument("--fast-policy", default="agent_s", choices=["greedy_delta", "lookahead_delta", "gnn_wms", "agent_s"])
    parser.add_argument("--fast-checkpoint", default="outputs/agent_s/v2_drift_agent_s_multiseed_model.pt")
    parser.add_argument("--eval-seeds", default="1007,1207,1407")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--agentd-min-prob-grid", default="0.22,0.28")
    parser.add_argument("--agentd-min-margin-grid", default="0.05,0.07")
    parser.add_argument("--agentd-base-gain-grid", default="0.6")
    parser.add_argument("--agentd-refresh-extra-gain", type=float, default=0.25)
    parser.add_argument("--agentd-aggressive-extra-gain", type=float, default=0.6)
    parser.add_argument("--agentd-max-wmd-gap-grid", default="0.75")
    parser.add_argument("--agentd-fallback-modes", default="keep_previous,wmd_if_gain")
    parser.add_argument("--agentd-fallback-gain-grid", default="0.6")
    parser.add_argument("--agent-s-min-prob", type=float, default=0.0)
    parser.add_argument("--agent-s-min-margin", type=float, default=0.05)
    parser.add_argument("--agent-s-fallback", default="greedy", choices=["none", "greedy"])
    parser.add_argument("--detail-output", default="outputs/logs/agentd_guard_calibration_detail.csv")
    parser.add_argument("--summary-output", default="outputs/logs/agentd_guard_calibration_summary.csv")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seeds = _parse_list(args.eval_seeds, int)
    min_prob_grid = _parse_list(args.agentd_min_prob_grid, float)
    min_margin_grid = _parse_list(args.agentd_min_margin_grid, float)
    base_gain_grid = _parse_list(args.agentd_base_gain_grid, float)
    max_wmd_gap_grid = _parse_list(args.agentd_max_wmd_gap_grid, float)
    fallback_modes = _parse_list(args.agentd_fallback_modes, str)
    fallback_gain_grid = _parse_list(args.agentd_fallback_gain_grid, float)

    detail_output = Path(args.detail_output)
    summary_output = Path(args.summary_output)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    per_seed_dir = detail_output.parent / f"{detail_output.stem}_per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    detail_rows_all: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for min_prob in min_prob_grid:
        for min_margin in min_margin_grid:
            for base_gain in base_gain_grid:
                for max_wmd_gap in max_wmd_gap_grid:
                    for fallback_mode in fallback_modes:
                        gains = fallback_gain_grid if fallback_mode == "wmd_if_gain" else [0.0]
                        for fallback_gain in gains:
                            detail_rows, summary_row = _candidate_rows(
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
                                agentd_min_prob=float(min_prob),
                                agentd_min_margin=float(min_margin),
                                agentd_base_gain=float(base_gain),
                                agentd_refresh_extra_gain=float(args.agentd_refresh_extra_gain),
                                agentd_aggressive_extra_gain=float(args.agentd_aggressive_extra_gain),
                                agentd_max_wmd_gap=float(max_wmd_gap),
                                agentd_fallback_mode=fallback_mode,
                                agentd_fallback_gain=float(fallback_gain),
                                agent_s_min_prob=float(args.agent_s_min_prob),
                                agent_s_min_margin=float(args.agent_s_min_margin),
                                agent_s_fallback=args.agent_s_fallback,
                                per_seed_dir=per_seed_dir,
                            )
                            detail_rows_all.extend(detail_rows)
                            summary_rows.append(summary_row)
                            print(
                                "min_prob={:.2f} min_margin={:.2f} base_gain={:.2f} gap={:.2f} fallback={} fg={:.2f} "
                                "mean={:.6f} win_vs_keep={:.2f} mean_delta_keep={:.6f}".format(
                                    float(min_prob),
                                    float(min_margin),
                                    float(base_gain),
                                    float(max_wmd_gap),
                                    fallback_mode,
                                    float(fallback_gain),
                                    float(summary_row["mean_guarded_total_cost"]),
                                    float(summary_row["win_rate_vs_keep_previous"]),
                                    float(summary_row["mean_delta_vs_keep_previous"]),
                                )
                            )

    detail_fieldnames = [
        "seed",
        "agentd_min_prob",
        "agentd_min_margin",
        "agentd_base_gain",
        "agentd_refresh_extra_gain",
        "agentd_aggressive_extra_gain",
        "agentd_max_wmd_gap",
        "agentd_fallback_mode",
        "agentd_fallback_gain",
        "guarded_avg_total_cost",
        "keep_previous_avg_total_cost",
        "dual_wmd_avg_total_cost",
        "raw_agentd_avg_total_cost",
        "delta_vs_keep_previous",
        "delta_vs_dual_wmd",
        "delta_vs_raw_agentd",
        "guard_trigger_rate",
        "guard_keep_previous_rate",
        "guard_wmd_fallback_rate",
        "guard_agentd_raw_rate",
    ]
    with open(detail_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows_all)

    summary_rows.sort(
        key=lambda row: (
            float(row["mean_guarded_total_cost"]),
            float(row["worst_delta_vs_keep_previous"]),
            -float(row["win_rate_vs_keep_previous"]),
        )
    )
    summary_fieldnames = [
        "agentd_min_prob",
        "agentd_min_margin",
        "agentd_base_gain",
        "agentd_refresh_extra_gain",
        "agentd_aggressive_extra_gain",
        "agentd_max_wmd_gap",
        "agentd_fallback_mode",
        "agentd_fallback_gain",
        "num_seeds",
        "mean_guarded_total_cost",
        "std_guarded_total_cost",
        "win_rate_vs_keep_previous",
        "mean_delta_vs_keep_previous",
        "worst_delta_vs_keep_previous",
        "win_rate_vs_dual_wmd",
        "mean_delta_vs_dual_wmd",
        "mean_delta_vs_raw_agentd",
        "mean_guard_trigger_rate",
        "mean_guard_keep_previous_rate",
        "mean_guard_wmd_fallback_rate",
        "mean_guard_agentd_raw_rate",
    ]
    with open(summary_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    best = summary_rows[0]
    print("Best Agent-D guard setting")
    print(
        "min_prob={:.2f} min_margin={:.2f} base_gain={:.2f} gap={:.2f} fallback={} fg={:.2f} mean={:.6f} "
        "win_vs_keep={:.2f} mean_delta_keep={:.6f} mean_delta_wmd={:.6f}".format(
            float(best["agentd_min_prob"]),
            float(best["agentd_min_margin"]),
            float(best["agentd_base_gain"]),
            float(best["agentd_max_wmd_gap"]),
            str(best["agentd_fallback_mode"]),
            float(best["agentd_fallback_gain"]),
            float(best["mean_guarded_total_cost"]),
            float(best["win_rate_vs_keep_previous"]),
            float(best["mean_delta_vs_keep_previous"]),
            float(best["mean_delta_vs_dual_wmd"]),
        )
    )
    print(f"detail_log={detail_output}")
    print(f"summary_log={summary_output}")


if __name__ == "__main__":
    main()
