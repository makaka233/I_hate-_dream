from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from edge_sim.evaluation.evaluate_dual_agent_multiseed import _parse_int_list, run_multiseed_evaluation
from edge_sim.evaluation.evaluate_wmd import load_cfg


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--wmd-checkpoint", default="outputs/wmd/v2_drift_wmd_v2_multiseed_model.pt")
    parser.add_argument("--agentd-checkpoint", default="outputs/agent_d/v2_drift_agent_d_v2_multiseed_model.pt")
    parser.add_argument("--fast-policy", default="agent_s", choices=["greedy_delta", "lookahead_delta", "gnn_wms", "agent_s"])
    parser.add_argument("--fast-checkpoint", default="outputs/agent_s/v2_drift_agent_s_multiseed_model.pt")
    parser.add_argument("--eval-seeds", default="2007,2107,2207,2307,2407,2507,2607,2707,2807,2907,3007,3107")
    parser.add_argument("--episodes", type=int, default=6)
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
    parser.add_argument("--load-labels", default="light,medium,heavy")
    parser.add_argument("--poisson-lambdas", default="0.6,1.0,1.6")
    parser.add_argument("--output-dir", default="outputs/logs/v2_load_sweep")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    eval_seeds = _parse_int_list(args.eval_seeds)
    labels = _parse_str_list(args.load_labels)
    lambdas = _parse_float_list(args.poisson_lambdas)
    if len(labels) != len(lambdas):
        raise ValueError("load-labels and poisson-lambdas must have the same length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, float | int | str]] = []

    for label, lam in zip(labels, lambdas):
        cfg_load = deepcopy(cfg)
        cfg_load["requests"] = dict(cfg_load["requests"])
        cfg_load["requests"]["poisson_lambda"] = float(lam)
        cfg_load["run_name"] = f"{cfg_load.get('run_name', 'v2')}_{label}"
        detail_output = output_dir / f"{label}_detail.csv"
        summary_output = output_dir / f"{label}_summary.csv"
        _, _, _, summary_rows = run_multiseed_evaluation(
            cfg=cfg_load,
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
            detail_output=detail_output,
            summary_output=summary_output,
        )
        for row in summary_rows:
            aggregate_rows.append(
                {
                    "load_label": label,
                    "poisson_lambda": float(lam),
                    **row,
                }
            )
        guarded = next(row for row in summary_rows if str(row["policy"]) == "dual_agentd_guarded_wms")
        print(
            "[{}] lambda={:.2f} guarded_mean={:.6f} win_vs_keep={:.2f} mean_delta_keep={:.6f}".format(
                label,
                float(lam),
                float(guarded["mean_avg_total_cost"]),
                float(guarded["win_rate_vs_keep_previous"]),
                float(guarded["mean_delta_vs_keep_previous"]),
            )
        )

    aggregate_output = output_dir / "aggregate_summary.csv"
    fieldnames = [
        "load_label",
        "poisson_lambda",
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
    with open(aggregate_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"aggregate_summary={aggregate_output}")


if __name__ == "__main__":
    main()
