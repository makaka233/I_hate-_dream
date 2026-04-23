from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from edge_sim.env.dynamic_deployment import deployment_change_count
from edge_sim.env.edge_env import EdgeEnv
from edge_sim.evaluation.evaluate_wms_gnn import load_gnn_wms, run_gnn_wms_planner_slot
from edge_sim.evaluation.policies import run_greedy_delta_slot, run_lookahead_delta_slot
from edge_sim.optim.kkt_allocator import KKTAllocator
from edge_sim.training.simulate_v2 import observed_source_service_demand
from edge_sim.training.wmd_utils import CANDIDATE_NAMES, candidate_pool, encode_candidate_features, feature_names


TARGET_COLUMNS = ["total_cost", "delay_sum", "migration_cost"]
KEEP_PREVIOUS_ID = CANDIDATE_NAMES.index("keep_previous")


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fast_evaluator(
    fast_policy: str,
    checkpoint_path: str | Path | None,
    device_name: str,
) -> tuple[object | None, torch.device]:
    device = torch.device(device_name)
    if fast_policy != "gnn_wms":
        return None, device
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when fast_policy='gnn_wms'.")
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
    else:
        raise ValueError(f"Unsupported fast_policy={fast_policy!r}")
    total_cost = float(delay_sum + migration_cost)
    return total_cost, float(delay_sum), float(migration_cost)


def build_dataset(
    cfg: dict,
    episodes: int,
    fast_slots: int,
    fast_policy: str,
    checkpoint_path: str | Path | None,
    future_weight: float,
    wm_margin: float,
    output_prefix: str | Path | None,
    device_name: str,
) -> tuple[Path, Path, dict[str, float]]:
    env = EdgeEnv(cfg)
    allocator = KKTAllocator(env.compute_cap, env.effective_bandwidth)
    dyn_cfg = cfg.get("dynamic_deployment", {})
    migration_weight = float(dyn_cfg.get("migration_weight", 0.1))
    loaded_fast_model, device = load_fast_evaluator(fast_policy, checkpoint_path, device_name)

    trace = [[env.sample_requests(slow_epoch=epoch) for _ in range(fast_slots)] for epoch in range(episodes)]
    previous_x = None
    previous_observed_matrix = None
    older_observed_matrix = None

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    decision_ids: list[int] = []
    candidate_ids: list[int] = []
    best_candidate_ids: list[int] = []
    hard_mask_rows: list[int] = []
    best_keep_gap_rows: list[float] = []
    best_second_margin_rows: list[float] = []
    meta_rows: list[dict[str, float | int | str]] = []
    feature_name_list = feature_names(env)
    decision_id = 0

    for epoch, batches in enumerate(trace):
        pool, state = candidate_pool(env, previous_x, previous_observed_matrix, older_observed_matrix, dyn_cfg)
        candidate_scores = []
        epoch_row_start = len(meta_rows)

        for candidate_name in CANDIDATE_NAMES:
            x = pool[candidate_name]
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
                device,
            )
            features = encode_candidate_features(env, previous_x, x, candidate_name, state)
            candidate_id = CANDIDATE_NAMES.index(candidate_name)
            feature_rows.append(features)
            target_rows.append(np.array([total_cost, delay_sum, migration_cost], dtype=np.float32))
            decision_ids.append(decision_id)
            candidate_ids.append(candidate_id)
            candidate_scores.append((candidate_name, total_cost))
            meta_rows.append(
                {
                    "decision_id": decision_id,
                    "epoch": epoch,
                    "candidate_name": candidate_name,
                    "candidate_id": candidate_id,
                    "total_cost": total_cost,
                    "delay_sum": delay_sum,
                    "migration_cost": migration_cost,
                    "migration_changes": migration_changes,
                    "history_demand_entropy": float(-(state["history_demand"] * np.log(state["history_demand"] + 1e-8)).sum()),
                    "trend_demand_entropy": float(-(state["trend_demand"] * np.log(state["trend_demand"] + 1e-8)).sum()),
                }
            )

        score_map = {name: float(score) for name, score in candidate_scores}
        sorted_scores = sorted(candidate_scores, key=lambda item: item[1])
        best_name, best_total_cost = sorted_scores[0]
        second_best_total_cost = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else float(best_total_cost)
        keep_previous_cost = float(score_map["keep_previous"])
        best_id = CANDIDATE_NAMES.index(best_name)
        hard_mask = int(best_id != KEEP_PREVIOUS_ID)
        best_keep_gap = max(keep_previous_cost - float(best_total_cost), 0.0)
        best_second_margin = max(float(second_best_total_cost) - float(best_total_cost), 0.0)
        for row in meta_rows[epoch_row_start:]:
            row["best_candidate_id"] = best_id
            row["best_candidate_name"] = best_name
            row["is_best"] = int(row["candidate_id"] == best_id)
            row["keep_previous_total_cost"] = keep_previous_cost
            row["best_total_cost"] = float(best_total_cost)
            row["best_keep_gap"] = best_keep_gap
            row["best_second_margin"] = best_second_margin
            row["hard_mask"] = hard_mask
            best_candidate_ids.append(best_id)
            hard_mask_rows.append(hard_mask)
            best_keep_gap_rows.append(best_keep_gap)
            best_second_margin_rows.append(best_second_margin)

        chosen_x = pool[best_name]
        older_observed_matrix = previous_observed_matrix
        previous_observed_matrix = observed_source_service_demand(batches, env.num_nodes, env.num_services)
        previous_x = chosen_x
        decision_id += 1

    run_name = cfg.get("run_name", "run")
    if output_prefix is None:
        output_prefix = Path("outputs") / "wmd" / f"{run_name}_{fast_policy}_dataset"
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(meta_rows[0].keys()) if meta_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    np.savez_compressed(
        npz_path,
        features=np.asarray(feature_rows, dtype=np.float32),
        targets=np.asarray(target_rows, dtype=np.float32),
        target_columns=np.asarray(TARGET_COLUMNS),
        feature_columns=np.asarray(feature_name_list),
        decision_ids=np.asarray(decision_ids, dtype=np.int64),
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        best_candidate_ids=np.asarray(best_candidate_ids, dtype=np.int64),
        hard_mask=np.asarray(hard_mask_rows, dtype=np.int64),
        best_keep_gap=np.asarray(best_keep_gap_rows, dtype=np.float32),
        best_second_margin=np.asarray(best_second_margin_rows, dtype=np.float32),
        candidate_names=np.asarray(CANDIDATE_NAMES),
    )

    summary = {
        "rows": float(len(feature_rows)),
        "decisions": float(decision_id),
        "avg_candidates_per_decision": float(len(feature_rows) / max(decision_id, 1)),
        "avg_best_total_cost": float(np.mean([min(group["total_cost"] for group in meta_rows[idx : idx + len(CANDIDATE_NAMES)]) for idx in range(0, len(meta_rows), len(CANDIDATE_NAMES))])) if meta_rows else 0.0,
        "hard_ratio": float(np.mean(hard_mask_rows)) if hard_mask_rows else 0.0,
        "avg_best_keep_gap": float(np.mean(best_keep_gap_rows)) if best_keep_gap_rows else 0.0,
        "avg_best_second_margin": float(np.mean(best_second_margin_rows)) if best_second_margin_rows else 0.0,
    }
    return csv_path, npz_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_drift.yaml")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--fast-slots", type=int, default=None)
    parser.add_argument("--fast-policy", default="gnn_wms", choices=["greedy_delta", "lookahead_delta", "gnn_wms"])
    parser.add_argument("--checkpoint", default="outputs/wms/v2_drift_gnn_wms_mixed_hard_model.pt")
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--wm-margin", type=float, default=0.002)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    fast_slots = int(args.fast_slots or cfg["system"]["slow_period"])
    checkpoint = None if args.fast_policy != "gnn_wms" else args.checkpoint
    csv_path, npz_path, summary = build_dataset(
        cfg=cfg,
        episodes=int(args.episodes),
        fast_slots=fast_slots,
        fast_policy=args.fast_policy,
        checkpoint_path=checkpoint,
        future_weight=float(args.future_weight),
        wm_margin=float(args.wm_margin),
        output_prefix=args.output_prefix,
        device_name=args.device,
    )
    print("WM-D dataset built")
    for key, value in summary.items():
        print(f"{key}={value:.4f}")
    print(f"csv={csv_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
