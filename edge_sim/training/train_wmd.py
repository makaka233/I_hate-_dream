from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from edge_sim.agents.deployment_world_model import DeploymentWorldModel


TARGET_COLUMNS = ["total_cost", "delay_sum", "migration_cost"]


def _load_datasets(
    paths: list[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_chunks = []
    target_chunks = []
    decision_chunks = []
    candidate_chunks = []
    best_chunks = []
    feature_columns = None
    target_columns = None
    decision_offset = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        if feature_columns is None:
            feature_columns = data["feature_columns"]
            target_columns = data["target_columns"]
        elif [str(x) for x in feature_columns] != [str(x) for x in data["feature_columns"]]:
            raise ValueError(f"feature_columns mismatch in {path}")
        elif [str(x) for x in target_columns] != [str(x) for x in data["target_columns"]]:
            raise ValueError(f"target_columns mismatch in {path}")

        feature_chunks.append(data["features"].astype(np.float32))
        target_chunks.append(data["targets"].astype(np.float32))
        ids = data["decision_ids"].astype(np.int64) + decision_offset
        decision_chunks.append(ids)
        candidate_chunks.append(data["candidate_ids"].astype(np.int64))
        best_chunks.append(data["best_candidate_ids"].astype(np.int64))
        decision_offset = int(ids.max()) + 1 if ids.size else decision_offset

    if feature_columns is None or target_columns is None:
        raise ValueError("At least one dataset is required.")
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(decision_chunks, axis=0),
        np.concatenate(candidate_chunks, axis=0),
        np.concatenate(best_chunks, axis=0),
    )


def _group_by_decision(
    features: np.ndarray,
    targets: np.ndarray,
    decision_ids: np.ndarray,
    candidate_ids: np.ndarray,
    best_candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_decisions = np.unique(decision_ids)
    num_candidates = int(candidate_ids.max()) + 1
    feature_dim = features.shape[1]
    target_dim = targets.shape[1]

    grouped_features = np.zeros((unique_decisions.size, num_candidates, feature_dim), dtype=np.float32)
    grouped_targets = np.zeros((unique_decisions.size, num_candidates, target_dim), dtype=np.float32)
    grouped_best = np.zeros(unique_decisions.size, dtype=np.int64)

    for out_idx, decision_id in enumerate(unique_decisions):
        idx = np.flatnonzero(decision_ids == decision_id)
        cand = candidate_ids[idx]
        grouped_features[out_idx, cand] = features[idx]
        grouped_targets[out_idx, cand] = targets[idx]
        grouped_best[out_idx] = int(best_candidate_ids[idx[0]])

    return grouped_features, grouped_targets, grouped_best


def _split_decisions(num_decisions: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_decisions)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_decisions * val_fraction)))
    return indices[val_count:], indices[:val_count]


def _weighted_group_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    per_decision = ((pred - target) ** 2).mean(dim=(1, 2))
    return (per_decision * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _ranking_ce(
    pred: torch.Tensor,
    true_best: torch.Tensor,
    rank_temperature: float,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    logits = -pred[:, :, 0] / max(rank_temperature, 1e-6)
    loss = F.cross_entropy(logits, true_best, reduction="none")
    return (loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _top1_accuracy(pred_total_cost: np.ndarray, true_total_cost: np.ndarray, true_best: np.ndarray) -> float:
    pred_best = np.argmin(pred_total_cost, axis=1)
    return float(np.mean(pred_best == true_best))


def _subset_accuracy(pred_total_cost: np.ndarray, true_best: np.ndarray, subset_mask: np.ndarray) -> float:
    if not np.any(subset_mask):
        return 0.0
    pred_best = np.argmin(pred_total_cost, axis=1)
    return float(np.mean(pred_best[subset_mask] == true_best[subset_mask]))


def _make_sample_weights(best_candidate: np.ndarray, keep_previous_id: int, hard_weight: float) -> np.ndarray:
    hard_mask = best_candidate != keep_previous_id
    weights = np.ones(best_candidate.shape[0], dtype=np.float32)
    weights[hard_mask] = float(hard_weight)
    return weights


def train(
    dataset_paths: list[str | Path],
    output_path: str | Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    rank_coef: float,
    rank_temperature: float,
    hard_weight: float,
    val_fraction: float,
    seed: int,
    device_name: str,
) -> tuple[Path, dict[str, float]]:
    features, targets, decision_ids, candidate_ids, best_candidate_ids = _load_datasets(dataset_paths)
    group_x, group_y, group_best = _group_by_decision(features, targets, decision_ids, candidate_ids, best_candidate_ids)

    train_idx, val_idx = _split_decisions(group_x.shape[0], val_fraction, seed)
    x_train = group_x[train_idx]
    y_train = group_y[train_idx]
    best_train = group_best[train_idx]
    x_val = group_x[val_idx]
    y_val = group_y[val_idx]
    best_val = group_best[val_idx]

    feature_mean = x_train.mean(axis=(0, 1), keepdims=True)
    feature_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    target_mean = y_train.mean(axis=(0, 1), keepdims=True)
    target_std = y_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train_n = (x_train - feature_mean) / feature_std
    y_train_n = (y_train - target_mean) / target_std
    x_val_n = (x_val - feature_mean) / feature_std
    y_val_n = (y_val - target_mean) / target_std

    keep_previous_id = 0
    train_weights = _make_sample_weights(best_train, keep_previous_id, hard_weight)
    val_weights = _make_sample_weights(best_val, keep_previous_id, hard_weight)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = DeploymentWorldModel(x_train.shape[-1], hidden_dim=hidden_dim, output_dim=len(TARGET_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_n),
        torch.from_numpy(y_train_n),
        torch.from_numpy(best_train.astype(np.int64)),
        torch.from_numpy(train_weights.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, bestb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            bestb = bestb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)
            flat_pred = model(xb.reshape(-1, xb.shape[-1]))
            pred = flat_pred.view(xb.shape[0], xb.shape[1], -1)
            mse_loss = _weighted_group_mse(pred, yb, wb)
            rank_loss = _ranking_ce(pred, bestb, rank_temperature, wb)
            loss = mse_loss + rank_coef * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            flat_val_pred = model(torch.from_numpy(x_val_n).to(device).reshape(-1, x_val_n.shape[-1])).cpu().numpy()
            val_pred_n = flat_val_pred.reshape(x_val_n.shape[0], x_val_n.shape[1], -1)
        per_group_mse = ((val_pred_n - y_val_n) ** 2).mean(axis=(1, 2))
        logits = -val_pred_n[:, :, 0] / max(rank_temperature, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
        rank_loss = -log_probs[np.arange(log_probs.shape[0]), best_val]
        val_loss = float(
            (per_group_mse * val_weights).sum() / max(float(val_weights.sum()), 1e-8)
            + rank_coef * (rank_loss * val_weights).sum() / max(float(val_weights.sum()), 1e-8)
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                "epoch={:04d} train_loss={:.6f} val_loss={:.6f}".format(
                    epoch,
                    float(np.mean(train_losses)) if train_losses else 0.0,
                    val_loss,
                )
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        flat_val_pred = model(torch.from_numpy(x_val_n).to(device).reshape(-1, x_val_n.shape[-1])).cpu().numpy()
    val_pred_n = flat_val_pred.reshape(x_val_n.shape[0], x_val_n.shape[1], -1)
    val_pred = val_pred_n * target_std + target_mean
    mae = np.abs(val_pred - y_val).mean(axis=(0, 1))
    hard_mask = best_val != keep_previous_id
    metrics = {
        "mae_total_cost": float(mae[0]),
        "mae_delay_sum": float(mae[1]),
        "mae_migration_cost": float(mae[2]),
        "top1_total_cost": _top1_accuracy(val_pred[:, :, 0], y_val[:, :, 0], best_val),
        "hard_ratio": float(np.mean(hard_mask)),
        "top1_total_cost_hard": _subset_accuracy(val_pred[:, :, 0], best_val, hard_mask),
        "top1_total_cost_easy": _subset_accuracy(val_pred[:, :, 0], best_val, ~hard_mask),
        "train_decisions": float(x_train.shape[0]),
        "val_decisions": float(x_val.shape[0]),
        "best_val_loss_norm": best_val_loss,
    }

    if output_path is None:
        first = Path(dataset_paths[0])
        suffix = "model" if len(dataset_paths) == 1 else f"{len(dataset_paths)}seed_model"
        output_path = Path("outputs") / "wmd" / f"{first.stem}_{suffix}.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(x_train.shape[-1]),
            "hidden_dim": int(hidden_dim),
            "target_columns": TARGET_COLUMNS,
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
            "rank_coef": float(rank_coef),
            "rank_temperature": float(rank_temperature),
            "hard_weight": float(hard_weight),
            "metrics": metrics,
        },
        output_path,
    )
    return output_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["outputs/wmd/v2_drift_gnn_wms_dataset.npz"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rank-coef", type=float, default=0.5)
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_path, metrics = train(
        dataset_paths=[Path(path) for path in args.dataset],
        output_path=args.output,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        rank_coef=float(args.rank_coef),
        rank_temperature=float(args.rank_temperature),
        hard_weight=float(args.hard_weight),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        device_name=args.device,
    )
    print("WM-D model trained")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
