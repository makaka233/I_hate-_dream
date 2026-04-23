from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_sim.agents.deployment_world_model import DeploymentWorldModel


TARGET_COLUMNS = ["total_cost", "delay_sum", "migration_cost"]


def _load_datasets(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _split_by_decision(decision_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(decision_ids)
    rng.shuffle(unique_ids)
    val_count = max(1, int(round(len(unique_ids) * val_fraction)))
    val_ids = set(int(x) for x in unique_ids[:val_count])
    val_mask = np.asarray([int(x) in val_ids for x in decision_ids], dtype=bool)
    return ~val_mask, val_mask


def _mae(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - target)
    return {
        "mae_total_cost": float(np.mean(err[:, 0])),
        "mae_delay_sum": float(np.mean(err[:, 1])),
        "mae_migration_cost": float(np.mean(err[:, 2])),
    }


def _top1_accuracy(pred_total_cost: np.ndarray, true_total_cost: np.ndarray, decision_ids: np.ndarray) -> float:
    correct = 0
    total = 0
    for decision_id in np.unique(decision_ids):
        idx = np.flatnonzero(decision_ids == decision_id)
        if idx.size == 0:
            continue
        pred_best = idx[int(np.argmin(pred_total_cost[idx]))]
        true_best = idx[int(np.argmin(true_total_cost[idx]))]
        correct += int(pred_best == true_best)
        total += 1
    return float(correct / max(total, 1))


def train(
    dataset_paths: list[str | Path],
    output_path: str | Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    val_fraction: float,
    seed: int,
    device_name: str,
) -> tuple[Path, dict[str, float]]:
    features, targets, decision_ids, _candidate_ids, _best_ids = _load_datasets(dataset_paths)
    train_mask, val_mask = _split_by_decision(decision_ids, val_fraction, seed)
    x_train = features[train_mask]
    y_train = targets[train_mask]
    x_val = features[val_mask]
    y_val = targets[val_mask]

    feature_mean = x_train.mean(axis=0, keepdims=True)
    feature_std = x_train.std(axis=0, keepdims=True) + 1e-6
    target_mean = y_train.mean(axis=0, keepdims=True)
    target_std = y_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = (x_train - feature_mean) / feature_std
    y_train_n = (y_train - target_mean) / target_std
    x_val_n = (x_val - feature_mean) / feature_std
    y_val_n = (y_val - target_mean) / target_std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = DeploymentWorldModel(features.shape[1], hidden_dim=hidden_dim, output_dim=len(TARGET_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred_n = model(torch.from_numpy(x_val_n).to(device)).cpu().numpy()
            val_loss = float(np.mean((val_pred_n - y_val_n) ** 2))
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
        val_pred_n = model(torch.from_numpy(x_val_n).to(device)).cpu().numpy()
    val_pred = val_pred_n * target_std + target_mean
    metrics = _mae(val_pred, y_val)
    metrics["top1_total_cost"] = _top1_accuracy(val_pred[:, 0], y_val[:, 0], decision_ids[val_mask])
    metrics["train_rows"] = float(x_train.shape[0])
    metrics["val_rows"] = float(x_val.shape[0])
    metrics["train_decisions"] = float(np.unique(decision_ids[train_mask]).size)
    metrics["val_decisions"] = float(np.unique(decision_ids[val_mask]).size)
    metrics["best_val_loss_norm"] = best_val_loss

    if output_path is None:
        first = Path(dataset_paths[0])
        suffix = "model" if len(dataset_paths) == 1 else f"{len(dataset_paths)}seed_model"
        output_path = Path("outputs") / "wmd" / f"{first.stem}_{suffix}.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(hidden_dim),
            "target_columns": TARGET_COLUMNS,
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
