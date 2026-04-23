from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_sim.agents.scheduler_world_model import SchedulerWorldModel


TARGET_COLUMNS = ["current_delta", "future_cost", "target_score"]


def _column_indices(names: np.ndarray, targets: list[str]) -> list[int]:
    lookup = {str(name): idx for idx, name in enumerate(names)}
    missing = [name for name in targets if name not in lookup]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    return [lookup[name] for name in targets]


def _split_by_decision(decision_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(decision_ids)
    rng.shuffle(unique_ids)
    val_count = max(1, int(round(len(unique_ids) * val_fraction)))
    val_ids = set(int(x) for x in unique_ids[:val_count])
    val_mask = np.asarray([int(x) in val_ids for x in decision_ids], dtype=bool)
    return ~val_mask, val_mask


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "mae_current_delta": float(np.mean(np.abs(err[:, 0]))),
        "mae_future_cost": float(np.mean(np.abs(err[:, 1]))),
        "mae_target_score": float(np.mean(np.abs(err[:, 2]))),
        "rmse_target_score": float(np.sqrt(np.mean(err[:, 2] ** 2))),
    }


def _top1_accuracy(pred_score: np.ndarray, true_score: np.ndarray, decision_ids: np.ndarray) -> float:
    correct = 0
    total = 0
    for decision_id in np.unique(decision_ids):
        idx = np.flatnonzero(decision_ids == decision_id)
        if idx.size == 0:
            continue
        pred_best = idx[int(np.argmin(pred_score[idx]))]
        true_best = idx[int(np.argmin(true_score[idx]))]
        correct += int(pred_best == true_best)
        total += 1
    return float(correct / max(total, 1))


def _load_datasets(dataset_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_chunks = []
    label_chunks = []
    decision_chunks = []
    feature_columns = None
    label_columns = None
    decision_offset = 0

    for path in dataset_paths:
        data = np.load(path, allow_pickle=True)
        if feature_columns is None:
            feature_columns = data["feature_columns"]
            label_columns = data["label_columns"]
        elif [str(x) for x in feature_columns] != [str(x) for x in data["feature_columns"]]:
            raise ValueError(f"Feature columns mismatch in {path}")
        elif [str(x) for x in label_columns] != [str(x) for x in data["label_columns"]]:
            raise ValueError(f"Label columns mismatch in {path}")

        feature_chunks.append(data["features"].astype(np.float32))
        label_chunks.append(data["labels"].astype(np.float32))
        ids = data["decision_ids"].astype(np.int64) + decision_offset
        decision_chunks.append(ids)
        decision_offset = int(ids.max()) + 1 if ids.size else decision_offset

    if feature_columns is None or label_columns is None:
        raise ValueError("At least one dataset is required.")
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
        np.concatenate(decision_chunks, axis=0),
        feature_columns,
        label_columns,
    )


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
    features, labels, decision_ids, feature_columns, label_columns = _load_datasets(dataset_paths)
    target_idx = _column_indices(label_columns, TARGET_COLUMNS)
    targets = labels[:, target_idx].astype(np.float32)

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
    model = SchedulerWorldModel(features.shape[1], hidden_dim=hidden_dim, output_dim=len(TARGET_COLUMNS)).to(device)
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
    metrics = _regression_metrics(val_pred, y_val)
    metrics["val_top1_accuracy"] = _top1_accuracy(val_pred[:, 2], y_val[:, 2], decision_ids[val_mask])
    metrics["train_rows"] = float(x_train.shape[0])
    metrics["val_rows"] = float(x_val.shape[0])
    metrics["train_decisions"] = float(np.unique(decision_ids[train_mask]).size)
    metrics["val_decisions"] = float(np.unique(decision_ids[val_mask]).size)
    metrics["best_val_loss_norm"] = best_val_loss

    if output_path is None:
        first_path = Path(dataset_paths[0])
        suffix = "model" if len(dataset_paths) == 1 else f"{len(dataset_paths)}seed_model"
        output_path = Path("outputs") / "wms" / f"{first_path.stem}_{suffix}.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(hidden_dim),
            "target_columns": TARGET_COLUMNS,
            "feature_columns": [str(x) for x in feature_columns],
            "label_columns": [str(x) for x in label_columns],
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
    parser.add_argument("--dataset", nargs="+", default=["outputs/wms/v2_drift_heuristic_lookahead_delta_d2.npz"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=120)
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
    print("WM-S model trained")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
