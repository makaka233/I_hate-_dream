from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_sim.agents.scheduler_world_model import SchedulerGNNWorldModel


TARGET_COLUMNS = ["current_delta", "future_cost", "target_score"]


def _load_datasets(paths: list[str | Path]) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "node_feat": [],
        "edge_attr": [],
        "candidate_edge_attr": [],
        "request_feat": [],
        "prev_node": [],
        "legal_mask": [],
        "targets": [],
        "best_action": [],
        "decision_ids": [],
    }
    edge_index = None
    target_columns = None
    decision_offset = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        if edge_index is None:
            edge_index = data["edge_index"].astype(np.int64)
            target_columns = data["target_columns"]
        elif not np.array_equal(edge_index, data["edge_index"].astype(np.int64)):
            raise ValueError(f"edge_index mismatch in {path}")
        elif [str(x) for x in target_columns] != [str(x) for x in data["target_columns"]]:
            raise ValueError(f"target_columns mismatch in {path}")

        for key in chunks:
            if key == "decision_ids":
                ids = data[key].astype(np.int64) + decision_offset
                chunks[key].append(ids)
                decision_offset = int(ids.max()) + 1 if ids.size else decision_offset
            else:
                chunks[key].append(data[key])

    if edge_index is None or target_columns is None:
        raise ValueError("At least one dataset is required.")

    loaded = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    loaded["edge_index"] = edge_index
    loaded["target_columns"] = np.asarray(target_columns)
    return loaded


def _split_indices(num_items: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_items * val_fraction)))
    return indices[val_count:], indices[:val_count]


def _normalizers(data: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, np.ndarray]:
    legal = data["legal_mask"][train_idx].astype(bool)
    targets = data["targets"][train_idx]
    target_values = targets[legal]
    return {
        "node_mean": data["node_feat"][train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32),
        "node_std": (data["node_feat"][train_idx].std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32),
        "edge_mean": data["edge_attr"][train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32),
        "edge_std": (data["edge_attr"][train_idx].std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32),
        "candidate_edge_mean": data["candidate_edge_attr"][train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32),
        "candidate_edge_std": (
            data["candidate_edge_attr"][train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        ).astype(np.float32),
        "request_mean": data["request_feat"][train_idx].mean(axis=0, keepdims=True).astype(np.float32),
        "request_std": (data["request_feat"][train_idx].std(axis=0, keepdims=True) + 1e-6).astype(np.float32),
        "target_mean": target_values.mean(axis=0, keepdims=True).astype(np.float32),
        "target_std": (target_values.std(axis=0, keepdims=True) + 1e-6).astype(np.float32),
    }


def _apply_normalizers(data: dict[str, np.ndarray], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "node_feat": ((data["node_feat"] - norm["node_mean"]) / norm["node_std"]).astype(np.float32),
        "edge_attr": ((data["edge_attr"] - norm["edge_mean"]) / norm["edge_std"]).astype(np.float32),
        "candidate_edge_attr": (
            (data["candidate_edge_attr"] - norm["candidate_edge_mean"]) / norm["candidate_edge_std"]
        ).astype(np.float32),
        "request_feat": ((data["request_feat"] - norm["request_mean"]) / norm["request_std"]).astype(np.float32),
        "targets": ((data["targets"] - norm["target_mean"]) / norm["target_std"]).astype(np.float32),
    }


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    mask = legal_mask.unsqueeze(-1).float()
    weighted = ((pred - target) ** 2) * mask * loss_weights.view(1, 1, -1)
    return weighted.sum() / torch.clamp(mask.sum() * loss_weights.sum(), min=1.0)


def _predict(
    model: SchedulerGNNWorldModel,
    edge_index: torch.Tensor,
    tensors: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            pred = model(
                torch.from_numpy(tensors["node_feat"][idx]).to(device),
                edge_index,
                torch.from_numpy(tensors["edge_attr"][idx]).to(device),
                torch.from_numpy(tensors["candidate_edge_attr"][idx]).to(device),
                torch.from_numpy(tensors["request_feat"][idx]).to(device),
                torch.from_numpy(tensors["prev_node"][idx]).to(device),
            )
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def _metrics(
    pred: np.ndarray,
    true: np.ndarray,
    legal_mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    future_weight: float,
) -> dict[str, float]:
    pred_raw = pred * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
    true_raw = true * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
    legal = legal_mask.astype(bool)
    err = pred_raw[legal] - true_raw[legal]
    true_score = np.where(legal, true_raw[:, :, 2], np.inf)
    pred_target_score = np.where(legal, pred_raw[:, :, 2], np.inf)
    exact_delta_pred_future = np.where(legal, true_raw[:, :, 0] + future_weight * pred_raw[:, :, 1], np.inf)

    true_best = np.argmin(true_score, axis=1)
    pred_best = np.argmin(pred_target_score, axis=1)
    exact_future_best = np.argmin(exact_delta_pred_future, axis=1)
    return {
        "mae_current_delta": float(np.mean(np.abs(err[:, 0]))),
        "mae_future_cost": float(np.mean(np.abs(err[:, 1]))),
        "mae_target_score": float(np.mean(np.abs(err[:, 2]))),
        "rmse_target_score": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "top1_pred_target_score": float(np.mean(pred_best == true_best)),
        "top1_exact_delta_pred_future": float(np.mean(exact_future_best == true_best)),
    }


def train(
    dataset_paths: list[str | Path],
    output_path: str | Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    heads: int,
    layers: int,
    request_dim: int,
    loss_weights: list[float],
    val_fraction: float,
    future_weight: float,
    seed: int,
    device_name: str,
) -> tuple[Path, dict[str, float]]:
    data = _load_datasets(dataset_paths)
    train_idx, val_idx = _split_indices(data["node_feat"].shape[0], val_fraction, seed)
    norm = _normalizers(data, train_idx)
    tensors = _apply_normalizers(data, norm)
    tensors["prev_node"] = data["prev_node"].astype(np.int64)
    tensors["legal_mask"] = data["legal_mask"].astype(np.bool_)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
    model = SchedulerGNNWorldModel(
        hidden_dim=hidden_dim,
        heads=heads,
        layers=layers,
        request_dim=request_dim,
        output_dim=len(TARGET_COLUMNS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_weight_t = torch.as_tensor(loss_weights, dtype=torch.float32, device=device)
    if loss_weight_t.numel() != len(TARGET_COLUMNS):
        raise ValueError(f"loss_weights must contain {len(TARGET_COLUMNS)} values.")

    train_ds = TensorDataset(
        torch.from_numpy(tensors["node_feat"][train_idx]),
        torch.from_numpy(tensors["edge_attr"][train_idx]),
        torch.from_numpy(tensors["candidate_edge_attr"][train_idx]),
        torch.from_numpy(tensors["request_feat"][train_idx]),
        torch.from_numpy(tensors["prev_node"][train_idx]),
        torch.from_numpy(tensors["targets"][train_idx]),
        torch.from_numpy(tensors["legal_mask"][train_idx]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for node_feat, edge_attr, cand_edge, req_feat, prev_node, targets, legal_mask in loader:
            node_feat = node_feat.to(device)
            edge_attr = edge_attr.to(device)
            cand_edge = cand_edge.to(device)
            req_feat = req_feat.to(device)
            prev_node = prev_node.to(device)
            targets = targets.to(device)
            legal_mask = legal_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(node_feat, edge_index, edge_attr, cand_edge, req_feat, prev_node)
            loss = _masked_mse(pred, targets, legal_mask, loss_weight_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred = _predict(model, edge_index, tensors, val_idx, batch_size, device)
        val_target = tensors["targets"][val_idx]
        val_mask = tensors["legal_mask"][val_idx]
        val_loss = float(np.mean(((val_pred - val_target) ** 2)[val_mask]))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                "epoch={:04d} train_loss={:.6f} val_loss={:.6f}".format(
                    epoch,
                    float(np.mean(train_losses)) if train_losses else 0.0,
                    val_loss,
                )
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = _predict(model, edge_index, tensors, val_idx, batch_size, device)
    metrics = _metrics(
        val_pred,
        tensors["targets"][val_idx],
        tensors["legal_mask"][val_idx],
        norm["target_mean"],
        norm["target_std"],
        future_weight,
    )
    metrics.update(
        {
            "train_decisions": float(train_idx.size),
            "val_decisions": float(val_idx.size),
            "best_val_loss_norm": best_val_loss,
        }
    )

    if output_path is None:
        first = Path(dataset_paths[0])
        suffix = "gnn_model" if len(dataset_paths) == 1 else f"{len(dataset_paths)}seed_gnn_model"
        output_path = Path("outputs") / "wms" / f"{first.stem}_{suffix}.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "edge_index": data["edge_index"],
            "target_columns": TARGET_COLUMNS,
            "normalizers": norm,
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "layers": int(layers),
            "request_dim": int(request_dim),
            "loss_weights": [float(x) for x in loss_weights],
            "metrics": metrics,
        },
        output_path,
    )
    return output_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["outputs/wms/v2_drift_heuristic_lookahead_delta_d2_gnn.npz"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--request-dim", type=int, default=32)
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[0.3, 1.0, 0.3])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--future-weight", type=float, default=0.8)
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
        heads=int(args.heads),
        layers=int(args.layers),
        request_dim=int(args.request_dim),
        loss_weights=[float(x) for x in args.loss_weights],
        val_fraction=float(args.val_fraction),
        future_weight=float(args.future_weight),
        seed=int(args.seed),
        device_name=args.device,
    )
    print("GNN WM-S model trained")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
