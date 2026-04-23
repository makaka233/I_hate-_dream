from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
        "greedy_action": [],
        "hard_mask": [],
        "hard_gap": [],
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
            elif key == "greedy_action":
                if key in data.files:
                    chunks[key].append(data[key].astype(np.int64))
                else:
                    legal = data["legal_mask"].astype(bool)
                    current_delta = np.where(legal, data["targets"][:, :, 0], np.inf)
                    chunks[key].append(np.argmin(current_delta, axis=1).astype(np.int64))
            elif key == "hard_mask":
                if key in data.files:
                    chunks[key].append(data[key].astype(np.bool_))
                else:
                    if "greedy_action" in data.files:
                        greedy_action = data["greedy_action"].astype(np.int64)
                    else:
                        legal = data["legal_mask"].astype(bool)
                        current_delta = np.where(legal, data["targets"][:, :, 0], np.inf)
                        greedy_action = np.argmin(current_delta, axis=1).astype(np.int64)
                    chunks[key].append((greedy_action != data["best_action"].astype(np.int64)).astype(np.bool_))
            elif key == "hard_gap":
                if key in data.files:
                    chunks[key].append(data[key].astype(np.float32))
                else:
                    legal = data["legal_mask"].astype(bool)
                    target_score = data["targets"][:, :, 2]
                    best_action = data["best_action"].astype(np.int64)
                    if "greedy_action" in data.files:
                        greedy_action = data["greedy_action"].astype(np.int64)
                    else:
                        current_delta = np.where(legal, data["targets"][:, :, 0], np.inf)
                        greedy_action = np.argmin(current_delta, axis=1).astype(np.int64)
                    best_score = target_score[np.arange(target_score.shape[0]), best_action]
                    greedy_score = target_score[np.arange(target_score.shape[0]), greedy_action]
                    chunks[key].append((greedy_score - best_score).astype(np.float32))
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
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    mask = legal_mask.unsqueeze(-1).float()
    weighted = ((pred - target) ** 2) * mask * loss_weights.view(1, 1, -1) * sample_weights.view(-1, 1, 1)
    denom = torch.clamp((legal_mask.float().sum(dim=1) * sample_weights).sum() * loss_weights.sum(), min=1.0)
    return weighted.sum() / denom


def _ranking_ce(
    pred: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    future_weight: float,
    rank_temperature: float,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    pred_raw = pred * target_std.view(1, 1, -1) + target_mean.view(1, 1, -1)
    target_raw = target * target_std.view(1, 1, -1) + target_mean.view(1, 1, -1)
    pred_score = target_raw[:, :, 0].detach() + future_weight * pred_raw[:, :, 1]
    true_score = target_raw[:, :, 2]

    pred_logits = (-pred_score / max(rank_temperature, 1e-6)).masked_fill(~legal_mask.bool(), -1e9)
    true_best = true_score.masked_fill(~legal_mask.bool(), 1e9).argmin(dim=1)
    loss = F.cross_entropy(pred_logits, true_best, reduction="none")
    return (loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _ranking_ce_np(
    pred: np.ndarray,
    target: np.ndarray,
    legal_mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    future_weight: float,
    rank_temperature: float,
    sample_weights: np.ndarray,
) -> float:
    pred_raw = pred * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
    target_raw = target * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
    legal = legal_mask.astype(bool)
    pred_score = target_raw[:, :, 0] + future_weight * pred_raw[:, :, 1]
    true_score = np.where(legal, target_raw[:, :, 2], np.inf)
    true_best = np.argmin(true_score, axis=1)

    logits = -pred_score / max(rank_temperature, 1e-6)
    logits = np.where(legal, logits, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
    losses = -log_probs[np.arange(log_probs.shape[0]), true_best]
    return float(np.sum(losses * sample_weights) / max(float(sample_weights.sum()), 1e-8))


def _make_sample_weights(hard_mask: np.ndarray, hard_gap: np.ndarray, hard_weight: float) -> np.ndarray:
    weights = np.ones(hard_mask.shape[0], dtype=np.float32)
    weights[hard_mask.astype(bool)] = float(hard_weight)
    positive_gap = np.maximum(hard_gap.astype(np.float32), 0.0)
    if np.any(positive_gap > 0):
        scale = float(np.percentile(positive_gap[positive_gap > 0], 75))
        if scale > 1e-8:
            weights += hard_mask.astype(np.float32) * np.clip(positive_gap / scale, 0.0, 1.5)
    return weights


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
    hard_mask: np.ndarray,
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
    hard = hard_mask.astype(bool)
    easy = ~hard

    def _subset_acc(choice: np.ndarray, truth: np.ndarray, subset: np.ndarray) -> float:
        if not np.any(subset):
            return 0.0
        return float(np.mean(choice[subset] == truth[subset]))

    return {
        "mae_current_delta": float(np.mean(np.abs(err[:, 0]))),
        "mae_future_cost": float(np.mean(np.abs(err[:, 1]))),
        "mae_target_score": float(np.mean(np.abs(err[:, 2]))),
        "rmse_target_score": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "top1_pred_target_score": float(np.mean(pred_best == true_best)),
        "top1_exact_delta_pred_future": float(np.mean(exact_future_best == true_best)),
        "hard_ratio": float(np.mean(hard)),
        "top1_pred_target_score_hard": _subset_acc(pred_best, true_best, hard),
        "top1_pred_target_score_easy": _subset_acc(pred_best, true_best, easy),
        "top1_exact_delta_pred_future_hard": _subset_acc(exact_future_best, true_best, hard),
        "top1_exact_delta_pred_future_easy": _subset_acc(exact_future_best, true_best, easy),
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
    rank_coef: float,
    rank_temperature: float,
    hard_weight: float,
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
    tensors["hard_mask"] = data["hard_mask"].astype(np.bool_)
    tensors["hard_gap"] = data["hard_gap"].astype(np.float32)
    sample_weights = _make_sample_weights(tensors["hard_mask"], tensors["hard_gap"], hard_weight)

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
    target_mean_t = torch.as_tensor(norm["target_mean"], dtype=torch.float32, device=device)
    target_std_t = torch.as_tensor(norm["target_std"], dtype=torch.float32, device=device)

    train_ds = TensorDataset(
        torch.from_numpy(tensors["node_feat"][train_idx]),
        torch.from_numpy(tensors["edge_attr"][train_idx]),
        torch.from_numpy(tensors["candidate_edge_attr"][train_idx]),
        torch.from_numpy(tensors["request_feat"][train_idx]),
        torch.from_numpy(tensors["prev_node"][train_idx]),
        torch.from_numpy(tensors["targets"][train_idx]),
        torch.from_numpy(tensors["legal_mask"][train_idx]),
        torch.from_numpy(sample_weights[train_idx]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for node_feat, edge_attr, cand_edge, req_feat, prev_node, targets, legal_mask, batch_weights in loader:
            node_feat = node_feat.to(device)
            edge_attr = edge_attr.to(device)
            cand_edge = cand_edge.to(device)
            req_feat = req_feat.to(device)
            prev_node = prev_node.to(device)
            targets = targets.to(device)
            legal_mask = legal_mask.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(node_feat, edge_index, edge_attr, cand_edge, req_feat, prev_node)
            mse_loss = _masked_mse(pred, targets, legal_mask, loss_weight_t, batch_weights)
            rank_loss = _ranking_ce(
                pred,
                targets,
                legal_mask,
                target_mean_t,
                target_std_t,
                future_weight,
                rank_temperature,
                batch_weights,
            )
            loss = mse_loss + rank_coef * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred = _predict(model, edge_index, tensors, val_idx, batch_size, device)
        val_target = tensors["targets"][val_idx]
        val_mask = tensors["legal_mask"][val_idx]
        val_weights = sample_weights[val_idx]
        val_sqerr = ((val_pred - val_target) ** 2).mean(axis=2)
        valid_counts = np.maximum(val_mask.sum(axis=1), 1)
        per_sample_mse = (val_sqerr * val_mask).sum(axis=1) / valid_counts
        val_mse = float(np.sum(per_sample_mse * val_weights) / max(float(val_weights.sum()), 1e-8))
        val_rank = _ranking_ce_np(
            val_pred,
            val_target,
            val_mask,
            norm["target_mean"],
            norm["target_std"],
            future_weight,
            rank_temperature,
            val_weights,
        )
        val_loss = val_mse + rank_coef * val_rank
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
        tensors["hard_mask"][val_idx],
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
    parser.add_argument("--rank-coef", type=float, default=0.2)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--hard-weight", type=float, default=3.0)
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
        rank_coef=float(args.rank_coef),
        rank_temperature=float(args.rank_temperature),
        hard_weight=float(args.hard_weight),
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
