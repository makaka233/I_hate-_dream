from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from edge_sim.agents.scheduler_policy import SchedulerGNNPolicy


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
                    greedy_action = data["greedy_action"].astype(np.int64)
                    chunks[key].append((greedy_action != data["best_action"].astype(np.int64)).astype(np.bool_))
            elif key == "hard_gap":
                if key in data.files:
                    chunks[key].append(data[key].astype(np.float32))
                else:
                    legal = data["legal_mask"].astype(bool)
                    target_score = data["targets"][:, :, 2]
                    best_action = data["best_action"].astype(np.int64)
                    greedy_action = data["greedy_action"].astype(np.int64)
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
    }


def _apply_normalizers(data: dict[str, np.ndarray], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "node_feat": ((data["node_feat"] - norm["node_mean"]) / norm["node_std"]).astype(np.float32),
        "edge_attr": ((data["edge_attr"] - norm["edge_mean"]) / norm["edge_std"]).astype(np.float32),
        "candidate_edge_attr": (
            (data["candidate_edge_attr"] - norm["candidate_edge_mean"]) / norm["candidate_edge_std"]
        ).astype(np.float32),
        "request_feat": ((data["request_feat"] - norm["request_mean"]) / norm["request_std"]).astype(np.float32),
    }


def _make_sample_weights(hard_mask: np.ndarray, hard_gap: np.ndarray, hard_weight: float) -> np.ndarray:
    weights = np.ones(hard_mask.shape[0], dtype=np.float32)
    weights[hard_mask.astype(bool)] = float(hard_weight)
    positive_gap = np.maximum(hard_gap.astype(np.float32), 0.0)
    if np.any(positive_gap > 0):
        scale = float(np.percentile(positive_gap[positive_gap > 0], 75))
        if scale > 1e-8:
            weights += hard_mask.astype(np.float32) * np.clip(positive_gap / scale, 0.0, 1.5)
    return weights


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor, legal_mask: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~legal_mask.bool(), -1e9)
    loss = F.cross_entropy(masked_logits, labels, reduction="none")
    return (loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _teacher_kl(
    logits: torch.Tensor,
    target_score: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    tau = max(float(temperature), 1e-6)
    masked_logits = logits.masked_fill(~legal_mask.bool(), -1e9)
    teacher_logits = (-target_score / tau).masked_fill(~legal_mask.bool(), -1e9)
    teacher_prob = torch.softmax(teacher_logits, dim=1)
    student_log_prob = torch.log_softmax(masked_logits / tau, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (tau * tau)
    return (loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _pairwise_rank_loss(
    logits: torch.Tensor,
    target_score: torch.Tensor,
    best_action: torch.Tensor,
    legal_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    rank_temperature: float,
) -> torch.Tensor:
    num_candidates = logits.shape[1]
    masked_logits = logits.masked_fill(~legal_mask.bool(), -1e9)
    best_logits = masked_logits.gather(1, best_action[:, None]).expand(-1, num_candidates)
    best_score = target_score.gather(1, best_action[:, None]).expand(-1, num_candidates)
    score_gap = torch.clamp(target_score - best_score, min=0.0)
    gap_scale = torch.clamp(score_gap[legal_mask.bool()].mean(), min=1e-6)
    pair_weights = score_gap / gap_scale
    mask = legal_mask.float() * (1.0 - F.one_hot(best_action, num_classes=num_candidates).to(dtype=logits.dtype))
    tau = max(float(rank_temperature), 1e-6)
    pair_loss = F.softplus((masked_logits - best_logits) / tau) * pair_weights * mask
    per_sample = pair_loss.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
    return (per_sample * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)


def _predict(
    model: SchedulerGNNPolicy,
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


def _subset_accuracy(pred: np.ndarray, truth: np.ndarray, subset: np.ndarray) -> float:
    if not np.any(subset):
        return 0.0
    return float(np.mean(pred[subset] == truth[subset]))


def _avg_regret(target_score: np.ndarray, pred_action: np.ndarray, best_action: np.ndarray) -> float:
    chosen = target_score[np.arange(target_score.shape[0]), pred_action]
    best = target_score[np.arange(target_score.shape[0]), best_action]
    return float(np.mean(chosen - best))


def _metrics(
    logits: np.ndarray,
    target_score: np.ndarray,
    legal_mask: np.ndarray,
    best_action: np.ndarray,
    hard_mask: np.ndarray,
) -> dict[str, float]:
    masked_logits = np.where(legal_mask.astype(bool), logits, -1e9)
    pred_action = np.argmax(masked_logits, axis=1)
    hard = hard_mask.astype(bool)
    return {
        "top1_accuracy": float(np.mean(pred_action == best_action)),
        "top1_accuracy_hard": _subset_accuracy(pred_action, best_action, hard),
        "top1_accuracy_easy": _subset_accuracy(pred_action, best_action, ~hard),
        "hard_ratio": float(np.mean(hard)),
        "avg_regret_target_score": _avg_regret(target_score, pred_action, best_action),
        "avg_regret_hard": _avg_regret(target_score[hard], pred_action[hard], best_action[hard]) if np.any(hard) else 0.0,
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
    teacher_coef: float,
    teacher_temperature: float,
    hard_weight: float,
    pairwise_coef: float,
    pairwise_temperature: float,
    val_fraction: float,
    seed: int,
    device_name: str,
) -> tuple[Path, dict[str, float]]:
    data = _load_datasets(dataset_paths)
    train_idx, val_idx = _split_indices(data["node_feat"].shape[0], val_fraction, seed)
    norm = _normalizers(data, train_idx)
    tensors = _apply_normalizers(data, norm)
    tensors["prev_node"] = data["prev_node"].astype(np.int64)
    tensors["legal_mask"] = data["legal_mask"].astype(np.bool_)
    tensors["best_action"] = data["best_action"].astype(np.int64)
    tensors["hard_mask"] = data["hard_mask"].astype(np.bool_)
    tensors["hard_gap"] = data["hard_gap"].astype(np.float32)
    tensors["target_score"] = data["targets"][:, :, TARGET_COLUMNS.index("target_score")].astype(np.float32)
    sample_weights = _make_sample_weights(tensors["hard_mask"], tensors["hard_gap"], hard_weight)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
    model = SchedulerGNNPolicy(
        hidden_dim=hidden_dim,
        heads=heads,
        layers=layers,
        request_dim=request_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.from_numpy(tensors["node_feat"][train_idx]),
        torch.from_numpy(tensors["edge_attr"][train_idx]),
        torch.from_numpy(tensors["candidate_edge_attr"][train_idx]),
        torch.from_numpy(tensors["request_feat"][train_idx]),
        torch.from_numpy(tensors["prev_node"][train_idx]),
        torch.from_numpy(tensors["legal_mask"][train_idx]),
        torch.from_numpy(tensors["best_action"][train_idx]),
        torch.from_numpy(tensors["target_score"][train_idx]),
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
        for node_feat, edge_attr, cand_edge, req_feat, prev_node, legal_mask, best_action, target_score, batch_weights in loader:
            node_feat = node_feat.to(device)
            edge_attr = edge_attr.to(device)
            cand_edge = cand_edge.to(device)
            req_feat = req_feat.to(device)
            prev_node = prev_node.to(device)
            legal_mask = legal_mask.to(device)
            best_action = best_action.to(device)
            target_score = target_score.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(node_feat, edge_index, edge_attr, cand_edge, req_feat, prev_node)
            ce_loss = _masked_ce(logits, best_action, legal_mask, batch_weights)
            teacher_loss = _teacher_kl(logits, target_score, legal_mask, teacher_temperature, batch_weights)
            pairwise_loss = _pairwise_rank_loss(
                logits,
                target_score,
                best_action,
                legal_mask,
                batch_weights,
                pairwise_temperature,
            )
            loss = ce_loss + float(teacher_coef) * teacher_loss + float(pairwise_coef) * pairwise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_logits = _predict(model, edge_index, tensors, val_idx, batch_size, device)
        val_target_score = tensors["target_score"][val_idx]
        val_mask = tensors["legal_mask"][val_idx]
        val_best_action = tensors["best_action"][val_idx]
        val_weights = sample_weights[val_idx]
        masked_val_logits = np.where(val_mask, val_logits, -1e9)
        ce_val = F.cross_entropy(
            torch.from_numpy(masked_val_logits.astype(np.float32)),
            torch.from_numpy(val_best_action.astype(np.int64)),
            reduction="none",
        ).numpy()
        teacher_val = (
            F.kl_div(
                torch.log_softmax(torch.from_numpy(masked_val_logits.astype(np.float32)) / max(float(teacher_temperature), 1e-6), dim=1),
                torch.softmax((-torch.from_numpy(val_target_score.astype(np.float32)) / max(float(teacher_temperature), 1e-6)).masked_fill(~torch.from_numpy(val_mask.astype(bool)), -1e9), dim=1),
                reduction="none",
            )
            .sum(dim=1)
            .numpy()
            * max(float(teacher_temperature), 1e-6) ** 2
        )
        pairwise_val = float(
            _pairwise_rank_loss(
                torch.from_numpy(val_logits.astype(np.float32)),
                torch.from_numpy(val_target_score.astype(np.float32)),
                torch.from_numpy(val_best_action.astype(np.int64)),
                torch.from_numpy(val_mask.astype(np.bool_)),
                torch.from_numpy(val_weights.astype(np.float32)),
                pairwise_temperature,
            ).item()
        )
        val_loss = float(
            ((ce_val + float(teacher_coef) * teacher_val) * val_weights).sum()
            / max(float(val_weights.sum()), 1e-8)
            + float(pairwise_coef) * pairwise_val
        )
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

    val_logits = _predict(model, edge_index, tensors, val_idx, batch_size, device)
    metrics = _metrics(
        val_logits,
        tensors["target_score"][val_idx],
        tensors["legal_mask"][val_idx],
        tensors["best_action"][val_idx],
        tensors["hard_mask"][val_idx],
    )
    metrics.update(
        {
            "train_decisions": float(train_idx.size),
            "val_decisions": float(val_idx.size),
            "best_val_loss": best_val_loss,
        }
    )

    if output_path is None:
        first = Path(dataset_paths[0])
        suffix = "agents_model" if len(dataset_paths) == 1 else f"{len(dataset_paths)}seed_agents_model"
        output_path = Path("outputs") / "agent_s" / f"{first.stem}_{suffix}.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "layers": int(layers),
            "request_dim": int(request_dim),
            "normalizers": {k: v.astype(np.float32) for k, v in norm.items()},
            "teacher_coef": float(teacher_coef),
            "teacher_temperature": float(teacher_temperature),
            "hard_weight": float(hard_weight),
            "pairwise_coef": float(pairwise_coef),
            "pairwise_temperature": float(pairwise_temperature),
            "metrics": metrics,
        },
        output_path,
    )
    return output_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["outputs/wms/v2_drift_heuristic_mixed_gnn_d2.npz"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--request-dim", type=int, default=32)
    parser.add_argument("--teacher-coef", type=float, default=0.3)
    parser.add_argument("--teacher-temperature", type=float, default=1.5)
    parser.add_argument("--hard-weight", type=float, default=2.5)
    parser.add_argument("--pairwise-coef", type=float, default=0.4)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
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
        heads=int(args.heads),
        layers=int(args.layers),
        request_dim=int(args.request_dim),
        teacher_coef=float(args.teacher_coef),
        teacher_temperature=float(args.teacher_temperature),
        hard_weight=float(args.hard_weight),
        pairwise_coef=float(args.pairwise_coef),
        pairwise_temperature=float(args.pairwise_temperature),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        device_name=args.device,
    )
    print("Agent-S trained")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
