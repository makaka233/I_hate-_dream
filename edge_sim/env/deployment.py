from __future__ import annotations

import numpy as np


def _can_place(x: np.ndarray, storage_req: np.ndarray, memory_req: np.ndarray, storage_cap: np.ndarray, memory_cap: np.ndarray, i: int, j: int, m: int) -> bool:
    if x[i, j, m] > 0:
        return False
    storage_used = (x[:, :, m] * storage_req).sum()
    memory_used = (x[:, :, m] * memory_req).sum()
    return storage_used + storage_req[i, j] <= storage_cap[m] and memory_used + memory_req[i, j] <= memory_cap[m]


def resource_usage(x: np.ndarray, storage_req: np.ndarray, memory_req: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    storage_used = (x * storage_req[:, :, None]).sum(axis=(0, 1))
    memory_used = (x * memory_req[:, :, None]).sum(axis=(0, 1))
    return storage_used.astype(np.float32), memory_used.astype(np.float32)


def deployment_summary(
    x: np.ndarray,
    service_stages: list[int],
    storage_req: np.ndarray,
    memory_req: np.ndarray,
    storage_cap: np.ndarray,
    memory_cap: np.ndarray,
) -> dict[str, float]:
    storage_used, memory_used = resource_usage(x, storage_req, memory_req)
    valid_replica_counts = []
    for i, stages in enumerate(service_stages):
        for j in range(stages):
            valid_replica_counts.append(float(x[i, j].sum()))

    node_counts = x.sum(axis=(0, 1))
    storage_util = storage_used / np.maximum(storage_cap, 1e-8)
    memory_util = memory_used / np.maximum(memory_cap, 1e-8)
    return {
        "total_replicas": float(sum(valid_replica_counts)),
        "min_stage_replicas": float(min(valid_replica_counts)),
        "max_stage_replicas": float(max(valid_replica_counts)),
        "avg_node_replicas": float(node_counts.mean()),
        "max_node_replicas": float(node_counts.max()),
        "avg_storage_util": float(storage_util.mean()),
        "max_storage_util": float(storage_util.max()),
        "avg_memory_util": float(memory_util.mean()),
        "max_memory_util": float(memory_util.max()),
    }


def format_deployment_summary(summary: dict[str, float]) -> str:
    return (
        "replicas total={total_replicas:.0f}, stage[min,max]=[{min_stage_replicas:.0f},{max_stage_replicas:.0f}], "
        "node avg/max={avg_node_replicas:.2f}/{max_node_replicas:.0f}, "
        "storage avg/max={avg_storage_util:.2f}/{max_storage_util:.2f}, "
        "memory avg/max={avg_memory_util:.2f}/{max_memory_util:.2f}"
    ).format(**summary)


def make_deployment(
    mode: str,
    service_stages: list[int],
    storage_req: np.ndarray,
    memory_req: np.ndarray,
    storage_cap: np.ndarray,
    memory_cap: np.ndarray,
    rng: np.random.Generator,
    min_replicas: int = 1,
    max_replicas: int = 2,
) -> np.ndarray:
    """Create a feasible deployment matrix x[i,j,m]."""

    num_services = len(service_stages)
    max_stages = max(service_stages)
    num_nodes = storage_cap.shape[0]
    x = np.zeros((num_services, max_stages, num_nodes), dtype=np.float32)

    if mode == "monolithic":
        for i, stages in enumerate(service_stages):
            best_node = None
            best_score = -np.inf
            for m in range(num_nodes):
                storage_need = storage_req[i, :stages].sum()
                memory_need = memory_req[i, :stages].sum()
                storage_used, memory_used = resource_usage(x, storage_req, memory_req)
                if storage_used[m] + storage_need <= storage_cap[m] and memory_used[m] + memory_need <= memory_cap[m]:
                    score = (storage_cap[m] - storage_used[m]) + (memory_cap[m] - memory_used[m])
                    if score > best_score:
                        best_score = score
                        best_node = m
            if best_node is None:
                raise RuntimeError(f"Cannot place monolithic service {i}; increase storage/memory capacity.")
            x[i, :stages, best_node] = 1.0
        return x

    stages_flat = [(i, j) for i, stages in enumerate(service_stages) for j in range(stages)]
    if mode == "random":
        rng.shuffle(stages_flat)

    def ranked_nodes(i: int, j: int) -> list[int]:
        storage_used, memory_used = resource_usage(x, storage_req, memory_req)
        deployed_count = x.sum(axis=(0, 1))
        scores = (storage_cap - storage_used) / storage_cap + (memory_cap - memory_used) / memory_cap
        if mode == "heuristic":
            scores -= 0.05 * deployed_count
            scores += 0.01 * rng.normal(size=num_nodes)
            return list(np.argsort(-scores))
        if mode == "fixed":
            return list(np.argsort(deployed_count))
        return list(rng.permutation(num_nodes))

    for replicas in range(min_replicas):
        for i, j in stages_flat:
            placed = False
            for m in ranked_nodes(i, j):
                if _can_place(x, storage_req, memory_req, storage_cap, memory_cap, i, j, m):
                    x[i, j, m] = 1.0
                    placed = True
                    break
            if not placed:
                raise RuntimeError(f"Cannot satisfy min replica for service stage ({i},{j}).")

    if mode in {"heuristic", "fixed", "random"} and max_replicas > min_replicas:
        # Add extra replicas when capacity allows. The modes differ only in node ranking.
        for i, j in stages_flat:
            while int(x[i, j].sum()) < max_replicas:
                placed = False
                for m in ranked_nodes(i, j):
                    if _can_place(x, storage_req, memory_req, storage_cap, memory_cap, i, j, m):
                        x[i, j, m] = 1.0
                        placed = True
                        break
                if not placed:
                    break

    return x
