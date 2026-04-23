from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edge_sim.env.request import Request


@dataclass
class AllocationResult:
    total_delay: float
    compute_delay: float
    transmission_delay: float
    gamma: np.ndarray
    link_load: np.ndarray
    f_alloc: dict[tuple[int, int, int], float] = field(default_factory=dict)
    r_alloc: dict[tuple[int, int, int], float] = field(default_factory=dict)
    infeasible: bool = False


def kkt_load_cost(gamma: np.ndarray, link_load: np.ndarray, compute_cap: np.ndarray, bandwidth: np.ndarray) -> float:
    comp = float(np.sum((gamma**2) / np.maximum(compute_cap, 1e-8)))
    active_links = link_load > 0
    if np.any(np.logical_and(active_links, bandwidth <= 0)):
        return 1e9
    tran = float(np.sum((link_load[active_links] ** 2) / np.maximum(bandwidth[active_links], 1e-8)))
    return comp + tran


def add_stage_to_load(
    gamma: np.ndarray,
    link_load: np.ndarray,
    request: Request,
    stage_idx: int,
    prev_node: int,
    node: int,
) -> None:
    gamma[node] += float(np.sqrt(max(request.compute[stage_idx], 0.0)))
    if prev_node != node:
        d_in = request.input_data if stage_idx == 0 else float(request.stage_data[stage_idx - 1])
        link_load[prev_node, node] += float(np.sqrt(max(d_in, 0.0)))


def path_to_load(
    requests: list[Request],
    schedules: dict[int, list[int]],
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.zeros(num_nodes, dtype=np.float32)
    link_load = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    by_id = {req.request_id: req for req in requests}
    for req_id, path in schedules.items():
        req = by_id[req_id]
        prev = req.source_node
        for j, node in enumerate(path):
            add_stage_to_load(gamma, link_load, req, j, prev, int(node))
            prev = int(node)
    return gamma, link_load


class KKTAllocator:
    """Closed-form lower-level allocation under fixed deployment and schedules."""

    def __init__(self, compute_cap: np.ndarray, bandwidth: np.ndarray):
        self.compute_cap = compute_cap.astype(np.float32)
        self.bandwidth = bandwidth.astype(np.float32)
        self.num_nodes = int(compute_cap.shape[0])

    def incremental_cost(
        self,
        gamma_before: np.ndarray,
        link_before: np.ndarray,
        gamma_after: np.ndarray,
        link_after: np.ndarray,
    ) -> float:
        return kkt_load_cost(gamma_after, link_after, self.compute_cap, self.bandwidth) - kkt_load_cost(
            gamma_before, link_before, self.compute_cap, self.bandwidth
        )

    def allocate(self, requests: list[Request], schedules: dict[int, list[int]]) -> AllocationResult:
        gamma, link_load = path_to_load(requests, schedules, self.num_nodes)
        infeasible = bool(np.any(np.logical_and(link_load > 0, self.bandwidth <= 0)))
        if infeasible:
            return AllocationResult(1e9, 1e9, 1e9, gamma, link_load, infeasible=True)

        compute_delay = float(np.sum((gamma**2) / np.maximum(self.compute_cap, 1e-8)))
        active_links = link_load > 0
        transmission_delay = float(
            np.sum((link_load[active_links] ** 2) / np.maximum(self.bandwidth[active_links], 1e-8))
        )

        f_alloc: dict[tuple[int, int, int], float] = {}
        r_alloc: dict[tuple[int, int, int], float] = {}
        by_id = {req.request_id: req for req in requests}

        for req_id, path in schedules.items():
            req = by_id[req_id]
            prev = req.source_node
            for j, node in enumerate(path):
                node = int(node)
                sqrt_c = float(np.sqrt(max(req.compute[j], 0.0)))
                if gamma[node] > 0:
                    f_alloc[(req_id, j, node)] = float(self.compute_cap[node] * sqrt_c / gamma[node])

                if prev != node:
                    d_in = req.input_data if j == 0 else float(req.stage_data[j - 1])
                    sqrt_d = float(np.sqrt(max(d_in, 0.0)))
                    if link_load[prev, node] > 0:
                        r_alloc[(req_id, j, node)] = float(
                            self.bandwidth[prev, node] * sqrt_d / link_load[prev, node]
                        )
                prev = node

        return AllocationResult(
            total_delay=compute_delay + transmission_delay,
            compute_delay=compute_delay,
            transmission_delay=transmission_delay,
            gamma=gamma,
            link_load=link_load,
            f_alloc=f_alloc,
            r_alloc=r_alloc,
            infeasible=False,
        )
