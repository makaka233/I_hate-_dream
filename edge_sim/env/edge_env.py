from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from edge_sim.env.deployment import make_deployment, resource_usage
from edge_sim.env.request import Request, RequestGenerator


class EdgeEnv:
    """Static V1 edge environment with batched request arrivals."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg["seed"]))

        sys_cfg = cfg["system"]
        self.num_nodes = int(sys_cfg["num_nodes"])
        self.users_per_node = int(sys_cfg["users_per_node"])
        self.service_stages = [int(x) for x in sys_cfg["service_stages"]]
        self.num_services = len(self.service_stages)
        self.max_stages = max(self.service_stages)
        self.slow_period = int(sys_cfg["slow_period"])

        self.adj, self.bandwidth = self._build_topology()
        self.reachability, self.effective_bandwidth = self._build_effective_links()
        self.compute_cap = self.rng.uniform(
            cfg["resources"]["compute_min"], cfg["resources"]["compute_max"], size=self.num_nodes
        ).astype(np.float32)
        self.storage_cap = np.full(self.num_nodes, cfg["resources"]["storage_capacity"], dtype=np.float32)
        self.memory_cap = np.full(self.num_nodes, cfg["resources"]["memory_capacity"], dtype=np.float32)

        self.service_storage = np.zeros((self.num_services, self.max_stages), dtype=np.float32)
        self.service_memory = np.zeros((self.num_services, self.max_stages), dtype=np.float32)
        for i, stages in enumerate(self.service_stages):
            self.service_storage[i, :stages] = self.rng.uniform(
                cfg["resources"]["service_storage_min"], cfg["resources"]["service_storage_max"], size=stages
            )
            self.service_memory[i, :stages] = self.rng.uniform(
                cfg["resources"]["service_memory_min"], cfg["resources"]["service_memory_max"], size=stages
            )

        self.request_generator = RequestGenerator(cfg, self.service_stages, self.users_per_node, self.rng)
        self.edge_index, self.edge_attr_static, self.edge_lookup = self._build_edges()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EdgeEnv":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def _build_topology(self) -> tuple[np.ndarray, np.ndarray]:
        topo = self.cfg["topology"]
        mode = topo.get("mode", "full_mesh")
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        bandwidth = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        def add_link(u: int, v: int, low: float, high: float) -> None:
            if u == v:
                return
            rate = self.rng.uniform(low, high)
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            bandwidth[u, v] = rate
            bandwidth[v, u] = rate

        if mode == "full_mesh":
            for u in range(self.num_nodes):
                for v in range(u + 1, self.num_nodes):
                    add_link(u, v, topo["bandwidth_min"], topo["bandwidth_max"])
        elif mode == "ring":
            for u in range(self.num_nodes):
                add_link(u, (u + 1) % self.num_nodes, topo["bandwidth_min"], topo["bandwidth_max"])
        elif mode == "clustered":
            num_clusters = int(topo.get("num_clusters", 2))
            clusters = np.array_split(np.arange(self.num_nodes), num_clusters)
            for cluster in clusters:
                for idx, u in enumerate(cluster):
                    for v in cluster[idx + 1 :]:
                        add_link(int(u), int(v), topo["bandwidth_min"], topo["bandwidth_max"])
            inter_low = float(topo.get("inter_bandwidth_min", topo["bandwidth_min"]))
            inter_high = float(topo.get("inter_bandwidth_max", topo["bandwidth_max"]))
            reps = [int(cluster[0]) for cluster in clusters]
            for idx, u in enumerate(reps):
                add_link(u, reps[(idx + 1) % len(reps)], inter_low, inter_high)
            # Add a second bridge between adjacent clusters when possible.
            for idx, cluster in enumerate(clusters):
                if len(cluster) > 1 and len(clusters[(idx + 1) % len(clusters)]) > 1:
                    add_link(int(cluster[1]), int(clusters[(idx + 1) % len(clusters)][1]), inter_low, inter_high)
        else:
            raise ValueError(f"Unsupported topology.mode={mode!r}")
        return adj, bandwidth.astype(np.float32)

    def _build_edges(self) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
        src, dst, attrs = [], [], []
        lookup: dict[tuple[int, int], int] = {}
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u == v or self.reachability[u, v] > 0:
                    lookup[(u, v)] = len(src)
                    src.append(u)
                    dst.append(v)
                    is_self = 1.0 if u == v else 0.0
                    bw = self.effective_bandwidth[u, v] if u != v else float(self.effective_bandwidth[self.effective_bandwidth > 0].mean())
                    attrs.append([is_self, bw / 120.0, 0.0, 0.0])
        return np.array([src, dst], dtype=np.int64), np.array(attrs, dtype=np.float32), lookup

    def _build_effective_links(self) -> tuple[np.ndarray, np.ndarray]:
        reach = np.eye(self.num_nodes, dtype=np.float32)
        eff = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for src in range(self.num_nodes):
            visited = np.zeros(self.num_nodes, dtype=bool)
            queue = [src]
            visited[src] = True
            while queue:
                u = queue.pop(0)
                for v in np.flatnonzero(self.adj[u] > 0):
                    if not visited[v]:
                        visited[v] = True
                        queue.append(int(v))
            reach[src, visited] = 1.0

        # Widest-path bottleneck bandwidth. This lets sparse topology influence
        # transmission delay while allowing multi-hop reachability in scheduling.
        for src in range(self.num_nodes):
            best = np.zeros(self.num_nodes, dtype=np.float32)
            best[src] = np.inf
            used = np.zeros(self.num_nodes, dtype=bool)
            for _ in range(self.num_nodes):
                candidates = np.where(~used, best, -1.0)
                u = int(np.argmax(candidates))
                if candidates[u] <= 0:
                    break
                used[u] = True
                for v in np.flatnonzero(self.adj[u] > 0):
                    bottleneck = min(best[u], self.bandwidth[u, v])
                    if bottleneck > best[v]:
                        best[v] = bottleneck
            for dst in range(self.num_nodes):
                if src != dst and reach[src, dst] > 0:
                    eff[src, dst] = best[dst]
        return reach, eff

    def make_deployment(self, mode: str | None = None) -> np.ndarray:
        dep_cfg = self.cfg["deployment"]
        return make_deployment(
            mode or dep_cfg["mode"],
            self.service_stages,
            self.service_storage,
            self.service_memory,
            self.storage_cap,
            self.memory_cap,
            self.rng,
            int(dep_cfg["min_replicas"]),
            int(dep_cfg["max_replicas"]),
        )

    def sample_requests(self, slow_epoch: int | None = None) -> list[Request]:
        return self.request_generator.sample_slot(self.num_nodes, slow_epoch=slow_epoch)

    def service_probabilities(self, slow_epoch: int | None = None) -> np.ndarray:
        return self.request_generator.service_probabilities(slow_epoch)

    def node_arrival_rates(self, slow_epoch: int | None = None) -> np.ndarray:
        return self.request_generator.node_arrival_rates(self.num_nodes, slow_epoch)

    def node_service_probabilities(self, slow_epoch: int | None = None) -> np.ndarray:
        return self.request_generator.node_service_probabilities(self.num_nodes, slow_epoch)

    def source_service_demand(self, slow_epoch: int | None = None) -> np.ndarray:
        return self.request_generator.source_service_demand(self.num_nodes, slow_epoch)

    def legal_nodes(self, x: np.ndarray, service_id: int, stage_idx: int, prev_node: int) -> np.ndarray:
        deployed = x[service_id, stage_idx] > 0.5
        reachable = (self.reachability[prev_node] > 0.5)
        reachable[prev_node] = True
        return np.logical_and(deployed, reachable)

    def graph_observation(
        self,
        x: np.ndarray,
        gamma_load: np.ndarray,
        link_load: np.ndarray,
        request: Request,
        stage_idx: int,
        prev_node: int,
    ) -> dict:
        storage_used, memory_used = resource_usage(x, self.service_storage, self.service_memory)
        service_id = request.service_id

        node_feat = np.zeros((self.num_nodes, 8), dtype=np.float32)
        node_feat[:, 0] = self.compute_cap / max(float(self.compute_cap.max()), 1.0)
        node_feat[:, 1] = gamma_load / np.maximum(self.compute_cap, 1e-6)
        node_feat[:, 2] = storage_used / np.maximum(self.storage_cap, 1e-6)
        node_feat[:, 3] = memory_used / np.maximum(self.memory_cap, 1e-6)
        node_feat[:, 4] = x[service_id, stage_idx]
        if stage_idx + 1 < self.service_stages[service_id]:
            node_feat[:, 5] = x[service_id, stage_idx + 1]
        node_feat[request.source_node, 6] = 1.0
        node_feat[prev_node, 7] = 1.0

        edge_attr = self.edge_attr_static.copy()
        for idx, (u, v) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            if u == v:
                edge_attr[idx, 2] = 0.0
            else:
                edge_attr[idx, 2] = link_load[u, v] / max(self.effective_bandwidth[u, v], 1e-6)
            edge_attr[idx, 3] = 1.0 if u == prev_node else 0.0

        d_in = request.input_data if stage_idx == 0 else float(request.stage_data[stage_idx - 1])
        req_feat = np.array(
            [
                service_id / max(self.num_services - 1, 1),
                stage_idx / max(self.max_stages - 1, 1),
                request.compute[stage_idx] / 10.0,
                d_in / 10.0,
                (request.num_stages - stage_idx - 1) / max(self.max_stages - 1, 1),
            ],
            dtype=np.float32,
        )
        edge_to_candidates = np.zeros((self.num_nodes, edge_attr.shape[1]), dtype=np.float32)
        for m in range(self.num_nodes):
            edge_to_candidates[m] = edge_attr[self.edge_lookup[(prev_node, m)]]

        return {
            "node_feat": node_feat,
            "edge_index": self.edge_index,
            "edge_attr": edge_attr,
            "candidate_edge_attr": edge_to_candidates,
            "request_feat": req_feat,
            "prev_node": np.array(prev_node, dtype=np.int64),
            "legal_mask": self.legal_nodes(x, service_id, stage_idx, prev_node).astype(np.bool_),
        }
