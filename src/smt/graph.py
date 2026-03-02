"""Network graph generation, path enumeration, and metric computation."""

from __future__ import annotations

import random
from collections.abc import Sequence

from smt.models import (
    AdjacencyList,
    NetworkTopology,
    NodeParams,
    PathMetrics,
)


def compute_path_metrics(
    path: list[int],
    node_params: dict[int, NodeParams],
) -> PathMetrics:
    """Compute per-path leakage and outcome probabilities from node params."""
    non_leak = 1.0
    p_val = 1.0
    p_nd = 1.0  # not-dropped

    for node in path[1:-1]:
        params = node_params.get(node)
        if params is None:
            continue
        non_leak *= 1.0 - params.p_int
        p_val *= params.f
        p_nd *= params.f + params.e  # 1 - d_v

    epsilon = 1.0 - non_leak
    p_drop = 1.0 - p_nd
    p_err = p_nd - p_val

    return PathMetrics(
        epsilon=epsilon,
        p_val=p_val,
        p_err=max(0.0, p_err),
        p_drop=max(0.0, p_drop),
    )


def find_all_paths(
    adj: AdjacencyList,
    src: int,
    dst: int,
    max_paths: int = 50,
) -> list[list[int]]:
    """Enumerate all simple paths from src to dst via iterative DFS."""
    paths: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(src, [src])]

    while stack and len(paths) < max_paths:
        node, path = stack.pop()
        if node == dst:
            paths.append(path)
            continue
        for neighbor in adj.get(node, []):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))

    return paths


def generate_layered_graph(
    nodes_per_layer: Sequence[int],
    edge_prob: float = 0.5,
    bandwidth_range: tuple[int, int] = (2, 8),
    seed: int | None = None,
) -> NetworkTopology:
    """Layered directed graph with probabilistic edges; guarantees connectivity."""
    rng = random.Random(seed)
    layers: list[list[int]] = []
    node_id = 1
    for n in nodes_per_layer:
        layer = list(range(node_id, node_id + n))
        layers.append(layer)
        node_id += n

    sender = layers[0][0]
    receiver = layers[-1][0]
    adj: AdjacencyList = {i: [] for i in range(1, node_id)}
    edge_bw: dict[tuple[int, int], int] = {}

    for i in range(len(layers) - 1):
        for u in layers[i]:
            for v in layers[i + 1]:
                if rng.random() < edge_prob:
                    adj[u].append(v)
                    edge_bw[(u, v)] = rng.randint(*bandwidth_range)

    # Guarantee connectivity: each node has at least one outgoing/incoming edge
    for i in range(len(layers) - 1):
        for u in layers[i]:
            if not adj[u]:
                v = rng.choice(layers[i + 1])
                adj[u].append(v)
                edge_bw[(u, v)] = rng.randint(*bandwidth_range)
    for i in range(1, len(layers)):
        for v in layers[i]:
            has_incoming = any(v in adj[u] for u in layers[i - 1])
            if not has_incoming:
                u = rng.choice(layers[i - 1])
                adj[u].append(v)
                edge_bw[(u, v)] = rng.randint(*bandwidth_range)

    return NetworkTopology(
        adjacency=adj,
        sender=sender,
        receiver=receiver,
        edge_bandwidths=edge_bw,
    )


def barabasi_albert_graph(
    n: int,
    m0: int,
    m: int,
    bandwidth_range: tuple[int, int] = (2, 8),
    seed: int | None = None,
) -> NetworkTopology:
    """Barabasi-Albert preferential attachment graph. Sender=1, receiver=2."""
    if m > m0:
        raise ValueError(f"m ({m}) must be <= m0 ({m0})")
    if n < m0:
        raise ValueError(f"n ({n}) must be >= m0 ({m0})")

    rng = random.Random(seed)

    adj: AdjacencyList = {
        i: [j for j in range(1, m0 + 1) if j != i] for i in range(1, m0 + 1)
    }
    edge_bw: dict[tuple[int, int], int] = {}
    for i in range(1, m0 + 1):
        for j in adj[i]:
            if (i, j) not in edge_bw:
                bw = rng.randint(*bandwidth_range)
                edge_bw[(i, j)] = bw
                edge_bw[(j, i)] = bw

    for new_node in range(m0 + 1, n + 1):
        total_degree = sum(len(edges) for edges in adj.values())
        weights = [len(adj[node]) / total_degree for node in sorted(adj.keys())]
        targets = rng.choices(sorted(adj.keys()), weights=weights, k=m)

        adj[new_node] = []
        for target in targets:
            bw = rng.randint(*bandwidth_range)
            adj[new_node].append(target)
            adj[target].append(new_node)
            edge_bw[(new_node, target)] = bw
            edge_bw[(target, new_node)] = bw

    return NetworkTopology(
        adjacency=adj,
        sender=1,
        receiver=2,
        edge_bandwidths=edge_bw,
    )


def assign_adversary_params(
    topology: NetworkTopology,
    n_compromised: int,
    p_int_range: tuple[float, float] = (0.05, 0.25),
    delta_range: tuple[float, float] = (0.1, 0.5),
    theta_range: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
) -> dict[int, NodeParams]:
    """Randomly select compromised nodes and assign adversary parameters."""
    rng = random.Random(seed)
    eligible = [
        n for n in topology.nodes if n != topology.sender and n != topology.receiver
    ]
    n_compromised = min(n_compromised, len(eligible))
    compromised = rng.sample(eligible, n_compromised)

    params: dict[int, NodeParams] = {}
    for node in compromised:
        p_int = round(rng.uniform(*p_int_range), 3)
        delta = round(rng.uniform(*delta_range), 3)
        theta = round(rng.uniform(*theta_range), 3)
        if delta + theta > 1.0:
            theta = round(1.0 - delta, 3)
        params[node] = NodeParams(p_int=p_int, delta=delta, theta=theta)

    return params


def build_topology(
    topology: NetworkTopology,
    max_paths: int = 50,
) -> NetworkTopology:
    """Enumerate paths and compute per-path metrics from node_params."""
    topology.paths = find_all_paths(
        topology.adjacency, topology.sender, topology.receiver, max_paths
    )
    topology.path_metrics = [
        compute_path_metrics(p, topology.node_params) for p in topology.paths
    ]
    return topology
