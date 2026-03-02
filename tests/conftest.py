"""Shared test fixtures for the SMT test suite."""

from __future__ import annotations

import pytest

from smt.graph import compute_path_metrics
from smt.models import NetworkTopology, NodeParams, PathMetrics


@pytest.fixture
def simple_3path_network() -> tuple[
    list[list[int]], dict[int, NodeParams], list[PathMetrics]
]:
    """3-path network from the Phase II notebook tests.

    Topology:
        S(1) -> A(2) -> D(5) -> R(3)
        S(1) -> B(4) -> D(5) -> R(3)
        S(1) -> C(6) -> R(3)
    """
    paths = [
        [1, 2, 5, 3],
        [1, 4, 5, 3],
        [1, 6, 3],
    ]
    node_params = {
        2: NodeParams(p_int=0.1, delta=0.3),
        4: NodeParams(p_int=0.15, delta=0.2),
        5: NodeParams(p_int=0.05, delta=0.5),
        6: NodeParams(p_int=0.2, delta=0.1),
    }
    metrics = [compute_path_metrics(p, node_params) for p in paths]
    return paths, node_params, metrics


@pytest.fixture
def simple_topology(
    simple_3path_network: tuple[list[list[int]], dict[int, NodeParams], list[PathMetrics]],
) -> NetworkTopology:
    """NetworkTopology wrapping the 3-path network."""
    paths, node_params, metrics = simple_3path_network
    adj = {
        1: [2, 4, 6],
        2: [5],
        3: [],
        4: [5],
        5: [3],
        6: [3],
    }
    edge_bw = {
        (1, 2): 5, (1, 4): 5, (1, 6): 5,
        (2, 5): 5, (4, 5): 5, (5, 3): 10, (6, 3): 5,
    }
    return NetworkTopology(
        adjacency=adj,
        sender=1,
        receiver=3,
        node_params=node_params,
        paths=paths,
        edge_bandwidths=edge_bw,
        path_metrics=metrics,
    )


@pytest.fixture
def small_prime() -> int:
    """A small prime for fast Shamir tests."""
    return 257
