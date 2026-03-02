"""Tests for smt.graph module."""

from __future__ import annotations

import pytest

from smt.graph import (
    assign_adversary_params,
    barabasi_albert_graph,
    build_topology,
    compute_path_metrics,
    find_all_paths,
    generate_layered_graph,
)
from smt.models import NodeParams


class TestFindAllPaths:
    def test_simple_graph(self):
        adj = {1: [2, 3], 2: [3], 3: []}
        paths = find_all_paths(adj, 1, 3)
        assert sorted(paths) == sorted([[1, 2, 3], [1, 3]])

    def test_no_path(self):
        adj = {1: [2], 2: [], 3: []}
        paths = find_all_paths(adj, 1, 3)
        assert paths == []

    def test_single_node(self):
        adj = {1: []}
        paths = find_all_paths(adj, 1, 1)
        assert paths == [[1]]

    def test_max_paths_cap(self):
        adj = {1: [2, 3, 4], 2: [5], 3: [5], 4: [5], 5: []}
        paths = find_all_paths(adj, 1, 5, max_paths=2)
        assert len(paths) == 2

    def test_cycle_avoidance(self):
        adj = {1: [2], 2: [3, 1], 3: []}
        paths = find_all_paths(adj, 1, 3)
        assert paths == [[1, 2, 3]]


class TestComputePathMetrics:
    def test_no_adversary(self):
        m = compute_path_metrics([1, 2, 3], {})
        assert m.epsilon == 0.0
        assert m.p_val == 1.0
        assert m.p_drop == 0.0

    def test_single_compromised_node(self):
        params = {2: NodeParams(p_int=0.2, delta=0.5)}
        m = compute_path_metrics([1, 2, 3], params)
        assert abs(m.epsilon - 0.2) < 1e-12
        assert abs(m.p_val - 0.9) < 1e-12
        assert abs(m.p_drop - 0.1) < 1e-12

    def test_multiple_nodes_phase2(self):
        params = {
            2: NodeParams(p_int=0.1, delta=0.3),
            5: NodeParams(p_int=0.05, delta=0.5),
        }
        m = compute_path_metrics([1, 2, 5, 3], params)
        expected_eps = 1 - (1 - 0.1) * (1 - 0.05)
        expected_pval = (1 - 0.1 * 0.3) * (1 - 0.05 * 0.5)
        assert abs(m.epsilon - expected_eps) < 1e-10
        assert abs(m.p_val - expected_pval) < 1e-10

    def test_with_tampering(self):
        params = {2: NodeParams(p_int=0.2, delta=0.3, theta=0.2)}
        m = compute_path_metrics([1, 2, 3], params)
        d = 0.2 * 0.3
        e = 0.2 * 0.2
        f = 1 - d - e
        assert abs(m.p_val - f) < 1e-12
        assert abs(m.p_drop - d) < 1e-12
        assert abs(m.p_err - e) < 1e-12


class TestGenerateLayeredGraph:
    def test_basic_connectivity(self):
        topo = generate_layered_graph([1, 3, 1], edge_prob=1.0, seed=42)
        assert topo.sender == 1
        assert topo.receiver == 5
        for node in [2, 3, 4]:
            assert node in topo.adjacency[1]
        for node in [2, 3, 4]:
            assert 5 in topo.adjacency[node]

    def test_guaranteed_connectivity(self):
        topo = generate_layered_graph([1, 5, 5, 1], edge_prob=0.01, seed=42)
        paths = find_all_paths(topo.adjacency, topo.sender, topo.receiver)
        assert len(paths) > 0

    def test_edge_bandwidths(self):
        topo = generate_layered_graph([1, 2, 1], edge_prob=1.0, seed=42)
        for _edge, bw in topo.edge_bandwidths.items():
            assert 2 <= bw <= 8

    def test_reproducibility(self):
        t1 = generate_layered_graph([1, 3, 1], seed=123)
        t2 = generate_layered_graph([1, 3, 1], seed=123)
        assert t1.adjacency == t2.adjacency


class TestBarabasiAlbertGraph:
    def test_node_count(self):
        topo = barabasi_albert_graph(10, 3, 2, seed=42)
        assert len(topo.adjacency) == 10

    def test_invalid_m(self):
        with pytest.raises(ValueError, match="m.*m0"):
            barabasi_albert_graph(10, 3, 5, seed=42)

    def test_sender_receiver(self):
        topo = barabasi_albert_graph(8, 3, 2, seed=42)
        assert topo.sender == 1
        assert topo.receiver == 2


class TestAssignAdversaryParams:
    def test_correct_count(self):
        topo = generate_layered_graph([1, 4, 1], edge_prob=1.0, seed=42)
        params = assign_adversary_params(topo, 2, seed=42)
        assert len(params) == 2

    def test_sender_receiver_excluded(self):
        topo = generate_layered_graph([1, 4, 1], edge_prob=1.0, seed=42)
        params = assign_adversary_params(topo, 4, seed=42)
        assert topo.sender not in params
        assert topo.receiver not in params

    def test_parameter_ranges(self):
        topo = generate_layered_graph([1, 10, 1], edge_prob=1.0, seed=42)
        params = assign_adversary_params(
            topo, 5, p_int_range=(0.1, 0.3), delta_range=(0.2, 0.4), seed=42
        )
        for p in params.values():
            assert 0.1 <= p.p_int <= 0.3
            assert 0.2 <= p.delta <= 0.4


class TestBuildTopology:
    def test_populates_paths_and_metrics(self):
        topo = generate_layered_graph([1, 3, 1], edge_prob=1.0, seed=42)
        topo.node_params = assign_adversary_params(topo, 2, seed=42)
        topo = build_topology(topo)
        assert len(topo.paths) > 0
        assert topo.path_metrics is not None
        assert len(topo.path_metrics) == len(topo.paths)
