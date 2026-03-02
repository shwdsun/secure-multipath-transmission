"""Tests for smt.simulation module."""

from __future__ import annotations

import pytest

from smt.graph import compute_path_metrics
from smt.models import NetworkTopology, NodeParams, PathMetrics
from smt.simulation import TransmissionSimulator


class TestTransmissionSimulator:
    @pytest.fixture
    def topology_no_adversary(self) -> NetworkTopology:
        adj = {1: [2, 3], 2: [4], 3: [4], 4: []}
        return NetworkTopology(
            adjacency=adj,
            sender=1,
            receiver=4,
            node_params={},
            paths=[[1, 2, 4], [1, 3, 4]],
            edge_bandwidths={(1, 2): 5, (1, 3): 5, (2, 4): 5, (3, 4): 5},
            path_metrics=[
                PathMetrics(epsilon=0.0, p_val=1.0),
                PathMetrics(epsilon=0.0, p_val=1.0),
            ],
        )

    @pytest.fixture
    def topology_with_dropping(self) -> NetworkTopology:
        node_params = {2: NodeParams(p_int=0.3, delta=0.5)}
        adj = {1: [2, 3], 2: [4], 3: [4], 4: []}
        paths = [[1, 2, 4], [1, 3, 4]]
        return NetworkTopology(
            adjacency=adj,
            sender=1,
            receiver=4,
            node_params=node_params,
            paths=paths,
            edge_bandwidths={(1, 2): 5, (1, 3): 5, (2, 4): 5, (3, 4): 5},
            path_metrics=[
                compute_path_metrics(p, node_params) for p in paths
            ],
        )

    def test_perfect_network(self, topology_no_adversary: NetworkTopology):
        sim = TransmissionSimulator(topology_no_adversary, prime=257, seed=42)
        result = sim.run(n_vec=[2, 2], threshold=3, n_trials=100)
        assert result.reliability == 1.0
        assert result.confidentiality_breach == 0.0

    def test_single_trial(self, topology_no_adversary: NetworkTopology):
        sim = TransmissionSimulator(topology_no_adversary, prime=257, seed=42)
        outcome = sim.simulate_trial([2, 2], threshold=3, secret=42)
        assert outcome.original_secret == 42
        assert outcome.reconstructed_secret == 42
        assert len(outcome.received_shares) == 4
        assert outcome.dropped_count == 0

    def test_dropping_reduces_reliability(
        self, topology_with_dropping: NetworkTopology
    ):
        sim = TransmissionSimulator(topology_with_dropping, prime=257, seed=42)
        result = sim.run(n_vec=[3, 3], threshold=5, n_trials=1000)
        assert result.reliability < 1.0
        assert result.avg_shares_received < 6.0

    def test_leakage_detection(
        self, topology_with_dropping: NetworkTopology
    ):
        sim = TransmissionSimulator(topology_with_dropping, prime=257, seed=42)
        result = sim.run(n_vec=[3, 3], threshold=2, n_trials=1000)
        assert result.avg_shares_leaked > 0

    def test_reproducibility(
        self, topology_with_dropping: NetworkTopology
    ):
        sim1 = TransmissionSimulator(topology_with_dropping, prime=257, seed=42)
        sim2 = TransmissionSimulator(topology_with_dropping, prime=257, seed=42)
        r1 = sim1.run([2, 2], threshold=2, n_trials=100)
        r2 = sim2.run([2, 2], threshold=2, n_trials=100)
        assert r1.reliability == r2.reliability
        assert r1.n_leaked == r2.n_leaked

    def test_high_redundancy_high_reliability(
        self, topology_with_dropping: NetworkTopology
    ):
        """With enough redundancy, reliability should be high even with drops."""
        sim = TransmissionSimulator(topology_with_dropping, prime=257, seed=42)
        result = sim.run(n_vec=[5, 5], threshold=2, n_trials=1000)
        assert result.reliability > 0.95
