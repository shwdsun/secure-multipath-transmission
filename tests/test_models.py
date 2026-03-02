"""Tests for smt.models data classes."""

import pytest

from smt.models import NetworkTopology, NodeParams, PathMetrics


class TestNodeParams:
    def test_derived_properties(self):
        p = NodeParams(p_int=0.1, delta=0.3, theta=0.2)
        assert abs(p.d - 0.03) < 1e-12
        assert abs(p.e - 0.02) < 1e-12
        assert abs(p.f - 0.95) < 1e-12

    def test_phase2_defaults(self):
        p = NodeParams(p_int=0.2, delta=0.5)
        assert p.theta == 0.0
        assert abs(p.d - 0.1) < 1e-12
        assert abs(p.e) < 1e-12
        assert abs(p.f - 0.9) < 1e-12

    def test_frozen(self):
        p = NodeParams(p_int=0.1, delta=0.3)
        with pytest.raises(AttributeError):
            p.p_int = 0.5  # type: ignore[misc]

    def test_invalid_p_int(self):
        with pytest.raises(ValueError, match="p_int"):
            NodeParams(p_int=1.5, delta=0.1)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            NodeParams(p_int=0.1, delta=-0.1)

    def test_delta_theta_sum(self):
        with pytest.raises(ValueError, match="delta.*theta"):
            NodeParams(p_int=0.5, delta=0.6, theta=0.6)

    def test_zero_interception(self):
        p = NodeParams(p_int=0.0, delta=1.0)
        assert p.d == 0.0
        assert p.f == 1.0


class TestPathMetrics:
    def test_valid_metrics(self):
        m = PathMetrics(epsilon=0.15, p_val=0.9, p_err=0.05, p_drop=0.05)
        assert abs(m.p_nd - 0.95) < 1e-12

    def test_phase1_metrics(self):
        m = PathMetrics(epsilon=0.2, p_val=1.0)
        assert m.p_err == 0.0
        assert m.p_drop == 0.0

    def test_invalid_sum(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            PathMetrics(epsilon=0.1, p_val=0.5, p_err=0.3, p_drop=0.1)


class TestNetworkTopology:
    def test_basic_properties(self):
        adj = {1: [2, 3], 2: [3], 3: []}
        topo = NetworkTopology(adjacency=adj, sender=1, receiver=3)
        assert topo.nodes == [1, 2, 3]
        assert topo.num_paths == 0

    def test_with_paths(self):
        adj = {1: [2, 3], 2: [3], 3: []}
        topo = NetworkTopology(
            adjacency=adj, sender=1, receiver=3,
            paths=[[1, 2, 3], [1, 3]],
        )
        assert topo.num_paths == 2
