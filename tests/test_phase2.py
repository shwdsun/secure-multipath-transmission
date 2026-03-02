"""Tests for smt.phases.phase2 module."""

from __future__ import annotations

import pytest

from smt.models import NodeParams, PathMetrics
from smt.phases.phase2 import Phase2Strategy
from smt.probability import pmf_sum_binomials, tail_prob


class TestPhase2Strategy:
    @pytest.fixture
    def strategy(
        self,
        simple_3path_network: tuple[
            list[list[int]], dict[int, NodeParams], list[PathMetrics]
        ],
    ) -> Phase2Strategy:
        _, _, metrics = simple_3path_network
        return Phase2Strategy(metrics, sigma=0.95, tau=0.01)

    def test_known_feasible(self, strategy: Phase2Strategy):
        """[3, 4, 0] should be feasible (from notebook tests)."""
        feasible, t = strategy.is_feasible([3, 4, 0])
        assert feasible is True
        assert t is not None

    def test_single_share_infeasible(self, strategy: Phase2Strategy):
        feasible, _ = strategy.is_feasible([1, 0, 0])
        assert feasible is False

    def test_minimal_tuple_count(self, strategy: Phase2Strategy):
        """The 3-path notebook test produces 21 minimal tuples."""
        tuples = strategy.generate_minimal_tuples(n_max=10)
        assert len(tuples) == 21

    def test_all_tuples_feasible(self, strategy: Phase2Strategy):
        tuples = strategy.generate_minimal_tuples(n_max=10)
        for n_vec, t in tuples:
            feasible, t_check = strategy.is_feasible(list(n_vec))
            assert feasible, f"{n_vec} should be feasible"
            assert t_check == t

    def test_all_tuples_minimal(self, strategy: Phase2Strategy):
        tuples = strategy.generate_minimal_tuples(n_max=10)
        for n_vec, _ in tuples:
            assert strategy.is_minimal(list(n_vec)), f"{n_vec} not minimal"

    def test_dual_constraints(self, strategy: Phase2Strategy):
        """Verify P[X_E >= t] <= tau and P[X_B >= t] >= sigma."""
        tuples = strategy.generate_minimal_tuples(n_max=10)
        eps = [m.epsilon for m in strategy.path_metrics]
        gam = [m.p_val for m in strategy.path_metrics]

        for n_vec, t in tuples:
            pmf_e = pmf_sum_binomials(list(n_vec), eps)
            pmf_b = pmf_sum_binomials(list(n_vec), gam)

            assert tail_prob(pmf_e, t) <= strategy.tau + 1e-12
            assert tail_prob(pmf_b, t) >= strategy.sigma - 1e-12

    def test_empty_network(self):
        strategy = Phase2Strategy([], sigma=0.95, tau=0.01)
        tuples = strategy.generate_minimal_tuples(n_max=5)
        assert len(tuples) == 0

    def test_high_security_reduces_throughput(
        self,
        simple_3path_network: tuple[
            list[list[int]], dict[int, NodeParams], list[PathMetrics]
        ],
    ):
        """Tighter tau should require more shares (higher total n)."""
        _, _, metrics = simple_3path_network
        loose = Phase2Strategy(metrics, sigma=0.95, tau=0.05)
        tight = Phase2Strategy(metrics, sigma=0.95, tau=0.001)

        tuples_loose = loose.generate_minimal_tuples(n_max=15)
        tuples_tight = tight.generate_minimal_tuples(n_max=15)

        min_n_loose = min(sum(nv) for nv, _ in tuples_loose)
        min_n_tight = min(sum(nv) for nv, _ in tuples_tight)
        assert min_n_tight >= min_n_loose
