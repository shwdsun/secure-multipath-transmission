"""Tests for smt.phases.phase1 module."""

from __future__ import annotations

import math

import pytest

from smt.models import NodeParams, PathMetrics
from smt.phases.phase1 import Phase1Strategy


class TestPhase1Strategy:
    @pytest.fixture
    def strategy(
        self,
        simple_3path_network: tuple[
            list[list[int]], dict[int, NodeParams], list[PathMetrics]
        ],
    ) -> Phase1Strategy:
        _, _, metrics = simple_3path_network
        return Phase1Strategy(metrics, sigma=0.95, tau=0.01)

    def test_feasibility_true(self, strategy: Phase1Strategy):
        feasible, k = strategy.is_feasible([2, 2, 2])
        assert feasible is True
        assert k == 6

    def test_feasibility_false_single(self, strategy: Phase1Strategy):
        feasible, _ = strategy.is_feasible([1, 0, 0])
        assert feasible is False

    def test_feasibility_zero(self, strategy: Phase1Strategy):
        feasible, _ = strategy.is_feasible([0, 0, 0])
        assert feasible is False

    def test_minimality(self, strategy: Phase1Strategy):
        tuples = strategy.generate_minimal_tuples(n_max=15)
        for n_vec, _k in tuples:
            # Every tuple should be feasible
            feas, _ = strategy.is_feasible(list(n_vec))
            assert feas, f"Tuple {n_vec} should be feasible"

            # Every tuple should be minimal
            assert strategy.is_minimal(list(n_vec)), f"Tuple {n_vec} not minimal"

    def test_confidentiality_bound(self, strategy: Phase1Strategy):
        tuples = strategy.generate_minimal_tuples(n_max=15)
        for n_vec, _ in tuples:
            log_product = sum(
                n_j * math.log(m.epsilon)
                for n_j, m in zip(n_vec, strategy.path_metrics, strict=False)
                if n_j > 0
            )
            assert log_product <= math.log(strategy.tau) + 1e-9

    def test_generates_nonempty(self, strategy: Phase1Strategy):
        tuples = strategy.generate_minimal_tuples(n_max=20)
        assert len(tuples) > 0

    def test_two_path_network(self):
        metrics = [
            PathMetrics(epsilon=0.3, p_val=1.0),
            PathMetrics(epsilon=0.5, p_val=1.0),
        ]
        strategy = Phase1Strategy(metrics, sigma=0.95, tau=0.01)
        tuples = strategy.generate_minimal_tuples(n_max=20)
        assert len(tuples) > 0

        for n_vec, _k in tuples:
            product = 1.0
            for n_j, m in zip(n_vec, metrics, strict=False):
                product *= m.epsilon ** n_j
            assert product <= 0.01 + 1e-12
