"""Tests for smt.probability module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from smt.probability import (
    find_feasible_threshold,
    find_t_rel,
    find_t_sec,
    pmf_sum_binomials,
    tail_prob,
)


class TestPmfSumBinomials:
    def test_single_binomial(self):
        pmf = pmf_sum_binomials([5], [0.3])
        expected = np.array([binom.pmf(k, 5, 0.3) for k in range(6)])
        np.testing.assert_allclose(pmf, expected, atol=1e-12)

    def test_two_binomials_mean(self):
        """E[Bin(2, 0.5) + Bin(3, 0.6)] = 1.0 + 1.8 = 2.8."""
        pmf = pmf_sum_binomials([2, 3], [0.5, 0.6])
        mean = sum(k * pmf[k] for k in range(len(pmf)))
        assert abs(mean - 2.8) < 1e-10

    def test_sums_to_one(self):
        pmf = pmf_sum_binomials([3, 4, 2], [0.2, 0.7, 0.5])
        assert abs(np.sum(pmf) - 1.0) < 1e-10

    def test_empty_components(self):
        pmf = pmf_sum_binomials([0, 5], [0.5, 0.3])
        expected = np.array([binom.pmf(k, 5, 0.3) for k in range(6)])
        np.testing.assert_allclose(pmf, expected, atol=1e-12)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            pmf_sum_binomials([1, 2], [0.5])

    def test_variance(self):
        """Variance of sum = sum of variances."""
        n_vec, p_vec = [4, 6], [0.3, 0.7]
        pmf = pmf_sum_binomials(n_vec, p_vec)
        mean = sum(k * pmf[k] for k in range(len(pmf)))
        var = sum((k - mean) ** 2 * pmf[k] for k in range(len(pmf)))
        expected_var = 4 * 0.3 * 0.7 + 6 * 0.7 * 0.3
        assert abs(var - expected_var) < 1e-10


class TestTailProb:
    def test_tail_at_zero(self):
        pmf = np.array([0.2, 0.3, 0.5])
        assert abs(tail_prob(pmf, 0) - 1.0) < 1e-12

    def test_tail_at_end(self):
        pmf = np.array([0.2, 0.3, 0.5])
        assert abs(tail_prob(pmf, 2) - 0.5) < 1e-12

    def test_tail_beyond_range(self):
        pmf = np.array([0.2, 0.3, 0.5])
        assert tail_prob(pmf, 10) == 0.0


class TestFindTSec:
    def test_basic(self):
        pmf = pmf_sum_binomials([5], [0.2])
        t = find_t_sec(pmf, 0.01)
        assert t is not None
        assert tail_prob(pmf, t) <= 0.01
        assert tail_prob(pmf, t - 1) > 0.01

    def test_impossible(self):
        pmf = pmf_sum_binomials([2], [0.99])
        t = find_t_sec(pmf, 1e-10)
        assert t is None or tail_prob(pmf, t) <= 1e-10


class TestFindTRel:
    def test_basic(self):
        pmf = pmf_sum_binomials([10], [0.9])
        t = find_t_rel(pmf, 0.95)
        assert t is not None
        assert tail_prob(pmf, t) >= 0.95
        assert tail_prob(pmf, t + 1) < 0.95


class TestFindFeasibleThreshold:
    def test_feasible_case(self):
        feasible, t = find_feasible_threshold(
            n_vec=[3, 4],
            epsilon_vec=[0.145, 0.1925],
            gamma_vec=[0.9457, 0.9457],
            sigma=0.95,
            tau=0.01,
        )
        assert feasible is True
        assert t is not None and t > 0

    def test_infeasible_high_leakage(self):
        feasible, _ = find_feasible_threshold(
            n_vec=[1],
            epsilon_vec=[0.99],
            gamma_vec=[0.5],
            sigma=0.99,
            tau=0.001,
        )
        assert feasible is False

    def test_empty_vec(self):
        feasible, _ = find_feasible_threshold([], [], [], 0.95, 0.01)
        assert feasible is False
