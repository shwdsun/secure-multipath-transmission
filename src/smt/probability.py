"""PMF of sums of binomials via convolution, tail probs, and threshold search.

Used for feasibility checks in Phase I-III (X_E = shares leaked, X_B = shares received).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def pmf_sum_binomials(n_vec: list[int], p_vec: list[float]) -> np.ndarray:
    """PMF of sum of independent Bin(n_j, p_j) via sequential convolution."""
    if len(n_vec) != len(p_vec):
        raise ValueError("n_vec and p_vec must have the same length")

    pmf = np.array([1.0])

    for n_j, p_j in zip(n_vec, p_vec, strict=True):
        if n_j == 0:
            continue
        binom_pmf = np.array([binom.pmf(k, n_j, p_j) for k in range(n_j + 1)])
        pmf = np.convolve(pmf, binom_pmf)

    return pmf


def tail_prob(pmf: np.ndarray, t: int) -> float:
    """P[X >= t] from PMF array."""
    if t <= 0:
        return 1.0
    if t >= len(pmf):
        return 0.0
    return float(np.sum(pmf[t:]))


def find_t_sec(pmf_eve: np.ndarray, tau: float) -> int | None:
    """Find the smallest threshold t such that P[X_E >= t] <= tau.

    This is the minimum Shamir threshold that ensures confidentiality:
    the adversary collects t or more shares with probability at most tau.

    Args:
        pmf_eve: PMF of X_E (shares leaked to adversary).
        tau: Maximum allowed leakage probability.

    Returns:
        Smallest valid t, or None if no such t exists.
    """
    n = len(pmf_eve) - 1
    for t in range(1, n + 1):
        if tail_prob(pmf_eve, t) <= tau:
            return t
    return None


def find_t_rel(pmf_bob: np.ndarray, sigma: float) -> int | None:
    """Find the largest threshold t such that P[X_B >= t] >= sigma.

    This is the maximum Shamir threshold that ensures reliability:
    Bob receives at least t shares with probability at least sigma.

    Args:
        pmf_bob: PMF of X_B (shares received by Bob).
        sigma: Minimum required reliability probability.

    Returns:
        Largest valid t, or None if no such t exists.
    """
    n = len(pmf_bob) - 1
    for t in range(n, 0, -1):
        if tail_prob(pmf_bob, t) >= sigma:
            return t
    return None


def find_feasible_threshold(
    n_vec: list[int],
    epsilon_vec: list[float],
    gamma_vec: list[float],
    sigma: float,
    tau: float,
) -> tuple[bool, int | None]:
    """Check Phase II feasibility for a given share assignment.

    Computes t_sec and t_rel from the distributions of X_E and X_B.
    A feasible threshold t exists iff t_sec <= t_rel.

    Args:
        n_vec: Share assignment vector [n_1, ..., n_M].
        epsilon_vec: Per-path leakage probabilities.
        gamma_vec: Per-path valid-arrival probabilities.
        sigma: Reliability target.
        tau: Confidentiality target.

    Returns:
        (feasible, t) where t is the canonical threshold t_sec if feasible.
    """
    if sum(n_vec) == 0:
        return False, None

    pmf_e = pmf_sum_binomials(n_vec, epsilon_vec)
    pmf_b = pmf_sum_binomials(n_vec, gamma_vec)

    t_sec = find_t_sec(pmf_e, tau)
    t_rel = find_t_rel(pmf_b, sigma)

    if t_sec is None or t_rel is None:
        return False, None

    if t_sec <= t_rel:
        return True, t_sec
    return False, None
