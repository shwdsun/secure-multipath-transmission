"""Phase I: passive leakage only (no drop/tamper).

(k,k)-Shamir; confidentiality constraint prod(epsilon_j^n_j) <= tau.
"""

from __future__ import annotations

import math

from smt.models import PathMetrics
from smt.phases.base import PhaseStrategy, SAVTuple


class Phase1Strategy(PhaseStrategy):
    """Phase I share allocation: passive leakage only.

    Since there is no dropping or tampering, reliability is automatic
    and we only enforce the confidentiality bound.
    """

    def __init__(
        self,
        path_metrics: list[PathMetrics],
        sigma: float,
        tau: float,
    ) -> None:
        super().__init__(path_metrics, sigma, tau)
        self._log_eps = [
            math.log(m.epsilon) if m.epsilon > 0 else float("-inf")
            for m in path_metrics
        ]
        self._log_tau = math.log(tau)

    def is_feasible(self, n_vec: list[int]) -> tuple[bool, int | None]:
        """Check Phase I confidentiality: prod(epsilon_j^n_j) <= tau.

        The threshold is always k = sum(n_vec) since (k,k) sharing is used.

        Returns:
            (feasible, k) where k = sum(n_vec).
        """
        if sum(n_vec) == 0:
            return False, None

        log_product = sum(
            n_j * le for n_j, le in zip(n_vec, self._log_eps, strict=False) if n_j > 0
        )
        k = sum(n_vec)

        if log_product <= self._log_tau:
            return True, k
        return False, None

    def generate_minimal_tuples(self, n_max: int = 10) -> set[SAVTuple]:
        """Generate all minimal Phase I SAVs using recursive log-domain search.

        Implements Algorithm 1 from the paper: for each path count from
        the last path, recursively solve for the remaining paths.

        Args:
            n_max: Maximum total shares (used as a safety bound).

        Returns:
            Set of (n_vec, k) tuples where k = sum(n_vec).
        """
        result: set[SAVTuple] = set()

        def _gen_sav(
            m: int, remaining_log_tau: float
        ) -> list[tuple[int, ...]]:
            """Recursive SAV generation for paths 0..m-1."""
            if m == 1:
                le = self._log_eps[0]
                if le >= 0:
                    return []
                n0 = math.ceil(remaining_log_tau / le)
                n0 = max(1, n0)
                return [(n0,)]

            le_m = self._log_eps[m - 1]
            if le_m >= 0:
                # Path m-1 has epsilon >= 1 (useless), skip it
                sub_results = _gen_sav(m - 1, remaining_log_tau)
                return [r + (0,) for r in sub_results]

            # n_m^only: smallest n such that epsilon_m^n <= tau_remaining
            n_only = math.ceil(remaining_log_tau / le_m)
            n_only = max(1, n_only)

            savs: list[tuple[int, ...]] = []

            for n_m in range(0, n_only):
                new_log_tau = remaining_log_tau - n_m * le_m
                sub_results = _gen_sav(m - 1, new_log_tau)

                for sub in sub_results:
                    total_shares = sum(sub) + n_m
                    if total_shares > n_max:
                        continue

                    if n_m == 0:
                        savs.append(sub + (0,))
                    else:
                        # Minimality check: would reducing n_m by 1 still work?
                        log_prod_sub = sum(
                            nj * lej
                            for nj, lej in zip(sub, self._log_eps[: m - 1], strict=False)
                            if nj > 0
                        )
                        if log_prod_sub + (n_m - 1) * le_m > self._log_tau:
                            savs.append(sub + (n_m,))

            # SAV using only path m
            if n_only <= n_max:
                only_sav = tuple(0 for _ in range(m - 1)) + (n_only,)
                savs.append(only_sav)

            return savs

        raw_savs = _gen_sav(self.M, self._log_tau)
        for sav in raw_savs:
            k = sum(sav)
            if k > 0:
                result.add((sav, k))

        return result
