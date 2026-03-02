"""Phase II: leakage + dropping (no tamper).

(n,t)-Shamir with dual constraints: P[X_B>=t]>=sigma, P[X_E>=t]<=tau.
"""

from __future__ import annotations

from collections import deque

from smt.models import PathMetrics
from smt.phases.base import PhaseStrategy, SAVTuple
from smt.probability import find_feasible_threshold


class Phase2Strategy(PhaseStrategy):
    """Phase II share allocation: passive leakage + active dropping.

    Finds minimal (n_vec, t) tuples satisfying both confidentiality
    and reliability under independent per-packet drop probabilities.
    """

    def __init__(
        self,
        path_metrics: list[PathMetrics],
        sigma: float,
        tau: float,
    ) -> None:
        super().__init__(path_metrics, sigma, tau)
        self._epsilon = [m.epsilon for m in path_metrics]
        self._gamma = [m.p_val for m in path_metrics]

    def is_feasible(self, n_vec: list[int]) -> tuple[bool, int | None]:
        """Check Phase II feasibility for a share assignment.

        Computes exact PMFs of X_E and X_B via 1D convolution and
        searches for a threshold t satisfying both constraints.

        Returns:
            (feasible, t) where t = t_sec if feasible.
        """
        return find_feasible_threshold(
            n_vec, self._epsilon, self._gamma, self.sigma, self.tau
        )

    def generate_minimal_tuples(self, n_max: int = 10) -> set[SAVTuple]:
        """Generate minimal Phase II SAV-tuples via BFS enumeration.

        Starting from unit vectors, the search expands infeasible
        assignments by incrementing one component at a time.  Feasible
        assignments are tested for minimality and collected.

        The BFS order (by total shares) guarantees that minimal tuples
        are discovered before any non-minimal supersets.

        Args:
            n_max: Maximum total shares per message.

        Returns:
            Set of minimal (n_vec, t) tuples.
        """
        result: set[SAVTuple] = set()
        queue: deque[tuple[int, ...]] = deque()
        visited: set[tuple[int, ...]] = set()

        for j in range(self.M):
            unit = tuple(1 if i == j else 0 for i in range(self.M))
            queue.append(unit)
            visited.add(unit)

        while queue:
            n_vec = queue.popleft()
            n_total = sum(n_vec)
            if n_total > n_max:
                continue

            feasible, t = self.is_feasible(list(n_vec))

            if feasible:
                assert t is not None
                if self.is_minimal(list(n_vec)):
                    result.add((n_vec, t))
            else:
                for j in range(self.M):
                    n_new = tuple(
                        n_vec[i] + (1 if i == j else 0) for i in range(self.M)
                    )
                    if n_new not in visited and sum(n_new) <= n_max:
                        visited.add(n_new)
                        queue.append(n_new)

        return result
