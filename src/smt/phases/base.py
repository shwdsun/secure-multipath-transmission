"""Abstract base for phase-specific security strategies (I, II, III)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from smt.models import PathMetrics

SAVTuple = tuple[tuple[int, ...], int]


class PhaseStrategy(ABC):
    """Base class for phase-specific share allocation strategies.

    Subclasses implement feasibility and tuple generation logic
    appropriate to their adversarial model.

    Args:
        path_metrics: Per-path security and reliability metrics.
        sigma: Reliability target -- P[receiver reconstructs correctly] >= sigma.
        tau: Confidentiality target -- P[adversary learns secret] <= tau.
    """

    def __init__(
        self,
        path_metrics: list[PathMetrics],
        sigma: float,
        tau: float,
    ) -> None:
        if not 0.0 < sigma <= 1.0:
            raise ValueError(f"sigma must be in (0, 1], got {sigma}")
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.path_metrics = path_metrics
        self.sigma = sigma
        self.tau = tau
        self.M = len(path_metrics)

    @abstractmethod
    def is_feasible(self, n_vec: list[int]) -> tuple[bool, int | None]:
        """Check whether a share assignment satisfies the phase's security constraints.

        Args:
            n_vec: Share assignment vector [n_1, ..., n_M].

        Returns:
            (feasible, threshold) -- the canonical threshold t if feasible, else None.
        """

    def is_minimal(self, n_vec: list[int]) -> bool:
        """Check whether a feasible SAV is minimal.

        A feasible n_vec is minimal if decrementing any positive
        component by 1 makes it infeasible.

        Args:
            n_vec: A feasible share assignment vector.

        Returns:
            True if minimal (no component can be reduced).
        """
        for i in range(self.M):
            if n_vec[i] > 0:
                n_prime = list(n_vec)
                n_prime[i] -= 1
                feasible, _ = self.is_feasible(n_prime)
                if feasible:
                    return False
        return True

    @abstractmethod
    def generate_minimal_tuples(self, n_max: int = 10) -> set[SAVTuple]:
        """Generate all minimal feasible SAV-tuples within a share budget.

        Args:
            n_max: Maximum total shares per message.

        Returns:
            Set of (n_vec_tuple, threshold) pairs.
        """
