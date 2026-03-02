"""Monte Carlo simulation of multipath secret sharing transmission.

Validates empirical reliability/confidentiality vs analytical predictions.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

from smt.models import NetworkTopology
from smt.shamir import ShamirSecretSharing, Share


@dataclass
class SimulationResult:
    """Aggregated results from a Monte Carlo simulation run.

    Attributes:
        n_trials: Number of simulation trials.
        n_reconstructed: Trials where receiver successfully reconstructed.
        n_leaked: Trials where adversary obtained >= t shares.
        reliability: Empirical P[reconstruction success].
        confidentiality_breach: Empirical P[adversary gets >= t shares].
        avg_shares_received: Mean number of shares received by Bob.
        avg_shares_leaked: Mean number of shares leaked to Eve.
    """

    n_trials: int
    n_reconstructed: int
    n_leaked: int
    reliability: float
    confidentiality_breach: float
    avg_shares_received: float
    avg_shares_leaked: float


@dataclass
class TrialOutcome:
    """Outcome of a single simulation trial."""

    received_shares: list[Share]
    leaked_share_count: int
    tampered_count: int
    dropped_count: int
    reconstructed_secret: int | None
    original_secret: int


class TransmissionSimulator:
    """Monte Carlo simulation engine for multipath secret sharing.

    Args:
        topology: Network topology with paths and node_params.
        prime: Field prime for Shamir SSS.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        prime: int = (1 << 127) - 1,
        seed: int | None = None,
    ) -> None:
        self.topology = topology
        self.sss = ShamirSecretSharing(prime)
        self.rng = random.Random(seed)
        self._prime = prime

    def simulate_trial(
        self,
        n_vec: list[int],
        threshold: int,
        secret: int | None = None,
    ) -> TrialOutcome:
        """Run a single transmission trial.

        Args:
            n_vec: Share assignment vector.
            threshold: Shamir reconstruction threshold.
            secret: Secret to share (random if None).

        Returns:
            TrialOutcome with per-trial details.
        """
        if secret is None:
            secret = self.rng.randrange(0, self._prime)

        n_total = sum(n_vec)
        shares = self.sss.share(secret, n_total, threshold)

        received: list[Share] = []
        leaked_count = 0
        tampered_count = 0
        dropped_count = 0

        share_idx = 0
        paths = self.topology.paths
        node_params = self.topology.node_params

        for j, n_j in enumerate(n_vec):
            path = paths[j]
            for _ in range(n_j):
                share = shares[share_idx]
                share_idx += 1

                intercepted = False
                dropped = False
                tampered = False

                for node in path[1:-1]:
                    params = node_params.get(node)
                    if params is None:
                        continue

                    if self.rng.random() < params.p_int:
                        intercepted = True

                        roll = self.rng.random()
                        if roll < params.delta:
                            dropped = True
                            break
                        elif roll < params.delta + params.theta:
                            tampered = True

                if intercepted:
                    leaked_count += 1

                if dropped:
                    dropped_count += 1
                elif tampered:
                    tampered_count += 1
                    corrupted_y = self.rng.randrange(0, self._prime)
                    received.append(Share(x=share.x, y=corrupted_y))
                else:
                    received.append(share)

        reconstructed = None
        if len(received) >= threshold:
            with contextlib.suppress(Exception):
                reconstructed = self.sss.reconstruct(received[:threshold])

        return TrialOutcome(
            received_shares=received,
            leaked_share_count=leaked_count,
            tampered_count=tampered_count,
            dropped_count=dropped_count,
            reconstructed_secret=reconstructed,
            original_secret=secret,
        )

    def run(
        self,
        n_vec: list[int],
        threshold: int,
        n_trials: int = 10000,
    ) -> SimulationResult:
        """Run multiple simulation trials and aggregate results.

        Args:
            n_vec: Share assignment vector.
            threshold: Shamir reconstruction threshold.
            n_trials: Number of Monte Carlo trials.

        Returns:
            SimulationResult with empirical statistics.
        """
        n_reconstructed = 0
        n_leaked = 0
        total_received = 0
        total_leaked = 0

        for _ in range(n_trials):
            outcome = self.simulate_trial(n_vec, threshold)

            if (
                outcome.reconstructed_secret is not None
                and outcome.reconstructed_secret == outcome.original_secret
            ):
                n_reconstructed += 1

            if outcome.leaked_share_count >= threshold:
                n_leaked += 1

            total_received += len(outcome.received_shares)
            total_leaked += outcome.leaked_share_count

        return SimulationResult(
            n_trials=n_trials,
            n_reconstructed=n_reconstructed,
            n_leaked=n_leaked,
            reliability=n_reconstructed / n_trials,
            confidentiality_breach=n_leaked / n_trials,
            avg_shares_received=total_received / n_trials,
            avg_shares_leaked=total_leaked / n_trials,
        )
