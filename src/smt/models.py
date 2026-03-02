"""Data models for network topology and adversary parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NodeParams:
    """Adversary parameters for a single compromised switch."""

    p_int: float
    delta: float = 0.0
    theta: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_int <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {self.p_int}")
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(f"delta must be in [0, 1], got {self.delta}")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        if self.delta + self.theta > 1.0 + 1e-12:
            raise ValueError(f"delta + theta must be <= 1, got {self.delta + self.theta}")

    @property
    def d(self) -> float:
        """Absolute drop probability."""
        return self.p_int * self.delta

    @property
    def e(self) -> float:
        """Absolute tamper-and-forward probability."""
        return self.p_int * self.theta

    @property
    def f(self) -> float:
        """Absolute intact-forward probability."""
        return 1.0 - self.d - self.e


@dataclass(frozen=True)
class PathMetrics:
    """Derived per-path metrics for security and reliability analysis."""

    epsilon: float
    p_val: float
    p_err: float = 0.0
    p_drop: float = 0.0

    def __post_init__(self) -> None:
        total = self.p_val + self.p_err + self.p_drop
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"p_val + p_err + p_drop must equal 1.0, got {total:.12f}"
            )

    @property
    def p_nd(self) -> float:
        """Probability a share is not dropped (arrives valid or tampered)."""
        return self.p_val + self.p_err


# Type aliases for adjacency representations
AdjacencyList = dict[int, list[int]]
WeightedAdjacencyList = dict[int, list[tuple[int, int]]]


@dataclass
class NetworkTopology:
    """Complete network specification for multipath security analysis.

    Attributes:
        adjacency: Node -> list of neighbor node IDs.
        sender: Sender node ID.
        receiver: Receiver node ID.
        node_params: Adversary parameters for compromised nodes.
        paths: Pre-computed list of sender-to-receiver paths.
        edge_bandwidths: Mapping from (u, v) edge to integer bandwidth capacity.
    """

    adjacency: AdjacencyList
    sender: int
    receiver: int
    node_params: dict[int, NodeParams] = field(default_factory=dict)
    paths: list[list[int]] = field(default_factory=list)
    edge_bandwidths: dict[tuple[int, int], int] = field(default_factory=dict)
    path_metrics: list[PathMetrics] | None = None

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def nodes(self) -> list[int]:
        return sorted(self.adjacency.keys())
