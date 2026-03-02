# Secure Multipath Transmission

**Probabilistic information-theoretic security through multipath secret sharing.**

A framework for establishing secure communication over untrusted network infrastructure, guaranteeing both confidentiality and reliability against computationally unbounded adversaries -- without pre-shared keys or computational hardness assumptions.

This work addresses a critical gap in modern network security: traditional cryptographic protocols (RSA, ECC) are vulnerable to quantum computing advances and cannot protect against hardware-level supply chain attacks. Our approach leverages Shamir's Secret Sharing over multipath network topologies to achieve **post-quantum**, **information-theoretic** security guarantees.

> **Note**: This repository contains code selectively migrated from a private research repository. The full codebase (including Phase III) will be released upon paper acceptance or thesis completion (est. April/May 2026). The current release demonstrates the framework architecture, core algorithms, and code quality. For academic inquiries or collaboration, please [open an issue](../../issues) or contact me directly.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Secure Multipath Transmission                  │
│                                                                 │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│  │  models   │   │   graph   │   │  shamir   │   │probability│  │
│  │           │   │           │   │           │   │           │  │
│  │NodeParams │   │ Topology  │   │ SSS over  │   │Convolution│  │
│  │PathMetrics│   │ PathFind  │   │   GF(p)   │   │Tail probs │  │
│  │ Topology  │   │ BA/Layer  │   │ Lagrange  │   │Thresholds │  │
│  └───────────┘   └───────────┘   └───────────┘   └───────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                          phases/                          │  │
│  │                                                           │  │
│  │  ┌───────────┐   ┌───────────┐   ┌─────────────────────┐  │  │
│  │  │  Phase I  │   │ Phase II  │   │      Phase III      │  │  │
│  │  │  Passive  │   │ Dropping  │   │  Active Adversary   │  │  │
│  │  │ (k,k)-SS  │   │ (n,t)-SS  │   │   [Private Repo]    │  │  │
│  │  │ Recursive │   │ BFS enum  │   │                     │  │  │
│  │  └───────────┘   └───────────┘   └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐   ┌────────────────────────────────────────┐   │
│  │  optimizer  │   │               simulation               │   │
│  │ ILP solver  │   │     Monte Carlo validation engine      │   │
│  │ PuLP/Gurobi │   │   End-to-end transmission simulation   │   │
│  └─────────────┘   └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Information-theoretic security**: Confidentiality and reliability guarantees based on information theory, not computational hardness -- immune to quantum attacks
- **Progressive defense framework**: Three phases countering increasingly capable adversaries (eavesdropping, dropping, and beyond)
- **Shamir Secret Sharing**: Full implementation over GF(p) with configurable field prime (default 2^127 - 1), Horner evaluation, and Lagrange interpolation
- **Exact probability analysis**: 1D convolution DP for computing distributions of sums of independent binomials; exact tail probability computation for security thresholds
- **Minimal SAV generation**: Efficient enumeration of minimal feasible share allocation vectors via recursive (Phase I) and BFS (Phase II) search
- **ILP throughput optimization**: Maximize secure messages per timeslot under link bandwidth constraints, with solver abstraction supporting PuLP (open-source) and Gurobi (commercial)
- **Monte Carlo validation**: End-to-end simulation engine that validates analytical predictions against empirical transmission outcomes

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/secure-multipath-transmission.git
cd secure-multipath-transmission

# Install with development dependencies
pip install -e ".[dev]"

# Optional: install Gurobi backend for ILP optimization
pip install gurobipy
```

**Requirements**: Python >= 3.10, numpy, scipy, PuLP

## Quick Start

```python
from smt.graph import build_topology
from smt.models import NetworkTopology, NodeParams
from smt.optimizer import ThroughputOptimizer
from smt.phases.phase2 import Phase2Strategy
from smt.simulation import TransmissionSimulator

# Define a network with adversarial nodes
topology = NetworkTopology(
    adjacency={1: [2, 4, 6], 2: [5], 3: [], 4: [5], 5: [3], 6: [3]},
    sender=1, receiver=3,
    node_params={
        2: NodeParams(p_int=0.10, delta=0.30),
        4: NodeParams(p_int=0.15, delta=0.20),
        5: NodeParams(p_int=0.05, delta=0.50),
        6: NodeParams(p_int=0.20, delta=0.10),
    },
    edge_bandwidths={(1,2): 5, (1,4): 5, (1,6): 5, (2,5): 5, (4,5): 5, (5,3): 10, (6,3): 5},
)
topology = build_topology(topology)

# Generate minimal secure share allocations
strategy = Phase2Strategy(topology.path_metrics, sigma=0.95, tau=0.01)
tuples = strategy.generate_minimal_tuples(n_max=10)
# -> 21 minimal SAV-tuples, each guaranteeing 95% reliability, <=1% leakage

# Maximize throughput under bandwidth constraints
optimizer = ThroughputOptimizer(topology)
result = optimizer.optimize(tuples)
# -> Optimal: 3 messages/timeslot

# Validate analytically-derived guarantees via Monte Carlo
sim = TransmissionSimulator(topology, prime=257, seed=42)
sim_result = sim.run([0, 5, 0], threshold=4, n_trials=10_000)
print(f"Reliability: {sim_result.reliability:.4f}")  # ~0.97
print(f"Leakage:     {sim_result.confidentiality_breach:.4f}")  # ~0.006
```

## Algorithm Overview

### Adversarial Model

We model compromised network switches with probabilistic interception and disruption capabilities. The framework handles progressively stronger adversaries across three phases.

### Phase I: Passive Leakage

Counters eavesdropping-only adversaries using (k, k)-Shamir secret sharing. All k shares are required for reconstruction, so the adversary must intercept every share to breach confidentiality. Minimal SAVs are generated via recursive enumeration in the log domain.

**Constraint**: `prod(epsilon_j^n_j) <= tau`

### Phase II: Active Dropping

Extends Phase I by tolerating packet drops via (n, t)-threshold secret sharing with t < n. The receiver can reconstruct from any t of n shares. The dual constraints ensure both reliability and confidentiality:

- **Reliability**: `P[X_B >= t] >= sigma` (Bob receives enough shares)
- **Confidentiality**: `P[X_E >= t] <= tau` (Eve doesn't get enough shares)

Distributions are computed exactly via 1D convolution of independent binomial PMFs.

### Phase III: Diverse Active Adversaries

Phase III extends the defense to a broader class of active adversaries with more diverse capabilities. It employs a more robust encoding and decoding strategy to maintain the (sigma, tau) guarantees under stronger adversarial assumptions.

The full Phase III design and implementation are part of ongoing thesis research, currently maintained in a private repository. The code will be migrated here upon paper acceptance or graduation (est. April/May 2026).

### Optimization

Given a set of minimal feasible SAV-tuples, we maximize secure throughput per timeslot via Integer Linear Programming under per-link bandwidth constraints. The solver supports both PuLP (open-source CBC) and Gurobi backends.

## Project Structure

```
src/smt/
├── models.py          # NodeParams, PathMetrics, NetworkTopology
├── graph.py           # Graph generation, path finding, metric computation
├── probability.py     # PMF convolution, tail probabilities, threshold search
├── shamir.py          # Shamir SSS over GF(p): share, reconstruct, byte-level API
├── phases/
│   ├── base.py        # Abstract PhaseStrategy interface
│   ├── phase1.py      # Recursive SAV generation (passive leakage)
│   ├── phase2.py      # BFS tuple generation (active dropping)
│   └── phase3.py      # Interface only (private repo -- pending publication)
├── optimizer.py       # ILP throughput maximization (PuLP / Gurobi)
└── simulation.py      # Monte Carlo transmission simulation
```

## Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=smt --cov-report=term-missing

# Run a specific module
pytest tests/test_shamir.py -v
```

**89 tests** covering all modules: data models, graph operations, probability computations, Shamir SSS (including byte-level sharing), Phase I/II strategies, ILP optimization, and Monte Carlo simulation.

## References

- A. Shamir, "How to share a secret," *Communications of the ACM*, 1979.
- A. Rashidi et al., "Securing communication over untrusted networks through adaptive multipath secret sharing," 2022.
- D. Dolev et al., "Perfectly secure message transmission," *JACM*, 1993.

## License

MIT License. See [LICENSE](LICENSE) for details.
