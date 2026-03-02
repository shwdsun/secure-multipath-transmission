#!/usr/bin/env python3
"""Quick start example: secure multipath transmission in 50 lines.

Demonstrates the core workflow:
  1. Define a network with adversarial nodes
  2. Generate minimal secure share allocations (Phase II)
  3. Maximize throughput via ILP
  4. Validate with Monte Carlo simulation
"""

from smt.graph import build_topology, compute_path_metrics
from smt.models import NetworkTopology, NodeParams, PathMetrics
from smt.optimizer import ThroughputOptimizer
from smt.phases.phase2 import Phase2Strategy
from smt.simulation import TransmissionSimulator

# --- 1. Define the network ---
#   S(1) ---> A(2) ---> D(5) ---> R(3)
#   S(1) ---> B(4) ---> D(5) ---> R(3)
#   S(1) ---> C(6) -----------> R(3)
topology = NetworkTopology(
    adjacency={1: [2, 4, 6], 2: [5], 3: [], 4: [5], 5: [3], 6: [3]},
    sender=1,
    receiver=3,
    node_params={
        2: NodeParams(p_int=0.10, delta=0.30),  # 10% intercept, 30% drop|intercept
        4: NodeParams(p_int=0.15, delta=0.20),
        5: NodeParams(p_int=0.05, delta=0.50),
        6: NodeParams(p_int=0.20, delta=0.10),
    },
    edge_bandwidths={
        (1, 2): 5, (1, 4): 5, (1, 6): 5,
        (2, 5): 5, (4, 5): 5, (5, 3): 10, (6, 3): 5,
    },
)
topology = build_topology(topology)

# --- 2. Generate minimal secure share allocations ---
sigma, tau = 0.95, 0.01  # 95% reliability, 1% max leakage
assert topology.path_metrics is not None
strategy = Phase2Strategy(topology.path_metrics, sigma=sigma, tau=tau)
tuples = strategy.generate_minimal_tuples(n_max=10)

print(f"Found {len(tuples)} minimal SAV-tuples (sigma={sigma}, tau={tau})")
for n_vec, t in sorted(tuples, key=lambda x: sum(x[0])):
    print(f"  shares={n_vec}, threshold={t}, total={sum(n_vec)}")

# --- 3. Maximize throughput under bandwidth constraints ---
optimizer = ThroughputOptimizer(topology)
result = optimizer.optimize(tuples)

print(f"\nOptimal throughput: {result.throughput:.0f} messages/timeslot")
print(f"Solver status: {result.status}")
for sav_tuple, count in result.allocation.items():
    print(f"  {count:.0f}x SAV {sav_tuple[0]} (t={sav_tuple[1]})")

# --- 4. Validate with Monte Carlo simulation ---
best_tuple = max(result.allocation, key=lambda x: result.allocation[x])
n_vec, t = best_tuple

sim = TransmissionSimulator(topology, prime=257, seed=42)
sim_result = sim.run(list(n_vec), t, n_trials=10_000)

print(f"\nMonte Carlo validation (10k trials, SAV={n_vec}, t={t}):")
print(f"  Reliability:    {sim_result.reliability:.4f}  (target >= {sigma})")
print(f"  Leakage rate:   {sim_result.confidentiality_breach:.4f}  (target <= {tau})")
print(f"  Avg received:   {sim_result.avg_shares_received:.2f}")
print(f"  Avg leaked:     {sim_result.avg_shares_leaked:.2f}")
