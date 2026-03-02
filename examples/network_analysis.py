#!/usr/bin/env python3
"""Network analysis example: comparing phases on a larger topology.

Generates a 12-node layered network, assigns adversary parameters,
and compares Phase I vs Phase II share allocation strategies in terms
of minimal share count and achievable throughput.
"""

import time

from smt.graph import (
    assign_adversary_params,
    build_topology,
    generate_layered_graph,
)
from smt.optimizer import ThroughputOptimizer
from smt.phases.phase1 import Phase1Strategy
from smt.phases.phase2 import Phase2Strategy
from smt.probability import pmf_sum_binomials, tail_prob

# --- 1. Generate a layered network ---
print("=" * 60)
print("Generating 12-node layered network")
print("=" * 60)

topology = generate_layered_graph(
    nodes_per_layer=[1, 3, 4, 3, 1],
    edge_prob=0.35,
    seed=42,
)
print(f"Sender: {topology.sender}, Receiver: {topology.receiver}")
print(f"Nodes: {topology.nodes}")
print(f"Adjacency:")
for node in sorted(topology.adjacency.keys()):
    if topology.adjacency[node]:
        print(f"  {node} -> {topology.adjacency[node]}")

# --- 2. Assign adversary parameters ---
topology.node_params = assign_adversary_params(
    topology, n_compromised=6, seed=42,
)
topology = build_topology(topology, max_paths=6)

print(f"\nCompromised nodes: {sorted(topology.node_params.keys())}")
for node, params in sorted(topology.node_params.items()):
    print(f"  Node {node}: p_int={params.p_int}, delta={params.delta}, "
          f"d={params.d:.4f}, f={params.f:.4f}")

assert topology.path_metrics is not None
print(f"\nPaths ({len(topology.paths)}):")
for i, (path, metrics) in enumerate(zip(topology.paths, topology.path_metrics)):
    print(f"  P{i+1}: {path}")
    print(f"       epsilon={metrics.epsilon:.4f}, p_val={metrics.p_val:.4f}, "
          f"p_drop={metrics.p_drop:.4f}")

# --- 3. Phase I analysis ---
print("\n" + "=" * 60)
print("Phase I: Passive Leakage Only")
print("=" * 60)

sigma, tau = 0.95, 0.01

phase1 = Phase1Strategy(topology.path_metrics, sigma=sigma, tau=tau)
t0 = time.perf_counter()
tuples_1 = phase1.generate_minimal_tuples(n_max=20)
t1 = time.perf_counter()

print(f"Generated {len(tuples_1)} minimal SAVs in {t1-t0:.3f}s")
if tuples_1:
    min_k = min(sum(nv) for nv, _ in tuples_1)
    max_k = max(sum(nv) for nv, _ in tuples_1)
    print(f"Share count range: [{min_k}, {max_k}]")

    opt1 = ThroughputOptimizer(topology)
    result1 = opt1.optimize(tuples_1)
    print(f"Phase I throughput: {result1.throughput:.0f} messages/timeslot")

# --- 4. Phase II analysis ---
print("\n" + "=" * 60)
print("Phase II: Passive Leakage + Active Dropping")
print("=" * 60)

phase2 = Phase2Strategy(topology.path_metrics, sigma=sigma, tau=tau)
t0 = time.perf_counter()
tuples_2 = phase2.generate_minimal_tuples(n_max=15)
t1 = time.perf_counter()

print(f"Generated {len(tuples_2)} minimal SAV-tuples in {t1-t0:.3f}s")
if tuples_2:
    min_n = min(sum(nv) for nv, _ in tuples_2)
    max_n = max(sum(nv) for nv, _ in tuples_2)
    print(f"Total share range: [{min_n}, {max_n}]")
    thresholds = sorted(set(t for _, t in tuples_2))
    print(f"Threshold values: {thresholds}")

    opt2 = ThroughputOptimizer(topology)
    result2 = opt2.optimize(tuples_2)
    print(f"Phase II throughput: {result2.throughput:.0f} messages/timeslot")

    # Show a few example tuples with their security analysis
    print("\nExample tuples with security analysis:")
    for n_vec, t in sorted(tuples_2, key=lambda x: sum(x[0]))[:5]:
        eps = [m.epsilon for m in topology.path_metrics]
        gam = [m.p_val for m in topology.path_metrics]
        pmf_e = pmf_sum_binomials(list(n_vec), eps)
        pmf_b = pmf_sum_binomials(list(n_vec), gam)
        print(f"  SAV={n_vec}, t={t}: "
              f"P[leak]={tail_prob(pmf_e, t):.6f}, "
              f"P[recv>={t}]={tail_prob(pmf_b, t):.6f}")

# --- 5. Comparison ---
print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)
print(f"{'Metric':<30} {'Phase I':>10} {'Phase II':>10}")
print("-" * 52)
if tuples_1 and tuples_2:
    min_k_1 = min(sum(nv) for nv, _ in tuples_1)
    min_n_2 = min(sum(nv) for nv, _ in tuples_2)
    print(f"{'Min shares/message':<30} {min_k_1:>10} {min_n_2:>10}")
    print(f"{'Minimal SAV count':<30} {len(tuples_1):>10} {len(tuples_2):>10}")
    print(f"{'Throughput (msg/slot)':<30} {result1.throughput:>10.0f} {result2.throughput:>10.0f}")
    print(f"{'Handles dropping?':<30} {'No':>10} {'Yes':>10}")
