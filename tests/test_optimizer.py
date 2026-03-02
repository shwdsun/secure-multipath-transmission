"""Tests for smt.optimizer module."""

from __future__ import annotations

import pytest

from smt.models import NetworkTopology
from smt.optimizer import OptimizationResult, SolverBackend, ThroughputOptimizer
from smt.phases.base import SAVTuple


class TestThroughputOptimizer:
    @pytest.fixture
    def optimizer(self, simple_topology: NetworkTopology) -> ThroughputOptimizer:
        return ThroughputOptimizer(simple_topology, backend=SolverBackend.PULP)

    def test_empty_tuples(self, optimizer: ThroughputOptimizer):
        result = optimizer.optimize(set())
        assert result.throughput == 0.0
        assert result.status == "No tuples"

    def test_single_tuple(self, optimizer: ThroughputOptimizer):
        tuples: set[SAVTuple] = {((1, 1, 1), 2)}
        result = optimizer.optimize(tuples)
        assert result.throughput >= 1.0
        assert result.status == "Optimal"

    def test_bandwidth_respected(
        self, simple_topology: NetworkTopology
    ):
        """Throughput should not exceed the minimum edge bandwidth."""
        opt = ThroughputOptimizer(simple_topology, backend=SolverBackend.PULP)
        tuples: set[SAVTuple] = {((5, 0, 0), 3)}
        result = opt.optimize(tuples)
        # Path 1 uses edges (1,2) bw=5 and (2,5) bw=5; 5 shares/msg
        # so at most 1 message
        assert result.throughput <= 1.0

    def test_multiple_tuples(self, optimizer: ThroughputOptimizer):
        tuples: set[SAVTuple] = {
            ((1, 0, 0), 1),
            ((0, 1, 0), 1),
            ((0, 0, 1), 1),
        }
        result = optimizer.optimize(tuples)
        assert result.throughput >= 3.0

    def test_result_structure(self, optimizer: ThroughputOptimizer):
        tuples: set[SAVTuple] = {((1, 1, 1), 2)}
        result = optimizer.optimize(tuples)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.allocation, dict)
        assert isinstance(result.status, str)

    def test_gurobi_import_error(
        self, simple_topology: NetworkTopology
    ):
        """Gurobi backend raises clear error if gurobipy is not installed."""
        opt = ThroughputOptimizer(simple_topology, backend=SolverBackend.GUROBI)
        try:
            opt.optimize({((1, 1, 1), 2)})
        except ImportError as e:
            assert "gurobipy" in str(e)
