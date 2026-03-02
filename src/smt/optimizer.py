"""ILP for secrecy throughput maximization.

PuLP (default) or Gurobi backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

from smt.models import NetworkTopology
from smt.phases.base import SAVTuple

logger = logging.getLogger(__name__)


class SolverBackend(Enum):
    PULP = auto()
    GUROBI = auto()


@dataclass
class OptimizationResult:
    """Result of the throughput optimization.

    Attributes:
        throughput: Optimal number of messages per timeslot.
        allocation: Mapping from SAV-tuple to number of messages using it.
        status: Solver status string.
    """

    throughput: float
    allocation: dict[SAVTuple, float]
    status: str


class ThroughputOptimizer:
    """ILP-based secrecy throughput maximizer.

    Formulation:
        maximize    sum_s x_s
        subject to  sum_s x_s * sum_j 1{e in P_j} * n_j^(s) <= b_e   for all e
                    x_s >= 0, integer

    Args:
        topology: Network topology with paths and edge bandwidths.
        backend: Solver backend to use.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        backend: SolverBackend = SolverBackend.PULP,
    ) -> None:
        self.topology = topology
        self.backend = backend

        self._edge_path_map = self._build_edge_path_map()

    def _build_edge_path_map(self) -> dict[tuple[int, int], list[int]]:
        """Map each edge to the indices of paths that traverse it."""
        edge_paths: dict[tuple[int, int], list[int]] = {}
        for j, path in enumerate(self.topology.paths):
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in edge_paths:
                    edge_paths[edge] = []
                edge_paths[edge].append(j)
        return edge_paths

    def optimize(
        self,
        tuples: set[SAVTuple],
        time_limit: float | None = None,
    ) -> OptimizationResult:
        """Solve the throughput maximization ILP.

        Args:
            tuples: Set of minimal feasible SAV-tuples.
            time_limit: Solver time limit in seconds (optional).

        Returns:
            OptimizationResult with optimal throughput and allocation.
        """
        if not tuples:
            return OptimizationResult(throughput=0.0, allocation={}, status="No tuples")

        if self.backend == SolverBackend.GUROBI:
            return self._solve_gurobi(tuples, time_limit)
        return self._solve_pulp(tuples, time_limit)

    def _solve_pulp(
        self,
        tuples: set[SAVTuple],
        time_limit: float | None,
    ) -> OptimizationResult:
        """Solve using PuLP with the CBC solver."""
        import pulp

        prob = pulp.LpProblem("SecrecyThroughput", pulp.LpMaximize)

        tuple_list = list(tuples)
        x = {
            i: pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer")
            for i in range(len(tuple_list))
        }

        prob += pulp.lpSum(x[i] for i in range(len(tuple_list)))

        for edge, bw in self.topology.edge_bandwidths.items():
            path_indices = self._edge_path_map.get(edge, [])
            if not path_indices:
                continue

            load = pulp.lpSum(
                x[s] * tuple_list[s][0][j]
                for s in range(len(tuple_list))
                for j in path_indices
            )
            prob += load <= bw

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        prob.solve(solver)

        allocation: dict[SAVTuple, float] = {}
        for i, tup in enumerate(tuple_list):
            val = x[i].varValue
            if val is not None and val > 0.5:
                allocation[tup] = float(val)

        throughput = pulp.value(prob.objective) or 0.0

        return OptimizationResult(
            throughput=throughput,
            allocation=allocation,
            status=pulp.LpStatus[prob.status],
        )

    def _solve_gurobi(
        self,
        tuples: set[SAVTuple],
        time_limit: float | None,
    ) -> OptimizationResult:
        """Solve using Gurobi (requires gurobipy)."""
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise ImportError(
                "Gurobi backend requires gurobipy. "
                "Install with: pip install gurobipy"
            ) from exc

        model = gp.Model("SecrecyThroughput")
        model.Params.LogToConsole = 0
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        tuple_list = list(tuples)
        x = {
            i: model.addVar(vtype=gp.GRB.INTEGER, lb=0, name=f"x_{i}")
            for i in range(len(tuple_list))
        }
        model.update()

        model.setObjective(
            gp.quicksum(x[i] for i in range(len(tuple_list))),
            gp.GRB.MAXIMIZE,
        )

        for edge, bw in self.topology.edge_bandwidths.items():
            path_indices = self._edge_path_map.get(edge, [])
            if not path_indices:
                continue

            load = gp.quicksum(
                x[s] * tuple_list[s][0][j]
                for s in range(len(tuple_list))
                for j in path_indices
            )
            model.addConstr(load <= bw)

        model.optimize()

        allocation: dict[SAVTuple, float] = {}
        status_str = "Unknown"

        if model.Status == gp.GRB.OPTIMAL:
            status_str = "Optimal"
            for i, tup in enumerate(tuple_list):
                val = x[i].X
                if val > 0.5:
                    allocation[tup] = float(val)
        elif model.Status == gp.GRB.INFEASIBLE:
            status_str = "Infeasible"
        elif model.Status == gp.GRB.TIME_LIMIT:
            status_str = "TimeLimit"
            for i, tup in enumerate(tuple_list):
                val = x[i].X
                if val > 0.5:
                    allocation[tup] = float(val)

        throughput = model.ObjVal if model.SolCount > 0 else 0.0

        return OptimizationResult(
            throughput=throughput,
            allocation=allocation,
            status=status_str,
        )
