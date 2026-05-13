"""
Benchmark metric dataclasses.

Each dataclass matches one section of the CLAUDE.md metrics spec:

  IKMetrics       — IK set size, safe set size, diversity score
  PlanningMetrics — success rate, time, nodes explored, path length
  ControlMetrics  — passivity violations, energy injection, tank trace
  ContactMetrics  — force error vs desired contact force
  BenchmarkResult — aggregates all of the above for one scenario run
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------
@dataclass
class IKMetrics:
    n_raw:          int    # raw IK solutions (before dedup / filtering)
    n_safe:         int    # safe IK solutions after filter_safe_set
    diversity_score: float  # mean pairwise L2 distance of safe set (joint space)

    @classmethod
    def compute(cls, Q_raw, Q_safe) -> "IKMetrics":
        import numpy as np

        n_raw = len(Q_raw) if Q_raw is not None else 0
        n_safe = len(Q_safe) if Q_safe else 0

        if n_safe > 1:
            Q = np.array(Q_safe)
            dists = []
            for i in range(len(Q)):
                for j in range(i + 1, len(Q)):
                    dists.append(float(np.linalg.norm(Q[i] - Q[j])))
            diversity = float(np.mean(dists))
        else:
            diversity = 0.0

        return cls(n_raw=n_raw, n_safe=n_safe, diversity_score=diversity)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
@dataclass
class PlanningMetrics:
    success:           bool
    time_to_solution:  float   # seconds; inf if failed
    nodes_explored:    int
    collision_checks:  int
    path_length:       float   # sum of joint-space segment lengths; 0 if failed
    iterations:        int

    @classmethod
    def from_plan_result(cls, result) -> "PlanningMetrics":
        """Build from a PlanResult (src.solver.planner.birrt.PlanResult)."""
        import numpy as np

        path_len = 0.0
        if result.success and result.path:
            path = result.path
            for i in range(len(path) - 1):
                path_len += float(np.linalg.norm(
                    np.asarray(path[i + 1]) - np.asarray(path[i])
                ))

        return cls(
            success=result.success,
            time_to_solution=result.time_to_solution if result.success else float("inf"),
            nodes_explored=result.nodes_explored,
            collision_checks=result.collision_checks,
            path_length=path_len,
            iterations=result.iterations,
        )


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
@dataclass
class ControlMetrics:
    n_steps:                int
    passivity_violations:   int    # steps where z > 0 (f_R injected energy)
    total_energy_injected:  float  # ∫ max(0, z) dt
    tank_energy_final:      float
    tank_energy_min:        float  # minimum over the whole run
    v_start:                float  # Lyapunov V at step 0
    v_end:                  float  # Lyapunov V at final step
    converged:              bool   # v_end < v_start

    @classmethod
    def from_results(cls, results) -> "ControlMetrics":
        """Build from list of ControlResult (src.solver.controller.impedance)."""
        vs     = [r.V           for r in results]
        tanks  = [r.tank_energy for r in results]
        viols  = sum(1 for r in results if r.passivity_violated)
        injected = sum(r.z for r in results if r.z > 0) * 0.01  # approximate ∫ z dt
        return cls(
            n_steps=len(results),
            passivity_violations=viols,
            total_energy_injected=injected,
            tank_energy_final=tanks[-1],
            tank_energy_min=min(tanks),
            v_start=vs[0],
            v_end=vs[-1],
            converged=vs[-1] < vs[0],
        )


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
@dataclass
class ContactMetrics:
    n_contacts:     int    # number of active contacts observed
    force_error_sq: float  # mean (|F| - F_desired)^2 over contacted steps

    @classmethod
    def from_contact_log(
        cls,
        contact_log: List[List[dict]],
        f_desired: float = 0.0,
    ) -> "ContactMetrics":
        import numpy as np

        n = sum(len(cs) for cs in contact_log)
        if n == 0:
            return cls(n_contacts=0, force_error_sq=0.0)

        errors = []
        for contacts in contact_log:
            for c in contacts:
                f_mag = float(np.linalg.norm(c["force"]))
                errors.append((f_mag - f_desired) ** 2)

        return cls(n_contacts=n, force_error_sq=float(np.mean(errors)))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    scenario:  str
    ik:        Optional[IKMetrics]       = None
    planning:  Optional[PlanningMetrics] = None
    control:   Optional[ControlMetrics]  = None
    contact:   Optional[ContactMetrics]  = None

    def to_dict(self) -> dict:
        d = {"scenario": self.scenario}
        if self.ik       is not None: d["ik"]       = asdict(self.ik)
        if self.planning is not None: d["planning"]  = asdict(self.planning)
        if self.control  is not None: d["control"]   = asdict(self.control)
        if self.contact  is not None: d["contact"]   = asdict(self.contact)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Scenario: {self.scenario}")
        print(f"{'='*60}")
        if self.ik:
            print(f"  IK     | raw={self.ik.n_raw:4d}  safe={self.ik.n_safe:4d}"
                  f"  diversity={self.ik.diversity_score:.3f}")
        if self.planning:
            p = self.planning
            status = "SUCCESS" if p.success else "FAILED "
            print(f"  Plan   | {status}  t={p.time_to_solution:.3f}s"
                  f"  nodes={p.nodes_explored:5d}  len={p.path_length:.2f}")
        if self.control:
            c = self.control
            conv = "converged" if c.converged else "diverged"
            print(f"  Ctrl   | {conv}  V: {c.v_start:.3f}→{c.v_end:.3f}"
                  f"  viol={c.passivity_violations}/{c.n_steps}"
                  f"  tank_min={c.tank_energy_min:.3f}")
        if self.contact:
            print(f"  Contact| n={self.contact.n_contacts}"
                  f"  force_err_sq={self.contact.force_error_sq:.3f}")
