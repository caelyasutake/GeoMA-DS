"""
Aggregate per-trial results into summary statistics.

Usage::

    results = load_jsonl("outputs/eval/per_trial_results.jsonl")
    agg = aggregate_results(results)
    save_aggregate(agg, Path("outputs/eval"))
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import TrialMetrics


# ---------------------------------------------------------------------------
# Per-metric stats
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    """Descriptive statistics for a single scalar metric."""
    name:   str
    n:      int   = 0
    mean:   Optional[float] = None
    std:    Optional[float] = None
    median: Optional[float] = None
    p25:    Optional[float] = None
    p75:    Optional[float] = None
    best:   Optional[float] = None
    worst:  Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "n": self.n,
            "mean": self.mean, "std": self.std,
            "median": self.median,
            "p25": self.p25, "p75": self.p75,
            "best": self.best, "worst": self.worst,
        }


def _stats(values: List[float], name: str = "") -> MetricStats:
    arr = [v for v in values if v is not None and np.isfinite(v)]
    if not arr:
        return MetricStats(name=name, n=0)
    a = np.array(arr)
    return MetricStats(
        name=name, n=len(a),
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        median=float(np.median(a)),
        p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)),
        best=float(np.min(a)),
        worst=float(np.max(a)),
    )


def _rate(flags: List[bool]) -> float:
    return float(np.mean(flags)) if flags else 0.0


# ---------------------------------------------------------------------------
# Condition aggregate
# ---------------------------------------------------------------------------

@dataclass
class ConditionAggregate:
    """All aggregated stats for one (scenario, condition) combination."""
    scenario:   str
    condition:  str
    n_trials:   int
    n_errors:   int

    # Key rates
    plan_success_rate:    float = 0.0
    terminal_success_rate: float = 0.0
    contact_established_rate: float = 0.0

    # IK
    ik_n_safe:         MetricStats = field(default_factory=lambda: MetricStats("ik_n_safe"))
    ik_diversity:      MetricStats = field(default_factory=lambda: MetricStats("ik_diversity"))
    ik_goal_rank:      MetricStats = field(default_factory=lambda: MetricStats("ik_goal_rank"))

    # Planning
    plan_time_s:       MetricStats = field(default_factory=lambda: MetricStats("plan_time_s"))
    plan_path_length:  MetricStats = field(default_factory=lambda: MetricStats("plan_path_length"))
    plan_nodes:        MetricStats = field(default_factory=lambda: MetricStats("plan_nodes"))
    plan_iterations:   MetricStats = field(default_factory=lambda: MetricStats("plan_iterations"))
    plan_col_checks:   MetricStats = field(default_factory=lambda: MetricStats("plan_col_checks"))

    # Execution
    final_goal_err:    MetricStats = field(default_factory=lambda: MetricStats("final_goal_err"))
    conv_time_s:       MetricStats = field(default_factory=lambda: MetricStats("conv_time_s"))
    exec_path_len:     MetricStats = field(default_factory=lambda: MetricStats("exec_path_len"))

    # Passivity
    clipped_ratio:     MetricStats = field(default_factory=lambda: MetricStats("clipped_ratio"))
    min_tank_energy:   MetricStats = field(default_factory=lambda: MetricStats("min_tank_energy"))
    mean_power_before: MetricStats = field(default_factory=lambda: MetricStats("mean_power_before"))
    beta_R_zero_frac:  MetricStats = field(default_factory=lambda: MetricStats("beta_R_zero_frac"))

    # Contact
    contact_maintained: MetricStats = field(default_factory=lambda: MetricStats("contact_maintained"))
    mean_contact_force: MetricStats = field(default_factory=lambda: MetricStats("mean_contact_force"))
    circle_rmse:        MetricStats = field(default_factory=lambda: MetricStats("circle_rmse"))
    radius_rmse:        MetricStats = field(default_factory=lambda: MetricStats("radius_rmse"))
    arc_ratio:          MetricStats = field(default_factory=lambda: MetricStats("arc_ratio"))
    height_error:       MetricStats = field(default_factory=lambda: MetricStats("height_error"))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "scenario": self.scenario,
            "condition": self.condition,
            "n_trials": self.n_trials,
            "n_errors": self.n_errors,
            "plan_success_rate": self.plan_success_rate,
            "terminal_success_rate": self.terminal_success_rate,
            "contact_established_rate": self.contact_established_rate,
        }
        for attr in [
            "ik_n_safe", "ik_diversity", "ik_goal_rank",
            "plan_time_s", "plan_path_length", "plan_nodes",
            "plan_iterations", "plan_col_checks",
            "final_goal_err", "conv_time_s", "exec_path_len",
            "clipped_ratio", "min_tank_energy", "mean_power_before", "beta_R_zero_frac",
            "contact_maintained", "mean_contact_force",
            "circle_rmse", "radius_rmse", "arc_ratio", "height_error",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = val.to_dict()
        return d


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def aggregate_results(
    trials: List[TrialMetrics],
) -> Dict[Tuple[str, str], ConditionAggregate]:
    """
    Group trials by (scenario, condition) and compute aggregate stats.

    Returns:
        Dict keyed by (scenario, condition) → ConditionAggregate.
    """
    groups: Dict[Tuple[str, str], List[TrialMetrics]] = defaultdict(list)
    for t in trials:
        groups[(t.scenario, t.condition)].append(t)

    aggs: Dict[Tuple[str, str], ConditionAggregate] = {}
    for (scen, cond), group in groups.items():
        agg = _aggregate_group(scen, cond, group)
        aggs[(scen, cond)] = agg

    return aggs


def _aggregate_group(
    scenario: str, condition: str, trials: List[TrialMetrics],
) -> ConditionAggregate:
    n_total  = len(trials)
    n_errors = sum(1 for t in trials if t.error is not None)

    agg = ConditionAggregate(
        scenario=scenario, condition=condition,
        n_trials=n_total, n_errors=n_errors,
    )

    # Rates
    plan_successes = [t.plan.success for t in trials if t.plan is not None]
    exec_successes = [t.execution.terminal_success for t in trials if t.execution is not None]
    contact_flags  = [t.contact.contact_established for t in trials if t.contact is not None]

    agg.plan_success_rate    = _rate(plan_successes)
    agg.terminal_success_rate = _rate(exec_successes)
    agg.contact_established_rate = _rate(contact_flags)

    # IK
    agg.ik_n_safe    = _stats([t.ik.n_safe           for t in trials if t.ik], "ik_n_safe")
    agg.ik_diversity = _stats([t.ik.ik_set_diversity  for t in trials if t.ik], "ik_diversity")
    agg.ik_goal_rank = _stats([t.ik.selected_goal_rank for t in trials if t.ik], "ik_goal_rank")

    # Planning (only successful plans)
    sp = [t.plan for t in trials if t.plan and t.plan.success]
    agg.plan_time_s      = _stats([p.time_s      for p in sp], "plan_time_s")
    agg.plan_path_length = _stats([p.path_length for p in sp], "plan_path_length")
    agg.plan_nodes       = _stats([p.nodes_explored for p in sp], "plan_nodes")
    agg.plan_iterations  = _stats([p.iterations  for p in sp], "plan_iterations")
    agg.plan_col_checks  = _stats([p.collision_checks for p in sp], "plan_col_checks")

    # Execution
    ex = [t.execution for t in trials if t.execution]
    agg.final_goal_err = _stats([e.final_goal_err for e in ex], "final_goal_err")
    agg.conv_time_s    = _stats(
        [e.convergence_time_s for e in ex if e.convergence_time_s is not None],
        "conv_time_s"
    )
    agg.exec_path_len  = _stats([e.exec_path_length for e in ex], "exec_path_len")

    # Passivity
    pa = [t.passivity for t in trials if t.passivity]
    agg.clipped_ratio     = _stats([p.clipped_ratio      for p in pa], "clipped_ratio")
    agg.min_tank_energy   = _stats([p.min_tank_energy    for p in pa], "min_tank_energy")
    agg.mean_power_before = _stats([p.mean_power_before  for p in pa], "mean_power_before")
    agg.beta_R_zero_frac  = _stats([p.beta_R_zero_fraction for p in pa], "beta_R_zero_frac")

    # Contact
    ct = [t.contact for t in trials if t.contact]
    agg.contact_maintained = _stats([c.contact_maintained_fraction for c in ct], "contact_maintained")
    agg.mean_contact_force = _stats([c.mean_contact_force          for c in ct], "mean_contact_force")
    agg.circle_rmse        = _stats([c.circle_tracking_rmse        for c in ct
                                     if c.circle_tracking_rmse != float("inf")], "circle_rmse")
    agg.radius_rmse        = _stats([c.circle_radius_rmse          for c in ct
                                     if c.circle_radius_rmse != float("inf")], "radius_rmse")
    agg.arc_ratio          = _stats([c.arc_completion_ratio        for c in ct], "arc_ratio")
    agg.height_error       = _stats([c.mean_height_error           for c in ct], "height_error")

    return agg


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_aggregate(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    output_dir: Path,
) -> None:
    """Save aggregate_results.json and summary_tables.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = {f"{scen}__{cond}": agg.to_dict() for (scen, cond), agg in aggs.items()}
    with open(output_dir / "aggregate_results.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, default=str)

    # CSV — one row per (scenario, condition)
    rows = []
    for (scen, cond), agg in aggs.items():
        row: Dict[str, Any] = {
            "scenario":   scen,
            "condition":  cond,
            "n_trials":   agg.n_trials,
            "n_errors":   agg.n_errors,
            "plan_success_rate":     agg.plan_success_rate,
            "terminal_success_rate": agg.terminal_success_rate,
            "contact_established_rate": agg.contact_established_rate,
        }
        for attr in [
            "plan_time_s", "plan_path_length", "final_goal_err",
            "clipped_ratio", "min_tank_energy",
            "contact_maintained", "circle_rmse", "arc_ratio",
        ]:
            ms: MetricStats = getattr(agg, attr, None)
            if ms and ms.n > 0:
                row[f"{attr}_mean"]   = ms.mean
                row[f"{attr}_std"]    = ms.std
                row[f"{attr}_median"] = ms.median
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_dir / "summary_tables.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
