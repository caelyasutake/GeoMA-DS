"""
eval_baselines — GeoMA-DS vs DS+CBF baseline comparison.

Two solvers evaluated across four canonical scenarios:

    geo_ma_ds          GeoMultiAttractorDS — production solver.
                       HJCD-IK batch → attractor classification/scoring →
                       geometric tangent/nullspace shaping → attractor switching.
                       Succeeds on all scenarios including homotopy barriers.
                       For barrier scenarios (cross_barrier etc.), boundary
                       escape waypoints are enabled automatically — no extra
                       flags are required.

    diffik_ds_cbf      DS+CBF baseline — comparison only.
                       Task-space DS + damped-LS differential IK + greedy CBF
                       velocity projection.  Maintains clearance but is trapped
                       by homotopy barriers (cross_barrier).

Three canonical scenarios:
    open_reach        No obstacles (free space)
    i_barrier         Frontal I-barrier (medium difficulty)
    cross_barrier     Y-Z cross, start top-right → goal bottom-left quadrant

Key results (robust profile, warm GPU):
    GeoMA-DS:   ~10 ms build / ~100 Hz planner rate / success on all scenarios
    DS+CBF:     0 ms build / instant / success only on open_reach + i_barrier

Usage (paper-facing commands — no extra flags needed)::

    python -m benchmarks.eval_baselines
    python -m benchmarks.eval_baselines --scenarios cross_barrier
    python -m benchmarks.eval_baselines --scenarios cross_barrier --methods geo_ma_ds diffik_ds_cbf
    python -m benchmarks.eval_baselines --steps 1000 --csv outputs/baselines.csv

Ablation / developer flags::

    --no-boundary-escape-waypoints   disable auto escape (measure IK-diversity-only baseline)
    --planner-profile fast           fast IK preset (128 restarts, 4 solutions)
    --enable-goal-shell-escape       legacy shell-escape alternative (not in paper)
"""

from __future__ import annotations

import argparse
import json
import sys
import csv
import datetime
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.scenarios.scenario_builders import (
    get_scenario,
    build_frontal_i_barrier_lr,
    build_frontal_yz_cross,
    free_space_scenario,
)
from src.solver.ds.factory import (
    _make_clearance_fn,
    build_geo_multi_attractor_ds,
    warmup_hjcdik,
)
from src.solver.ds.geo_multi_attractor_ds import GeoMultiAttractorDSConfig
from src.solver.planner.collision import _panda_link_positions, _panda_fk_batch
from src.solver.ik.coverage_expansion import (
    CoverageConfig,
    _grasptarget_pos,
    _grasptarget_jacobian_fd,
)


# ---------------------------------------------------------------------------
# Simulation constants (shared by all methods)
# ---------------------------------------------------------------------------
_DT             = 0.02        # seconds per control step
_N_STEPS        = 1500        # max steps per trial
_MAX_QDOT       = 2.0         # rad/s velocity saturation
_SUCCESS_THRESH = 0.05        # m — grasptarget-to-goal distance
_STALL_WINDOW   = 150         # steps to check for stall (~3s at 50 Hz)
_STALL_THRESH   = 1e-3        # m — minimum EE progress over window to not stall

# Joint limits for Panda
_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


# ---------------------------------------------------------------------------
# Clearance gradient (central finite differences)
# ---------------------------------------------------------------------------

def _clearance_gradient(q: np.ndarray, clearance_fn, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros(7)
    for j in range(7):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        grad[j] = (clearance_fn(qp) - clearance_fn(qm)) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    method:              str
    scenario:            str
    success:             bool
    conv_step:           int        # step at which success was achieved (-1 = none)
    stalled:             bool
    dnf_reason:          str        # "" | "stall" | "collision" | "joint_limit"
    final_grasp_err_m:   float
    min_clearance_m:     float
    collision_count:     int
    mean_step_ms:        float
    hz:                  float           # alias for control_hz (backward compat)
    control_hz:          float           # runtime loop frequency: n_steps / run_s
    planner_ms:          float           # one-time build/planner phase (ms)
    planner_hz:          Optional[float] # builds/s; None when planner_s == 0
    end_to_end_hz:       float           # n_steps / (planner_s + run_s)
    planner_source:      str             # "online_ik" | "precomputed_ik" | "disabled"
    ik_source:           str             # "online" | "precomputed" | "disabled"
    ik_generation_ms:    float           # time spent in HJCD-IK during build (0 if precomputed)
    cbf_active_fraction: float      # fraction of steps CBF modified q_dot
    max_alpha:           float      # max obs_blend_alpha (GeoMA-DS only, else 0.0)
    n_switches:          int        # attractor switches (GeoMA-DS only, else 0)
    joint_path_length:   float      # sum of ||Δq|| over trajectory (rad)
    task_path_length_m:  float      # sum of ||Δx_EE|| over trajectory (m)
    ik_rewarm_ms:        float = 0.0              # per-build lightweight rewarm (excluded from planner_ms)
    benchmark_ordering:  str  = "build_all_first" # "build_all_first" | "interleaved_build_run"
    n_switch_route:      int  = 0                 # switches that changed homotopy class mid-route
    n_switch_near_goal:  int  = 0                 # near-goal chatter switches (should be 0 with lockout)
    n_switch_obstacle:   int  = 0                 # switches in obstacle-influence region
    n_switch_forced:     int  = 0                 # forced stall switches (diagnostic mode only)
    n_esc_sw:            int  = 0                 # goal-shell escape switches
    n_esc_cyc:           int  = 0                 # escape family cycles (boundary escape)
    n_rec_sw:            int  = 0                 # clearance recovery switches
    n_back_sw:           int  = 0                 # backtrack/staging switches
    n_pregoal_sw:            int   = 0    # pre-goal attractor switches (online mode)
    n_reentry_fail:          int   = 0    # online escape IK generation failures
    classify_ms:             float = 0.0  # build-time family classification (ms)
    setup_ms:                float = 0.0  # build-time setup/scoring overhead (ms)
    runtime_plan_peak_ms:    float = 0.0  # worst single replanning event latency (ms)
    runtime_plan_avg_ms:     float = 0.0  # average latency per replanning event (ms)
    runtime_plan_events:     int   = 0    # number of online replanning events
    trial_idx:               int   = 0
    seed:                    int   = 0


# ---------------------------------------------------------------------------
# Multi-trial helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import hjcdik
        if hasattr(hjcdik, "set_seed"):
            hjcdik.set_seed(seed)
    except Exception:
        pass


def trial_success(result: "TrialResult", goal_tolerance: float = _SUCCESS_THRESH) -> bool:
    return (
        result.conv_step >= 0
        and result.collision_count == 0
        and result.min_clearance_m >= 0.0
        and result.final_grasp_err_m <= goal_tolerance
        and not result.stalled
    )


_SUMMARY_CONTINUOUS = [
    "conv_step", "final_grasp_err_m", "min_clearance_m", "planner_ms",
    "planner_hz", "ik_generation_ms", "classify_ms", "setup_ms",
    "control_hz", "end_to_end_hz", "mean_step_ms",
    "runtime_plan_events", "runtime_plan_peak_ms", "runtime_plan_avg_ms",
    "joint_path_length", "task_path_length_m",
]


def _compute_group_stats(rows: List["TrialResult"]) -> dict:
    N = len(rows)
    successes = [r for r in rows if trial_success(r)]
    num_success = len(successes)
    stats: dict = {
        "N": N,
        "num_success": num_success,
        "success_rate": num_success / N if N > 0 else 0.0,
        "num_collision": sum(1 for r in rows if r.collision_count > 0),
        "num_stalled":   sum(1 for r in rows if r.stalled),
        "num_timeout":   sum(1 for r in rows if r.conv_step < 0 and not r.stalled and r.collision_count == 0),
        "metrics": {},
    }
    for key in _SUMMARY_CONTINUOUS:
        if key == "conv_step":
            vals = [getattr(r, key) for r in successes if getattr(r, key, -1) >= 0]
        else:
            vals = [v for r in rows for v in [getattr(r, key, None)]
                    if v is not None and v == v]
        if not vals:
            stats["metrics"][key] = {"mean": None, "std": None, "median": None,
                                     "p95": None, "min": None, "max": None}
            continue
        arr = sorted(vals)
        mean_ = sum(arr) / len(arr)
        std_  = statistics.stdev(arr) if len(arr) > 1 else 0.0
        med_  = statistics.median(arr)
        p95_  = arr[int(0.95 * len(arr))] if len(arr) >= 2 else arr[-1]
        stats["metrics"][key] = {
            "mean": mean_, "std": std_, "median": med_,
            "p95": p95_, "min": arr[0], "max": arr[-1],
        }
    return stats



# ---------------------------------------------------------------------------
# Scenario builder helpers
# ---------------------------------------------------------------------------

def _build_open_reach():
    """Frontal I-barrier easy spec with obstacles removed (free-space reach)."""
    from src.scenarios.scenario_schema import ScenarioSpec
    spec = build_frontal_i_barrier_lr("easy")
    return ScenarioSpec(
        name="open_reach",
        q_start=spec.q_start.copy(),
        target_pose=spec.target_pose,
        ik_goals=[g.copy() for g in spec.ik_goals],
        obstacles=[],
        goal_radius=spec.goal_radius,
        planner=spec.planner,
        controller=spec.controller,
        visualization=spec.visualization,
    )


def _build_i_barrier() -> "ScenarioSpec":
    spec = build_frontal_i_barrier_lr("medium")
    spec.name = "i_barrier"
    return spec


def _build_cross_barrier() -> "ScenarioSpec":
    spec = build_frontal_yz_cross()
    spec.name = "cross_barrier"
    return spec


_SCENARIO_BUILDERS = {
    "open_reach":     _build_open_reach,
    "i_barrier":      _build_i_barrier,
    "cross_barrier":  _build_cross_barrier,
}

CANONICAL_SCENARIOS = list(_SCENARIO_BUILDERS.keys())
CANONICAL_METHODS   = ["diffik_ds_cbf", "geo_ma_ds"]
ALL_KNOWN_METHODS   = CANONICAL_METHODS

# Aliases for external callers
ALL_SCENARIOS  = CANONICAL_SCENARIOS
ALL_METHODS    = CANONICAL_METHODS

# IK planner profiles: preset bundles for batch_size / num_solutions / filter_mode / ik_source
_PLANNER_PROFILES = {
    "fast":        {"ik_batch_size": 128,  "ik_num_solutions": 8, "ik_filter_mode": "minimal", "ik_source": "online",
                    "enable_yz_expansion": False, "online_escape_batch": 64, "online_pregoal_batch": 32,
                    "online_escape_candidate_policy": "score_first", "online_ik_max_ms": 0.0,
                    "family_classifier_mode": "yz_cross_endpoint_fast"},
    "reactive":    {"ik_batch_size": 256,  "ik_num_solutions": 4, "ik_filter_mode": "minimal", "ik_source": "online",
                    "online_escape_candidate_policy": "score_first", "online_ik_max_ms": 0.0},
    "robust":      {"ik_batch_size": 1000, "ik_num_solutions": 8, "ik_filter_mode": "safe",    "ik_source": "online",
                    "enable_yz_expansion": False, "online_escape_batch": 256, "online_pregoal_batch": 64,
                    "online_escape_candidate_policy": "score_first", "online_ik_max_ms": 0.0},
    "precomputed": {"ik_batch_size": 1000, "ik_num_solutions": 8, "ik_filter_mode": "safe",    "ik_source": "precomputed",
                    "online_escape_candidate_policy": "score_first", "online_ik_max_ms": 0.0},
}
# Per-build rewarm: restores GPU P-state and memory allocator before each IK call.
# Must match the production batch_size — a smaller size only warms CUDA init, not
# buffer allocation, so the GPU clock may not reach P0 before the real call starts.
_IK_REWARM_BATCH_SIZE     = 512   # overridden at call sites with the actual ik_batch_size
_IK_REWARM_NUM_SOLUTIONS  = 1


# ---------------------------------------------------------------------------
# Build metadata extraction helper
# ---------------------------------------------------------------------------

def _extract_build_meta(method: str, solver, ik_source: str):
    """Return (planner_source, ik_source_result, ik_gen_ms, classify_ms) from a built solver."""
    if method.startswith("diffik"):
        return "disabled", "disabled", 0.0, 0.0
    if method.startswith("geo_ma"):
        bd = getattr(getattr(solver, "_ds", None), "planner_breakdown", {})
        ik_src = bd.get("ik_source", ik_source)
        ik_gen = bd.get("ik_generation_ms", 0.0)
        classify = bd.get("ik_to_attractors_ms", 0.0)
        planner_src = "online_ik" if ik_src == "online" else "precomputed_ik"
        return planner_src, ik_src, ik_gen, classify
    return "computed", "disabled", 0.0, 0.0


# ---------------------------------------------------------------------------
# DS+CBF baseline solver
# ---------------------------------------------------------------------------

class DiffIKDSCBF:
    """
    DS+CBF baseline: task-space DS with damped-LS differential IK and greedy
    CBF velocity projection.

    This is the comparison baseline — it maintains positive clearance but
    cannot escape homotopy barriers (gets trapped when the only collision-free
    approach requires passing through an obstacle-blocked region).

    Velocity field:
        ẋ = K_x * (x_goal - x)
        q_dot = J^+ ẋ,  J^+ = J^T (J J^T + λ²I)^-1

    CBF projection (greedy single-pass, no QP):
        activation when clearance < d_safe + d_buffer
        if ∇h^T q_dot + α*h < 0:
            q_dot -= violation * ∇h / (‖∇h‖² + ε)
    """
    name = "diffik_ds_cbf"

    def __init__(
        self,
        target:      np.ndarray,
        clearance_fn,
        K_x:         float = 2.0,
        lam:         float = 0.05,
        d_safe:      float = 0.03,
        d_buffer:    float = 0.10,
        alpha:       float = 8.0,
        grad_eps:    float = 1e-3,
    ):
        self.target       = target.copy()
        self.K_x          = K_x
        self.lam          = lam
        self.clearance_fn = clearance_fn
        self.d_safe       = d_safe
        self.d_buffer     = d_buffer
        self.alpha        = alpha
        self.grad_eps     = grad_eps

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        x     = _grasptarget_pos(q)
        err   = self.target - x
        x_dot = self.K_x * err

        J   = _grasptarget_jacobian_fd(q)               # (3, 7)
        JJT = J @ J.T + (self.lam ** 2) * np.eye(3)
        Jp  = J.T @ np.linalg.inv(JJT)                  # (7, 3) damped pseudoinverse
        q_dot = Jp @ x_dot

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        cl  = self.clearance_fn(q)
        h   = cl - self.d_safe
        cbf_active = False

        if cl < self.d_safe + self.d_buffer:
            grad_h = _clearance_gradient(q, self.clearance_fn, eps=self.grad_eps)
            lhs    = float(np.dot(grad_h, q_dot)) + self.alpha * float(h)
            if lhs < 0.0:
                denom  = float(np.dot(grad_h, grad_h)) + 1e-12
                q_dot  = q_dot - lhs * grad_h / denom
                cbf_active = True

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        return q_dot, {
            "grasp_err_m": float(np.linalg.norm(err)),
            "cbf_active":  cbf_active,
        }


# ---------------------------------------------------------------------------
# GeoMA-DS wrapper
# ---------------------------------------------------------------------------

class GeoMADSSolver:
    """Thin wrapper that drives GeoMultiAttractorDS like the local solvers."""

    def __init__(self, name: str, ds):
        self.name = name
        self._ds  = ds

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        q_dot, result = self._ds.compute(q, dt=_DT)

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        return q_dot, {
            "grasp_err_m":           float(result.goal_error),
            "cbf_active":            False,
            "alpha":                 float(result.obs_blend_alpha),
            "n_switches":            int(result.n_switches),
            "n_switch_near_goal":    int(result.n_switch_near_goal),
            "n_switch_obstacle":     int(result.n_switch_obstacle_region),
            "n_switch_route":        int(result.n_switch_route),
            "n_switch_forced_stall": int(result.n_switch_forced_stall),
            "n_escape_switches":       int(result.n_escape_switches),
            "n_escape_family_cycles":  int(result.n_escape_family_cycles),
            "n_recovery_switches":     int(result.n_recovery_switches),
            "n_backtrack_switches":    int(result.n_backtrack_switches),
            "n_pregoal_switches":           int(result.n_pregoal_switches),
            "n_reentry_fail":               int(result.n_reentry_fail),
            "runtime_planner_ms":           float(result.runtime_planner_ms),
            "runtime_planner_ms_max":       float(result.runtime_planner_ms_max),
            "runtime_planner_event_count":  int(result.runtime_planner_event_count),
        }


# ---------------------------------------------------------------------------
# Per-trial cross_barrier summary helper
# ---------------------------------------------------------------------------

def _print_cross_summary(scen_name: str, result: "TrialResult", solver) -> None:
    """Print a compact per-trial diagnostic for cross_barrier scenario."""
    if scen_name != "cross_barrier":
        return
    _ds = getattr(solver, "_ds", None)
    _cfg = getattr(_ds, "config", None)
    _pg_beta = list(getattr(_cfg, "pregoal_beta_sweep", [0.80])) if _cfg is not None else [0.80]
    print(
        f"[cross_summary "
        f"trial={result.trial_idx} seed={result.seed} "
        f"success={result.success} conv_step={result.conv_step} "
        f"final_grasp_err={result.final_grasp_err_m:.4f} "
        f"min_clearance={result.min_clearance_m:.4f} "
        f"n_escape_events={result.runtime_plan_events} "
        f"pregoal_beta_sweep={_pg_beta} "
        f"pregoal_accepted={getattr(_ds, '_pregoal_accepted_count', -1)} "
        f"pregoal_rejected={getattr(_ds, '_pregoal_rejected_count', -1)} "
        f"pregoal_last_clearance={getattr(_ds, '_pregoal_last_clearance', -1.0):.3f} "
        f"pregoal_last_task_err={getattr(_ds, '_pregoal_last_task_err', -1.0):.3f} "
        f"pregoal_final_dist={getattr(_ds, '_pregoal_final_dist_rad', -1.0):.3f} "
        f"pregoal_regen_count={getattr(_ds, '_pregoal_regen_count', 0)} "
        f"reentry_blocked={getattr(_ds, '_n_goal_reentry_blocked', 0)}]",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Universal trial runner
# ---------------------------------------------------------------------------

def run_trial(
    solver,
    spec,
    clearance_fn,
    n_steps: int = _N_STEPS,
    planner_s: float = 0.0,
    planner_source: str = "computed",
    ik_source: str = "disabled",
    ik_generation_ms: float = 0.0,
    ik_rewarm_ms: float = 0.0,
    benchmark_ordering: str = "build_all_first",
) -> Tuple[TrialResult, np.ndarray]:
    """
    Simulate one trial and return (TrialResult, q_history).

    q_history: (T, 7) joint trajectory, including q_start as step 0.
    The same clearance_fn is used by all methods.
    Collision (clearance < 0): halt immediately, report DNF.
    Success: grasptarget distance to target < _SUCCESS_THRESH.
    Stall: |progress| < _STALL_THRESH over last _STALL_WINDOW steps.
    """
    target_pos = np.array(spec.target_pose["position"], dtype=float)

    q      = np.asarray(spec.q_start, dtype=float)
    q_prev = q.copy()
    q_hist: List[np.ndarray] = [q.copy()]

    conv_step    = -1
    min_cl       = float("inf")
    collision_n  = 0
    cbf_active_n = 0
    max_alpha    = 0.0
    n_switches   = 0
    n_switch_route     = 0
    n_switch_near_goal = 0
    n_switch_obstacle  = 0
    n_switch_forced    = 0
    n_esc_sw           = 0
    n_esc_cyc          = 0
    n_rec_sw           = 0
    n_back_sw          = 0
    n_pregoal_sw            = 0
    n_reentry_fail          = 0
    runtime_planner_ms_total = 0.0
    runtime_planner_ms_max   = 0.0
    runtime_planner_ev_cnt   = 0

    # Read build-time timing breakdown from solver (geo_ma only)
    _bd            = getattr(getattr(solver, "_ds", None), "planner_breakdown", {})
    _classify_ms   = float(_bd.get("ik_to_attractors_ms", 0.0))
    _setup_ms      = float(_bd.get("static_scoring_ms", 0.0)
                           + _bd.get("clearance_closure_ms", 0.0)
                           + _bd.get("other_ms", 0.0))

    joint_path   = 0.0
    task_path    = 0.0
    x_prev       = _grasptarget_pos(q)

    # Stall detection buffer
    err_buf: List[float] = []

    step_times: List[float] = []
    stalled    = False
    dnf_reason = ""

    t0_trial = time.perf_counter()

    for step in range(n_steps):
        t0 = time.perf_counter()

        q_dot, diag = solver.compute(q)
        step_times.append((time.perf_counter() - t0) * 1e3)

        # Integrate
        q_new = q + _DT * q_dot
        q_new = np.clip(q_new, _Q_LO, _Q_HI)

        # Clearance check — halt on penetration (robot cannot pass through obstacles)
        cl = float(clearance_fn(q_new))
        if cl < min_cl:
            min_cl = cl
        if cl < 0.0:
            collision_n += 1
            dnf_reason   = "collision"
            break

        # CBF / alpha tracking
        if diag.get("cbf_active", False):
            cbf_active_n += 1
        max_alpha        = max(max_alpha,        float(diag.get("alpha",              0.0)))
        n_switches       = max(n_switches,       int(diag.get("n_switches",           0)))
        n_switch_route     = max(n_switch_route,     int(diag.get("n_switch_route",         0)))
        n_switch_near_goal = max(n_switch_near_goal, int(diag.get("n_switch_near_goal",     0)))
        n_switch_obstacle  = max(n_switch_obstacle,  int(diag.get("n_switch_obstacle",      0)))
        n_switch_forced    = max(n_switch_forced,    int(diag.get("n_switch_forced_stall",   0)))
        n_esc_sw           = max(n_esc_sw,           int(diag.get("n_escape_switches",        0)))
        n_esc_cyc          = max(n_esc_cyc,          int(diag.get("n_escape_family_cycles",   0)))
        n_rec_sw           = max(n_rec_sw,           int(diag.get("n_recovery_switches",      0)))
        n_back_sw          = max(n_back_sw,          int(diag.get("n_backtrack_switches",     0)))
        n_pregoal_sw            = max(n_pregoal_sw,            int(diag.get("n_pregoal_switches",          0)))
        n_reentry_fail          = max(n_reentry_fail,          int(diag.get("n_reentry_fail",              0)))
        runtime_planner_ms_total  = max(runtime_planner_ms_total, float(diag.get("runtime_planner_ms", 0.0)))
        runtime_planner_ms_max   = max(runtime_planner_ms_max,  float(diag.get("runtime_planner_ms_max",    0.0)))
        runtime_planner_ev_cnt   = max(runtime_planner_ev_cnt,  int(diag.get("runtime_planner_event_count", 0)))

        # Path length accumulators
        dq  = np.linalg.norm(q_new - q)
        x_n = _grasptarget_pos(q_new)
        dx  = float(np.linalg.norm(x_n - x_prev))
        joint_path += float(dq)
        task_path  += dx
        x_prev      = x_n

        q = q_new
        q_hist.append(q.copy())

        # Success check
        grasp_err = float(np.linalg.norm(target_pos - _grasptarget_pos(q)))
        if grasp_err < _SUCCESS_THRESH and conv_step < 0:
            conv_step = step

        # Stall detection (only check before convergence)
        if conv_step < 0:
            err_buf.append(grasp_err)
            if len(err_buf) > _STALL_WINDOW:
                err_buf.pop(0)
            if len(err_buf) == _STALL_WINDOW:
                progress = err_buf[0] - err_buf[-1]
                if abs(progress) < _STALL_THRESH:
                    # Don't break if the DS is actively running its recovery pipeline;
                    # backtrack/escape need time to execute before declaring stall.
                    _ds_obj  = getattr(solver, "_ds", None)
                    _ds_busy = _ds_obj is not None and (
                        getattr(_ds_obj, "_backtrack_mode_active", False)
                        or getattr(_ds_obj, "_escape_mode_active",  False)
                        or getattr(_ds_obj, "_pre_goal_mode_active", False)
                    )
                    if not _ds_busy:
                        stalled    = True
                        dnf_reason = "stall"
                        break

    grasp_err_final = float(np.linalg.norm(target_pos - _grasptarget_pos(q)))

    _fc = ""   # failure classification string (populated below when stalled)

    # Failure classification for stalled boundary-escape runs (Steps 13)
    if stalled and hasattr(solver, "_ds") and hasattr(solver._ds, "_escape_mode_active"):
        _ds = solver._ds
        _esc_atts = [a for a in _ds.attractors if a.kind == "escape"]
        _deferred = getattr(_ds, "_deferred_escape_candidates", [])
        if not _esc_atts:
            if _deferred:
                _fc = "A_online: deferred escape candidates existed but online generation was not called"
            else:
                _fc = "A: no escape families available"
        elif _ds._n_escape_switches == 0:
            _fc = "B: escape switch never triggered"
        elif _ds._n_escape_switches == 0 and getattr(_ds, "_n_backtrack_switches", 0) > 0:
            _fc = "B1: backtrack did not move far enough into escape basin"
        elif _ds._n_escape_family_cycles == 0 and _ds._n_escape_switches > 0:
            _fc = "C: escape family tried but no progress"
        elif _ds._escape_exhausted:
            _fc = "D: all escape families exhausted"
        elif not _ds._escape_mode_active and _ds._n_escape_switches > 0:
            _fc = "E: escape reached but return-to-goal failed"
        elif (
            getattr(_ds, "_escape_blocked_count", 0) > 0
            and getattr(_ds, "_grad_recovery_best_clearance", 0.0) < 0.005
        ):
            _fc = "F: recovery failed to improve clearance"
        elif getattr(_ds, "_n_backtrack_switches", 0) > 0:
            _fc = "B2: backtrack succeeded but escape family still failed"
        else:
            _fc = "C: escape family tried but no further progress"
        print(f"[failure_classification] {_fc}", flush=True)

    n_steps_run = len(step_times)
    mean_ms     = float(np.mean(step_times)) if step_times else 0.0
    run_s_total = time.perf_counter() - t0_trial

    control_hz    = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    total_s       = planner_s + run_s_total
    end_to_end_hz = float(n_steps_run) / total_s if total_s > 0 else 0.0
    planner_hz    = (1.0 / planner_s if planner_s > 1e-9 else None) if planner_source != "disabled" else None
    cbf_frac      = cbf_active_n / max(n_steps_run, 1)

    # Planner timing breakdown (always, when build data available)
    if _bd:
        _other = max(0.0, planner_s * 1000.0 - ik_generation_ms - _classify_ms - _setup_ms)
        print(
            f"[planner_breakdown]"
            f" init={planner_s*1000:.1f}ms"
            f" ik={ik_generation_ms:.1f}ms"
            f" classify={_classify_ms:.1f}ms"
            f" setup={_setup_ms:.1f}ms"
            f" other={_other:.1f}ms",
            flush=True,
        )

    # Report key online planner timing metrics
    if runtime_planner_ev_cnt > 0:
        _plan_peak_hz = 1000.0 / runtime_planner_ms_max if runtime_planner_ms_max > 0 else float("inf")
        _init_ms_str = f"{planner_s*1000:.1f}"
        _init_hz_str = f"{1000.0/(planner_s*1000.0):.0f}" if planner_s > 1e-9 else "N/A"
        print(
            f"[timing_report]"
            f" planner_init_ms={_init_ms_str}"
            f" planner_init_Hz={_init_hz_str}"
            f" ctrl_Hz={control_hz:.0f}"
            f" e2e_Hz={end_to_end_hz:.0f}"
            f" runtime_plan_events={runtime_planner_ev_cnt}"
            f" runtime_plan_peak_ms={runtime_planner_ms_max:.1f}"
            f" runtime_plan_peak_Hz={_plan_peak_hz:.0f}",
            flush=True,
        )

    result = TrialResult(
        method              = solver.name,
        scenario            = spec.name,
        success             = conv_step >= 0,
        conv_step           = conv_step,
        stalled             = stalled,
        dnf_reason          = dnf_reason,
        final_grasp_err_m   = grasp_err_final,
        min_clearance_m     = min_cl if min_cl < float("inf") else 0.0,
        collision_count     = collision_n,
        mean_step_ms        = mean_ms,
        hz                  = control_hz,
        control_hz          = control_hz,
        planner_ms          = planner_s * 1000.0,
        planner_hz          = planner_hz,
        end_to_end_hz       = end_to_end_hz,
        planner_source      = planner_source,
        ik_source           = ik_source,
        ik_generation_ms    = ik_generation_ms,
        cbf_active_fraction = cbf_frac,
        max_alpha           = max_alpha,
        n_switches          = n_switches,
        joint_path_length   = joint_path,
        task_path_length_m  = task_path,
        ik_rewarm_ms        = ik_rewarm_ms,
        benchmark_ordering  = benchmark_ordering,
        n_switch_route      = n_switch_route,
        n_switch_near_goal  = n_switch_near_goal,
        n_switch_obstacle   = n_switch_obstacle,
        n_switch_forced     = n_switch_forced,
        n_esc_sw            = n_esc_sw,
        n_esc_cyc           = n_esc_cyc,
        n_rec_sw            = n_rec_sw,
        n_back_sw           = n_back_sw,
        n_pregoal_sw             = n_pregoal_sw,
        n_reentry_fail           = n_reentry_fail,
        classify_ms              = _classify_ms,
        setup_ms                 = _setup_ms,
        runtime_plan_peak_ms     = runtime_planner_ms_max,
        runtime_plan_avg_ms      = (runtime_planner_ms_total / runtime_planner_ev_cnt
                                    if runtime_planner_ev_cnt > 0 else 0.0),
        runtime_plan_events      = runtime_planner_ev_cnt,
    )

    # Near-goal failure diagnostic: arm ended close in task space but didn't converge.
    _near_goal_thr = 0.10
    if not result.success and grasp_err_final < _near_goal_thr:
        _ds_ng = getattr(solver, "_ds", None)
        _ak_ng  = getattr(_ds_ng, "_active_escape_family", None) or "unknown"
        _cl_ng  = result.min_clearance_m
        print(
            f"[near_goal_failure "
            f"task_err={grasp_err_final:.4f} "
            f"active_kind={getattr(_ds_ng, 'attractors', [{}])[getattr(_ds_ng, '_active_idx', 0) if _ds_ng and _ds_ng.attractors else 0].kind if _ds_ng and _ds_ng.attractors else 'N/A'} "
            f"clearance={_cl_ng:.4f} "
            f"pregoal_accepted={getattr(_ds_ng, '_pregoal_accepted_count', -1)} "
            f"pregoal_final_dist={getattr(_ds_ng, '_pregoal_final_dist_rad', -1.0):.3f} "
            f"reentry_blocked={getattr(_ds_ng, '_n_goal_reentry_blocked', 0)} "
            f"failure_class={_fc!r}]",
            flush=True,
        )

    return result, np.array(q_hist)


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def build_solver(
    method: str,
    spec,
    ik_source: str = "online",
    ik_batch_size: int = 1000,
    ik_num_solutions: int = 8,
    ik_filter_mode: str = "safe",
    force_stall_switch: bool = False,
    forced_stall_switch_mode: str = "cycle",
    enable_goal_shell_escape: bool = False,
    enable_simple_escape_waypoint: bool = False,
    enable_clearance_recovery:        bool = False,
    enable_boundary_escape_waypoints: Optional[bool] = None,  # None = auto per scenario
    boundary_escape_max_waypoints:    int  = 7,
    boundary_escape_margin_m:         float = 0.08,
    boundary_escape_min_clearance_m:  float = 0.05,
    boundary_escape_build_mode:       str  = "prebuild",
    enable_backtrack_staging:         bool = False,
    attractor_generation_mode:        str  = "online",
    enable_pre_goal_waypoint:         bool = False,
    enable_yz_expansion:              bool  = True,
    online_escape_ik_batch_size:      int   = 64,
    online_pregoal_ik_batch_size:     int   = 64,
    online_escape_candidate_policy:   str   = "score_first",
    online_ik_max_ms:                 float = 0.0,
    family_classifier_mode:           str   = "",
    defer_initial_ik:                 bool  = False,
):
    target = np.array(spec.target_pose["position"], dtype=float)
    obstacles    = spec.collision_obstacles()
    clearance_fn = _make_clearance_fn(obstacles)  # returns +inf when obstacles is empty

    if method == "diffik_ds_cbf":
        return DiffIKDSCBF(target, clearance_fn), clearance_fn

    if method == "geo_ma_ds":
        _is_yz_cross  = "yz_cross" in spec.name.lower() or spec.name == "cross_barrier"
        _has_obstacles = bool(spec.collision_obstacles())
        _default_cls  = "yz_cross_endpoint_fast" if _is_yz_cross else "goal_frame_midlink"
        _classifier   = family_classifier_mode if family_classifier_mode else _default_cls
        _lm_mode      = "combined" if _is_yz_cross else "auto"
        _label_detail = "lateral"  # only used for goal_frame_midlink mode

        # Boundary escape: auto-enable for yz_cross barrier scenarios (paper default).
        # Pass False to disable (ablation), True to force-enable, None to auto-detect.
        if enable_boundary_escape_waypoints is None:
            _use_boundary_esc = _is_yz_cross and _has_obstacles
        else:
            _use_boundary_esc = bool(enable_boundary_escape_waypoints) and _has_obstacles

        _use_simple_esc    = enable_simple_escape_waypoint and _is_yz_cross and not _use_boundary_esc
        _use_shell_esc     = (enable_goal_shell_escape and _is_yz_cross
                              and not _use_simple_esc and not _use_boundary_esc)
        _use_recovery      = enable_clearance_recovery and _is_yz_cross and _use_simple_esc
        # Map attractor_generation_mode to boundary_escape_build_mode
        _esc_build_mode = (
            "on_stall"
            if (attractor_generation_mode == "online" and _use_boundary_esc)
            else boundary_escape_build_mode
        )
        _ds_config = GeoMultiAttractorDSConfig(
            max_speed=_MAX_QDOT,   # match trial loop's velocity cap so lookahead is accurate
            enable_timing=False,
            enable_stall_escape_switch=False,  # enabled below only if has_below_family
            enable_forced_stall_switch=force_stall_switch and _is_yz_cross,
            forced_stall_switch_mode=forced_stall_switch_mode,
            enable_goal_shell_escape=_use_shell_esc,
            goal_shell_sample_mode="yz_cross",
            goal_shell_radius_m=0.20,
            goal_shell_max_waypoints=3,
            goal_shell_escape_boost=100.0,   # diagnostic: large boost to force switching
            enable_simple_escape_waypoint=_use_simple_esc,
            escape_waypoint_cross_center_z=0.50 if _is_yz_cross else 0.0,
            escape_waypoint_stall_window_steps=200 if _is_yz_cross else 25,
            enable_clearance_recovery=_use_recovery,
            enable_boundary_escape_waypoints=_use_boundary_esc,
            boundary_escape_max_waypoints=boundary_escape_max_waypoints,
            boundary_escape_margin_m=boundary_escape_margin_m,
            boundary_escape_min_clearance_m=boundary_escape_min_clearance_m,
            boundary_escape_cross_center_z=0.50 if _is_yz_cross else 0.0,
            # Fire escape sooner (50 steps ≈ 1s) so the arm hasn't yet been driven
            # into the obstacle; 200 steps was too late — arm already pinned at ~1mm.
            boundary_escape_stall_window_steps=50 if _is_yz_cross else 25,
            boundary_escape_build_mode=_esc_build_mode,
            # Fully retreat to q_start before attempting escape: this ensures the arm
            # has good clearance (far from cross) when the escape velocity field fires.
            # With backtrack_target_mode="q_start", the arm returns to its initial
            # configuration (high clearance) before routing to the escape waypoint.
            enable_backtrack_staging=(_use_boundary_esc and _is_yz_cross) or (enable_backtrack_staging and _use_boundary_esc),
            backtrack_target_mode="q_start" if (_use_boundary_esc and _is_yz_cross) else "partial_to_start",
            # Stronger nullspace gain for yz_cross: pushes arm links away from
            # the cross horizontal bar while routing to the below-cross waypoint.
            K_null=2.0 if (_use_boundary_esc and _is_yz_cross) else 0.5,
            attractor_generation_mode=attractor_generation_mode,
            enable_pre_goal_waypoint=(enable_pre_goal_waypoint and _use_boundary_esc),
            online_escape_ik_batch_size=online_escape_ik_batch_size,
            online_pregoal_ik_batch_size=online_pregoal_ik_batch_size,
            online_escape_candidate_policy=online_escape_candidate_policy,
            online_ik_max_ms=online_ik_max_ms,
        )
        ds = build_geo_multi_attractor_ds(
            spec,
            config=_ds_config,
            coverage_config=None,
            ik_source=ik_source,
            ik_batch_size=ik_batch_size,
            ik_num_solutions=ik_num_solutions,
            ik_filter_mode=ik_filter_mode,
            midlink_label_detail=_label_detail,
            family_classifier_mode=_classifier,
            yz_threshold_mode="adaptive",
            yz_cross_landmark_mode=_lm_mode,
            enable_yz_quadrant_expansion=(_is_yz_cross and enable_yz_expansion),
            defer_initial_ik=defer_initial_ik,
        )
        if _is_yz_cross:
            _yz_diag = getattr(ds, "attractor_diagnostics", {}).get("yz_cross", {})
            if _yz_diag.get("has_below_family", False):
                ds.config.enable_stall_escape_switch = True
            else:
                if _use_boundary_esc:
                    print("[boundary_escape active — stall handled by escape state machine]", end=" ", flush=True)
                elif not _use_simple_esc:
                    print("[stall escape disabled: no below family]", end=" ", flush=True)
                else:
                    print("[simple_escape active: not gated on below family]", end=" ", flush=True)
            if force_stall_switch:
                print(f"[forced_stall_switch mode={forced_stall_switch_mode}]", end=" ", flush=True)
            _gs_diag = getattr(ds, "attractor_diagnostics", {}).get("goal_shell", {})
            if _gs_diag.get("enabled"):
                _n_esc_att = _gs_diag.get("num_escape_attractors", 0)
                print(f"[goal_shell n_esc_att={_n_esc_att}]", end=" ", flush=True)
        bd = getattr(ds, "planner_breakdown", None)
        if bd:
            ik_calls   = bd.get("ik_num_calls", 1)
            mode       = bd.get("classifier_mode", "unknown")
            sample     = bd.get("sample_mode", "")
            ik_src     = bd.get("ik_source", ik_source)
            ik_ms      = bd.get("ik_generation_ms", 0.0)
            cl_ms      = bd.get("ik_to_attractors_ms", 0)
            sc_ms      = bd.get("static_scoring_ms", 0)
            sample_tag = f" sample={sample}" if sample else ""
            if ik_src == "online":
                calls_warn = " [!extra_calls]" if ik_calls > 1 else ""
                hjcd_v  = bd.get("hjcd_ms", 0.0)
                dedup_v = bd.get("dedup_ms", 0.0)
                pose_v  = bd.get("pose_filter_ms", 0.0)
                jl_v    = bd.get("joint_limit_filter_ms", 0.0)
                cl_v    = bd.get("clearance_filter_ms", 0.0)
                n_valid = bd.get("ik_num_after_filter", 0)
                ik_sub  = (f"[hjcd={hjcd_v:.0f} dedup={dedup_v:.0f} "
                           f"pose={pose_v:.0f} jl={jl_v:.0f} cl={cl_v:.0f} n={n_valid}]")
                ik_tag  = f" ik={ik_ms:.0f}ms{ik_sub} calls={ik_calls}{calls_warn}"
            else:
                ik_tag = " ik=precomputed"
            ld_tag = f" labels={bd.get('label_detail', 'lateral')}"
            print(f"[classifier={mode}{sample_tag}{ik_tag} classify={cl_ms:.0f}ms score={sc_ms:.0f}ms{ld_tag}]", end=" ", flush=True)
        ad = getattr(ds, "attractor_diagnostics", None)
        if ad:
            fc     = ad.get("family_counts", {})
            fc_str = " ".join(f"{fam.split(':',1)[-1]}×{cnt}" for fam, cnt in sorted(fc.items()))
            print(f"[attractors={ad['num_attractors']} families={fc_str}]", end=" ", flush=True)
            yz = ad.get("yz_cross")
            if yz:
                lm   = yz.get("selected_landmark", "?")
                lmm  = yz.get("landmark_mode", "auto")
                tau  = f"tau_y={yz.get('tau_y',0):.3f} tau_z={yz.get('tau_z',0):.3f}"
                yfc  = " ".join(f"{k}×{v}" for k, v in sorted(yz.get("family_counts", {}).items()))
                has_b = "yes" if yz.get("has_below_family") else "NO"
                print(f"[yz_cross: mode={lmm} lm={lm} {tau} below={has_b} | {yfc}]", end=" ", flush=True)
        return GeoMADSSolver("geo_ma_ds", ds), clearance_fn

    raise ValueError(f"Unknown method {method!r}")


# ---------------------------------------------------------------------------
# MuJoCo rendering
# ---------------------------------------------------------------------------

_CAM_DEFAULT = dict(
    height=480, width=640,
    cam_lookat=(0.3, 0.0, 0.4),
    cam_distance=2.2,
    cam_azimuth=140.0,
    cam_elevation=-25.0,
)


def _build_render_env(spec):
    """Build a MuJoCoRenderEnv from a ScenarioSpec (visual obstacles + red goal sphere)."""
    from src.simulation.mujoco_env import MuJoCoRenderEnv, RenderConfig
    from src.simulation.panda_scene import load_panda_scene

    obs_dict  = spec.obstacles_as_panda_scene_dict()
    ee_target = np.array(spec.target_pose["position"], dtype=float) if spec.target_pose else None
    model     = load_panda_scene(obstacles=obs_dict, ee_target=ee_target)
    cam       = RenderConfig(**_CAM_DEFAULT)
    return MuJoCoRenderEnv(model, cam)


def _render_frames(env, q_history: np.ndarray, every_n: int = 2) -> List[np.ndarray]:
    """Render every_n-th frame of q_history. Returns [] if GL unavailable."""
    frames: List[np.ndarray] = []
    indices = list(range(0, len(q_history), every_n))
    if len(q_history) - 1 not in indices:
        indices.append(len(q_history) - 1)
    for idx in indices:
        frames.append(env.render_at(q_history[idx]))
    if frames and float(frames[0].mean()) < 1.0:
        print("    [warn] GL returned black frames — skipping video output")
        return []
    return frames


def _save_video(frames: List[np.ndarray], out_path: Path, fps: int) -> Optional[str]:
    """Save frames as MP4 (imageio) or GIF (PIL). Returns the path written."""
    if not frames:
        return None
    # Try MP4
    try:
        import imageio
        p = out_path.with_suffix(".mp4")
        writer = imageio.get_writer(str(p), fps=fps, codec="libx264",
                                    pixelformat="yuv420p", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"    → {p}")
        return str(p)
    except Exception:
        pass
    # Fall back to GIF
    try:
        from PIL import Image
        p = out_path.with_suffix(".gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(p), save_all=True, append_images=imgs[1:],
                     duration=max(1, int(1000 / fps)), loop=0)
        print(f"    → {p}")
        return str(p)
    except Exception as e:
        print(f"    [warn] could not save video: {e}")
        return None


def _save_snapshots(env, q_history: np.ndarray, out_dir: Path, labels=("start", "mid", "final")):
    """Save start / mid / final PNG snapshots."""
    try:
        from PIL import Image
    except ImportError:
        return
    n = len(q_history)
    idxs = {"start": 0, "mid": n // 2, "final": n - 1}
    for label in labels:
        idx   = idxs.get(label, 0)
        frame = env.render_at(q_history[idx])
        if frame is not None and float(frame.mean()) > 1.0:
            p = out_dir / f"snap_{label}.png"
            Image.fromarray(frame).save(str(p))


def render_trial(
    spec,
    method_name: str,
    q_history: np.ndarray,
    out_dir: Path,
    every_n: int = 2,
    fps: int = 30,
) -> Optional[str]:
    """
    Render one trial to GIF/MP4 + start/mid/final PNG snapshots.

    Returns the path of the video file, or None if rendering unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = _build_render_env(spec)
    except Exception as exc:
        print(f"    [warn] could not build render env: {exc}")
        return None

    frames = _render_frames(env, q_history, every_n=every_n)
    _save_snapshots(env, q_history, out_dir)
    stem   = out_dir / method_name
    return _save_video(frames, stem, fps=fps)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

_COL_W = {
    "method":              20,
    "scenario":            28,
    "success":              7,
    "conv_step":            9,
    "final_grasp_err_m":   13,
    "min_clearance_m":     12,
    "collision_count":     10,
    "mean_step_ms":        11,
    "planner_ms":          11,
    "planner_hz":          11,
    "ik_generation_ms":    13,
    "control_hz":           9,
    "end_to_end_hz":       10,
    "cbf_active_fraction": 10,
    "max_alpha":            9,
    "n_switches":          10,
    "n_switch_route":       9,
    "n_switch_near_goal":  11,
    "n_switch_forced":     11,
    "n_esc_sw":             9,
    "n_esc_cyc":            9,
    "n_rec_sw":             9,
    "n_back_sw":            9,
    "joint_path_length":   13,
    "task_path_length_m":  14,
    "stalled":              8,
    "ik_rewarm_ms":        12,
    "classify_ms":         11,
    "setup_ms":            10,
    "runtime_plan_peak_ms": 14,
    "runtime_plan_avg_ms":  13,
    "runtime_plan_events":  13,
}

_HEADERS = {
    "method":              "Method",
    "scenario":            "Scenario",
    "success":             "OK",
    "conv_step":           "conv_step",
    "final_grasp_err_m":   "grasp_err_m",
    "min_clearance_m":     "min_cl_m",
    "collision_count":     "collisions",
    "mean_step_ms":        "step_ms",
    "planner_ms":          "planner_ms",
    "planner_hz":          "planner_Hz",
    "ik_generation_ms":    "ik_gen_ms",
    "control_hz":          "ctrl_Hz",
    "end_to_end_hz":       "e2e_Hz",
    "cbf_active_fraction": "cbf_frac",
    "max_alpha":           "max_alpha",
    "n_switches":          "n_switch",
    "n_switch_route":      "n_sw_rt",
    "n_switch_near_goal":  "n_sw_ng",
    "n_switch_forced":     "n_sw_forced",
    "n_esc_sw":            "n_esc_sw",
    "n_esc_cyc":           "n_esc_cyc",
    "n_rec_sw":            "n_rec_sw",
    "n_back_sw":           "n_back_sw",
    "joint_path_length":   "joint_path",
    "task_path_length_m":  "task_path_m",
    "stalled":             "stalled",
    "ik_rewarm_ms":        "ik_rewarm_ms",
    "classify_ms":         "classify_ms",
    "setup_ms":            "setup_ms",
    "runtime_plan_peak_ms": "replan_peak_ms",
    "runtime_plan_avg_ms":  "replan_avg_ms",
    "runtime_plan_events":  "replan_events",
}


def _fmt(val, key: str) -> str:
    if isinstance(val, bool):
        return "yes" if val else "no"
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if key in ("hz", "control_hz", "end_to_end_hz", "planner_hz"):
            return f"{val:.0f}"
        if key in ("planner_ms", "ik_generation_ms", "ik_rewarm_ms",
                   "classify_ms", "setup_ms",
                   "runtime_plan_peak_ms", "runtime_plan_avg_ms"):
            return f"{val:.1f}"
        if key == "final_grasp_err_m":
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def print_table(results: List[TrialResult]) -> None:
    keys = list(_COL_W.keys())
    header = "  ".join(f"{_HEADERS[k]:<{_COL_W[k]}}" for k in keys)
    sep    = "  ".join("-" * _COL_W[k] for k in keys)
    print()
    print(header)
    print(sep)
    for r in results:
        d = asdict(r)
        row = "  ".join(f"{_fmt(d[k], k):<{_COL_W[k]}}" for k in keys)
        print(row)
    print()


# ---------------------------------------------------------------------------
# Interpretation note (local-minima context)
# ---------------------------------------------------------------------------

def _print_interpretation(
    results: List[TrialResult],
    ik_warmup_ms: float = 0.0,
    warmup_enabled: bool = False,
    rewarm_enabled: bool = True,
    benchmark_ordering: str = "build_all_first",
) -> None:
    print("=" * 80)
    print("Interpretation notes")
    print("=" * 80)
    cbf_fails = [r for r in results if r.method == "diffik_ds_cbf" and not r.success]
    geo_ok    = [r for r in results if r.method.startswith("geo_ma") and r.success]
    geo_fail  = [r for r in results if r.method.startswith("geo_ma") and not r.success]

    if cbf_fails:
        scenarios_failed = sorted({r.scenario for r in cbf_fails})
        print(f"  DS+CBF baseline failed to converge on: {scenarios_failed}")
        print("  → The baseline maintains clearance but is trapped by homotopy barriers.")

    if geo_ok:
        barrier_ok = [r for r in geo_ok if r.scenario != "open_reach"]
        if barrier_ok:
            print(f"  GeoMA-DS succeeded on canonical barrier scenarios "
                  f"({len(barrier_ok)} / {len(barrier_ok)+len(geo_fail)} barrier trials).")
            print("  → GeoMA-DS uses IK-derived attractors, geometric modulation, and temporary "
                  "obstacle-boundary escape attractors to find alternative routes.")
        else:
            print(f"  GeoMA-DS succeeded on {len(geo_ok)} / {len(geo_ok)+len(geo_fail)} trials.")

    geo_online = [r for r in results if r.ik_source == "online"]
    if geo_online:
        print()
        if warmup_enabled and ik_warmup_ms > 0:
            print(f"  IK timing: HJCD-IK CUDA cold warmup: {ik_warmup_ms:.0f} ms (one-time, excluded from planner_ms).")
        else:
            print("  IK timing: no CUDA warmup — first GeoMA-DS scenario may include cold-start CUDA init in planner_ms.")
        rewarm_rows = [r for r in geo_online if r.ik_rewarm_ms > 0]
        if rewarm_rows:
            mean_rw = float(np.mean([r.ik_rewarm_ms for r in rewarm_rows]))
            print(f"  Per-build rewarm: {mean_rw:.0f} ms avg (excluded from planner_ms; restores GPU warm state).")
            print("  planner_ms = warm online IK + filtering + classification + scoring.")
        elif rewarm_enabled:
            print("  Per-build rewarm: enabled but no rewarm was needed (no geo_ma online builds).")
        else:
            print("  Per-build rewarm disabled (--no-rewarm-ik-before-build); "
                  "GeoMA-DS planner_ms may include GPU idle-state recovery latency.")
        print(f"  Benchmark ordering: {benchmark_ordering}.")
    print()


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------

def save_csv(results: List[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [f.name for f in fields(TrialResult)]
    lines = [",".join(keys)]
    for r in results:
        d = asdict(r)
        lines.append(",".join(str(d[k]) for k in keys))
    path.write_text("\n".join(lines))
    print(f"[baselines] CSV → {path}")


def save_json(results: List[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"[baselines] JSON → {path}")


def _save_summary_csv(groups: List[dict], path: "Path") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = ["scenario", "method", "N", "num_success", "success_rate",
                 "num_collision", "num_stalled", "num_timeout"]
    stat_cols = []
    for key in _SUMMARY_CONTINUOUS:
        for stat in ("mean", "std", "median", "p95", "min", "max"):
            stat_cols.append(f"{key}_{stat}")
    header = base_cols + stat_cols
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for g in groups:
            row = [g["scenario"], g["method"], g["N"], g["num_success"],
                   g["success_rate"], g["num_collision"], g["num_stalled"], g["num_timeout"]]
            for key in _SUMMARY_CONTINUOUS:
                m = g["metrics"].get(key, {})
                for stat in ("mean", "std", "median", "p95", "min", "max"):
                    row.append(m.get(stat))
            writer.writerow(row)
    print(f"[baselines] summary CSV → {path}")


def _save_summary_json(
    groups: List[dict],
    path: "Path",
    trials: int,
    seed: int,
    scenarios: List[str],
    methods: List[str],
    planner_profile: str,
    cuda_warmup_ms: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = None
    payload = {
        "metadata": {
            "trials": trials,
            "seed": seed,
            "scenarios": scenarios,
            "methods": methods,
            "planner_profile": planner_profile,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": git_commit,
            "cuda_warmup_ms": cuda_warmup_ms,
        },
        "groups": groups,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[baselines] summary JSON → {path}")



# ---------------------------------------------------------------------------
# Main loop helpers
# ---------------------------------------------------------------------------

_IK_ENSURE_WARM_TARGET_MS  = 1500.0   # wall-time budget for per-trial GPU warm-up loop
_IK_ENSURE_WARM_SETTLED_MS = 8.0     # a single rewarm call below this → GPU is at P0


def _ensure_gpu_warm(
    batch_size: int = _IK_REWARM_BATCH_SIZE,
    target_wall_ms: float = _IK_ENSURE_WARM_TARGET_MS,
    settled_ms: float = _IK_ENSURE_WARM_SETTLED_MS,
) -> float:
    """
    Restore GPU to P0 state before a timed HJCD-IK call.

    Runs warmup_hjcdik in a tight loop until either:
      (a) a single call takes < settled_ms (GPU clock reached P0), or
      (b) target_wall_ms of wall time has elapsed (gives up — GPU stays P0 on warm runs).

    Background: NVIDIA GPUs drop from P0 to P6-P8 after ~3-5s of idle.  A single
    short rewarm call only partially restores the clock (P8→P6).  Sustained GPU
    activity for ~750-1000ms is required to climb from P6 back to P0.  A tight
    loop provides this without Python call gaps large enough for the driver to stop
    boosting.

    Returns total elapsed wall time in milliseconds.
    """
    t0 = time.perf_counter()
    while True:
        t_call = time.perf_counter()
        warmup_hjcdik(batch_size=batch_size, num_solutions=_IK_REWARM_NUM_SOLUTIONS)
        call_ms = (time.perf_counter() - t_call) * 1000.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if call_ms < settled_ms or elapsed_ms >= target_wall_ms:
            return elapsed_ms


def _do_rewarm_if_needed(
    method: str,
    ik_source: str,
    rewarm_enabled: bool,
    inline_prefix: Optional[str],
    batch_size: int = _IK_REWARM_BATCH_SIZE,
) -> float:
    """
    Per-build safety rewarm: runs one warmup call.  On warm GPU this takes <10ms.
    On cold GPU (after _ensure_gpu_warm not yet called) it takes ~150ms.

    This is kept as a lightweight safety net for the first build after the per-trial
    _ensure_gpu_warm call.  The per-trial call does the heavy lifting.

    Returns rewarm_ms (0.0 if skipped).
    """
    if not (method.startswith("geo_ma") and ik_source == "online" and rewarm_enabled):
        return 0.0
    t_rw = time.perf_counter()
    warmup_hjcdik(batch_size=batch_size, num_solutions=_IK_REWARM_NUM_SOLUTIONS)
    rewarm_ms = (time.perf_counter() - t_rw) * 1000.0
    if inline_prefix is not None:
        print(f"{inline_prefix}rewarm={rewarm_ms:.0f}ms building solver ...", end=" ", flush=True)
    return rewarm_ms


def _print_trial_status(result: TrialResult) -> None:
    status = "OK" if result.success else ("STALL" if result.stalled else "DNF")
    plan_hz_str = f"/{result.planner_hz:.0f}Hz" if result.planner_hz is not None else "/N/A"
    print(
        f"{status}  conv={result.conv_step}  "
        f"grasp={result.final_grasp_err_m:.2e}m  "
        f"cl={result.min_clearance_m:.4f}m  "
        f"planner={result.planner_ms:.0f}ms{plan_hz_str}  "
        f"ctrl={result.control_hz:.0f}Hz  "
        f"e2e={result.end_to_end_hz:.0f}Hz"
    )


def _print_timing_targets(result: TrialResult, require_fast: bool = False) -> None:
    if result.runtime_plan_events == 0:
        return
    _planner_hz   = result.planner_hz
    _peak_ms      = result.runtime_plan_peak_ms
    _avg_ms       = result.runtime_plan_avg_ms
    planner_ok    = _planner_hz is not None and _planner_hz > 100
    replan_pk_ok  = _peak_ms <= 10.0
    replan_av_ok  = _avg_ms  <= 10.0
    _pk_hz  = 1000.0 / _peak_ms if _peak_ms > 0 else float("inf")
    _av_hz  = 1000.0 / _avg_ms  if _avg_ms  > 0 else float("inf")
    _hz_str = f"{_planner_hz:.0f}" if _planner_hz is not None else "N/A"
    print(f"[timing_target] planner_Hz={_hz_str} ok={planner_ok}")
    print(f"[timing_target] replan_peak_ms={_peak_ms:.1f} replan_peak_Hz={_pk_hz:.0f} ok={replan_pk_ok}")
    print(f"[timing_target] replan_avg_ms={_avg_ms:.1f} replan_avg_Hz={_av_hz:.0f} ok={replan_av_ok}")
    if require_fast:
        assert planner_ok,   f"planner_Hz={_hz_str} not > 100"
        assert replan_pk_ok, f"replan_peak_ms={_peak_ms:.1f} not <= 10.0"
        assert replan_av_ok, f"replan_avg_ms={_avg_ms:.1f} not <= 10.0"


def _do_render(args, spec, method: str, scen_name: str, q_hist: np.ndarray) -> None:
    vid_dir = args.out_dir / scen_name / method
    print(f"    rendering {len(q_hist)} frames → {vid_dir} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    render_trial(spec, method, q_hist, vid_dir, every_n=args.every_n, fps=args.fps)
    print(f"({time.perf_counter()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="DiffIK-CBF vs GeoMA-DS baselines")
    parser.add_argument("--steps",     type=int,  default=_N_STEPS,
                        help=f"Max steps per trial (default {_N_STEPS})")
    parser.add_argument("--scenarios", nargs="+", default=CANONICAL_SCENARIOS,
                        choices=list(_SCENARIO_BUILDERS.keys()),
                        help="Scenarios to evaluate")
    parser.add_argument("--methods",   nargs="+", default=None,
                        choices=CANONICAL_METHODS,
                        help="Explicit method list (default: all canonical methods)")
    parser.add_argument("--csv",  type=Path, default=None,
                        help="Save results to CSV")
    parser.add_argument("--json", type=Path, default=None,
                        help="Save results to JSON")
    parser.add_argument("--render", action="store_true",
                        help="Render each trial to GIF/MP4 via MuJoCo off-screen renderer")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baselines/videos"),
                        help="Root directory for rendered videos (default: outputs/baselines/videos)")
    parser.add_argument("--fps",      type=int, default=30,
                        help="Frames per second for rendered video (default: 30)")
    parser.add_argument("--every-n",  type=int, default=2,
                        help="Render every N-th simulation step (default: 2)")
    parser.add_argument("--ik-source", choices=["online", "precomputed"], default=None,
                        help="IK source override (overrides --planner-profile): 'online' or 'precomputed'")
    parser.add_argument("--no-warmup-ik", action="store_true",
                        help="Disable HJCD-IK CUDA warmup (first scenario includes cold-start overhead)")
    parser.add_argument("--planner-profile", choices=list(_PLANNER_PROFILES.keys()), default="fast",
                        help="IK planner preset: fast (128/8/minimal, no YZ expansion), "
                             "reactive (256/4/minimal), robust (1000/8/safe), "
                             "precomputed (uses spec.ik_goals). Default: fast")
    parser.add_argument("--ik-batch-size", type=int, default=None,
                        help="HJCD-IK batch_size override (overrides --planner-profile default)")
    parser.add_argument("--ik-num-solutions", type=int, default=None,
                        help="HJCD-IK num_solutions override (overrides --planner-profile default)")
    parser.add_argument("--no-rewarm-ik-before-build", action="store_true",
                        help="Disable per-build HJCD rewarm before each GeoMA-DS solver build "
                             "(rewarm is on by default to prevent GPU idle-state artifacts)")
    parser.add_argument("--build-order",
                        choices=["geo_first", "build_all_first", "interleaved"],
                        default="geo_first",
                        help="Build ordering strategy (default: geo_first). "
                             "geo_first: all GeoMA-DS solvers built consecutively before CPU baselines — "
                             "gives the most reliable warm-GPU planner timing. "
                             "build_all_first: per-scenario build-all-then-run. "
                             "interleaved: legacy build-then-run per method per scenario.")
    parser.add_argument("--interleaved-build-run", action="store_true",
                        help="[deprecated] Alias for --build-order interleaved.")
    parser.add_argument("--force-stall-switch", action="store_true",
                        help="Enable forced attractor switching on stall (cross_barrier diagnostic)")
    parser.add_argument("--forced-stall-switch-mode",
                        choices=["cycle", "best_nonactive", "random"], default="cycle",
                        help="Forced stall switch mode: cycle (default), best_nonactive, random")
    parser.add_argument("--enable-goal-shell-escape", action="store_true", default=False,
                        help="Enable goal-shell escape attractors for cross_barrier")
    parser.add_argument("--no-goal-shell-escape", action="store_true", default=False,
                        help="Disable goal-shell escape attractors (overrides --enable-goal-shell-escape)")
    parser.add_argument("--enable-simple-escape-waypoint", action="store_true", default=False,
                        help="Enable simple scene-aware escape waypoint for cross_barrier")
    parser.add_argument("--no-simple-escape-waypoint", action="store_true", default=False,
                        help="Disable simple escape waypoint (overrides --enable-simple-escape-waypoint)")
    parser.add_argument("--enable-clearance-recovery", action="store_true", default=False,
                        help="Enable clearance recovery stage before portal escape (cross_barrier)")
    parser.add_argument("--no-clearance-recovery", action="store_true", default=False,
                        help="Disable clearance recovery (overrides --enable-clearance-recovery)")
    parser.add_argument("--enable-boundary-escape-waypoints", action="store_true", default=False,
                        help="[ablation] Force-enable boundary escape waypoints even for non-barrier scenarios")
    parser.add_argument("--no-boundary-escape-waypoints", action="store_true", default=False,
                        help="[ablation] Disable boundary escape waypoints (overrides auto-enable for yz_cross)")
    parser.add_argument("--boundary-escape-max-waypoints", type=int, default=7,
                        help="Maximum number of boundary escape waypoints (default: 7)")
    parser.add_argument("--boundary-escape-margin", type=float, default=0.08,
                        help="Clearance margin (m) added outside cross extents for waypoints (default: 0.08)")
    parser.add_argument("--boundary-escape-min-clearance", type=float, default=0.05,
                        help="Minimum waypoint clearance (m) for boundary escape candidates (default: 0.05)")
    parser.add_argument("--boundary-escape-build-mode", type=str, default="prebuild",
                        choices=["prebuild", "on_stall"],
                        help="When to run boundary escape IK: 'prebuild' (default) or 'on_stall'")
    parser.add_argument("--enable-backtrack-staging", action="store_true", default=False,
                        help="Enable backtrack/staging attractor between clearance recovery and escape portal")
    parser.add_argument("--no-backtrack-staging", action="store_true", default=False,
                        help="Disable backtrack staging (overrides --enable-backtrack-staging)")
    parser.add_argument("--attractor-generation-mode",
                        choices=["online", "prebuild_debug", "precomputed"], default="online",
                        help="'online': generate escape/backtrack/pre_goal IK at runtime (fair timing). "
                             "'precomputed': use spec.ik_goals for initial attractors (sets ik_source=precomputed, "
                             "excludes initial IK from planner_ms). "
                             "'prebuild_debug': pre-build all attractors (unfair timing, for debugging only).")
    parser.add_argument("--no-pre-goal-waypoint", action="store_false", dest="enable_pre_goal_waypoint",
                        help="Disable online pre-goal IK waypoint (default: enabled for boundary escape scenarios)")
    parser.set_defaults(enable_pre_goal_waypoint=True)
    parser.add_argument("--online-escape-ik-batch-size", type=int, default=None,
                        help="Override HJCD batch_size for online escape IK events (default: from profile, 64)")
    parser.add_argument("--online-pregoal-ik-batch-size", type=int, default=None,
                        help="Override HJCD batch_size for online pre-goal IK events (default: from profile, 32)")
    parser.add_argument("--require-fast-planner-timing", action="store_true", default=False,
                        help="Assert planner_Hz>100, replan_peak_ms<10, replan_avg_ms<10 after each run")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per (scenario, method) pair (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; trial k uses seed+k (default: 0)")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="Write grouped summary statistics to CSV")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Write grouped summary statistics + metadata to JSON")
    args = parser.parse_args(argv)

    methods = args.methods if args.methods is not None else list(CANONICAL_METHODS)

    # Resolve IK parameters: profile provides defaults, individual args override
    _profile              = _PLANNER_PROFILES[args.planner_profile]
    ik_source             = args.ik_source        if args.ik_source        is not None else _profile["ik_source"]
    ik_batch_size         = args.ik_batch_size    if args.ik_batch_size    is not None else _profile["ik_batch_size"]
    ik_num_solutions      = args.ik_num_solutions if args.ik_num_solutions is not None else _profile["ik_num_solutions"]
    ik_filter_mode        = _profile["ik_filter_mode"]
    _profile_yz_expansion        = _profile.get("enable_yz_expansion",             True)
    _online_esc_batch            = (args.online_escape_ik_batch_size
                                    if args.online_escape_ik_batch_size is not None
                                    else _profile.get("online_escape_batch", 64))
    _online_pg_batch             = (args.online_pregoal_ik_batch_size
                                    if args.online_pregoal_ik_batch_size is not None
                                    else _profile.get("online_pregoal_batch", 64))
    _online_esc_candidate_policy = _profile.get("online_escape_candidate_policy",   "score_first")
    _online_ik_max_ms            = _profile.get("online_ik_max_ms",                 0.0)
    _family_classifier_mode      = _profile.get("family_classifier_mode",            "")
    _defer_initial_ik            = _profile.get("defer_initial_ik", False)

    # "precomputed" attractor_generation_mode: skip initial HJCD-IK, use spec.ik_goals
    if args.attractor_generation_mode == "precomputed" and args.ik_source is None:
        ik_source = "precomputed"
    # The actual attractor_generation_mode passed to build_solver controls escape IK mode;
    # "precomputed" is a CLI alias for ik_source=precomputed, not a new escape mode.
    _eff_attractor_mode = "online" if args.attractor_generation_mode == "precomputed" else args.attractor_generation_mode

    if ik_source == "precomputed":
        print("[warn] precomputed IK: planner timing excludes online IK generation "
              "and should not be used as the final fair benchmark.")
    if args.planner_profile != "fast" or ik_batch_size != 128 or ik_num_solutions != 8:
        print(f"[profile] {args.planner_profile}  "
              f"batch={ik_batch_size}  solutions={ik_num_solutions}  filter={ik_filter_mode}")

    rewarm_ik_before_build = not args.no_rewarm_ik_before_build
    build_order = "interleaved" if args.interleaved_build_run else args.build_order
    benchmark_ordering = build_order

    if args.no_rewarm_ik_before_build:
        print("[warn] --no-rewarm-ik-before-build: GeoMA-DS planner timing may include GPU idle-state "
              "recovery latency from preceding CPU-only baselines.")
    if build_order == "interleaved":
        print("[info] build-order=interleaved: legacy build-then-run per method per scenario. "
              "Use only for cold/idle-state experiments.")
    elif build_order == "build_all_first":
        print("[info] build-order=build_all_first: per-scenario build-all-then-run. "
              "Consider --build-order geo_first for most reliable warm-GPU planner timing.")

    # HJCD-IK CUDA warmup (absorbs one-time CUDA context init before timed scenarios)
    ik_warmup_ms = 0.0
    warmup_enabled = (
        not args.no_warmup_ik
        and any(m.startswith("geo_ma") for m in methods)
        and ik_source == "online"
    )
    if warmup_enabled:
        print("[warmup] HJCD-IK CUDA warmup running ...", end=" ", flush=True)
        ik_warmup_ms = warmup_hjcdik()
        print(f"{ik_warmup_ms:.0f}ms")
    elif any(m.startswith("geo_ma") for m in methods) and ik_source == "online" and args.no_warmup_ik:
        print("[warn] --no-warmup-ik: first GeoMA-DS scenario includes cold-start CUDA overhead in planner_ms.")

    results: List[TrialResult] = []

    _goal_shell_escape      = args.enable_goal_shell_escape and not args.no_goal_shell_escape
    _simple_escape_waypoint = (args.enable_simple_escape_waypoint
                               and not args.no_simple_escape_waypoint)
    _clearance_recovery     = (
        (args.enable_clearance_recovery or _simple_escape_waypoint)
        and not args.no_clearance_recovery
    )
    # None = auto-detect per scenario (yz_cross enables automatically).
    # False = ablation: disable even where auto would enable.
    # True  = ablation: force-enable even for non-barrier scenarios.
    if args.no_boundary_escape_waypoints:
        _boundary_escape: Optional[bool] = False
    elif args.enable_boundary_escape_waypoints:
        _boundary_escape = True
    else:
        _boundary_escape = None  # auto
    _build_kw = dict(
        ik_source=ik_source, ik_batch_size=ik_batch_size,
        ik_num_solutions=ik_num_solutions, ik_filter_mode=ik_filter_mode,
        force_stall_switch=args.force_stall_switch,
        forced_stall_switch_mode=args.forced_stall_switch_mode,
        enable_goal_shell_escape=_goal_shell_escape,
        enable_simple_escape_waypoint=_simple_escape_waypoint,
        enable_clearance_recovery=_clearance_recovery,
        enable_boundary_escape_waypoints=_boundary_escape,
        boundary_escape_max_waypoints=args.boundary_escape_max_waypoints,
        boundary_escape_margin_m=args.boundary_escape_margin,
        boundary_escape_min_clearance_m=args.boundary_escape_min_clearance,
        boundary_escape_build_mode=args.boundary_escape_build_mode,
        enable_backtrack_staging=(args.enable_backtrack_staging
                                  and not args.no_backtrack_staging),
        attractor_generation_mode=_eff_attractor_mode,
        enable_pre_goal_waypoint=args.enable_pre_goal_waypoint,
        enable_yz_expansion=_profile_yz_expansion,
        online_escape_ik_batch_size=_online_esc_batch,
        online_pregoal_ik_batch_size=_online_pg_batch,
        online_escape_candidate_policy=_online_esc_candidate_policy,
        online_ik_max_ms=_online_ik_max_ms,
        family_classifier_mode=_family_classifier_mode,
        defer_initial_ik=_defer_initial_ik,
    )

    for trial_idx in range(args.trials):
        if args.trials > 1:
            _set_seed(args.seed + trial_idx)
            print()
            print("=" * 80)
            print(f"Trial {trial_idx + 1}/{args.trials}  (seed={args.seed + trial_idx})")
            print("=" * 80)
        _trial_results: List[TrialResult] = []

        # Per-trial GPU warm-up: run a tight rewarm loop until GPU clock reaches P0.
        # After a 30-90s CPU-only simulation the GPU drops to P6-P8; a single short
        # rewarm call only partially restores the clock.  The loop runs for up to
        # _IK_ENSURE_WARM_TARGET_MS ms, stopping early once a call settles below
        # _IK_ENSURE_WARM_SETTLED_MS (GPU is at P0).  On an already-warm GPU this
        # completes in <20ms; on a cold GPU it takes ~800-1000ms.
        if (rewarm_ik_before_build
                and any(m.startswith("geo_ma") for m in methods)
                and ik_source == "online"
                and trial_idx > 0):
            t_ew = time.perf_counter()
            _ensure_gpu_warm(batch_size=ik_batch_size)
            _ew_ms = (time.perf_counter() - t_ew) * 1000.0
            print(f"  [gpu warm-up] {_ew_ms:.0f}ms (GPU restored to P0 for trial {trial_idx+1})")

        if build_order == "geo_first":
            geo_methods   = [m for m in methods if m.startswith("geo_ma")]
            other_methods = [m for m in methods if not m.startswith("geo_ma")]
            specs     = {name: _SCENARIO_BUILDERS[name]() for name in args.scenarios}
            all_built = {name: {} for name in args.scenarios}

            if geo_methods:
                print(f"\nbuild phase: geo_ma solvers ({len(args.scenarios)} scenarios) ...")
                for method in geo_methods:
                    for scen_name in args.scenarios:
                        spec = specs[scen_name]
                        rewarm_ms = _do_rewarm_if_needed(method, ik_source, rewarm_ik_before_build, None,
                                                         batch_size=ik_batch_size)
                        rewarm_tag = f"rewarm={rewarm_ms:.0f}ms " if rewarm_ms > 0 else ""
                        print(f"  [{scen_name}/{method}] {rewarm_tag}building ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            solver, clearance_fn = build_solver(method, spec, **_build_kw)
                        except Exception as exc:
                            print(f"FAILED: {exc}")
                            all_built[scen_name][method] = None
                            continue
                        build_s = time.perf_counter() - t0
                        ps, iks, ikgen, clsms = _extract_build_meta(method, solver, ik_source)
                        plan_hz = (1.0 / build_s if build_s > 1e-9 else None) if ps != "disabled" else None
                        plan_hz_str = f"/{plan_hz:.0f}Hz" if plan_hz is not None else "/N/A"
                        _cls_str = f"  [ik={ikgen:.0f}ms classify={clsms:.0f}ms]" if clsms > 0 else ""
                        print(f"planner={build_s*1e3:.0f}ms{plan_hz_str}{_cls_str}")
                        all_built[scen_name][method] = (solver, clearance_fn, build_s, ps, iks, ikgen, rewarm_ms)

            if other_methods:
                print(f"\nbuild phase: other solvers ({len(args.scenarios)} scenarios) ...")
                for method in other_methods:
                    for scen_name in args.scenarios:
                        spec = specs[scen_name]
                        print(f"  [{scen_name}/{method}] building ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            solver, clearance_fn = build_solver(method, spec, **_build_kw)
                        except Exception as exc:
                            print(f"FAILED: {exc}")
                            all_built[scen_name][method] = None
                            continue
                        build_s = time.perf_counter() - t0
                        ps, iks, ikgen, clsms = _extract_build_meta(method, solver, ik_source)
                        plan_hz = (1.0 / build_s if build_s > 1e-9 else None) if ps != "disabled" else None
                        plan_hz_str = f"/{plan_hz:.0f}Hz" if plan_hz is not None else "/N/A"
                        _cls_str = f"  [ik={ikgen:.0f}ms classify={clsms:.0f}ms]" if clsms > 0 else ""
                        print(f"planner={build_s*1e3:.0f}ms{plan_hz_str}{_cls_str}")
                        all_built[scen_name][method] = (solver, clearance_fn, build_s, ps, iks, ikgen, 0.0)

            print(f"\nrun phase ...")
            for scen_name in args.scenarios:
                spec = specs[scen_name]
                print(f"\n{'='*60}")
                print(f"Scenario: {scen_name}  (q_start=[{', '.join(f'{v:.3f}' for v in spec.q_start)}])")
                print(f"{'='*60}")
                for method in methods:
                    entry = all_built[scen_name].get(method)
                    if entry is None:
                        continue
                    solver, clearance_fn, build_s, ps, iks, ikgen, rewarm_ms = entry
                    print(f"  [{method}] running {args.steps} steps ...", end=" ", flush=True)
                    result, q_hist = run_trial(
                        solver, spec, clearance_fn,
                        n_steps=args.steps, planner_s=build_s,
                        planner_source=ps, ik_source=iks, ik_generation_ms=ikgen,
                        ik_rewarm_ms=rewarm_ms, benchmark_ordering=benchmark_ordering,
                    )
                    _print_trial_status(result)
                    _print_timing_targets(result, require_fast=args.require_fast_planner_timing)
                    result.trial_idx = trial_idx
                    result.seed = args.seed + trial_idx
                    results.append(result)
                    _trial_results.append(result)
                    if args.render:
                        _do_render(args, spec, method, scen_name, q_hist)

        else:
            for scen_name in args.scenarios:
                spec = _SCENARIO_BUILDERS[scen_name]()
                print(f"\n{'='*60}")
                print(f"Scenario: {scen_name}  (q_start=[{', '.join(f'{v:.3f}' for v in spec.q_start)}])")
                print(f"{'='*60}")

                if build_order == "interleaved":
                    for method in methods:
                        rewarm_ms = _do_rewarm_if_needed(
                            method, ik_source, rewarm_ik_before_build, inline_prefix=f"  [{method}] ",
                            batch_size=ik_batch_size,
                        )
                        if rewarm_ms == 0.0:
                            print(f"  [{method}] building solver ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            solver, clearance_fn = build_solver(method, spec, **_build_kw)
                        except Exception as exc:
                            print(f"FAILED to build: {exc}")
                            continue
                        build_s = time.perf_counter() - t0
                        ps, iks, ikgen, _clsms = _extract_build_meta(method, solver, ik_source)
                        print(f"({build_s*1e3:.0f} ms)  running {args.steps} steps ...", end=" ", flush=True)
                        result, q_hist = run_trial(
                            solver, spec, clearance_fn,
                            n_steps=args.steps, planner_s=build_s,
                            planner_source=ps, ik_source=iks, ik_generation_ms=ikgen,
                            ik_rewarm_ms=rewarm_ms, benchmark_ordering=benchmark_ordering,
                        )
                        _print_trial_status(result)
                        _print_timing_targets(result, require_fast=args.require_fast_planner_timing)
                        result.trial_idx = trial_idx
                        result.seed = args.seed + trial_idx
                        results.append(result)
                        _trial_results.append(result)
                        _print_cross_summary(scen_name, result, solver)
                        if args.render:
                            _do_render(args, spec, method, scen_name, q_hist)

                else:  # build_all_first
                    built = {}
                    print(f"  building solvers ...")
                    for method in methods:
                        rewarm_ms = _do_rewarm_if_needed(
                            method, ik_source, rewarm_ik_before_build, inline_prefix=None,
                            batch_size=ik_batch_size,
                        )
                        rewarm_tag = f"rewarm={rewarm_ms:.0f}ms " if rewarm_ms > 0 else ""
                        print(f"    [{method}] {rewarm_tag}building ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            solver, clearance_fn = build_solver(method, spec, **_build_kw)
                        except Exception as exc:
                            print(f"FAILED: {exc}")
                            built[method] = None
                            continue
                        build_s = time.perf_counter() - t0
                        ps, iks, ikgen, clsms = _extract_build_meta(method, solver, ik_source)
                        plan_hz = (1.0 / build_s if build_s > 1e-9 else None) if ps != "disabled" else None
                        plan_hz_str = f"/{plan_hz:.0f}Hz" if plan_hz is not None else "/N/A"
                        _cls_str = f"  [ik={ikgen:.0f}ms classify={clsms:.0f}ms]" if clsms > 0 else ""
                        print(f"planner={build_s*1e3:.0f}ms{plan_hz_str}{_cls_str}")
                        built[method] = (solver, clearance_fn, build_s, ps, iks, ikgen, rewarm_ms)

                    print(f"  running trials ...")
                    for method in methods:
                        entry = built.get(method)
                        if entry is None:
                            continue
                        solver, clearance_fn, build_s, ps, iks, ikgen, rewarm_ms = entry
                        print(f"    [{method}] running {args.steps} steps ...", end=" ", flush=True)
                        result, q_hist = run_trial(
                            solver, spec, clearance_fn,
                            n_steps=args.steps, planner_s=build_s,
                            planner_source=ps, ik_source=iks, ik_generation_ms=ikgen,
                            ik_rewarm_ms=rewarm_ms, benchmark_ordering=benchmark_ordering,
                        )
                        _print_trial_status(result)
                        _print_timing_targets(result, require_fast=args.require_fast_planner_timing)
                        result.trial_idx = trial_idx
                        result.seed = args.seed + trial_idx
                        results.append(result)
                        _trial_results.append(result)
                        _print_cross_summary(scen_name, result, solver)
                        if args.render:
                            _do_render(args, spec, method, scen_name, q_hist)

    print_table(results)
    _print_interpretation(
        results,
        ik_warmup_ms=ik_warmup_ms,
        warmup_enabled=warmup_enabled,
        rewarm_enabled=rewarm_ik_before_build,
        benchmark_ordering=benchmark_ordering,
    )

    if args.trials > 1:
        from itertools import groupby
        sorted_results = sorted(results, key=lambda r: (r.scenario, r.method))
        groups = []
        for (scen, meth), grp in groupby(sorted_results, key=lambda r: (r.scenario, r.method)):
            grp_list = list(grp)
            g = _compute_group_stats(grp_list)
            g["scenario"] = scen
            g["method"] = meth
            groups.append(g)

        _SUM_COLS = [
            ("scenario",            30, "Scenario"),
            ("method",              22, "Method"),
            ("N",                    4, "N"),
            ("success",             10, "Success"),
            ("planner_ms",          18, "Planner ms"),
            ("control_hz",          16, "Ctrl Hz"),
            ("min_clearance_m",     16, "Min cl m"),
            ("final_grasp_err_m",   22, "Grasp err m"),
            ("runtime_plan_events",   16, "Replan events"),
            ("runtime_plan_peak_ms",  18, "Replan peak ms"),
            ("runtime_plan_avg_ms",   18, "Replan avg ms"),
        ]
        print()
        print("=" * 80)
        print(f"Quantitative summary over {args.trials} trials (seed={args.seed})")
        print("=" * 80)
        hdr = "  ".join(f"{label:<{w}}" for _, w, label in _SUM_COLS)
        print(hdr)
        print("  ".join("-" * w for _, w, _ in _SUM_COLS))
        for g in groups:
            m = g["metrics"]
            def _ms(key, _m=m):
                mv = _m.get(key, {})
                mn, sd = mv.get("mean"), mv.get("std")
                return "N/A" if mn is None else f"{mn:.1f} +- {sd:.1f}"
            def _hz(key, _m=m):
                mv = _m.get(key, {})
                mn, sd = mv.get("mean"), mv.get("std")
                return "N/A" if mn is None else f"{mn:.0f} +- {sd:.0f}"
            def _cl(key, _m=m):
                mv = _m.get(key, {})
                mn, sd = mv.get("mean"), mv.get("std")
                return "N/A" if mn is None else f"{mn:.4f} +- {sd:.4f}"
            def _ge(key, _m=m):
                mv = _m.get(key, {})
                mn, sd = mv.get("mean"), mv.get("std")
                return "N/A" if mn is None else f"{mn:.2e} +- {sd:.2e}"
            def _ev(key, _m=m):
                mv = _m.get(key, {})
                mn, sd = mv.get("mean"), mv.get("std")
                return "N/A" if mn is None else f"{mn:.1f} +- {sd:.1f}"
            N_g = g["N"]
            ns = g["num_success"]
            vals = {
                "scenario":            g["scenario"],
                "method":              g["method"],
                "N":                   str(N_g),
                "success":             f"{ns}/{N_g}",
                "planner_ms":          _ms("planner_ms"),
                "control_hz":          _hz("control_hz"),
                "min_clearance_m":     _cl("min_clearance_m"),
                "final_grasp_err_m":   _ge("final_grasp_err_m"),
                "runtime_plan_events":   _ev("runtime_plan_events"),
                "runtime_plan_peak_ms":  _ms("runtime_plan_peak_ms"),
                "runtime_plan_avg_ms":   _ms("runtime_plan_avg_ms"),
            }
            row = "  ".join(f"{vals[k]:<{w}}" for k, w, _ in _SUM_COLS)
            print(row)
        print()

        if args.summary_csv:
            _save_summary_csv(groups, args.summary_csv)
        if args.summary_json:
            _save_summary_json(
                groups, args.summary_json,
                trials=args.trials, seed=args.seed,
                scenarios=list(args.scenarios), methods=list(methods),
                planner_profile=args.planner_profile,
                cuda_warmup_ms=ik_warmup_ms,
            )
    else:
        if args.summary_csv:
            _save_summary_csv(
                [dict(_compute_group_stats([r]), scenario=r.scenario, method=r.method)
                 for r in results],
                args.summary_csv,
            )
        if args.summary_json:
            _save_summary_json(
                [dict(_compute_group_stats([r]), scenario=r.scenario, method=r.method)
                 for r in results],
                args.summary_json,
                trials=args.trials, seed=args.seed,
                scenarios=list(args.scenarios), methods=list(methods),
                planner_profile=args.planner_profile,
                cuda_warmup_ms=ik_warmup_ms,
            )

    if args.csv:
        save_csv(results, args.csv)
    if args.json:
        save_json(results, args.json)


if __name__ == "__main__":
    main()
