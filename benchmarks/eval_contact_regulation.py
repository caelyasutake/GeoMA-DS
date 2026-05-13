"""
Contact & Passivity Regulation Benchmark — Tasks A, B, C.

Compares five control methods on contact quality, force regulation, and wall
tracing in a kinematic simulation (Euler-integrated joint positions, no physics
engine, analytic contact spring model).

Methods
-------
  diffik_ds_impedance       DiffIK-DS + contact normal projection
  diffik_ds_cbf_impedance   DiffIK-DS + greedy CBF + contact projection
  geo_ma_ds_impedance       GeoMA-DS + contact projection (β=1, no tank)
  geo_ma_ds_passivity       GeoMA-DS + EnergyTank gating β_tan, β_null
  geo_ma_ds_passivity_cbf   GeoMA-DS + passivity + greedy CBF

Tasks
-----
  A  wall_contact_regulation   approach → contact → regulate normal force
  B  wall_line_trace            contact maintained → horizontal trace (3 K_x levels)
  C  obstacle_aware_trace       wall trace + cylinder obstacle avoidance

Outputs
-------
  outputs/final/contact_passivity/<task>/
      metrics.json          per-method scalar metrics
      force_plot.png        normal force proxy over time, per method
      tank_energy_plot.png  tank energy over time (passivity methods)
      trajectory_plot.png   EE xy path during trace phase
      clearance_plot.png    min clearance over time (Task C)
      summary.md            comparison table

Usage::

    python -m benchmarks.eval_contact_regulation
    python -m benchmarks.eval_contact_regulation --tasks A B
    python -m benchmarks.eval_contact_regulation --methods geo_ma_ds_passivity geo_ma_ds_passivity_cbf
    python -m benchmarks.eval_contact_regulation --out-dir outputs/contact_reg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scenarios.scenario_schema import Obstacle, ScenarioSpec
from src.solver.ds.factory import (
    _make_clearance_fn,
    _make_batch_from_lp_fn,
    ik_goals_to_attractors,
)
from src.solver.ds.geo_multi_attractor_ds import (
    GeoMultiAttractorDS,
    GeoMultiAttractorDSConfig,
    IKAttractor,
)
from src.solver.planner.collision import _panda_link_positions
from src.solver.ik.coverage_expansion import (
    _grasptarget_pos,
    _grasptarget_jacobian_fd,
)
from src.solver.tank.tank import EnergyTank, TankConfig


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
_DT             = 0.002   # s — 500 Hz kinematic update
_APPROACH_STEPS = 800     # ~1.6 s  (need ~0.95 s to cover 0.11 m with K_x=2.0)
_HOLD_STEPS     = 400     # ~0.8 s contact hold
_TRACE_STEPS    = 500     # ~1.0 s tracing
_MAX_QDOT       = 1.5     # rad/s velocity saturation

# Wall geometry (horizontal platform)
_WALL_Z_TOP     = 0.37    # m — top surface height
_CONTACT_THRESH = 0.005   # m — contact detection zone below wall surface
_K_SPRING       = 200.0   # N/m — spring constant for normal force proxy
_K_RESTORE      = 15.0    # velocity gain to restore when penetrated

# Trace path (Task B and C)
_TRACE_X_START  = 0.40    # m
_TRACE_X_END    = 0.52    # m
_TRACE_Y        = 0.0     # m

# Task C obstacle (cylinder near the trace path)
_OBS_CYL_POS    = [0.50, 0.15, 0.45]
_OBS_CYL_RADIUS = 0.04    # m
_OBS_CYL_HHALF  = 0.12    # m — half-height

# Joint limits (Panda)
_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_HI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# Ready pose and wall approach goal
# _Q_WALL_APPROACH: grasptarget at ~[0.40, 0.0, 0.355] — 15mm below wall surface.
# The approach target is inside the wall so exponential-decay DS reliably crosses
# the contact threshold (z <= 0.375) during the approach phase.
# The contact constraint prevents actual penetration via spring restoration.
# Computed by 500-step DiffIK from Q_READY to target [0.40, 0.0, 0.355].
_Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_Q_WALL_APPROACH = np.array([0.0, -0.3861, 0.0, -2.3727, 0.0, 1.8481, 0.785])


# ---------------------------------------------------------------------------
# Wall obstacle builder
# ---------------------------------------------------------------------------

def _wall_obstacle() -> Obstacle:
    return Obstacle(
        name="contact_wall",
        type="box",
        position=[0.40, 0.0, 0.35],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        size=[0.25, 0.25, 0.02],
        rgba=(0.30, 0.55, 0.85, 0.85),
        collision_enabled=True,
        friction=(0.05, 0.005, 0.0001),
    )


def _cylinder_obstacle() -> Obstacle:
    return Obstacle(
        name="side_cylinder",
        type="cylinder",
        position=_OBS_CYL_POS,
        size=[_OBS_CYL_RADIUS, _OBS_CYL_HHALF],
        rgba=(0.85, 0.35, 0.20, 0.85),
        collision_enabled=True,
    )


# ---------------------------------------------------------------------------
# Clearance gradient (shared utility)
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
# Contact constraint helper
# ---------------------------------------------------------------------------

def apply_contact_constraint(
    q_dot: np.ndarray,
    q: np.ndarray,
    wall_z_top: float = _WALL_Z_TOP,
    k_restore: float = _K_RESTORE,
) -> Tuple[np.ndarray, float, bool]:
    """
    Project out the wall-normal (z) velocity component and add spring restoration
    when the EE has penetrated the wall.

    Returns (q_dot_constrained, normal_force_proxy_N, in_contact).

    normal_force_proxy_N: K_spring * |z_vel_removed| — the normal force needed to
    cancel the downward velocity the controller was trying to apply.  Positive
    when the controller is pushing into the wall (good contact regulation).
    """
    J = _grasptarget_jacobian_fd(q)  # (3, 7)
    J_z = J[2, :]                    # z-row (7,)
    grasp_z = float(_grasptarget_pos(q)[2])

    in_contact = grasp_z <= wall_z_top + _CONTACT_THRESH

    if not in_contact:
        return q_dot.copy(), 0.0, False

    # Project out z-velocity (no driving into or away from wall)
    norm_sq = float(J_z @ J_z) + 1e-12
    z_vel_scalar = float(J_z @ q_dot) / norm_sq

    # EE z-velocity (m/s) — negative = pushing downward into wall.
    z_vel_ee = float(J_z @ q_dot)  # m/s

    # Normal force proxy: virtual damper model F = c * |z_vel_into_wall|.
    # Positive when the controller is driving the EE into the wall surface.
    # _K_SPRING here acts as a damper coefficient (N/(m/s)).
    force_proxy = max(0.0, -z_vel_ee) * _K_SPRING

    q_dot = q_dot - (z_vel_ee / norm_sq) * J_z  # remove all z-velocity

    # Spring restoration when penetrated (prevents numerical drift)
    penetration = max(0.0, wall_z_top - grasp_z)
    if penetration > 0.0:
        q_dot = q_dot + k_restore * penetration * J_z / norm_sq

    return q_dot, force_proxy, True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContactTrialResult:
    method:                  str
    task:                    str
    success_approach:        bool    # EE reached contact zone
    time_to_contact_steps:   int     # -1 if never
    contact_maintained_frac: float   # fraction of hold+trace phase in contact
    contact_depth_mean_m:    float   # mean force proxy during contact (N)
    contact_depth_std_m:     float   # std force proxy
    normal_force_mean_N:     float   # same as contact_depth_mean_m (alias for clarity)
    normal_force_std_N:      float   # same as contact_depth_std_m
    trace_rmse_m:            float   # RMSE from trace path (0 for Task A)
    trace_completion:        float   # fraction of trace distance covered
    min_clearance_m:         float   # min clearance to all obstacles
    n_posture_switches:      int     # GeoMA-DS only, else 0
    beta_mean:               float   # mean tank beta (passivity methods, else 1.0)
    tank_energy_min:         float   # minimum tank energy (passivity methods)
    tank_energy_mean:        float   # mean tank energy
    passivity_violations:    int     # steps with z > 0
    energy_injected_J:       float   # total energy injected by f_R
    mean_step_ms:            float
    hz:                      float
    # Time series (not in JSON summary, used for plots)
    _force_history:          List[float] = None
    _tank_history:           List[float] = None
    _grasp_xy_history:       List[np.ndarray] = None
    _clearance_history:      List[float] = None
    _beta_history:           List[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Method: DiffIK-DS + contact projection
# ---------------------------------------------------------------------------

class DiffIKContactSolver:
    name = "diffik_ds_impedance"

    def __init__(
        self,
        target: np.ndarray,
        K_x: float = 2.0,
        lam: float = 0.05,
        clearance_fn=None,
    ):
        self.target = target.copy()
        self.K_x    = K_x
        self.lam    = lam
        self.clearance_fn = clearance_fn

    def set_target(self, target: np.ndarray) -> None:
        self.target = target.copy()

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        x   = _grasptarget_pos(q)
        err = self.target - x
        x_dot = self.K_x * err

        J   = _grasptarget_jacobian_fd(q)
        JJT = J @ J.T + (self.lam ** 2) * np.eye(3)
        Jp  = J.T @ np.linalg.inv(JJT)
        q_dot = Jp @ x_dot

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        return q_dot, {"cbf_active": False, "alpha": 0.0, "n_switches": 0,
                        "beta": 1.0, "tank_energy": 1.0}


class DiffIKCBFContactSolver(DiffIKContactSolver):
    name = "diffik_ds_cbf_impedance"

    def __init__(
        self,
        target: np.ndarray,
        clearance_fn,
        K_x: float = 2.0,
        lam: float = 0.05,
        d_safe: float = 0.03,
        d_buffer: float = 0.10,
        alpha: float = 8.0,
        grad_eps: float = 1e-3,
    ):
        super().__init__(target, K_x=K_x, lam=lam, clearance_fn=clearance_fn)
        self.name      = "diffik_ds_cbf_impedance"
        self.d_safe    = d_safe
        self.d_buffer  = d_buffer
        self.alpha     = alpha
        self.grad_eps  = grad_eps

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        q_dot, diag = super().compute(q)

        if self.clearance_fn is not None:
            cl  = float(self.clearance_fn(q))
            h   = cl - self.d_safe
            if cl < self.d_safe + self.d_buffer:
                grad_h = _clearance_gradient(q, self.clearance_fn, eps=self.grad_eps)
                lhs    = float(np.dot(grad_h, q_dot)) + self.alpha * float(h)
                if lhs < 0.0:
                    denom  = float(np.dot(grad_h, grad_h)) + 1e-12
                    q_dot  = q_dot - lhs * grad_h / denom
                    diag["cbf_active"] = True

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        return q_dot, diag


# ---------------------------------------------------------------------------
# GeoMA-DS contact wrapper (no passivity)
# ---------------------------------------------------------------------------

class GeoMAContactSolver:
    name = "geo_ma_ds_impedance"

    def __init__(self, ds: GeoMultiAttractorDS):
        self._ds = ds
        self._target_q: Optional[np.ndarray] = None

    def set_target_q(self, q_goal: np.ndarray) -> None:
        self._target_q = q_goal.copy()
        # Update the first attractor's goal in-place (single-attractor contact scenario)
        self._ds.attractors[0].q_goal = q_goal.copy()
        self._ds._active_idx = 0

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        q_dot, result = self._ds.compute(q, dt=_DT)
        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)
        return q_dot, {
            "cbf_active": False,
            "alpha": float(result.obs_blend_alpha),
            "n_switches": int(result.n_switches),
            "beta": 1.0,
            "tank_energy": 1.0,
        }


# ---------------------------------------------------------------------------
# GeoMA-DS + passivity tank
# ---------------------------------------------------------------------------

class GeoMAPassivitySolver:
    name = "geo_ma_ds_passivity"

    def __init__(
        self,
        ds: GeoMultiAttractorDS,
        tank: EnergyTank,
        cbf_clearance_fn=None,
        d_safe: float = 0.03,
        d_buffer: float = 0.10,
        cbf_alpha: float = 8.0,
    ):
        self._ds = ds
        self._tank = tank
        self._cbf_fn = cbf_clearance_fn
        self.d_safe   = d_safe
        self.d_buffer = d_buffer
        self.cbf_alpha = cbf_alpha

    def set_target_q(self, q_goal: np.ndarray) -> None:
        self._ds.attractors[0].q_goal = q_goal.copy()
        self._ds._active_idx = 0

    def compute(self, q: np.ndarray) -> Tuple[np.ndarray, dict]:
        _, result = self._ds.compute(q, dt=_DT)

        beta = self._tank.beta_R

        # Passivity-gated combination: f_attractor always active
        q_dot = result.f_attractor + beta * result.f_tangent + beta * result.f_null

        # CBF safety filter (passivity_cbf subclass)
        cbf_active = False
        if self._cbf_fn is not None:
            cl = float(self._cbf_fn(q))
            h  = cl - self.d_safe
            if cl < self.d_safe + self.d_buffer:
                grad_h = _clearance_gradient(q, self._cbf_fn)
                lhs    = float(np.dot(grad_h, q_dot)) + self.cbf_alpha * float(h)
                if lhs < 0.0:
                    denom  = float(np.dot(grad_h, grad_h)) + 1e-12
                    q_dot  = q_dot - lhs * grad_h / denom
                    cbf_active = True

        speed = float(np.linalg.norm(q_dot))
        if speed > _MAX_QDOT:
            q_dot = q_dot * (_MAX_QDOT / speed)

        # Tank update: z = qdot^T (f_tangent + f_null)
        # (f_attractor is conservative, not tracked for passivity)
        qdot_actual = q_dot  # kinematic: actual velocity = desired velocity
        z = float(qdot_actual @ (result.f_tangent + result.f_null))
        D = 5.0 * np.eye(7)  # nominal damping for tank charging
        self._tank.step(z, qdot_actual, D, _DT)

        return q_dot, {
            "cbf_active": cbf_active,
            "alpha": float(result.obs_blend_alpha),
            "n_switches": int(result.n_switches),
            "beta": beta,
            "tank_energy": self._tank.energy,
        }


class GeoMAPassivityCBFSolver(GeoMAPassivitySolver):
    name = "geo_ma_ds_passivity_cbf"

    def __init__(self, ds: GeoMultiAttractorDS, tank: EnergyTank, cbf_clearance_fn):
        super().__init__(ds, tank, cbf_clearance_fn=cbf_clearance_fn)
        self.name = "geo_ma_ds_passivity_cbf"


# ---------------------------------------------------------------------------
# Trace target generator
# ---------------------------------------------------------------------------

def _trace_target_pos(step: int, total_steps: int) -> np.ndarray:
    """Linear trace from x_start to x_end at wall height z."""
    t = min(1.0, step / max(1, total_steps))
    x = _TRACE_X_START + t * (_TRACE_X_END - _TRACE_X_START)
    return np.array([x, _TRACE_Y, _WALL_Z_TOP])


def _trace_target_joint(
    trace_pos: np.ndarray,
    q_ref: np.ndarray,
    lam: float = 0.05,
) -> np.ndarray:
    """
    Compute a joint-space target that achieves trace_pos as grasptarget.
    Uses one damped-LS IK step from q_ref (not iterated, just for attractor update).
    """
    x   = _grasptarget_pos(q_ref)
    err = trace_pos - x
    J   = _grasptarget_jacobian_fd(q_ref)
    JJT = J @ J.T + lam ** 2 * np.eye(3)
    Jp  = J.T @ np.linalg.inv(JJT)
    dq  = Jp @ err
    q_target = np.clip(q_ref + dq, _Q_LO, _Q_HI)
    return q_target


# ---------------------------------------------------------------------------
# Core trial runner
# ---------------------------------------------------------------------------

def run_contact_trial(
    solver,
    approach_target_q: np.ndarray,
    task: str,
    clearance_fn,
    n_approach: int = _APPROACH_STEPS,
    n_hold: int     = _HOLD_STEPS,
    n_trace: int    = _TRACE_STEPS,
) -> ContactTrialResult:
    """
    Run one contact trial through approach → hold → (trace).

    Task 'A': approach + hold only.
    Task 'B'/'C': approach + hold + trace.
    """
    q = _Q_READY.copy()
    qdot_prev = np.zeros(7)

    force_hist   : List[float] = []
    tank_hist    : List[float] = []
    beta_hist    : List[float] = []
    xy_hist      : List[np.ndarray] = []
    cl_hist      : List[float] = []
    contact_log  : List[bool]  = []
    depth_log    : List[float] = []

    step_times: List[float] = []

    # Set solver to approach target
    _set_target(solver, approach_target_q, q)

    time_to_contact = -1
    n_switches      = 0
    total_steps     = n_approach + n_hold + (n_trace if task in ("B", "C") else 0)

    # ---- Phase 1: APPROACH ------------------------------------------------
    for step in range(n_approach):
        t0 = time.perf_counter()
        q_dot, diag = solver.compute(q)
        step_times.append((time.perf_counter() - t0) * 1e3)

        n_switches = max(n_switches, diag.get("n_switches", 0))

        q_dot, pen, in_contact = apply_contact_constraint(q_dot, q)
        q = np.clip(q + _DT * q_dot, _Q_LO, _Q_HI)

        cl = float(clearance_fn(q))
        grasp = _grasptarget_pos(q)
        force_hist.append(pen)
        tank_hist.append(float(diag.get("tank_energy", 1.0)))
        beta_hist.append(float(diag.get("beta", 1.0)))
        xy_hist.append(grasp[:2].copy())
        cl_hist.append(cl)
        contact_log.append(in_contact)
        depth_log.append(pen)

        if in_contact and time_to_contact < 0:
            time_to_contact = step

    approach_success = time_to_contact >= 0

    # ---- Phase 2: HOLD ----------------------------------------------------
    _set_target(solver, approach_target_q, q)
    for step in range(n_hold):
        t0 = time.perf_counter()
        q_dot, diag = solver.compute(q)
        step_times.append((time.perf_counter() - t0) * 1e3)

        n_switches = max(n_switches, diag.get("n_switches", 0))

        q_dot, pen, in_contact = apply_contact_constraint(q_dot, q)
        q = np.clip(q + _DT * q_dot, _Q_LO, _Q_HI)

        cl = float(clearance_fn(q))
        grasp = _grasptarget_pos(q)
        force_hist.append(pen)
        tank_hist.append(float(diag.get("tank_energy", 1.0)))
        beta_hist.append(float(diag.get("beta", 1.0)))
        xy_hist.append(grasp[:2].copy())
        cl_hist.append(cl)
        contact_log.append(in_contact)
        depth_log.append(pen)

    # ---- Phase 3: TRACE (Tasks B and C) ------------------------------------
    trace_errs: List[float] = []
    trace_completed = 0.0

    if task in ("B", "C"):
        for step in range(n_trace):
            trace_pos = _trace_target_pos(step, n_trace)
            trace_q   = _trace_target_joint(trace_pos, q)
            _set_target(solver, trace_q, q)

            t0 = time.perf_counter()
            q_dot, diag = solver.compute(q)
            step_times.append((time.perf_counter() - t0) * 1e3)

            n_switches = max(n_switches, diag.get("n_switches", 0))

            q_dot, pen, in_contact = apply_contact_constraint(q_dot, q)
            q = np.clip(q + _DT * q_dot, _Q_LO, _Q_HI)

            cl = float(clearance_fn(q))
            grasp = _grasptarget_pos(q)

            # Trace error: distance from EE to desired trace point
            trace_err = float(np.linalg.norm(grasp[:2] - trace_pos[:2]))
            trace_errs.append(trace_err)

            force_hist.append(pen)
            tank_hist.append(float(diag.get("tank_energy", 1.0)))
            beta_hist.append(float(diag.get("beta", 1.0)))
            xy_hist.append(grasp[:2].copy())
            cl_hist.append(cl)
            contact_log.append(in_contact)
            depth_log.append(pen)

        # Trace completion: fraction of x-range covered
        final_x = float(_grasptarget_pos(q)[0])
        x_range = _TRACE_X_END - _TRACE_X_START
        trace_completed = float(
            np.clip((final_x - _TRACE_X_START) / max(x_range, 1e-6), 0.0, 1.0)
        )

    # ---- Aggregate metrics ------------------------------------------------
    hold_trace_contact = contact_log[n_approach:]
    contact_frac = (
        float(np.mean(hold_trace_contact)) if hold_trace_contact else 0.0
    )

    depths_arr = np.array(depth_log[n_approach:])
    d_mean = float(np.mean(depths_arr)) if len(depths_arr) > 0 else 0.0
    d_std  = float(np.std(depths_arr))  if len(depths_arr) > 0 else 0.0

    tank_arr = np.array(tank_hist)

    passivity_violations = int(getattr(
        getattr(solver, "_tank", None), "passivity_violation_count", 0
    ))
    energy_injected = float(getattr(
        getattr(solver, "_tank", None), "total_energy_injected", 0.0
    ))

    mean_ms = float(np.mean(step_times)) if step_times else 0.0

    return ContactTrialResult(
        method=solver.name,
        task=task,
        success_approach=approach_success,
        time_to_contact_steps=time_to_contact,
        contact_maintained_frac=contact_frac,
        contact_depth_mean_m=d_mean,
        contact_depth_std_m=d_std,
        normal_force_mean_N=d_mean,   # d_mean is already a force proxy in N
        normal_force_std_N=d_std,
        trace_rmse_m=float(np.sqrt(np.mean(np.array(trace_errs) ** 2))) if trace_errs else 0.0,
        trace_completion=trace_completed,
        min_clearance_m=float(np.min(cl_hist)) if cl_hist else float("inf"),
        n_posture_switches=n_switches,
        beta_mean=float(np.mean(beta_hist)) if beta_hist else 1.0,
        tank_energy_min=float(np.min(tank_arr)) if len(tank_arr) > 0 else 1.0,
        tank_energy_mean=float(np.mean(tank_arr)) if len(tank_arr) > 0 else 1.0,
        passivity_violations=passivity_violations,
        energy_injected_J=energy_injected,
        mean_step_ms=mean_ms,
        hz=float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        _force_history=force_hist,
        _tank_history=tank_hist,
        _grasp_xy_history=xy_hist,
        _clearance_history=cl_hist,
        _beta_history=beta_hist,
    )


def _set_target(solver, q_goal: np.ndarray, q_current: np.ndarray) -> None:
    """Update the solver's current goal target (handles all solver types)."""
    grasp_goal = _grasptarget_pos(q_goal)
    if hasattr(solver, "set_target_q"):
        solver.set_target_q(q_goal)
    elif hasattr(solver, "set_target"):
        solver.set_target(grasp_goal)
    elif hasattr(solver, "target"):
        solver.target = grasp_goal.copy()


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def build_solver(
    method: str,
    all_obstacles: List[Obstacle],
    avoidance_obstacles: List[Obstacle],
    approach_target_q: np.ndarray,
):
    """
    Construct the named solver.

    all_obstacles:        used for CBF safety + clearance logging
    avoidance_obstacles:  used for GeoMA-DS geometric shaping (excludes wall
                          contact surface, so shaping only activates for
                          Task C's cylinder obstacle)
    """
    all_clearance_fn = _make_clearance_fn(all_obstacles) if all_obstacles else (lambda q: 1.0)
    avoid_clearance_fn = _make_clearance_fn(avoidance_obstacles) if avoidance_obstacles else None
    approach_pos = _grasptarget_pos(approach_target_q)

    if method == "diffik_ds_impedance":
        return DiffIKContactSolver(
            target=approach_pos,
            K_x=2.0,
            clearance_fn=all_clearance_fn,
        )

    if method == "diffik_ds_cbf_impedance":
        return DiffIKCBFContactSolver(
            target=approach_pos,
            clearance_fn=all_clearance_fn,
            K_x=2.0,
        )

    # GeoMA-DS: attractor clearance uses ALL obstacles; geometric shaping
    # uses AVOIDANCE obstacles only (wall excluded — handled by contact constraint)
    attractors = ik_goals_to_attractors([approach_target_q], all_clearance_fn)
    geo_clearance_fn = avoid_clearance_fn if avoidance_obstacles else (lambda q: 1.0)
    batch_fn = _make_batch_from_lp_fn(avoidance_obstacles) if avoidance_obstacles else None

    ds_config = GeoMultiAttractorDSConfig(
        K_c=2.0,
        K_null=0.5,
        clearance_enter=0.15,
        clearance_full=0.04,
        lookahead_safety=False,  # no lookahead — contact constraint handles wall
    )

    if method == "geo_ma_ds_impedance":
        ds = GeoMultiAttractorDS(
            attractors, geo_clearance_fn, config=ds_config, batch_from_lp_fn=batch_fn
        )
        return GeoMAContactSolver(ds)

    if method == "geo_ma_ds_passivity":
        ds   = GeoMultiAttractorDS(
            attractors, geo_clearance_fn, config=ds_config, batch_from_lp_fn=batch_fn
        )
        tank = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0))
        return GeoMAPassivitySolver(ds, tank)

    if method == "geo_ma_ds_passivity_cbf":
        ds   = GeoMultiAttractorDS(
            attractors, geo_clearance_fn, config=ds_config, batch_from_lp_fn=batch_fn
        )
        tank = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0))
        return GeoMAPassivityCBFSolver(ds, tank, all_clearance_fn)

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _build_task_scenario(task: str) -> Tuple[List[Obstacle], List[Obstacle], np.ndarray]:
    """
    Returns (all_obstacles, avoidance_obstacles, approach_target_q).

    avoidance_obstacles: obstacles GeoMA-DS shapes around.
    The wall is a contact surface — handled by the contact constraint, not by
    GeoMA-DS geometric shaping.  Only Task C adds the cylinder for avoidance.
    """
    wall = _wall_obstacle()
    if task == "C":
        cyl = _cylinder_obstacle()
        all_obstacles = [wall, cyl]
        avoidance_obstacles = [cyl]   # GeoMA-DS avoids cylinder, not wall
    else:
        all_obstacles = [wall]
        avoidance_obstacles = []      # no side obstacles for Tasks A and B
    return all_obstacles, avoidance_obstacles, _Q_WALL_APPROACH.copy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "diffik_ds_impedance":      "#6B6B6B",
    "diffik_ds_cbf_impedance":  "#E07B39",
    "geo_ma_ds_impedance":      "#43A047",
    "geo_ma_ds_passivity":      "#2176AE",
    "geo_ma_ds_passivity_cbf":  "#7B2D8B",
}
_METHOD_LABELS = {
    "diffik_ds_impedance":      "DiffIK-DS",
    "diffik_ds_cbf_impedance":  "DiffIK-DS+CBF",
    "geo_ma_ds_impedance":      "GeoMA-DS",
    "geo_ma_ds_passivity":      "GeoMA-DS+Passivity",
    "geo_ma_ds_passivity_cbf":  "GeoMA-DS+Pass+CBF",
}


def _time_axis(n: int) -> np.ndarray:
    return np.arange(n) * _DT


def plot_force(results: List[ContactTrialResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in results:
        if r._force_history is None:
            continue
        t = _time_axis(len(r._force_history))
        ax.plot(t, r._force_history, label=_METHOD_LABELS.get(r.method, r.method),
                color=_METHOD_COLORS.get(r.method, "#888888"), alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal force proxy (N)")
    ax.set_title(f"Task {results[0].task} — Normal force (K_spring × penetration)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "force_plot.png", dpi=120)
    plt.close(fig)


def plot_tank_energy(results: List[ContactTrialResult], out_dir: Path) -> None:
    passivity_results = [r for r in results if "passivity" in r.method]
    if not passivity_results:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in passivity_results:
        if r._tank_history is None:
            continue
        t = _time_axis(len(r._tank_history))
        ax.plot(t, r._tank_history, label=_METHOD_LABELS.get(r.method, r.method),
                color=_METHOD_COLORS.get(r.method, "#888888"), alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tank energy (J)")
    ax.set_title(f"Task {results[0].task} — Passivity tank energy")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="s_min=0")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tank_energy_plot.png", dpi=120)
    plt.close(fig)


def plot_trajectory(results: List[ContactTrialResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        if r._grasp_xy_history is None:
            continue
        xy = np.array(r._grasp_xy_history)
        ax.plot(xy[:, 0], xy[:, 1], label=_METHOD_LABELS.get(r.method, r.method),
                color=_METHOD_COLORS.get(r.method, "#888888"), alpha=0.8, linewidth=1.2)

    if results[0].task in ("B", "C"):
        ax.plot([_TRACE_X_START, _TRACE_X_END], [_TRACE_Y, _TRACE_Y],
                "k--", linewidth=1.5, label="Desired trace")
    if results[0].task == "C":
        theta = np.linspace(0, 2 * np.pi, 64)
        ax.fill(
            _OBS_CYL_POS[0] + _OBS_CYL_RADIUS * np.cos(theta),
            _OBS_CYL_POS[1] + _OBS_CYL_RADIUS * np.sin(theta),
            color="#E53935", alpha=0.4, label="Cylinder obstacle",
        )

    ax.set_xlabel("EE x (m)")
    ax.set_ylabel("EE y (m)")
    ax.set_title(f"Task {results[0].task} — EE xy trajectory (wall surface)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_plot.png", dpi=120)
    plt.close(fig)


def plot_clearance(results: List[ContactTrialResult], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in results:
        if r._clearance_history is None:
            continue
        t = _time_axis(len(r._clearance_history))
        ax.plot(t, r._clearance_history, label=_METHOD_LABELS.get(r.method, r.method),
                color=_METHOD_COLORS.get(r.method, "#888888"), alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min clearance (m)")
    ax.set_title(f"Task {results[0].task} — Obstacle clearance")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="Collision boundary")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clearance_plot.png", dpi=120)
    plt.close(fig)


def write_summary_md(
    results: List[ContactTrialResult],
    task: str,
    out_path: Path,
) -> None:
    lines = [
        f"# Contact Regulation — Task {task}\n",
        "",
        "| Method | Contact% | Force mean (N) | Force std (N) | Trace RMSE (m) | Min cl (m) | Hz |",
        "|--------|----------|----------------|---------------|----------------|------------|-----|",
    ]
    for r in results:
        row = (
            f"| {_METHOD_LABELS.get(r.method, r.method)}"
            f" | {r.contact_maintained_frac * 100:.0f}%"
            f" | {r.normal_force_mean_N:.2f}"
            f" | {r.normal_force_std_N:.2f}"
            f" | {r.trace_rmse_m:.4f}"
            f" | {r.min_clearance_m:.3f}"
            f" | {r.hz:.0f} |"
        )
        lines.append(row)
    lines.append("")
    if any("passivity" in r.method for r in results):
        lines.append("## Passivity Metrics\n")
        lines.append("| Method | beta mean | Tank min (J) | Violations | Energy inj (J) |")
        lines.append("|--------|-----------|--------------|------------|----------------|")
        for r in results:
            if "passivity" not in r.method:
                continue
            row = (
                f"| {_METHOD_LABELS.get(r.method, r.method)}"
                f" | {r.beta_mean:.3f}"
                f" | {r.tank_energy_min:.3f}"
                f" | {r.passivity_violations}"
                f" | {r.energy_injected_J:.4f} |"
            )
            lines.append(row)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "diffik_ds_impedance",
    "diffik_ds_cbf_impedance",
    "geo_ma_ds_impedance",
    "geo_ma_ds_passivity",
    "geo_ma_ds_passivity_cbf",
]
ALL_TASKS = ["A", "B", "C"]

_TASK_LABELS = {
    "A": "wall_contact_regulation",
    "B": "wall_line_trace",
    "C": "obstacle_aware_trace",
}


def run_task(
    task: str,
    methods: List[str],
    out_dir: Path,
    verbose: bool = True,
) -> List[ContactTrialResult]:
    all_obstacles, avoidance_obstacles, approach_target_q = _build_task_scenario(task)
    clearance_fn = _make_clearance_fn(all_obstacles) if all_obstacles else (lambda q: 1.0)

    task_dir = out_dir / _TASK_LABELS[task]
    task_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[Task {task}: {_TASK_LABELS[task]}] {len(methods)} methods")

    results: List[ContactTrialResult] = []
    for method in methods:
        if verbose:
            print(f"  {method} ...", end="", flush=True)
        t0 = time.perf_counter()

        solver = build_solver(method, all_obstacles, avoidance_obstacles, approach_target_q)
        result = run_contact_trial(
            solver=solver,
            approach_target_q=approach_target_q,
            task=task,
            clearance_fn=clearance_fn,
        )
        elapsed = time.perf_counter() - t0

        if verbose:
            status = "contact" if result.success_approach else "NO_CONTACT"
            print(
                f" {status}  maint={result.contact_maintained_frac:.2f}"
                f"  F={result.normal_force_mean_N:.1f}N"
                f"  cl={result.min_clearance_m:.3f}m"
                f"  {result.hz:.0f}Hz"
                f"  ({elapsed:.1f}s)"
            )

        results.append(result)

    # Save metrics.json
    metrics = {r.method: r.to_dict() for r in results}
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save plots
    plot_force(results, task_dir)
    plot_tank_energy(results, task_dir)
    plot_trajectory(results, task_dir)
    plot_clearance(results, task_dir)
    write_summary_md(results, task, task_dir / "summary.md")

    if verbose:
        print(f"  -> {task_dir}")

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Contact regulation benchmark")
    parser.add_argument("--tasks",    nargs="+", default=ALL_TASKS,
                        choices=ALL_TASKS, metavar="TASK")
    parser.add_argument("--methods",  nargs="+", default=ALL_METHODS)
    parser.add_argument("--out-dir",  type=Path,
                        default=Path("outputs/final/contact_passivity"))
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[ContactTrialResult]] = {}
    for task in args.tasks:
        all_results[task] = run_task(
            task, args.methods, out_dir, verbose=not args.quiet
        )

    if not args.quiet:
        print(f"\nAll outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
