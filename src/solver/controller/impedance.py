"""
Phase 5 — DS Impedance Controller

Implements the passive impedance control law:

    τ = G(q) − D (ẋ − ẋ_d)

With desired velocity:

    ẋ_d = f_c(x) + β_R · f_R(x) + f_n(x)

Where:
    G(q)  — gravity compensation torque (injectable; zero if not provided)
    D     — positive-definite damping matrix
    ẋ     — current joint velocity
    ẋ_d   — desired joint velocity from DS + tank
    β_R   — passivity gate from EnergyTank (0 or 1)
    f_c   — conservative DS component (always included)
    f_R   — residual DS component (gated by tank)
    f_n   — null-space / joint-centering term (optional)

Passivity guarantee (joint space):
    The dissipated power ẋᵀ D ẋ flows into the tank, which then controls
    whether f_R is active.  When the tank is depleted (β_R = 0) the system
    reduces to a purely conservative DS, which is passive by construction.

Reference:
    Kronander & Billard (2016) "Passive Interaction Control with Dynamical Systems"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from src.solver.ds.path_ds import PathDS
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import (
    PassivityFilterConfig,
    filter_residual,
    compute_epsilon,
    orthogonalize_residual,
)

if TYPE_CHECKING:
    from src.scenarios.scenario_schema import Obstacle
    from src.solver.controller.cbf_filter import CBFConfig, CBFDiagnostics
    from src.solver.ds.modulation import ModulationConfig, ModulationDiagnostics
    from src.solver.controller.hard_shield import HardShieldConfig, HardShieldDiagnostics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ControllerConfig:
    """Parameters for the impedance controller."""

    # Damping gains — scalar converted to D = d_gain * I
    d_gain: float = 5.0

    # Null-space joint-centering gain (0 = disabled)
    f_n_gain: float = 0.1

    # Joint center for null-space term (defaults to joint midpoint at runtime)
    q_center: Optional[np.ndarray] = None

    # Gravity compensation callable: (q) -> tau_gravity, shape (n_joints,)
    # None → zero gravity compensation
    gravity_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # Passivity filter applied to f_R before the energy tank gate.
    # None → filter disabled (backward compatible).
    passivity_filter: Optional[PassivityFilterConfig] = None

    # Phase 5B: orthogonalize f_R before the passivity filter so that
    # ẋᵀ f_R = 0 exactly (using unit-vector projection).  Default True.
    orthogonalize_residual: bool = True

    # Phase 5B: scale f_R_neutral by alpha before the passivity filter.
    # alpha ∈ [0.3, 0.7] reduces residual amplitude → filter clips rarely.
    # 1.0 = no scaling (backward compatible).
    alpha: float = 0.5

    # Optional CBF safety filter.  None → disabled (backward compatible).
    # Set to a CBFConfig instance to enable online obstacle avoidance.
    cbf: Optional["CBFConfig"] = None

    # Optional task-space modulation obstacle avoidance.  None → disabled.
    # When enabled, applies M(x_ee) @ (J @ qdot_d) → J^+ + null-space.
    modulation: Optional["ModulationConfig"] = None

    # Optional hard non-contact safety shield.  None → disabled.
    # Applied after modulation; performs binary line-search to guarantee
    # clearance(q + dt*qdot) >= d_hard_min before every simulation step.
    hard_shield: Optional["HardShieldConfig"] = None


# ---------------------------------------------------------------------------
# Per-step output
# ---------------------------------------------------------------------------
@dataclass
class ControlResult:
    """Output of one controller step."""

    tau: np.ndarray          # control torque, shape (n_joints,)
    xdot_d: np.ndarray       # desired velocity, shape (n_joints,)
    beta_R: float            # tank passivity gate (smooth, ∈ [0,1])
    z: float                 # passivity metric ẋᵀ f_R (nominal, pre-filter)
    tank_energy: float       # s after this step
    passivity_violated: bool # True when z > 0 (f_R injected energy)
    V: float                 # Lyapunov value at current x
    # Passivity filter diagnostics (zero / False when filter is disabled)
    pf_clipped:      bool  = False   # True when f_R was projected
    pf_power_nom:    float = 0.0     # ẋᵀ f_R before filter
    pf_power_filtered: float = 0.0  # ẋᵀ f_R after filter
    # CBF filter diagnostics (None when CBF is disabled)
    cbf: Optional["CBFDiagnostics"] = None
    # Modulation diagnostics (None when modulation is disabled)
    modulation: Optional["ModulationDiagnostics"] = None
    # Hard shield diagnostics (None when shield is disabled)
    hard_shield: Optional["HardShieldDiagnostics"] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_D(n_joints: int, d_gain: float) -> np.ndarray:
    """Build isotropic damping matrix D = d_gain * I."""
    return d_gain * np.eye(n_joints)


def _null_space_term(
    q: np.ndarray,
    q_center: np.ndarray,
    K_n: float,
) -> np.ndarray:
    """
    Joint-centering null-space velocity:  f_n = K_n * (q_center − q).

    Keeps the robot near a preferred configuration while f_c/f_R handle
    the primary task.  Phase 6 can project this into the task null space
    using the Jacobian.
    """
    return K_n * (q_center - q)


# ---------------------------------------------------------------------------
# Public step function
# ---------------------------------------------------------------------------
def step(
    q: np.ndarray,
    qdot: np.ndarray,
    ds: PathDS,
    tank: EnergyTank,
    dt: float,
    config: Optional[ControllerConfig] = None,
    obstacles: Optional[List["Obstacle"]] = None,
    phase: str = "",
    clearance: Optional[float] = None,
    ds_mode: str = "normal",
    ds_escape_target: Optional[np.ndarray] = None,
    escape_velocity: Optional[np.ndarray] = None,
    nominal_velocity_override: Optional[np.ndarray] = None,
    q_goal: Optional[np.ndarray] = None,
    goal_clearance: Optional[float] = None,
) -> ControlResult:
    """
    Execute one control step.

    Args:
        q:                Current joint position, shape (n_joints,).
        qdot:             Current joint velocity, shape (n_joints,).
        ds:               PathDS instance (from Phase 4).
        tank:             EnergyTank instance (from Phase 5 tank module).
        dt:               Control time step (seconds).
        config:           ControllerConfig; safe defaults when None.
        obstacles:        Obstacle list for CBF safety filter.
        phase:            Current task phase string.
        clearance:        Minimum signed obstacle clearance at current state (m).
        ds_mode:          PathDS mode string: "normal", "reduced_fc",
                          "escape_clearance", "backtrack", "bridge_target".
        ds_escape_target: Joint-space bridge target (used in "bridge_target" mode).
        escape_velocity:  Pre-computed escape joint velocity to add on top of
                          the DS output (e.g. from EscapePolicy).  None = unused.
        nominal_velocity_override: When provided, replaces the DS-computed
                          desired velocity entirely (before CBF/gravity).
                          Used by the path-tube tracking controller to enforce
                          strict path following near obstacles.  None = unused.
        q_goal:           Goal joint configuration for goal-aware CBF margin
                          tapering (GoalAwareCBFConfig).  Passed through to
                          CBFSafetyFilter.filter().  None = no tapering.
        goal_clearance:   Signed obstacle clearance at q_goal (m).  Used as
                          the safety gate for final approach mode — ensures the
                          mode only activates when the goal itself is collision-
                          free.  None = gate disabled (treated as positive).

    Returns:
        ControlResult with torque, desired velocity, and all metrics.
    """
    if config is None:
        config = ControllerConfig()

    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    n = len(q)

    D = _build_D(n, config.d_gain)

    # ---- 1. Get passivity gate from tank --------------------------------
    beta_R = tank.beta_R

    # ---- 2. Compute DS velocity (mode-aware) ----------------------------
    # Use PathDS.f() which handles all escape modes internally.
    fc_raw = ds.f(q, beta_R=0.0, clearance=clearance,
                  mode=ds_mode, escape_target=ds_escape_target)
    # fc_raw = f_c component only (beta_R=0 omits f_R)
    fc = fc_raw
    fr = ds.f_R(q)

    # ---- 3. Phase 5B: orthogonalize + scale f_R → energy-neutral ----------
    if config.orthogonalize_residual:
        fr = orthogonalize_residual(qdot, fr)
    if config.alpha != 1.0:
        fr = config.alpha * fr

    # ---- 4. Passivity filter on f_R (proactive energy constraint) -------
    pf_clipped       = False
    pf_power_nom     = 0.0
    pf_power_filtered = 0.0
    if config.passivity_filter is not None:
        epsilon = tank.epsilon   # state-dependent threshold ε(s)
        pf_result = filter_residual(qdot, fr, epsilon=epsilon,
                                    config=config.passivity_filter)
        fr               = pf_result.fR_filtered
        pf_clipped       = pf_result.clipped
        pf_power_nom     = pf_result.power_nom
        pf_power_filtered = pf_result.power_filtered
    else:
        pf_power_nom = float(qdot @ fr)

    # ---- 5. Null-space term f_n -----------------------------------------
    if config.f_n_gain > 0.0:
        q_center = (
            config.q_center
            if config.q_center is not None
            else 0.5 * (ds.path[0] + ds.x_goal)
        )
        fn = _null_space_term(q, q_center, config.f_n_gain)
    else:
        fn = np.zeros(n)

    # ---- 5b. Final approach: suppress f_R + cap speed near verified-safe goal
    _in_final_approach = False
    if (ds.config.final_approach_enabled
            and q_goal is not None
            and goal_clearance is not None
            and goal_clearance > 0.0):
        _goal_err_fa = float(np.linalg.norm(q - np.asarray(q_goal, dtype=float)))
        if _goal_err_fa < ds.config.approach_radius:
            _in_final_approach = True
            fr = fr * ds.config.residual_scale_near_goal

    # ---- 6. Desired velocity (beta_R is now smooth ∈ [0,1]) -------------
    xdot_d = fc + beta_R * fr + fn
    # Escape velocity override: add external escape command when active
    if escape_velocity is not None:
        xdot_d = xdot_d + np.asarray(escape_velocity, dtype=float)
    # Path-tube override: replaces DS-computed xdot_d entirely when provided.
    # The override still goes through passivity filter (already applied to fr)
    # and the CBF safety layer below, preserving all safety guarantees.
    if nominal_velocity_override is not None:
        xdot_d = np.asarray(nominal_velocity_override, dtype=float)

    # ---- 6a. Final approach speed cap (after combining all terms) -----------
    # Skip if nominal_velocity_override is active — that mode manages its own speed.
    if _in_final_approach and nominal_velocity_override is None:
        _speed = float(np.linalg.norm(xdot_d))
        if _speed > ds.config.approach_max_speed:
            xdot_d = xdot_d * (ds.config.approach_max_speed / _speed)

    # ---- 6b. Task-space modulation (optional) ----------------------------
    mod_diag: Optional["ModulationDiagnostics"] = None
    if config.modulation is not None and config.modulation.enabled and obstacles:
        from src.solver.ds.modulation import apply_modulation
        xdot_d, mod_diag = apply_modulation(q, xdot_d, obstacles, config.modulation)

    # ---- 6c. Hard non-contact safety shield (optional) -------------------
    shield_diag: Optional["HardShieldDiagnostics"] = None
    if config.hard_shield is not None and config.hard_shield.enabled and obstacles:
        from src.solver.controller.hard_shield import enforce_hard_clearance
        xdot_d, shield_diag = enforce_hard_clearance(
            q, xdot_d, dt, obstacles, config.hard_shield
        )

    # ---- 6d. CBF safety filter (optional) --------------------------------
    cbf_diag: Optional["CBFDiagnostics"] = None
    if config.cbf is not None and config.cbf.enabled and obstacles:
        from src.solver.controller.cbf_filter import CBFSafetyFilter
        _cbf = CBFSafetyFilter(config.cbf)
        xdot_d, cbf_diag = _cbf.filter(q, xdot_d, obstacles, phase, q_goal=q_goal)

    # ---- 7. Gravity compensation ----------------------------------------
    if config.gravity_fn is not None:
        G = np.asarray(config.gravity_fn(q), dtype=float)
    else:
        G = np.zeros(n)

    # ---- 8. Control torque  τ = G(q) − D (ẋ − ẋ_d) --------------------
    tau = G - D @ (qdot - xdot_d)

    # ---- 9. Passivity metric  z = ẋᵀ (β_R · f_R) (actual injected power) -
    # Use the same `fr` that entered xdot_d (post-orthogonalize, post-alpha,
    # post-passivity-filter, post-final-approach-scale).  When final approach
    # zeroes fr the tank is not drained.  beta_R matches the gate used above.
    z = float(np.dot(qdot, beta_R * fr))

    # ---- 10. Tank update ------------------------------------------------
    tank.step(z, qdot, D, dt)

    return ControlResult(
        tau=tau,
        xdot_d=xdot_d,
        beta_R=beta_R,
        z=z,
        tank_energy=tank.energy,
        passivity_violated=(z > 0.0),
        V=ds.V(q),
        pf_clipped=pf_clipped,
        pf_power_nom=pf_power_nom,
        pf_power_filtered=pf_power_filtered,
        cbf=cbf_diag,
        modulation=mod_diag,
        hard_shield=shield_diag,
    )


# ---------------------------------------------------------------------------
# Multi-step simulation helper (for testing / benchmarking)
# ---------------------------------------------------------------------------
def simulate(
    q0: np.ndarray,
    qdot0: np.ndarray,
    ds: PathDS,
    tank: EnergyTank,
    dt: float,
    n_steps: int,
    config: Optional[ControllerConfig] = None,
    dynamics_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
    obstacles: Optional[List["Obstacle"]] = None,
    phase: str = "",
    q_goal: Optional[np.ndarray] = None,
    goal_clearance: Optional[float] = None,
) -> List[ControlResult]:
    """
    Simulate the closed-loop system for n_steps.

    By default uses a simple Euler integrator:  q += qdot * dt,
    qdot += tau / m * dt  (unit mass).  Supply ``dynamics_fn(q, qdot, tau)``
    to override with a proper rigid-body model.

    Args:
        q0, qdot0:      Initial joint state.
        ds:             PathDS.
        tank:           EnergyTank (mutated in place).
        dt:             Time step (s).
        n_steps:        Number of steps.
        config:         ControllerConfig.
        dynamics_fn:    Optional (q, qdot, tau) -> qdot_dot (acceleration).
        obstacles:      Obstacle list forwarded to the CBF safety filter.
        phase:          Task phase string forwarded to the CBF safety filter.
        q_goal:         Goal joint configuration forwarded to the CBF safety
                        filter for goal-aware margin tapering.  None = disabled.
        goal_clearance: Signed obstacle clearance at q_goal (m), forwarded to
                        step() as the final approach safety gate.  None = gate
                        disabled (treated as positive).

    Returns:
        List of ControlResult, one per step.
    """
    q = np.asarray(q0, dtype=float).copy()
    qdot = np.asarray(qdot0, dtype=float).copy()
    results: List[ControlResult] = []

    for _ in range(n_steps):
        res = step(q, qdot, ds, tank, dt, config, obstacles=obstacles, phase=phase,
                   q_goal=q_goal, goal_clearance=goal_clearance)
        results.append(res)

        # Integrate forward
        if dynamics_fn is not None:
            qdot_dot = dynamics_fn(q, qdot, res.tau)
        else:
            # Default: unit-mass double integrator
            qdot_dot = res.tau   # tau = acceleration for unit mass

        q = q + qdot * dt
        qdot = qdot + qdot_dot * dt

    return results
