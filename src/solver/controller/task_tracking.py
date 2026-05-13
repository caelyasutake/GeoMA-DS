"""
Task-Space Tracking Controller for Contact-Circle Trajectories

Implements a task-space impedance controller that explicitly tracks a
Cartesian circle reference while regulating contact force along the
surface normal.

Control law
-----------
The desired EE velocity is decomposed into:

    Tangential (tracking the circle):
        xdot_t = P_t @ (xdot_d − K_p·e_x − K_v·e_v)

    Normal (contact force regulation):
        xdot_n = P_n @ (K_f·(F_n − F_desired)·n)
        Sign: F_n < F_desired → negative along n → EE moves toward surface ✓

    Combined:
        xdot_des = xdot_t + alpha_contact · xdot_n

Where:
    P_t = I − n·nᵀ     tangential projector (removes normal component)
    P_n = n·nᵀ         normal projector
    e_x = ee_pos − x_d  position error
    e_v = J·qdot − xdot_d  velocity error
    F_n = F_contact · n     measured normal force

Joint torques:
    qdot_des = J_pinv @ xdot_des          (damped pseudoinverse)
    tau = G(q) − D·(qdot − qdot_des)     (gravity comp + damping)

Passivity enforcement
---------------------
The error-correction term (residual) is filtered before being added to
the conservative (feed-forward circle velocity) component.  The energy
tank gates the residual when depleted — the same pattern as impedance.py.

    qdot_conservative = J_pinv @ P_t @ xdot_d      (feed-forward only)
    qdot_residual     = qdot_des − qdot_conservative (correction term)

The passivity filter is applied to qdot_residual; beta_R gates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from src.solver.ds.contact_ds import CircleReference
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import (
    PassivityFilterConfig,
    filter_residual,
    orthogonalize_residual,
)

if TYPE_CHECKING:
    from src.scenarios.scenario_schema import Obstacle
    from src.solver.controller.cbf_filter import CBFConfig, CBFDiagnostics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TaskTrackingConfig:
    """
    Parameters for the task-space tracking controller.

    Attributes:
        K_p:          Position error gain (task space, 1/s).
        K_v:          Velocity error gain (task space, dimensionless).
        K_f:          Contact force regulation gain (m/(s·N)).
        F_desired:    Desired normal contact force (N, positive = into surface).
        alpha_contact: Blend weight [0,1] for the normal contact term.
                      1.0 = full normal regulation; 0.0 = no normal term.
        damping:      Joint-space damping gain (N·m·s/rad).
        lambda_reg:   Damped pseudoinverse regularisation (prevents singularity).
        gravity_fn:   Callable q → tau_g (7,).  None → zero gravity comp.
        orthogonalize_residual: Remove velocity-aligned component from qdot_res.
        alpha:        Scale factor on qdot_residual before passivity filter.
    """

    K_p:          float = 50.0
    K_v:          float = 10.0
    K_f:          float = 0.0    # Jacobian-transpose force gain (set to 0 by default).
                                  # K_f > 0 enables feedback: tau += Jᵀ·K_f·(F_n−F_des)·n.
                                  # WARNING: K_f > 0 causes bouncing on rigid MuJoCo
                                  # contacts because the constraint solver reports F_n≈50N
                                  # (arm weight), generating a large upward impulse that
                                  # lifts the arm off the wall repeatedly → NaN QACC.
                                  # Use F_contact_bias instead for stable contact.
    F_desired:    float = 3.0
    F_contact_bias: float = 0.0  # Constant downward bias force (N) applied via Jᵀ.
                                  # tau_bias = Jᵀ·(−F_contact_bias·n).  With n=[0,0,1]
                                  # this is a steady −z push into the surface.
                                  # At equilibrium the wall reaction = F_contact_bias N,
                                  # so contact_mag ≈ F_contact_bias > 0.  Use 1–4 N.
    alpha_contact: float = 0.8
    damping:      float = 5.0
    lambda_reg:   float = 0.01
    gravity_fn:   Optional[Callable[[np.ndarray], np.ndarray]] = None
    orthogonalize_residual: bool = True
    alpha:        float = 0.5
    xdot_max:     float = 0.3    # max EE speed (m/s) — prevents huge joint torques
    # Separate per-direction saturations (applied before combining).
    # v_tan_max limits circle-tracking speed; v_norm_max limits contact-approach
    # speed and is applied INDEPENDENTLY so the normal drive is never starved by
    # a large tangential command.
    v_tan_max:    float = 0.25   # max tangential component (m/s)
    v_norm_max:   float = 0.20   # max normal component (m/s)
    tau_limit:    float = 87.0   # per-joint torque clamp (N·m) — prevents NaN in QACC

    # Optional CBF safety filter.  None → disabled (backward compatible).
    cbf: Optional["CBFConfig"] = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class TaskControlResult:
    """
    Output of one task-space controller step.

    Attributes:
        tau:          Joint torques (7,).
        tank_energy:  Tank energy s after this step.
        beta_R:       Passivity gate ∈ [0,1].
        z:            Passivity metric qdot·qdot_residual (pre-filter).
        pf_clipped:   True if the passivity filter projected the residual.
        pf_power_nom: qdot·qdot_res before filter.
        qdot_des:     Full desired joint velocity (7,).
        xdot_des:     Desired EE velocity in task space (3,).
    """

    tau:          np.ndarray   # (7,)
    tank_energy:  float
    beta_R:       float
    z:            float
    pf_clipped:   bool
    pf_power_nom: float
    qdot_des:     np.ndarray   # (7,)
    xdot_des:     np.ndarray   # (3,)
    # CBF filter diagnostics (None when CBF is disabled)
    cbf: Optional["CBFDiagnostics"] = None


# ---------------------------------------------------------------------------
# Damped pseudoinverse
# ---------------------------------------------------------------------------
def _damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """
    Damped pseudoinverse  J† = Jᵀ (J Jᵀ + λ I)⁻¹.

    Args:
        J:   Linear Jacobian (3, n).
        lam: Regularisation parameter λ > 0.

    Returns:
        J†, shape (n, 3).
    """
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam * np.eye(m))


# ---------------------------------------------------------------------------
# Primary step function
# ---------------------------------------------------------------------------
def task_space_step(
    q:          np.ndarray,       # (7,)  current joint config
    qdot:       np.ndarray,       # (7,)  current joint velocity
    ee_pos:     np.ndarray,       # (3,)  current EE position
    ref:        CircleReference,  # from circle_on_plane_reference
    F_contact:  np.ndarray,       # (3,)  contact force (world frame)
    jacobian:   np.ndarray,       # (6,7) full geometric Jacobian
    tank:       EnergyTank,
    dt:         float,
    config:     Optional[TaskTrackingConfig] = None,
    passivity_filter_cfg: Optional[PassivityFilterConfig] = None,
    obstacles:  Optional[List["Obstacle"]] = None,
    phase:      str = "",
) -> TaskControlResult:
    """
    Execute one task-space tracking controller step.

    Args:
        q:                    Current joint configuration (7,).
        qdot:                 Current joint velocity (7,).
        ee_pos:               Current EE Cartesian position (3,).
        ref:                  Circle reference at current time t.
        F_contact:            Resultant contact force in world frame (3,).
        jacobian:             Full (6,7) geometric Jacobian from env.jacobian(q).
        tank:                 EnergyTank instance (mutated in place).
        dt:                   Control timestep (s).
        config:               TaskTrackingConfig; safe defaults when None.
        passivity_filter_cfg: PassivityFilterConfig; safe defaults when None.
        obstacles:            Obstacle list for CBF safety filter.
        phase:                Task phase string (for phase_dependent contact mode).

    Returns:
        TaskControlResult with torques and diagnostics.
    """
    if config is None:
        config = TaskTrackingConfig()
    if passivity_filter_cfg is None:
        passivity_filter_cfg = PassivityFilterConfig()

    q    = np.asarray(q,    dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    n    = len(q)

    # ---- Surface frame projectors ----------------------------------------
    normal = ref.normal           # (3,) unit surface normal
    P_t = np.eye(3) - np.outer(normal, normal)   # tangential projector
    P_n = np.outer(normal, normal)               # normal projector

    # ---- Linear Jacobian (position rows only) ----------------------------
    J_lin = jacobian[:3, :]       # (3, 7)

    # ---- Tracking errors --------------------------------------------------
    e_x = ee_pos - ref.x_d                    # (3,) position error
    ee_vel = J_lin @ qdot                      # (3,) current EE velocity
    e_v = ee_vel - ref.xdot_d                 # (3,) velocity error

    # ---- Tangential tracking: follow circle (velocity-based) --------------
    xdot_tangential = P_t @ (ref.xdot_d - config.K_p * e_x - config.K_v * e_v)

    # Saturate tangential speed to prevent huge joint torques.
    if config.v_tan_max > 0.0:
        tan_speed = float(np.linalg.norm(xdot_tangential))
        if tan_speed > config.v_tan_max:
            xdot_tangential = xdot_tangential * (config.v_tan_max / tan_speed)
    if config.xdot_max > 0.0:
        speed = float(np.linalg.norm(xdot_tangential))
        if speed > config.xdot_max:
            xdot_tangential = xdot_tangential * (config.xdot_max / speed)

    # ---- Desired EE velocity = tangential only ----------------------------
    # Normal (contact) is handled via direct Jacobian-transpose force below,
    # NOT through velocity tracking.  Velocity-based normal control bounces
    # on rigid contacts because any small penetration creates a large impulse.
    xdot_des = xdot_tangential   # (3,) — purely tangential

    # ---- Map to joint space via damped pseudoinverse ---------------------
    J_pinv = _damped_pinv(J_lin, config.lambda_reg)   # (7, 3)
    qdot_des = J_pinv @ xdot_des                       # (7,)

    # ---- Passivity: split into conservative and residual ------------------
    # Conservative: pure feed-forward circle velocity (no error correction)
    xdot_conservative = P_t @ ref.xdot_d
    qdot_conservative = J_pinv @ xdot_conservative    # (7,)

    # Residual: the error-correction term (tangential only)
    qdot_residual = qdot_des - qdot_conservative      # (7,)

    # ---- Normal contact regulation via Jacobian-transpose force ----------
    # Two mechanisms are available (can be combined):
    #
    # 1. Feedback (K_f > 0):
    #      τ_fb = Jᵀ · alpha_contact · K_f · (F_n − F_desired) · n
    #    When F_n < F_desired → pushes into surface (−z). ✓
    #    When F_n > F_desired → lifts away (+z). ✓
    #    CAUTION: rigid MuJoCo contacts report F_n ≈ 50 N (arm weight), so
    #    K_f as small as 2 produces 94 N upward impulse → bouncing → NaN QACC.
    #    Only safe with very small K_f (< 0.05) or compliant contact models.
    #
    # 2. Constant bias (F_contact_bias > 0):
    #      τ_bias = Jᵀ · (−F_contact_bias · n)
    #    Steady downward push.  At equilibrium wall reaction = F_contact_bias N,
    #    so contact_mag ≈ F_contact_bias.  No feedback → no bouncing. ✓
    #    Recommended: 1–4 N.
    F_n = float(np.dot(F_contact, normal))    # measured normal force
    tau_contact = J_lin.T @ (
        config.alpha_contact * config.K_f * (F_n - config.F_desired) * normal
        - config.F_contact_bias * normal
    )    # (7,)

    # ---- Optional orthogonalise + scale residual -------------------------
    if config.orthogonalize_residual:
        qdot_residual = orthogonalize_residual(qdot, qdot_residual)
    if config.alpha != 1.0:
        qdot_residual = config.alpha * qdot_residual

    # ---- Passivity filter on residual ------------------------------------
    beta_R = tank.beta_R
    epsilon = tank.epsilon

    pf_result = filter_residual(
        qdot, qdot_residual,
        epsilon=epsilon,
        config=passivity_filter_cfg,
    )
    qdot_residual_filtered = pf_result.fR_filtered
    pf_clipped   = pf_result.clipped
    pf_power_nom = pf_result.power_nom

    # ---- Desired velocity: conservative + gated residual -----------------
    qdot_des_final = qdot_conservative + beta_R * qdot_residual_filtered

    # ---- CBF safety filter (optional) ------------------------------------
    cbf_diag: Optional["CBFDiagnostics"] = None
    if config.cbf is not None and config.cbf.enabled and obstacles:
        from src.solver.controller.cbf_filter import CBFSafetyFilter
        _cbf = CBFSafetyFilter(config.cbf)
        qdot_des_final, cbf_diag = _cbf.filter(q, qdot_des_final, obstacles, phase)

    # ---- Damping matrix --------------------------------------------------
    D = config.damping * np.eye(n)

    # ---- Gravity compensation --------------------------------------------
    if config.gravity_fn is not None:
        G = np.asarray(config.gravity_fn(q), dtype=float)
    else:
        G = np.zeros(n)

    # ---- Control torque: gravity comp + direct contact force + velocity damping
    tau = G + tau_contact - D @ (qdot - qdot_des_final)

    # ---- Clamp torques to prevent NaN propagation through MuJoCo QACC ----
    if config.tau_limit > 0.0:
        tau = np.clip(tau, -config.tau_limit, config.tau_limit)

    # ---- Passivity metric z = qdot · qdot_residual (pre-filter) ---------
    z = float(qdot @ qdot_residual)

    # ---- Tank update -----------------------------------------------------
    tank.step(z, qdot, D, dt)

    return TaskControlResult(
        tau=tau,
        tank_energy=tank.energy,
        beta_R=beta_R,
        z=z,
        pf_clipped=pf_clipped,
        pf_power_nom=pf_power_nom,
        qdot_des=qdot_des_final,
        xdot_des=xdot_des,
        cbf=cbf_diag,
    )
