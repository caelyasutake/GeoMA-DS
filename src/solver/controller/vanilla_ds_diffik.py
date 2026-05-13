"""
Vanilla DS + Differential IK + Modulation controller.

The simplest possible task-space reactive controller:

    xdot_nom  = -Kp * (x_ee - x_goal)          # task-space DS attractor
    xdot_mod  = M(x_ee) @ xdot_nom              # modulation avoidance
    qdot_des  = J^+ @ xdot_mod + N @ qdot_null  # damped diff IK + null-space

Intentionally minimal — no global planner, no PathDS, no family switching.
May get stuck in local minima. That is acceptable; it is the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.solver.ds.modulation import (
    ModulationConfig, ModulationDiagnostics, combined_modulation,
)
from src.scenarios.scenario_schema import Obstacle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VanillaDSConfig:
    """Task-space DS attractor parameters."""
    pos_gain:              float = 2.0    # Kp: xdot = -Kp * (x - x_goal)
    max_task_speed:        float = 0.25   # m/s — clip xdot before diff IK
    goal_tolerance:        float = 0.01   # m — declare convergence


@dataclass
class DiffIKConfig:
    """Damped differential IK mapping parameters."""
    damping:               float = 1e-2   # regularisation λ in (JJ^T + λI)
    max_joint_speed:       float = 1.0    # rad/s — clip qdot after IK
    use_nullspace_posture: bool  = True
    nullspace_gain:        float = 0.05   # k_null in qdot_null = -k * (q - q_nom)


@dataclass
class VanillaDSDiffIKDiagnostics:
    """Per-step diagnostics returned by the controller."""
    ee_pos:              np.ndarray  = field(default_factory=lambda: np.zeros(3))
    goal_error_m:        float       = 0.0   # ||x_ee - x_goal||
    xdot_nom_norm:       float       = 0.0   # ||xdot_nom||
    xdot_mod_norm:       float       = 0.0   # ||xdot_mod||
    qdot_norm:           float       = 0.0   # ||qdot_des||
    jacobian_cond:       float       = 0.0   # condition number of J (3x7 pos block)
    goal_reached:        bool        = False
    modulation:          ModulationDiagnostics = field(
                             default_factory=ModulationDiagnostics)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class VanillaDSDiffIKController:
    """
    One-step controller: task-space DS → modulation → damped diff IK.

    Parameters
    ----------
    ds_config:         VanillaDSConfig
    diffik_config:     DiffIKConfig
    modulation_config: ModulationConfig  (None = no modulation)
    q_nominal:         Preferred posture for null-space regularisation.
                       Defaults to q_start at first call if not provided.
    """

    def __init__(
        self,
        ds_config:         Optional[VanillaDSConfig]   = None,
        diffik_config:     Optional[DiffIKConfig]      = None,
        modulation_config: Optional[ModulationConfig]  = None,
        q_nominal:         Optional[np.ndarray]        = None,
    ):
        self.ds      = ds_config      or VanillaDSConfig()
        self.diffik  = diffik_config  or DiffIKConfig()
        self.mod_cfg = modulation_config
        self._q_nominal = q_nominal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        q:          np.ndarray,
        x_goal:     np.ndarray,
        J:          np.ndarray,         # (3, n_joints) — position Jacobian
        x_ee:       np.ndarray,         # (3,) — current EE position
        obstacles:  Optional[List[Obstacle]] = None,
    ) -> Tuple[np.ndarray, VanillaDSDiffIKDiagnostics]:
        """
        Compute desired joint velocity for one control step.

        Args:
            q:         Current joint configuration, shape (n_joints,).
            x_goal:    Target EE position, shape (3,).
            J:         Position Jacobian dEE/dq, shape (3, n_joints).
            x_ee:      Current EE world position, shape (3,).
            obstacles: Collision-enabled obstacles for modulation.

        Returns:
            qdot_des:  Desired joint velocity, shape (n_joints,).
            diag:      Per-step diagnostics.
        """
        q      = np.asarray(q,      dtype=float)
        x_goal = np.asarray(x_goal, dtype=float)
        x_ee   = np.asarray(x_ee,   dtype=float)
        n      = len(q)

        # Store first q as nominal posture if not given
        if self._q_nominal is None:
            self._q_nominal = q.copy()

        diag = VanillaDSDiffIKDiagnostics(ee_pos=x_ee.copy())

        # ---- 1. Task-space DS attractor ----------------------------------
        err         = x_ee - x_goal
        goal_err    = float(np.linalg.norm(err))
        diag.goal_error_m = goal_err
        diag.goal_reached = goal_err < self.ds.goal_tolerance

        xdot_nom = -self.ds.pos_gain * err
        speed    = float(np.linalg.norm(xdot_nom))
        if speed > self.ds.max_task_speed:
            xdot_nom = xdot_nom * (self.ds.max_task_speed / speed)
        diag.xdot_nom_norm = float(np.linalg.norm(xdot_nom))

        # ---- 2. Modulation obstacle avoidance ----------------------------
        mod_diag = ModulationDiagnostics()
        xdot_mod = xdot_nom.copy()
        if self.mod_cfg is not None and obstacles:
            col_obs = [o for o in obstacles if o.collision_enabled]
            if col_obs:
                M, mod_diag = combined_modulation(x_ee, col_obs, self.mod_cfg)
                xdot_mod = M @ xdot_nom
        diag.modulation    = mod_diag
        diag.xdot_mod_norm = float(np.linalg.norm(xdot_mod))

        # ---- 3. Damped differential IK -----------------------------------
        lam    = self.diffik.damping
        JJT    = J @ J.T + lam * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)  # (n, 3)

        qdot_des = J_pinv @ xdot_mod

        # ---- 4. Null-space posture regularisation ------------------------
        if self.diffik.use_nullspace_posture and self._q_nominal is not None:
            N        = np.eye(n) - J_pinv @ J
            qdot_null = -self.diffik.nullspace_gain * (q - self._q_nominal)
            qdot_des  = qdot_des + N @ qdot_null

        # ---- 5. Joint speed clipping ------------------------------------
        jspeed = float(np.linalg.norm(qdot_des))
        if jspeed > self.diffik.max_joint_speed:
            qdot_des = qdot_des * (self.diffik.max_joint_speed / jspeed)

        diag.qdot_norm     = float(np.linalg.norm(qdot_des))
        diag.jacobian_cond = float(np.linalg.cond(J)) if J.shape[0] <= J.shape[1] else 0.0

        return qdot_des, diag
