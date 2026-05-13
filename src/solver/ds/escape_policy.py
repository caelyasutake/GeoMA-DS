"""
Escape-mode policy for Multi-IK-DS.

When the robot is trapped in a local minimum (center-post conflict), this
module provides escape velocity commands that can temporarily oppose the
nominal DS flow in order to increase clearance and exit the trap basin.

Supported modes
---------------
NORMAL          – standard PathDS, no override
REDUCED_FC      – suppress f_c (goal attraction) entirely; keep f_R only
ESCAPE_CLEARANCE – move in direction of increasing obstacle clearance
BACKTRACK       – retrace recent joint trajectory in reverse
BRIDGE_TARGET   – follow a temporary bridge attractor in joint space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np


class EscapeMode(Enum):
    NORMAL           = "normal"
    REDUCED_FC       = "reduced_fc"
    ESCAPE_CLEARANCE = "escape_clearance"
    BACKTRACK        = "backtrack"
    BRIDGE_TARGET    = "bridge_target"


@dataclass
class EscapePolicyConfig:
    allow_backtracking:          bool  = True
    max_backtrack_steps:         int   = 100
    backtrack_progress_fraction: float = 0.15   # fraction of history to retrace
    clearance_ascent_weight:     float = 1.0
    family_bias_weight:          float = 0.3
    fd_eps:                      float = 1e-4   # finite-difference step for gradient
    escape_clearance_target:     float = 0.06   # clearance to reach before resuming
    max_escape_steps:            int   = 200
    history_len:                 int   = 300     # max joint trajectory steps stored
    backtrack_gain:              float = 2.0     # velocity gain during backtracking
    clearance_gain:              float = 3.0     # velocity gain for clearance ascent
    bridge_gain:                 float = 2.0     # P-gain toward bridge target


class EscapePolicy:
    """
    Stateful escape-mode controller.

    Usage
    -----
    1. Call ``push_trajectory(q)`` every normal execution step.
    2. When a trap is detected, call ``start_escape(mode, ...)``.
    3. Each subsequent step, call ``escape_velocity(q, clearance_fn, fk_fn, q_min, q_max)``
       to get qdot_escape (joint-space).
    4. Call ``is_escaped(clearance)`` to decide when to resume normal mode.
    """

    def __init__(self, config: EscapePolicyConfig | None = None) -> None:
        self.config = config or EscapePolicyConfig()
        self._traj: List[np.ndarray] = []       # rolling joint history
        self._backtrack_ptr: int = 0            # index into reversed history
        self._backtrack_buf: List[np.ndarray] = []
        self.mode: EscapeMode = EscapeMode.NORMAL
        self.escape_steps: int = 0
        self.bridge_target: Optional[np.ndarray] = None  # joint-space bridge goal
        self.preferred_escape_dir: Optional[np.ndarray] = None  # family-bias direction (joint)

    # ------------------------------------------------------------------
    # Trajectory history
    # ------------------------------------------------------------------

    def push_trajectory(self, q: np.ndarray) -> None:
        """Record the current joint config; call every execution step."""
        self._traj.append(q.copy())
        if len(self._traj) > self.config.history_len:
            self._traj.pop(0)

    # ------------------------------------------------------------------
    # Escape activation
    # ------------------------------------------------------------------

    def start_escape(
        self,
        mode: EscapeMode,
        bridge_target: Optional[np.ndarray] = None,
        preferred_escape_dir: Optional[np.ndarray] = None,
    ) -> None:
        """Enter escape mode."""
        self.mode = mode
        self.escape_steps = 0
        self.bridge_target = bridge_target
        self.preferred_escape_dir = preferred_escape_dir

        if mode == EscapeMode.BACKTRACK and self.config.allow_backtracking:
            n_back = max(1, int(len(self._traj) * self.config.backtrack_progress_fraction))
            n_back = min(n_back, self.config.max_backtrack_steps)
            self._backtrack_buf = list(reversed(self._traj[-n_back:]))
            self._backtrack_ptr = 0

    def reset(self) -> None:
        """Return to normal mode."""
        self.mode = EscapeMode.NORMAL
        self.escape_steps = 0
        self.bridge_target = None
        self.preferred_escape_dir = None

    # ------------------------------------------------------------------
    # Velocity computation
    # ------------------------------------------------------------------

    def escape_velocity(
        self,
        q: np.ndarray,
        clearance_fn: Optional[Callable[[np.ndarray], float]] = None,
        n_dof: int = 7,
        q_min: Optional[np.ndarray] = None,
        q_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return an escape joint velocity for the current mode.

        Parameters
        ----------
        q            : current joint config (n_dof,)
        clearance_fn : callable q -> scalar clearance (required for ESCAPE_CLEARANCE)
        n_dof        : robot DOF
        q_min/q_max  : joint limits (used to clamp finite-difference queries)

        Returns
        -------
        qdot_escape : (n_dof,) joint velocity command
        """
        self.escape_steps += 1
        cfg = self.config

        if self.mode == EscapeMode.REDUCED_FC:
            # Zero escape velocity — caller suppresses f_c; f_R still acts
            return np.zeros(n_dof)

        if self.mode == EscapeMode.ESCAPE_CLEARANCE:
            return self._clearance_ascent(q, clearance_fn, n_dof, q_min, q_max)

        if self.mode == EscapeMode.BACKTRACK:
            return self._backtrack_velocity(q, n_dof)

        if self.mode == EscapeMode.BRIDGE_TARGET:
            return self._bridge_velocity(q, n_dof)

        return np.zeros(n_dof)

    # ------------------------------------------------------------------
    # Internal sub-policies
    # ------------------------------------------------------------------

    def _clearance_ascent(
        self,
        q: np.ndarray,
        clearance_fn: Optional[Callable[[np.ndarray], float]],
        n_dof: int,
        q_min: Optional[np.ndarray],
        q_max: Optional[np.ndarray],
    ) -> np.ndarray:
        if clearance_fn is None:
            return np.zeros(n_dof)

        eps = self.config.fd_eps
        grad = np.zeros(n_dof)
        c0 = clearance_fn(q)
        for i in range(n_dof):
            q_p = q.copy()
            q_p[i] += eps
            if q_max is not None:
                q_p[i] = min(q_p[i], q_max[i])
            grad[i] = (clearance_fn(q_p) - c0) / eps

        # Add family-bias component
        if self.preferred_escape_dir is not None:
            w_c = self.config.clearance_ascent_weight
            w_b = self.config.family_bias_weight
            v = w_c * grad + w_b * self.preferred_escape_dir
        else:
            v = grad

        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm * self.config.clearance_gain

        return v

    def _backtrack_velocity(self, q: np.ndarray, n_dof: int) -> np.ndarray:
        if not self._backtrack_buf or self._backtrack_ptr >= len(self._backtrack_buf):
            return np.zeros(n_dof)

        target = self._backtrack_buf[self._backtrack_ptr]
        delta = target - q
        dist = np.linalg.norm(delta)
        if dist < 1e-3:
            self._backtrack_ptr += 1
            if self._backtrack_ptr >= len(self._backtrack_buf):
                return np.zeros(n_dof)
            target = self._backtrack_buf[self._backtrack_ptr]
            delta = target - q
            dist = np.linalg.norm(delta)

        if dist < 1e-9:
            return np.zeros(n_dof)

        return (delta / dist) * self.config.backtrack_gain

    def _bridge_velocity(self, q: np.ndarray, n_dof: int) -> np.ndarray:
        if self.bridge_target is None:
            return np.zeros(n_dof)
        delta = self.bridge_target - q
        norm = np.linalg.norm(delta)
        if norm < 1e-9:
            return np.zeros(n_dof)
        return delta * self.config.bridge_gain  # proportional; caller may clip

    # ------------------------------------------------------------------
    # Exit condition
    # ------------------------------------------------------------------

    def is_escaped(self, current_clearance: float) -> bool:
        """Return True when clearance is sufficient to resume normal mode."""
        if self.mode == EscapeMode.BACKTRACK:
            # Done when we've replayed all backtrack targets
            return self._backtrack_ptr >= len(self._backtrack_buf)
        return current_clearance >= self.config.escape_clearance_target

    def budget_exhausted(self) -> bool:
        return self.escape_steps >= self.config.max_escape_steps
