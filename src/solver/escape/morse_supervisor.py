"""Slow-rate supervisor for FastMorseEscapeController.

Runs every control step but refreshes expensive cached quantities only every
update_period_steps. Activates on trap detection; escalates to existing
EscapePolicy after max_reactive_escape_steps with no progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.evaluation.stall_detection import TrapDiagnostics
from src.solver.ik.goal_selection import IKGoalInfo


@dataclass
class MorseSupervisorConfig:
    enabled: bool = True
    update_period_steps: int = 10
    activate_on_trap: bool = True
    deactivate_when_trap_clears: bool = True

    # Escalation thresholds
    max_reactive_escape_steps: int = 100
    min_progress_after_steps: int = 20
    min_clearance_gain_required: float = 0.005
    min_goal_progress_required: float = 0.005

    # Deactivate when goal is close
    goal_error_deactivate_threshold: float = 0.1


@dataclass
class MorseSupervisorState:
    active: bool = False
    activation_step: Optional[int] = None
    last_update_step: int = -1
    active_source_tag: Optional[str] = None
    cached_ik_target: Optional[np.ndarray] = None
    cached_tangent_dirs: List[np.ndarray] = field(default_factory=list)
    cached_backtrack_dir: Optional[np.ndarray] = None
    steps_active: int = 0
    clearance_at_activation: Optional[float] = None
    goal_error_at_activation: Optional[float] = None


class MorseEscapeSupervisor:
    def __init__(self, config: MorseSupervisorConfig) -> None:
        self.config = config
        self.state = MorseSupervisorState()
        self._last_clearance: float = 0.0
        self._last_goal_error: float = float("inf")
        self._q_history: List[np.ndarray] = []
        self._escalate_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        q: np.ndarray,
        trap_diag: TrapDiagnostics,
        clearance_fn: Callable[[np.ndarray], float],
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        ik_infos: List[IKGoalInfo],
        goal_error: float,
    ) -> None:
        """Call once per control step inside the stall-detection block."""
        if not self.config.enabled:
            return

        # Track recent positions for backtrack direction
        self._q_history.append(q.copy())
        if len(self._q_history) > 20:
            self._q_history.pop(0)

        self._last_clearance = clearance_fn(q)
        self._last_goal_error = goal_error

        if not self.state.active:
            if self.config.activate_on_trap and trap_diag.trapped:
                self.state.active = True
                self.state.activation_step = step
                self.state.steps_active = 0
                self.state.clearance_at_activation = self._last_clearance
                self.state.goal_error_at_activation = goal_error
                self._escalate_requested = False
                self._refresh_cache(step, q, jacobian_fn, clearance_fn, ik_infos)
            return

        # Already active — update counters and check exit conditions
        self.state.steps_active += 1

        if goal_error < self.config.goal_error_deactivate_threshold:
            self.deactivate("goal_reached")
            return

        if self.config.deactivate_when_trap_clears and not trap_diag.trapped:
            self.deactivate("trap_cleared")
            return

        # Escalation check
        if self.state.steps_active >= self.config.min_progress_after_steps:
            cl_gain   = self._last_clearance  - (self.state.clearance_at_activation  or 0.0)
            goal_gain = (self.state.goal_error_at_activation or float("inf")) - self._last_goal_error
            if (cl_gain   < self.config.min_clearance_gain_required and
                    goal_gain < self.config.min_goal_progress_required):
                self._escalate_requested = True

        if self.state.steps_active >= self.config.max_reactive_escape_steps:
            self._escalate_requested = True

        # Periodic cache refresh
        if step - self.state.last_update_step >= self.config.update_period_steps:
            self._refresh_cache(step, q, jacobian_fn, clearance_fn, ik_infos)

    def deactivate(self, reason: str = "") -> None:
        """Deactivate the supervisor. reason is stored in active_source_tag."""
        self.state.active = False
        self.state.active_source_tag = reason
        self._escalate_requested = False

    def should_escalate(self) -> bool:
        """True when reactive escape has failed and we must hand off to fallback."""
        return self._escalate_requested and self.state.active

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_cache(
        self,
        step: int,
        q: np.ndarray,
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        clearance_fn: Callable[[np.ndarray], float],
        ik_infos: List[IKGoalInfo],
    ) -> None:
        self.state.last_update_step = step

        # Best IK target (lowest score_prior = best quality)
        if ik_infos:
            best = min(ik_infos, key=lambda x: x.score_prior)
            self.state.cached_ik_target = best.q_goal.copy()

        # Obstacle tangent directions via clearance gradient in task space
        J = jacobian_fn(q)
        J_pinv = np.linalg.pinv(J)
        eps = 1e-4
        c0 = clearance_fn(q)
        grad_q = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            q_p = q.copy()
            q_p[i] += eps
            grad_q[i] = (clearance_fn(q_p) - c0) / eps

        # Chain rule: task-space gradient = J_pinv^T @ joint-space gradient
        normal_task = (J_pinv.T @ grad_q)[:3]
        norm = float(np.linalg.norm(normal_task))
        if norm > 1e-9:
            normal_task = normal_task / norm
            arb = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(normal_task, arb))) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            t1 = np.cross(normal_task, arb)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(normal_task, t1)
            tangents: List[np.ndarray] = []
            for t in [t1, -t1, t2, -t2]:
                t6 = np.zeros(6)
                t6[:3] = t
                dq = J_pinv @ t6
                n = float(np.linalg.norm(dq))
                if n > 1e-9:
                    tangents.append(dq / n)
            self.state.cached_tangent_dirs = tangents
        else:
            self.state.cached_tangent_dirs = []

        # Backtrack direction: point from current position toward oldest tracked
        if len(self._q_history) >= 2:
            d = self._q_history[0] - self._q_history[-1]
            n = float(np.linalg.norm(d))
            self.state.cached_backtrack_dir = (d / n) if n > 1e-9 else None
        else:
            self.state.cached_backtrack_dir = None
