"""Fast online Morse escape controller — designed for 300–500 Hz servo loop.

Uses one-step direction scoring only. Forbidden slow-path symbols are never
imported or called: the rollout scorer, the full Morse planner, Lanczos
eigenvector methods, and BiRRT are all excluded from this module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.solver.ik.goal_selection import IKGoalInfo


@dataclass
class FastMorseEscapeConfig:
    enabled: bool = True

    # Timing guard
    max_compute_time_ms: float = 2.0

    # Candidate sources
    max_candidates: int = 8
    use_ds_direction: bool = True
    use_clearance_gradient: bool = True
    use_obstacle_tangents: bool = True
    use_ik_nullspace: bool = True
    use_backtrack: bool = True
    use_joint_basis: bool = False
    use_random: bool = False

    # Control output
    escape_speed: float = 0.25
    max_qdot_norm: float = 0.5
    smoothing_alpha: float = 0.7

    # One-step prediction
    prediction_dt: float = 0.01
    finite_diff_eps: float = 1e-3

    # Scoring weights
    w_clearance_gain: float = 5.0
    w_goal_progress: float = 2.0
    w_joint_limit: float = 1.0
    w_motion: float = 0.05
    w_collision: float = 1000.0

    # Safety
    collision_clearance_threshold: float = 0.0


@dataclass
class FastEscapeCandidate:
    direction: np.ndarray       # unit vector in joint space (n_dof,)
    source_tag: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FastEscapeResult:
    active: bool
    qdot_escape: Optional[np.ndarray]
    selected_source_tag: Optional[str]
    selected_score: Optional[float]
    n_candidates: int
    compute_time_ms: float
    diagnostics: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


def _joint_limit_cost(q: np.ndarray) -> float:
    """Soft penalty for joint configs outside [-2.9, 2.9] rad (Panda range)."""
    lo = np.maximum(0.0, -2.9 - q)
    hi = np.maximum(0.0, q - 2.9)
    return float(np.sum(lo ** 2 + hi ** 2))


class FastMorseEscapeController:
    def __init__(self, config: FastMorseEscapeConfig) -> None:
        self.config = config
        self._prev_qdot: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Call when supervisor deactivates to clear smoothing state."""
        self._prev_qdot = None

    def compute_qdot(
        self,
        q: np.ndarray,
        ds_qdot: Optional[np.ndarray],
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        clearance_fn: Callable[[np.ndarray], float],
        goal_error_fn: Callable[[np.ndarray], float],
        ik_infos: List[IKGoalInfo],
        cached_tangent_dirs: List[np.ndarray],
        backtrack_dir: Optional[np.ndarray],
    ) -> FastEscapeResult:
        """Compute one escape joint velocity within the compute-time budget.

        Returns FastEscapeResult with active=False if disabled, no safe
        candidate found, or clearance is already below threshold.
        """
        t0 = time.perf_counter()
        cfg = self.config

        if not cfg.enabled:
            return FastEscapeResult(
                active=False, qdot_escape=None, selected_source_tag=None,
                selected_score=None, n_candidates=0,
                compute_time_ms=0.0, diagnostics={},
            )

        c0 = clearance_fn(q)
        e0 = goal_error_fn(q)

        # Build cheap candidate set (no rollouts)
        candidates: List[FastEscapeCandidate] = []
        J = jacobian_fn(q)
        J_pinv = np.linalg.pinv(J)
        N = np.eye(q.shape[0]) - J_pinv @ J   # nullspace projector

        if cfg.use_ds_direction and ds_qdot is not None:
            d = _unit(np.asarray(ds_qdot, dtype=float))
            if d is not None:
                candidates.append(FastEscapeCandidate(d, "ds"))

        if cfg.use_clearance_gradient:
            grad = np.zeros(q.shape[0])
            for idx in range(q.shape[0]):
                q_p = q.copy()
                q_p[idx] += cfg.finite_diff_eps
                grad[idx] = (clearance_fn(q_p) - c0) / cfg.finite_diff_eps
            d = _unit(grad)
            if d is not None:
                candidates.append(FastEscapeCandidate(d, "clearance_grad"))

        if cfg.use_ik_nullspace and ik_infos:
            # score_prior: lower = better reachability; min() picks the best escape target
            best_ik = min(ik_infos, key=lambda x: x.score_prior)
            raw = best_ik.q_goal - q
            d_ns = _unit(N @ raw)
            d = d_ns if d_ns is not None else _unit(raw)
            if d is not None:
                candidates.append(FastEscapeCandidate(
                    d, "nullspace_ik",
                    metadata={"family": best_ik.family_label},
                ))

        if cfg.use_obstacle_tangents:
            for t in cached_tangent_dirs[:4]:
                d = _unit(np.asarray(t, dtype=float))
                if d is not None:
                    candidates.append(FastEscapeCandidate(d, "tangent"))

        if cfg.use_backtrack and backtrack_dir is not None:
            d = _unit(np.asarray(backtrack_dir, dtype=float))
            if d is not None:
                candidates.append(FastEscapeCandidate(d, "backtrack"))

        candidates = candidates[: cfg.max_candidates]

        # One-step scoring: predict q_pred and evaluate metrics
        best_score = -float("inf")
        best_cand: Optional[FastEscapeCandidate] = None

        for cand in candidates:
            q_pred = q + cfg.prediction_dt * cfg.escape_speed * cand.direction
            c_pred = clearance_fn(q_pred)

            if c_pred < cfg.collision_clearance_threshold:
                continue   # reject unsafe

            clearance_gain = c_pred - c0
            goal_progress  = e0 - goal_error_fn(q_pred)
            jl_cost        = _joint_limit_cost(q_pred)
            motion_cost    = float(np.dot(cand.direction, cand.direction))

            score = (
                cfg.w_clearance_gain * clearance_gain
                + cfg.w_goal_progress * goal_progress
                - cfg.w_joint_limit  * jl_cost
                - cfg.w_motion       * motion_cost
            )
            if score > best_score:
                best_score = score
                best_cand  = cand

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if best_cand is None:
            return FastEscapeResult(
                active=False, qdot_escape=None, selected_source_tag=None,
                selected_score=None, n_candidates=len(candidates),
                compute_time_ms=elapsed_ms, diagnostics={"c0": c0},
            )

        # Build output velocity with norm clamping and temporal smoothing
        qdot = cfg.escape_speed * best_cand.direction
        norm = float(np.linalg.norm(qdot))
        if norm > cfg.max_qdot_norm:
            qdot = qdot * (cfg.max_qdot_norm / norm)

        if self._prev_qdot is not None and self._prev_qdot.shape == qdot.shape:
            qdot = cfg.smoothing_alpha * self._prev_qdot + (1.0 - cfg.smoothing_alpha) * qdot
        self._prev_qdot = qdot.copy()

        return FastEscapeResult(
            active=True,
            qdot_escape=qdot,
            selected_source_tag=best_cand.source_tag,
            selected_score=best_score,
            n_candidates=len(candidates),
            compute_time_ms=elapsed_ms,
            diagnostics={"c0": c0, "e0": e0},
        )
