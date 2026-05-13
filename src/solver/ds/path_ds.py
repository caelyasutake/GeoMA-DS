"""
Phase 4: DS Path Generation

Converts a joint-space RRT path into a Dynamical System (DS) velocity field:

    f(x) = f_c(x) + beta_R * f_R(x)

Conservative component (Lyapunov-stable, passivity guaranteed):
    V(x)    = 0.5 * K_c * ||x - x*||^2
    f_c(x)  = -∇V = -K_c * (x - x*)

Residual component (path-following, energy-bounded by the tank):
    f_R(x) = w(x) * [ K_r * τ(x) + K_n * (π(x) - x) ]

Where:
    π(x)  — projection of x onto the nearest path segment
    τ(x)  — unit tangent at π(x)
    w(x)  — goal-proximity weight (→ 0 as x → x*) so f_R vanishes at the goal

beta_R ∈ {0, 1} is set by the energy tank (Phase 5).
The passivity monitor: z(x, ẋ) = ẋᵀ f_R(x).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DSConfig:
    """Gain parameters for the path DS."""

    K_c: float = 2.0    # conservative gain (gradient of Lyapunov); increased for Phase 9
    K_r: float = 1.0    # path-tangent gain in f_R
    K_n: float = 0.5    # path-normal (attraction toward path) gain in f_R
    goal_radius: float = 0.05   # radius inside which w(x) blends to 0 (rad)

    # Proximity-aware f_c scaling: when clearance < near_obs_threshold, scale
    # the conservative component down to c_scale_near_obs to prevent corner-cutting.
    near_obs_threshold: float = 0.06   # clearance below which scaling kicks in (m)
    c_scale_near_obs:   float = 0.25   # minimum f_c scale when very close to obstacle

    # Velocity saturation: clamp the total DS output norm to max_speed (rad/s).
    # Prevents the arm from building up momentum on long paths and overshooting.
    # inf = disabled (backward compatible default).
    max_speed: float = float("inf")

    # Final approach mode: suppress f_R and cap speed near a verified-safe goal
    final_approach_enabled:   bool  = False
    approach_radius:          float = 0.25
    approach_max_speed:       float = 0.3
    residual_scale_near_goal: float = 0.0


# ---------------------------------------------------------------------------
# Segment geometry helpers
# ---------------------------------------------------------------------------
def _project_on_segment(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Project x onto segment [a, b].

    Returns:
        proj:  Closest point on segment.
        t:     Parameter ∈ [0, 1].
        dist:  ||x - proj||.
    """
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-12:
        proj = a.copy()
        return proj, 0.0, float(np.linalg.norm(x - proj))
    t = float(np.clip(np.dot(x - a, ab) / len_sq, 0.0, 1.0))
    proj = a + t * ab
    return proj, t, float(np.linalg.norm(x - proj))


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector; return v unchanged if near-zero."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v.copy()


# ---------------------------------------------------------------------------
# PathDS
# ---------------------------------------------------------------------------
class PathDS:
    """
    Dynamical System built from a joint-space path [q_0, …, q_N].

    Usage::

        ds = PathDS(path, config=DSConfig())
        xdot = ds.f(x)                    # full DS (beta_R = 1)
        xdot = ds.f(x, beta_R=0.5)        # tank-modulated
        fc   = ds.f_c(x)
        fr   = ds.f_R(x)
        v    = ds.V(x)
        z    = ds.passivity_metric(x, xdot)  # ẋᵀ f_R
    """

    def __init__(
        self,
        path: List[np.ndarray],
        config: Optional[DSConfig] = None,
    ) -> None:
        if len(path) < 2:
            raise ValueError("Path must have at least 2 waypoints.")

        self.config = config or DSConfig()
        self.path = [np.asarray(q, dtype=float) for q in path]
        self.x_goal = self.path[-1].copy()
        self._n = len(self.path)

        # Precompute per-segment unit tangents
        self._tangents: List[np.ndarray] = []
        self._seg_lengths: List[float] = []
        for i in range(self._n - 1):
            delta = self.path[i + 1] - self.path[i]
            length = float(np.linalg.norm(delta))
            self._seg_lengths.append(length)
            self._tangents.append(_unit(delta))

        # Cumulative arc length (for diagnostics / progress)
        self.arc_lengths = np.concatenate([[0.0], np.cumsum(self._seg_lengths)])
        self.total_length = float(self.arc_lengths[-1])

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------
    def nearest_on_path(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find the closest point on the path to x.

        Returns:
            proj:       Closest point on path (on a segment), shape (n_joints,).
            tangent:    Unit tangent of that segment, shape (n_joints,).
            seg_idx:    Index of the nearest segment (0 … N-2).
        """
        x = np.asarray(x, dtype=float)
        best_dist = np.inf
        best_proj = self.path[0].copy()
        best_tangent = self._tangents[0].copy()
        best_seg = 0

        for i in range(self._n - 1):
            proj, _, dist = _project_on_segment(x, self.path[i], self.path[i + 1])
            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                best_tangent = self._tangents[i]
                best_seg = i

        return best_proj, best_tangent, best_seg

    def _goal_weight(self, x: np.ndarray) -> float:
        """
        Smooth weight w(x) ∈ [0, 1] that decays to 0 near the goal.

        w(x) = min(1, ||x - x*|| / goal_radius)

        This ensures f_R vanishes at x* so the DS rests at the attractor.
        """
        dist_to_goal = float(np.linalg.norm(x - self.x_goal))
        r = self.config.goal_radius
        return min(1.0, dist_to_goal / r) if r > 0 else 1.0

    def progress(self, x: np.ndarray) -> float:
        """
        Return normalized progress along path ∈ [0, 1].
        0 = at start, 1 = at goal.
        """
        if self.total_length < 1e-12:
            return 1.0
        _, _, seg_idx = self.nearest_on_path(x)
        proj, t, _ = _project_on_segment(
            x, self.path[seg_idx], self.path[seg_idx + 1]
        )
        arc = self.arc_lengths[seg_idx] + t * self._seg_lengths[seg_idx]
        return float(np.clip(arc / self.total_length, 0.0, 1.0))

    # ------------------------------------------------------------------
    # DS components
    # ------------------------------------------------------------------
    def f_c(self, x: np.ndarray) -> np.ndarray:
        """
        Conservative component: f_c(x) = -∇V = -K_c * (x - x*).

        Lyapunov-stable: V̇ = ∇V · f_c = -K_c ||x - x*||² ≤ 0.
        """
        x = np.asarray(x, dtype=float)
        return -self.config.K_c * (x - self.x_goal)

    def f_R(self, x: np.ndarray) -> np.ndarray:
        """
        Residual (path-following) component.

        f_R(x) = w(x) * [ K_r * τ(π(x)) + K_n * (π(x) - x) ]

        The weight w(x) ensures f_R → 0 as x → x* (no residual at attractor).
        Energy injection z = ẋᵀ f_R is monitored by the tank (Phase 5).
        """
        x = np.asarray(x, dtype=float)
        proj, tangent, _ = self.nearest_on_path(x)
        w = self._goal_weight(x)

        tangent_term = self.config.K_r * tangent
        normal_term = self.config.K_n * (proj - x)
        return w * (tangent_term + normal_term)

    def f(
        self,
        x: np.ndarray,
        beta_R: float = 1.0,
        clearance: Optional[float] = None,
        mode: str = "normal",
        escape_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full DS velocity field with optional escape-mode overrides.

        Args:
            x:             Current joint configuration.
            beta_R:        Tank modulation scalar ∈ [0, 1].
            clearance:     Minimum obstacle clearance (m).  When below
                           near_obs_threshold, f_c is scaled down.
            mode:          One of "normal", "reduced_fc", "escape_clearance",
                           "backtrack", "bridge_target".  In non-normal modes
                           f_c is suppressed so the escape policy controls motion.
            escape_target: Joint-space target used in "bridge_target" mode
                           (ignored in other modes).

        Returns:
            Desired joint velocity ẋ_d, shape (n_joints,).
        """
        fc = self.f_c(x)

        # --- Escape-mode overrides ---
        if mode == "reduced_fc":
            # Suppress conservative pull entirely; only path-following remains
            fc = np.zeros_like(fc)
        elif mode in ("escape_clearance", "backtrack"):
            # Suppress f_c so escape policy drives motion without fighting it
            fc = np.zeros_like(fc)
        elif mode == "bridge_target" and escape_target is not None:
            # Replace f_c with attraction to bridge joint target
            fc = -self.config.K_c * (x - np.asarray(escape_target, dtype=float))
        elif clearance is not None and clearance < self.config.near_obs_threshold:
            # Normal proximity-aware scaling
            scale = max(
                self.config.c_scale_near_obs,
                clearance / self.config.near_obs_threshold,
            )
            fc = scale * fc

        v = fc + float(beta_R) * self.f_R(x)
        if self.config.max_speed < float("inf"):
            speed = float(np.linalg.norm(v))
            if speed > self.config.max_speed:
                v = v * (self.config.max_speed / speed)
        return v

    # ------------------------------------------------------------------
    # Lyapunov function
    # ------------------------------------------------------------------
    def V(self, x: np.ndarray) -> float:
        """
        Lyapunov function: V(x) = 0.5 * K_c * ||x - x*||².

        V(x*) = 0, V(x) > 0 for x ≠ x*.
        """
        x = np.asarray(x, dtype=float)
        diff = x - self.x_goal
        return float(0.5 * self.config.K_c * np.dot(diff, diff))

    def V_dot(self, x: np.ndarray, xdot: np.ndarray) -> float:
        """
        Time derivative of V: V̇ = ∇V · ẋ = K_c * (x - x*) · ẋ.

        For x = f_c(x): V̇ = -K_c ||x - x*||² ≤ 0 (stable).
        """
        x = np.asarray(x, dtype=float)
        xdot = np.asarray(xdot, dtype=float)
        return float(self.config.K_c * np.dot(x - self.x_goal, xdot))

    # ------------------------------------------------------------------
    # Passivity metric
    # ------------------------------------------------------------------
    def passivity_metric(self, x: np.ndarray, xdot: np.ndarray) -> float:
        """
        z = ẋᵀ f_R(x).

        Positive  → f_R injects energy into the system (monitored by tank).
        Negative  → f_R dissipates energy (always safe).

        Used by the energy tank in Phase 5.
        """
        x = np.asarray(x, dtype=float)
        xdot = np.asarray(xdot, dtype=float)
        return float(np.dot(xdot, self.f_R(x)))
