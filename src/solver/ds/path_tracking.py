"""
Path-tube tracking controller for Multi-IK-DS.

Provides a joint-space velocity that keeps the robot inside a corridor around
the planned Bi-RRT path.  Used as the primary nominal velocity when obstacle
clearance is low, replacing the ordinary PathDS output.

Control law
-----------
    qdot_path = k_t * t_hat + k_p * (q_proj - q)

where:
    t_hat  = unit tangent of the nearest path segment (forward progress)
    q_proj = nearest point on the path (attraction back into tube)

The anchor-point version advances a waypoint anchor and pulls toward it:
    qdot_path = k_t * normalize(q_anchor_next - q_anchor) + k_p * (q_anchor - q)

Hysteresis prevents anchor from jumping forward faster than the robot
moves, which would recreate the shortcutting problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PathTubeConfig:
    """Parameters for the path-tube tracking controller."""
    enabled:                bool  = True
    switch_clearance:       float = 0.08   # enter path-tube mode below this (m)
    hard_switch_clearance:  float = 0.05   # DS fully suppressed below this (m)
    exit_clearance:         float = 0.10   # hysteretic exit: resume DS above this (m)
    tube_radius_q:          float = 0.20   # max allowed deviation from path (rad)
    tangent_gain:           float = 1.0    # k_t: forward progress weight
    path_attract_gain:      float = 2.0    # k_p: attraction back toward path
    anchor_advance_dist:    float = 0.08   # advance anchor only when this close (rad)
    max_anchor_jump:        int   = 3      # max waypoints to jump per step
    progress_hysteresis_steps: int = 20   # steps before anchor can advance again
    debug:                  bool  = False  # enable per-step console logging


# ---------------------------------------------------------------------------
# Pure geometry helpers (stateless)
# ---------------------------------------------------------------------------

def nearest_path_index(q: np.ndarray, path: List[np.ndarray]) -> int:
    """
    Return the index of the nearest waypoint in ``path`` to joint config ``q``.
    """
    q = np.asarray(q, dtype=float)
    best_idx, best_dist = 0, float("inf")
    for i, wp in enumerate(path):
        d = float(np.linalg.norm(q - wp))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def nearest_path_segment(
    q: np.ndarray,
    path: List[np.ndarray],
) -> Tuple[int, int]:
    """
    Return (i, i+1) indices of the nearest segment in ``path`` to ``q``.

    Finds the segment whose closest point to ``q`` is smallest.
    """
    q = np.asarray(q, dtype=float)
    best_seg = 0
    best_dist = float("inf")
    for i in range(len(path) - 1):
        a = np.asarray(path[i],   dtype=float)
        b = np.asarray(path[i+1], dtype=float)
        ab = b - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-12:
            proj = a.copy()
        else:
            t = float(np.clip(np.dot(q - a, ab) / len_sq, 0.0, 1.0))
            proj = a + t * ab
        d = float(np.linalg.norm(q - proj))
        if d < best_dist:
            best_dist = d
            best_seg = i
    return best_seg, best_seg + 1


def path_tangent(path: List[np.ndarray], seg_idx: int) -> np.ndarray:
    """
    Return unit tangent of segment ``seg_idx`` → ``seg_idx+1``.
    Returns zero vector if segment has zero length.
    """
    a = np.asarray(path[seg_idx],   dtype=float)
    b = np.asarray(path[seg_idx+1], dtype=float)
    delta = b - a
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-12 else delta.copy()


def path_tracking_error(
    q: np.ndarray,
    path: List[np.ndarray],
    seg_idx: int,
) -> np.ndarray:
    """
    Return the vector from ``q`` to the nearest point on segment ``seg_idx``.

    This is the attraction term ``(q_proj - q)`` in the control law.
    """
    q = np.asarray(q, dtype=float)
    a = np.asarray(path[seg_idx],   dtype=float)
    b = np.asarray(path[seg_idx+1], dtype=float)
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-12:
        return a - q
    t = float(np.clip(np.dot(q - a, ab) / len_sq, 0.0, 1.0))
    proj = a + t * ab
    return proj - q


def distance_to_path_q(q: np.ndarray, path: List[np.ndarray]) -> float:
    """
    Return the minimum joint-space distance from ``q`` to any segment in path.
    """
    if not path:
        return 0.0
    q = np.asarray(q, dtype=float)
    if len(path) == 1:
        return float(np.linalg.norm(q - np.asarray(path[0], dtype=float)))
    seg_i, _ = nearest_path_segment(q, path)
    err = path_tracking_error(q, path, seg_i)
    return float(np.linalg.norm(err))


# ---------------------------------------------------------------------------
# Stateful tracker
# ---------------------------------------------------------------------------

class PathTubeTracker:
    """
    Stateful path-tube tracking controller.

    Usage
    -----
    1. Construct once after planning: ``tracker = PathTubeTracker(path, config)``
    2. Each step, call ``qdot_nom(q, clearance)`` to get the tube-tracking
       nominal joint velocity.
    3. Blend with DS output based on clearance (handled by the caller).

    The tracker maintains an *anchor index* — a waypoint the robot is
    currently heading toward.  The anchor advances only when the robot is
    sufficiently close, preventing premature shortcutting.
    """

    def __init__(
        self,
        path: List[np.ndarray],
        config: Optional[PathTubeConfig] = None,
    ) -> None:
        if len(path) < 2:
            raise ValueError("Path must have at least 2 waypoints.")
        self.path = [np.asarray(q, dtype=float) for q in path]
        self.config = config or PathTubeConfig()
        self._n = len(self.path)

        # Anchor state
        self._anchor_idx: int = 0
        self._steps_since_advance: int = 0

        # Per-step diagnostics (populated by qdot_nom)
        self.last_seg_idx:        int   = 0
        self.last_distance_q:     float = 0.0
        self.last_anchor_frozen:  bool  = False
        self.anchor_advance_count: int  = 0
        self.anchor_freeze_count:  int  = 0
        self.tube_exit_count:      int  = 0

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    def _try_advance_anchor(self, q: np.ndarray) -> None:
        """Advance the anchor toward the goal if conditions allow."""
        cfg = self.config
        if self._anchor_idx >= self._n - 1:
            return  # already at end

        # Hysteresis: do not advance if too soon since last advance
        if self._steps_since_advance < cfg.progress_hysteresis_steps:
            self._steps_since_advance += 1
            self.last_anchor_frozen = True
            self.anchor_freeze_count += 1
            return

        # Advance only if close enough to the current anchor
        dist_to_anchor = float(
            np.linalg.norm(q - self.path[self._anchor_idx])
        )
        if dist_to_anchor > cfg.anchor_advance_dist:
            self.last_anchor_frozen = True
            self.anchor_freeze_count += 1
            return

        # Advance (capped)
        jump = 0
        while (self._anchor_idx < self._n - 1
               and jump < cfg.max_anchor_jump
               and float(np.linalg.norm(q - self.path[self._anchor_idx])) < cfg.anchor_advance_dist):
            self._anchor_idx += 1
            jump += 1

        if jump > 0:
            self.last_anchor_frozen = False
            self._steps_since_advance = 0
            self.anchor_advance_count += jump
        else:
            self.last_anchor_frozen = True
            self.anchor_freeze_count += 1

    # ------------------------------------------------------------------
    # Nominal velocity
    # ------------------------------------------------------------------

    def qdot_nom(self, q: np.ndarray, clearance: Optional[float] = None) -> np.ndarray:
        """
        Compute the path-tube tracking nominal joint velocity.

        Control law:
            qdot_path = k_t * t_hat + k_p * (q_proj - q)

        where t_hat and q_proj come from the segment toward the current anchor.

        Parameters
        ----------
        q         : current joint config (n_dof,)
        clearance : current obstacle clearance (m); used only for logging

        Returns
        -------
        qdot_path : (n_dof,) joint velocity
        """
        q = np.asarray(q, dtype=float)
        cfg = self.config

        # Advance anchor if warranted
        self._try_advance_anchor(q)

        # Find nearest segment from current position toward anchor
        # Restrict search to segments not yet passed (up to anchor)
        anchor = min(self._anchor_idx, self._n - 2)
        seg_i, _ = nearest_path_segment(q, self.path[:anchor + 2])
        seg_i = min(seg_i, anchor)

        self.last_seg_idx = seg_i

        # Tangent (forward direction)
        t_hat = path_tangent(self.path, seg_i)

        # Attraction term (back toward path)
        q_proj_minus_q = path_tracking_error(q, self.path, seg_i)
        dist_q = float(np.linalg.norm(q_proj_minus_q))
        self.last_distance_q = dist_q

        # Track tube exits
        if dist_q > cfg.tube_radius_q:
            self.tube_exit_count += 1

        # Combined control law
        qdot = cfg.tangent_gain * t_hat + cfg.path_attract_gain * q_proj_minus_q

        if cfg.debug:
            print(
                f"[PathTube] seg={seg_i}/{self._n-2} "
                f"anchor={self._anchor_idx} "
                f"dist_q={dist_q:.4f} "
                f"clearance={clearance:.4f if clearance is not None else 'N/A'} "
                f"frozen={self.last_anchor_frozen}"
            )

        return qdot

    # ------------------------------------------------------------------
    # Blend weight computation
    # ------------------------------------------------------------------

    def blend_weights(self, clearance: float) -> Tuple[float, float]:
        """
        Return (w_path, w_ds) blending weights based on obstacle clearance.

        When clearance < hard_switch_clearance: w_path=1, w_ds=0 (full tube).
        When clearance > switch_clearance: w_path=0, w_ds=1 (full DS).
        Linear blend in between.

        Parameters
        ----------
        clearance : minimum signed obstacle clearance (m)

        Returns
        -------
        (w_path, w_ds) : blend weights summing to 1
        """
        cfg = self.config
        if clearance <= cfg.hard_switch_clearance:
            return 1.0, 0.0
        if clearance >= cfg.switch_clearance:
            return 0.0, 1.0
        # Linear blend
        t = (cfg.switch_clearance - clearance) / (
            cfg.switch_clearance - cfg.hard_switch_clearance
        )
        t = float(np.clip(t, 0.0, 1.0))
        return t, 1.0 - t

    # ------------------------------------------------------------------
    # Mode query (with hysteresis)
    # ------------------------------------------------------------------

    def in_tube_mode(self, clearance: float, currently_in_tube: bool) -> bool:
        """
        Return whether to be in PATH_TUBE_TRACKING mode.

        Uses hysteresis: enter at switch_clearance, exit at exit_clearance.

        Parameters
        ----------
        clearance        : current obstacle clearance (m)
        currently_in_tube: True if already in tube mode this step

        Returns
        -------
        bool : True → use path-tube tracking; False → use free DS
        """
        cfg = self.config
        if not cfg.enabled:
            return False
        if currently_in_tube:
            return clearance < cfg.exit_clearance
        return clearance < cfg.switch_clearance
