"""
Bridge / escape subgoal generation for Multi-IK-DS.

When a direct path to the final IK goal is blocked by the center post,
this module generates intermediate EE poses that route around the obstacle,
solves IK for each pose, and returns collision-safe bridge joint configs.

Bridge goals are used in BRIDGE_TARGET escape mode: the robot first moves
to a bridge config (clearing the obstacle band), then continues to the
final goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BridgeGoalConfig:
    y_clearance_offsets: List[float] = field(default_factory=lambda: [0.08, 0.12, 0.06])
    z_offsets:           List[float] = field(default_factory=lambda: [0.0, 0.05, -0.05])
    x_pullback:          float       = 0.10   # retract in X before crossing
    ik_max_iter:         int         = 200
    ik_step:             float       = 0.05   # Jacobian IK step size
    ik_tol:              float       = 5e-3   # EE position tolerance (m)
    min_clearance:       float       = 0.025  # min clearance for a bridge goal to be valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_center_post_y(spec) -> Optional[float]:
    """Return the Y position of the center_post obstacle, if present."""
    obstacles = getattr(spec, "obstacles", None) or []
    for obs in obstacles:
        name = getattr(obs, "name", "") or ""
        if "center_post" in name or "center" in name:
            pos = getattr(obs, "pos", None)
            if pos is not None and len(pos) >= 2:
                return float(pos[1])
    return None


def _ee_from_fk(q: np.ndarray, fk_fn: Callable) -> np.ndarray:
    """Return EE position (3,) from FK callable q -> link_positions."""
    positions = fk_fn(q)
    return np.asarray(positions[-1], dtype=float)  # index 7 = hand body


# ---------------------------------------------------------------------------
# Jacobian IK
# ---------------------------------------------------------------------------

def jacobian_ik(
    ee_target: np.ndarray,
    q_init: np.ndarray,
    fk_fn: Callable,
    jacobian_fn: Optional[Callable],
    config: BridgeGoalConfig,
    q_min: Optional[np.ndarray] = None,
    q_max: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Iterative Jacobian-transpose IK to reach a 3-D EE position.

    Parameters
    ----------
    ee_target  : desired EE position (3,)
    q_init     : initial joint config (n_dof,)
    fk_fn      : callable q -> list of link positions
    jacobian_fn: callable q -> (3, n_dof) position Jacobian, or None
                 (if None, finite-difference Jacobian is used)
    config     : BridgeGoalConfig
    q_min/max  : joint limits

    Returns
    -------
    q_sol  : converged joint config or None on failure
    err    : final EE position error (m)
    """
    q = q_init.copy()
    n_dof = len(q)
    eps = 1e-4  # finite-difference step

    for _ in range(config.ik_max_iter):
        ee = _ee_from_fk(q, fk_fn)
        delta = ee_target - ee
        err = np.linalg.norm(delta)
        if err < config.ik_tol:
            return q, float(err)

        # Build position Jacobian (3, n_dof)
        if jacobian_fn is not None:
            J = np.asarray(jacobian_fn(q), dtype=float)
            if J.shape != (3, n_dof):
                J = J[:3, :n_dof]
        else:
            J = np.zeros((3, n_dof))
            for i in range(n_dof):
                q_p = q.copy()
                q_p[i] += eps
                J[:, i] = (_ee_from_fk(q_p, fk_fn) - ee) / eps

        # Jacobian-transpose update
        dq = J.T @ delta
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 1e-9:
            dq = dq / dq_norm * config.ik_step

        q = q + dq
        if q_min is not None:
            q = np.maximum(q, q_min)
        if q_max is not None:
            q = np.minimum(q, q_max)

    ee = _ee_from_fk(q, fk_fn)
    return None, float(np.linalg.norm(ee_target - ee))


# ---------------------------------------------------------------------------
# Bridge EE pose generation
# ---------------------------------------------------------------------------

def generate_bridge_ee_poses(
    start_ee: np.ndarray,
    goal_ee: np.ndarray,
    center_post_y: float,
    config: BridgeGoalConfig,
) -> List[np.ndarray]:
    """
    Generate candidate bridge EE positions that route around the center post.

    Strategy: retract in X to x_pullback behind start, then offset in Y to
    clear the post, at various Z heights.

    Parameters
    ----------
    start_ee        : current EE position (3,)
    goal_ee         : final goal EE position (3,)
    center_post_y   : Y coordinate of center_post obstacle
    config          : BridgeGoalConfig

    Returns
    -------
    List of candidate EE positions (3,)
    """
    poses = []
    x_retract = start_ee[0] - config.x_pullback
    z_base = (start_ee[2] + goal_ee[2]) / 2.0

    for y_off in config.y_clearance_offsets:
        for z_off in config.z_offsets:
            # Place bridge on the side of the post that the goal is on
            goal_side_y = goal_ee[1]
            if goal_side_y >= center_post_y:
                y_bridge = center_post_y + y_off
            else:
                y_bridge = center_post_y - y_off
            pose = np.array([x_retract, y_bridge, z_base + z_off])
            poses.append(pose)

    return poses


# ---------------------------------------------------------------------------
# Top-level: generate valid bridge joint configs
# ---------------------------------------------------------------------------

def generate_bridge_goals(
    spec,
    q_current: np.ndarray,
    start_ee: np.ndarray,
    goal_ee: np.ndarray,
    fk_fn: Callable,
    jacobian_fn: Optional[Callable],
    col_fn: Callable,          # q -> float (min clearance; >0 = collision-free)
    config: Optional[BridgeGoalConfig] = None,
    q_min: Optional[np.ndarray] = None,
    q_max: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Generate collision-safe bridge joint configurations.

    Parameters
    ----------
    spec        : scenario spec (used to find center_post position)
    q_current   : current joint config
    start_ee    : current EE position
    goal_ee     : final goal EE position
    fk_fn       : q -> list of link positions
    jacobian_fn : q -> (3, n_dof) Jacobian (or None for FD)
    col_fn      : q -> min_clearance (positive = safe)
    config      : BridgeGoalConfig (defaults if None)
    q_min/max   : joint limits

    Returns
    -------
    List of valid bridge joint configs (may be empty)
    """
    if config is None:
        config = BridgeGoalConfig()

    center_post_y = _find_center_post_y(spec)
    if center_post_y is None:
        # Fallback: use midpoint Y between start and goal
        center_post_y = float((start_ee[1] + goal_ee[1]) / 2.0)

    candidate_poses = generate_bridge_ee_poses(start_ee, goal_ee, center_post_y, config)

    valid_bridges: List[np.ndarray] = []
    for ee_target in candidate_poses:
        q_sol, err = jacobian_ik(
            ee_target, q_current, fk_fn, jacobian_fn, config, q_min, q_max
        )
        if q_sol is None or err > config.ik_tol * 2:
            continue
        clearance = col_fn(q_sol)
        if clearance >= config.min_clearance:
            valid_bridges.append(q_sol)

    return valid_bridges
