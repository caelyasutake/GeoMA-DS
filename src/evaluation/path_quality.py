"""
Path quality analysis for Multi-IK-DS evaluation.

Provides utilities to assess how executable a planned joint-space path is
before handing it to the DS+CBF execution stack.

Key functions
-------------
path_clearance_stats  — sample dense points along a planned path and compute
                        aggregate clearance statistics (min, mean, near-obstacle
                        fraction, per-obstacle minima).

interpolation_clearance — evaluate straight-line joint-space interpolation from
                          q_start to q_goal as a cheap pre-filter proxy.

path_risk_score       — combine clearance stats into a scalar "execution risk"
                        where higher = harder to execute safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.solver.planner.collision import _panda_link_positions, _LINK_RADII
from src.scenarios.scenario_schema import Obstacle


# ---------------------------------------------------------------------------
# Link clearance primitive (shared with trial_runner but kept local for speed)
# ---------------------------------------------------------------------------

def _link_clearance_to_obstacle(
    link_pos: np.ndarray,
    link_radius: float,
    obs: Obstacle,
) -> float:
    """Signed clearance between a link sphere and one obstacle. Positive = free."""
    p = np.array(obs.position, dtype=float)
    t = obs.type.lower()
    if t == "box":
        half = np.array(obs.size, dtype=float)
        closest = np.clip(link_pos, p - half, p + half)
        return float(np.linalg.norm(link_pos - closest)) - link_radius
    elif t == "sphere":
        return float(np.linalg.norm(link_pos - p)) - link_radius - float(obs.size[0])
    elif t == "cylinder":
        cyl_r, cyl_hh = float(obs.size[0]), float(obs.size[1])
        dx, dy = link_pos[0] - p[0], link_pos[1] - p[1]
        dz = link_pos[2] - p[2]
        r_xy = float(np.sqrt(dx * dx + dy * dy))
        cz = float(np.clip(dz, -cyl_hh, cyl_hh))
        if r_xy < 1e-9:
            closest = np.array([p[0], p[1], p[2] + cz])
        elif r_xy <= cyl_r:
            closest = np.array([p[0] + dx, p[1] + dy, p[2] + cz])
        else:
            scale = cyl_r / r_xy
            closest = np.array([p[0] + dx * scale, p[1] + dy * scale, p[2] + cz])
        return float(np.linalg.norm(link_pos - closest)) - link_radius
    return float("inf")


def _min_clearance_at_q(q: np.ndarray, obs_list: List[Obstacle]) -> float:
    """Minimum signed clearance across all links and all obstacles at config q."""
    link_positions = _panda_link_positions(np.asarray(q, dtype=float))
    min_cl = float("inf")
    for i, lp in enumerate(link_positions):
        r = float(_LINK_RADII.get(i, 0.08))
        for obs in obs_list:
            cl = _link_clearance_to_obstacle(np.array(lp, dtype=float), r, obs)
            if cl < min_cl:
                min_cl = cl
    return min_cl


def _per_obs_clearance_at_q(
    q: np.ndarray,
    obs_list: List[Obstacle],
) -> Dict[str, float]:
    """Per-obstacle minimum clearance at config q."""
    link_positions = _panda_link_positions(np.asarray(q, dtype=float))
    result: Dict[str, float] = {}
    for obs in obs_list:
        min_cl = float("inf")
        for i, lp in enumerate(link_positions):
            r = float(_LINK_RADII.get(i, 0.08))
            cl = _link_clearance_to_obstacle(np.array(lp, dtype=float), r, obs)
            if cl < min_cl:
                min_cl = cl
        result[obs.name] = min_cl
    return result


# ---------------------------------------------------------------------------
# Path clearance analysis
# ---------------------------------------------------------------------------

@dataclass
class PathClearanceStats:
    """Clearance statistics for a planned or interpolated path."""
    min_clearance:          float = float("inf")
    mean_clearance:         float = float("inf")
    near_obstacle_fraction: float = 0.0       # fraction of samples below threshold
    n_samples:              int   = 0
    per_obstacle_min:       Dict[str, float] = field(default_factory=dict)
    # Threshold used for near_obstacle_fraction
    threshold:              float = 0.02


def path_clearance_stats(
    path: List[np.ndarray],
    obs_list: List[Obstacle],
    n_samples: int = 50,
    near_obstacle_threshold: float = 0.02,
) -> PathClearanceStats:
    """
    Sample n_samples evenly-spaced points along a joint-space path and compute
    aggregate clearance statistics.

    Args:
        path:                   List of joint configs forming the path.
        obs_list:               Collision-enabled obstacles.
        n_samples:              Number of evenly-spaced samples along the path.
        near_obstacle_threshold: Clearance below this counts as "near obstacle".

    Returns:
        PathClearanceStats.
    """
    if not obs_list or not path or len(path) < 2:
        return PathClearanceStats(n_samples=0, threshold=near_obstacle_threshold)

    # Build cumulative arc length for uniform sampling
    segs = [float(np.linalg.norm(np.asarray(path[i+1], dtype=float)
                                  - np.asarray(path[i], dtype=float)))
            for i in range(len(path) - 1)]
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]
    if total < 1e-12:
        return PathClearanceStats(n_samples=0, threshold=near_obstacle_threshold)

    sample_arcs = np.linspace(0.0, total, n_samples)
    clearances: List[float] = []
    per_obs_all: Dict[str, List[float]] = {o.name: [] for o in obs_list}

    for s in sample_arcs:
        # Find segment
        seg = int(np.searchsorted(arc, s, side="right") - 1)
        seg = min(seg, len(path) - 2)
        t = (s - arc[seg]) / max(segs[seg], 1e-12)
        q = (1.0 - t) * np.asarray(path[seg], dtype=float) + t * np.asarray(path[seg+1], dtype=float)

        per = _per_obs_clearance_at_q(q, obs_list)
        cl = min(per.values()) if per else float("inf")
        clearances.append(cl)
        for name, v in per.items():
            per_obs_all[name].append(v)

    per_obs_min = {name: float(np.min(vals)) if vals else float("inf")
                   for name, vals in per_obs_all.items()}

    return PathClearanceStats(
        min_clearance=float(np.min(clearances)),
        mean_clearance=float(np.mean(clearances)),
        near_obstacle_fraction=float(np.mean([c < near_obstacle_threshold for c in clearances])),
        n_samples=n_samples,
        per_obstacle_min=per_obs_min,
        threshold=near_obstacle_threshold,
    )


def interpolation_clearance(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    obs_list: List[Obstacle],
    n_steps: int = 20,
) -> PathClearanceStats:
    """
    Evaluate clearance along a straight-line joint-space interpolation
    from q_start to q_goal. Used as a cheap pre-filter proxy.

    A low clearance here means the goal lies in a region where the DS
    may try to cut through an obstacle.
    """
    path = [q_start + t * (q_goal - q_start) for t in np.linspace(0.0, 1.0, n_steps)]
    return path_clearance_stats(path, obs_list, n_samples=n_steps)


# ---------------------------------------------------------------------------
# Execution risk score
# ---------------------------------------------------------------------------

@dataclass
class PathRiskScore:
    """Scalar execution risk and component breakdown."""
    total:                  float = 0.0   # lower = better (more executable)
    clearance_term:         float = 0.0
    near_obstacle_term:     float = 0.0
    length_term:            float = 0.0
    interp_clearance_term:  float = 0.0
    feasible:               bool  = True  # False if planning failed


def path_risk_score(
    path: Optional[List[np.ndarray]],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    obs_list: List[Obstacle],
    plan_success: bool = True,
    n_path_samples: int = 40,
    n_interp_steps: int = 15,
    near_threshold: float = 0.025,
    w_clearance: float = 2.0,
    w_near_obs: float = 1.5,
    w_length: float = 0.1,
    w_interp: float = 1.0,
) -> PathRiskScore:
    """
    Compute a scalar execution risk score for a planned path.

    Lower score = more executable by DS+CBF.

    Args:
        path:          Planned joint-space path (None or empty = infeasible).
        q_start:       Start config.
        q_goal:        Goal config.
        obs_list:      Collision-enabled obstacles.
        plan_success:  Whether planning succeeded.
        n_path_samples: Samples along planned path.
        n_interp_steps: Samples for straight-line interpolation.
        near_threshold: Clearance threshold for "near obstacle" penalty.
        w_*:           Penalty weights.

    Returns:
        PathRiskScore (lower total = better).
    """
    if not plan_success or not path or len(path) < 2:
        return PathRiskScore(total=1e6, feasible=False)

    if not obs_list:
        path_len = sum(float(np.linalg.norm(
            np.asarray(path[i+1], dtype=float) - np.asarray(path[i], dtype=float)
        )) for i in range(len(path) - 1))
        return PathRiskScore(total=w_length * path_len, length_term=w_length * path_len, feasible=True)

    stats = path_clearance_stats(path, obs_list, n_samples=n_path_samples,
                                 near_obstacle_threshold=near_threshold)
    interp = interpolation_clearance(q_start, q_goal, obs_list, n_steps=n_interp_steps)

    path_len = sum(float(np.linalg.norm(
        np.asarray(path[i+1], dtype=float) - np.asarray(path[i], dtype=float)
    )) for i in range(len(path) - 1))

    eps = 1e-3
    cl_term    = w_clearance / max(stats.min_clearance, eps)
    near_term  = w_near_obs  * stats.near_obstacle_fraction
    len_term   = w_length    * path_len
    interp_term = w_interp   / max(interp.min_clearance, eps)

    total = cl_term + near_term + len_term + interp_term
    return PathRiskScore(
        total=total,
        clearance_term=cl_term,
        near_obstacle_term=near_term,
        length_term=len_term,
        interp_clearance_term=interp_term,
        feasible=True,
    )
