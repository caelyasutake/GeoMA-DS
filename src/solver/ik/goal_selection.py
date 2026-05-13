"""
Family-aware IK goal selection for Multi-IK-DS.

Converts a flat list of IK goal configurations into a structured set of
posture-family candidates, each with quality scores, so the planner can
select and rank goals by executability rather than just feasibility.

Key types
---------
IKGoalInfo        — per-goal metadata: family label, clearance scores, rank.
classify_ik_goals — build IKGoalInfo list from raw goals + obstacle context.
select_family_representatives — pick one representative per family (+ optional top-k).
rank_goals_within_family — sort goals inside a family by a scalar score.

Family classifiers
------------------
"goal_frame_midlink" — fast default: task-relative perpendicular offset of a
                       mid-chain landmark relative to the start-goal line.
                       Does not compute obstacle distances. ~1 ms typical.
"elbow_y"            — mean Panda elbow Y along straight-line interpolation.
"closest_passage"    — obstacle-relative side of the closest robot-obstacle
                       interaction along the interpolated path. ~30-40 ms.
"auto"               — prefer goal_frame_midlink; fall back to elbow_y when
                       all labels are degenerate.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.solver.planner.collision import (
    _panda_link_positions, _panda_fk_batch,
    _LINK_RADII,
    _sphere_box_signed_dist,
    _precompute_obs_rotations,
    _batch_clearance_from_lp,
)
from src.scenarios.scenario_schema import Obstacle

# Link index for elbow (link4 in 0-indexed chain, same as ik_family_analysis)
_ELBOW_IDX = 3
_CENTRE_THRESH = 0.04   # metres — matches ik_family_analysis
_DEFAULT_MIDLINK_IDX = 4   # link5 — forearm region, good mid-chain landmark


def _has_active_obstacles(obs_list) -> bool:
    """True when obs_list has at least one non-disabled obstacle."""
    if not obs_list:
        return False
    return any(not getattr(o, "disabled", False) for o in obs_list)


# ---------------------------------------------------------------------------
# IKGoalInfo
# ---------------------------------------------------------------------------

@dataclass
class IKGoalInfo:
    """Per-goal metadata produced by classify_ik_goals."""
    q_goal:              np.ndarray
    goal_idx:            int        # index into the original ik_goals list
    family_label:        str        # "elbow_fwd" | "elbow_center" | "elbow_back" | closest-passage label
    mean_elbow_y:        float      # mean elbow Y along straight-line path
    static_clearance:    float      # clearance at the goal config itself
    interp_min_clearance: float     # min clearance along straight-line interp
    manipulability:      float      # det(J J^T)^0.5 proxy via joint-space metric
    score_prior:         float      # combined score: lower = better candidate
    is_collision_free:   bool       # static collision check passed
    # Escape-side reasoning
    preferred_escape_side: str  = ""   # e.g. "positive_y", "negative_y", "backtrack"
    family_escape_score:   float = 0.0  # lower = easier to escape from this family
    # Closest-passage metadata (populated when family_method == "closest_passage")
    family_method:           str             = "elbow_y"
    closest_obstacle_name:   Optional[str]   = None
    closest_link_name:       Optional[str]   = None
    closest_distance:        Optional[float] = None
    closest_step_idx:        Optional[int]   = None
    closest_rel_local:       Optional[np.ndarray] = None
    lateral_bin:             Optional[str]   = None
    vertical_bin:            Optional[str]   = None


# ---------------------------------------------------------------------------
# Helpers — elbow-Y classifier
# ---------------------------------------------------------------------------

def _mean_elbow_y(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    n_steps: int = 10,
) -> float:
    """Mean Y-position of elbow link along straight-line q_start → q_goal."""
    ys = []
    for t in np.linspace(0.0, 1.0, n_steps):
        q = q_start + t * (q_goal - q_start)
        ys.append(_panda_link_positions(q)[_ELBOW_IDX][1])
    return float(np.mean(ys))


def _classify_family(mean_y: float) -> str:
    if mean_y > _CENTRE_THRESH:
        return "elbow_fwd"
    elif mean_y < -_CENTRE_THRESH:
        return "elbow_back"
    return "elbow_center"


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _make_goal_frame(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (e1, e2, e3) orthonormal frame.

    e1: task progress axis from x_start to x_goal.
    e2, e3: perpendicular axes for posture-family classification.

    Falls back to world frame when x_start ≈ x_goal (degenerate).
    """
    e1_raw = x_goal - x_start
    norm = float(np.linalg.norm(e1_raw))
    if norm < eps:
        return (np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([0., 0., 1.]))
    e1 = e1_raw / norm
    world_up = np.array([0., 0., 1.])
    ref = world_up if abs(float(np.dot(e1, world_up))) <= 0.95 else np.array([1., 0., 0.])
    e2 = np.cross(ref, e1)
    e2 = e2 / float(np.linalg.norm(e2))
    e3 = np.cross(e1, e2)
    return e1, e2, e3


def _min_clearance_at_q(q: np.ndarray, obs_list: List[Obstacle]) -> float:
    """Minimum signed clearance across all links at config q."""
    from src.evaluation.path_quality import _link_clearance_to_obstacle
    link_pos = _panda_link_positions(q)
    min_cl = float("inf")
    for i, lp in enumerate(link_pos):
        r = float(_LINK_RADII.get(i, 0.08))
        for obs in obs_list:
            cl = _link_clearance_to_obstacle(np.array(lp, dtype=float), r, obs)
            if cl < min_cl:
                min_cl = cl
    return min_cl


def _interp_min_clearance(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    obs_list: List[Obstacle],
    n_steps: int = 15,
) -> float:
    """Min clearance along straight-line interpolation from q_start to q_goal."""
    from src.evaluation.path_quality import _link_clearance_to_obstacle
    min_cl = float("inf")
    for t in np.linspace(0.0, 1.0, n_steps):
        q = q_start + t * (q_goal - q_start)
        link_pos = _panda_link_positions(q)
        for i, lp in enumerate(link_pos):
            r = float(_LINK_RADII.get(i, 0.08))
            for obs in obs_list:
                cl = _link_clearance_to_obstacle(np.array(lp, dtype=float), r, obs)
                if cl < min_cl:
                    min_cl = cl
    return min_cl


def _manipulability(q: np.ndarray) -> float:
    """
    Simplified manipulability proxy: determinant of the 3×7 position Jacobian
    (finite-difference over FK positions) as a quality heuristic.
    High = more dexterous.
    """
    eps = 1e-4
    q = np.asarray(q, dtype=float)
    # FK-based position Jacobian approximation (3×7)
    p0 = _panda_link_positions(q)[-1]
    J = np.zeros((3, len(q)))
    for j in range(len(q)):
        qe = q.copy()
        qe[j] += eps
        pe = _panda_link_positions(qe)[-1]
        J[:, j] = (pe - p0) / eps
    # Yoshikawa manipulability: sqrt(det(J J^T))
    JJT = J @ J.T
    val = float(np.sqrt(max(0.0, np.linalg.det(JJT))))
    return val


# ---------------------------------------------------------------------------
# Helpers — closest-passage classifier
# ---------------------------------------------------------------------------

_LINK_NAMES: Dict[int, str] = {
    0: "link1", 1: "link2", 2: "link3", 3: "link4",
    4: "link5", 5: "link6", 6: "link7", 7: "hand",
}

# Lateral bin tokens used by _parse_lateral_bin to detect closest-passage labels.
_LATERAL_BINS = frozenset({"pos_y", "neg_y", "center_y"})


def signed_bin(
    value: float,
    threshold: float,
    pos_name: str,
    neg_name: str,
    center_name: str,
) -> str:
    """Classify a scalar into one of three named bins around zero."""
    if value > threshold:
        return pos_name
    elif value < -threshold:
        return neg_name
    return center_name


def _parse_lateral_bin(label: str) -> Optional[str]:
    """Extract lateral bin token from a colon-separated closest-passage label, or None."""
    for part in label.split(":"):
        if part in _LATERAL_BINS:
            return part
    return None


def _too_many_singletons(labels: List[str], threshold: float = 0.75) -> bool:
    """True when the fraction of singleton-count family labels meets or exceeds threshold."""
    counts = Counter(labels)
    n_singletons = sum(1 for c in counts.values() if c == 1)
    return n_singletons >= threshold * len(labels)


def _link_obs_clearance(
    link_pos: np.ndarray,
    link_radius: float,
    obs: Obstacle,
    R_inv: Optional[np.ndarray] = None,
) -> float:
    """Signed clearance from a link sphere to a single obstacle. Positive = free."""
    obs_pos = np.array(obs.position, dtype=float)
    t = obs.type.lower()
    if t == "box":
        half = np.array(obs.size, dtype=float)
        return _sphere_box_signed_dist(link_pos, link_radius, obs_pos, half, R_inv)
    elif t == "sphere":
        return float(np.linalg.norm(link_pos - obs_pos)) - link_radius - float(obs.size[0])
    elif t == "cylinder":
        cyl_r, cyl_hh = float(obs.size[0]), float(obs.size[1])
        dx = link_pos[0] - obs_pos[0]
        dy = link_pos[1] - obs_pos[1]
        dz = link_pos[2] - obs_pos[2]
        r_xy = max(float(np.sqrt(dx * dx + dy * dy)), 1e-9)
        scale = min(cyl_r / r_xy, 1.0)
        cz = float(np.clip(dz, -cyl_hh, cyl_hh))
        closest = np.array([obs_pos[0] + dx * scale, obs_pos[1] + dy * scale, obs_pos[2] + cz])
        return float(np.linalg.norm(link_pos - closest)) - link_radius
    return float("inf")


def _closest_passage_info(
    link_pos_grid: np.ndarray,
    obs_list: List[Obstacle],
    obs_R_invs: List[Optional[np.ndarray]],
    lateral_axis: int = 1,
    vertical_axis: int = 2,
    side_threshold_m: float = 0.04,
    vertical_threshold_m: float = 0.04,
) -> List[dict]:
    """
    For each goal, find the (step, link, obstacle) triple with minimum clearance.

    Args:
        link_pos_grid: (n_goals, n_steps, 8, 3) FK link positions.
        obs_list:      Obstacle list parallel to obs_R_invs.
        obs_R_invs:    Precomputed R.T per obstacle from _precompute_obs_rotations.

    Returns:
        List of dicts (one per goal) with keys:
          min_clearance, step_idx, link_idx, obs_idx,
          link_pos, obs_center, rel_local,
          lateral_bin, vertical_bin, obs_name, link_name,
          full_label, medium_label, simple_label.
    """
    n_goals, n_steps = link_pos_grid.shape[:2]

    results = []
    for g in range(n_goals):
        best_cl = float("inf")
        best_step = best_link = best_obs = 0

        for step in range(n_steps):
            for li in range(8):
                lp = link_pos_grid[g, step, li]
                lr = float(_LINK_RADII.get(li, 0.08))
                for oi, obs in enumerate(obs_list):
                    cl = _link_obs_clearance(lp, lr, obs, obs_R_invs[oi])
                    if cl < best_cl:
                        best_cl = cl
                        best_step = step
                        best_link = li
                        best_obs  = oi

        lp_best  = link_pos_grid[g, best_step, best_link].copy()
        obs_best = obs_list[best_obs]
        obs_ctr  = np.array(obs_best.position, dtype=float)
        R_inv    = obs_R_invs[best_obs]

        rel_world = lp_best - obs_ctr
        rel_local = (R_inv @ rel_world) if R_inv is not None else rel_world.copy()

        lat  = signed_bin(float(rel_local[lateral_axis]),   side_threshold_m,
                          "pos_y", "neg_y", "center_y")
        vert = signed_bin(float(rel_local[vertical_axis]), vertical_threshold_m,
                          "above", "below", "center_z")

        obs_name  = obs_best.name if obs_best.name else f"obs{best_obs}"
        link_name = _LINK_NAMES.get(best_link, f"link{best_link}")

        results.append({
            "min_clearance":  best_cl,
            "step_idx":       best_step,
            "link_idx":       best_link,
            "obs_idx":        best_obs,
            "link_pos":       lp_best,
            "obs_center":     obs_ctr,
            "rel_local":      rel_local,
            "lateral_bin":    lat,
            "vertical_bin":   vert,
            "obs_name":       obs_name,
            "link_name":      link_name,
            "full_label":     f"{obs_name}:{link_name}:{lat}:{vert}",
            "medium_label":   f"{obs_name}:{lat}:{vert}",
            "simple_label":   f"{lat}:{vert}",
        })

    return results


# ---------------------------------------------------------------------------
# Elbow-Y implementation (extracted for dispatch)
# ---------------------------------------------------------------------------

def _classify_ik_goals_elbow_y(
    q_start: np.ndarray,
    ik_goals: List[np.ndarray],
    obs_list: List[Obstacle],
    col_fn: Optional[Callable[[np.ndarray], bool]] = None,
    margin: float = 0.0,
    w_clearance: float = 2.0,
    w_interp: float = 1.0,
    w_manip: float = 0.5,
) -> List[IKGoalInfo]:
    """Original per-goal elbow-Y family classifier."""
    q_start = np.asarray(q_start, dtype=float)
    infos: List[IKGoalInfo] = []

    for idx, q_raw in enumerate(ik_goals):
        q = np.asarray(q_raw, dtype=float)

        elbow_y = _mean_elbow_y(q_start, q)
        family  = _classify_family(elbow_y)

        if obs_list:
            static_cl = _min_clearance_at_q(q, obs_list) - margin
            interp_cl = _interp_min_clearance(q_start, q, obs_list) - margin
        else:
            static_cl = float("inf")
            interp_cl = float("inf")

        free = bool(col_fn(q)) if col_fn is not None else static_cl > 0.0
        manip = _manipulability(q)

        eps = 1e-3
        score = (
            w_clearance / max(static_cl, eps)
            + w_interp  / max(interp_cl, eps)
            - w_manip   * manip
        )

        if family == "elbow_fwd":
            escape_side  = "positive_y"
            escape_score = max(0.0, -elbow_y)
        elif family == "elbow_back":
            escape_side  = "negative_y"
            escape_score = max(0.0, elbow_y)
        else:
            escape_side  = "backtrack"
            escape_score = 0.5

        infos.append(IKGoalInfo(
            q_goal=q,
            goal_idx=idx,
            family_label=family,
            mean_elbow_y=elbow_y,
            static_clearance=static_cl,
            interp_min_clearance=interp_cl,
            manipulability=manip,
            score_prior=score,
            is_collision_free=free,
            preferred_escape_side=escape_side,
            family_escape_score=escape_score,
        ))

    return infos


# ---------------------------------------------------------------------------
# Closest-passage classifier
# ---------------------------------------------------------------------------

def classify_ik_goals_closest_passage(
    q_start: np.ndarray,
    ik_goals: List[np.ndarray],
    obs_list: List[Obstacle],
    col_fn: Optional[Callable[[np.ndarray], bool]] = None,
    margin: float = 0.0,
    w_clearance: float = 2.0,
    w_interp: float = 1.0,
    w_manip: float = 0.5,
    n_interp_steps: int = 25,
    compute_manipulability: bool = False,
    lateral_axis: int = 1,
    vertical_axis: int = 2,
    side_threshold_m: float = 0.04,
    vertical_threshold_m: float = 0.04,
    include_link_in_label: bool = False,
    singleton_threshold: float = 0.75,
    verbose: bool = False,
    link_pos_grid: Optional[np.ndarray] = None,
) -> List[IKGoalInfo]:
    """
    Classify IK goals by obstacle-relative closest-passage signatures.

    Samples the start-to-goal joint interpolation, runs batched FK, finds the
    closest robot-obstacle interaction for each candidate, and labels it by
    which side of the relevant obstacle the robot passes.

    Falls back to elbow-Y if obs_list is empty.
    """
    q_start = np.asarray(q_start, dtype=float)
    n_goals = len(ik_goals)
    if n_goals == 0:
        return []

    if not obs_list:
        if verbose:
            logging.info("closest_passage: no obstacles; falling back to elbow_y.")
        return _classify_ik_goals_elbow_y(
            q_start, ik_goals, obs_list,
            col_fn=col_fn, margin=margin,
            w_clearance=w_clearance, w_interp=w_interp, w_manip=w_manip,
        )

    qs = np.stack([np.asarray(g, dtype=float) for g in ik_goals])  # (n_goals, 7)

    # Batched FK over all goals × steps: (n_goals, n_steps, 8, 3)
    # Skip if caller already provided a pre-built grid (avoids duplicate FK pass).
    if link_pos_grid is None:
        t_vals    = np.linspace(0.0, 1.0, n_interp_steps)
        diff      = qs - q_start[np.newaxis, :]
        interp_qs = (q_start[np.newaxis, np.newaxis, :]
                     + t_vals[np.newaxis, :, np.newaxis] * diff[:, np.newaxis, :])
        link_pos_grid = _panda_fk_batch(interp_qs.reshape(-1, 7)).reshape(
            n_goals, n_interp_steps, 8, 3
        )
    else:
        n_interp_steps = link_pos_grid.shape[1]

    obs_R_invs = _precompute_obs_rotations(obs_list)

    # Closest-passage info — also gives interp min clearance per goal
    passage_infos = _closest_passage_info(
        link_pos_grid, obs_list, obs_R_invs,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
        side_threshold_m=side_threshold_m,
        vertical_threshold_m=vertical_threshold_m,
    )
    interp_cl = np.array([p["min_clearance"] - margin for p in passage_infos])

    # Static clearance at goal config (last interpolation step, t=1)
    static_cl = np.full(n_goals, float("inf"))
    for g in range(n_goals):
        for li in range(8):
            lp = link_pos_grid[g, -1, li]
            lr = float(_LINK_RADII.get(li, 0.08))
            for oi, obs in enumerate(obs_list):
                cl = _link_obs_clearance(lp, lr, obs, obs_R_invs[oi]) - margin
                if cl < static_cl[g]:
                    static_cl[g] = cl

    if compute_manipulability and w_manip > 0.0:
        manip_vals = np.array([_manipulability(qs[i]) for i in range(n_goals)])
    else:
        manip_vals = np.ones(n_goals)

    col_free = (np.array([bool(col_fn(qs[i])) for i in range(n_goals)])
                if col_fn is not None else static_cl > 0.0)

    # Choose label granularity, compress if over-fragmented
    if include_link_in_label:
        labels = [p["full_label"] for p in passage_infos]
        label_method = "full"
    else:
        labels = [p["medium_label"] for p in passage_infos]
        label_method = "medium"

    if _too_many_singletons(labels, singleton_threshold):
        if verbose:
            logging.info(
                "closest_passage: %d unique labels for %d candidates at %s; "
                "compressing to medium labels.",
                len(set(labels)), n_goals, label_method,
            )
        labels = [p["medium_label"] for p in passage_infos]
        label_method = "medium"

    if _too_many_singletons(labels, singleton_threshold):
        if verbose:
            logging.info(
                "closest_passage: still %d unique labels for %d candidates; "
                "compressing to simple labels.",
                len(set(labels)), n_goals,
            )
        labels = [p["simple_label"] for p in passage_infos]
        label_method = "simple"

    if verbose:
        for idx in range(n_goals):
            p = passage_infos[idx]
            logging.info(
                "IK candidate %d:\n"
                "  method: closest_passage (%s)\n"
                "  family: %s\n"
                "  closest_obstacle: %s\n"
                "  closest_link: %s\n"
                "  closest_distance: %.3f\n"
                "  closest_step: %d / %d\n"
                "  rel_local: %s",
                idx, label_method, labels[idx],
                p["obs_name"], p["link_name"], p["min_clearance"],
                p["step_idx"], n_interp_steps,
                np.round(p["rel_local"], 3).tolist(),
            )

    eps = 1e-3
    infos: List[IKGoalInfo] = []
    for idx in range(n_goals):
        sc = float(static_cl[idx])
        ic = float(interp_cl[idx])
        ma = float(manip_vals[idx])
        p  = passage_infos[idx]

        score = (
            w_clearance / max(sc, eps)
            + w_interp  / max(ic, eps)
            - w_manip   * ma
        )

        infos.append(IKGoalInfo(
            q_goal=qs[idx],
            goal_idx=idx,
            family_label=labels[idx],
            mean_elbow_y=0.0,
            static_clearance=sc,
            interp_min_clearance=ic,
            manipulability=ma,
            score_prior=score,
            is_collision_free=bool(col_free[idx]),
            preferred_escape_side=p["lateral_bin"],
            family_escape_score=0.5,
            family_method="closest_passage",
            closest_obstacle_name=p["obs_name"],
            closest_link_name=p["link_name"],
            closest_distance=p["min_clearance"],
            closest_step_idx=p["step_idx"],
            closest_rel_local=p["rel_local"],
            lateral_bin=p["lateral_bin"],
            vertical_bin=p["vertical_bin"],
        ))

    return infos


# ---------------------------------------------------------------------------
# Goal-frame mid-link classifier (fast default)
# ---------------------------------------------------------------------------

def classify_ik_goals_goal_frame_midlink(
    q_start: np.ndarray,
    ik_goals: List[np.ndarray],
    obs_list: List[Obstacle],
    col_fn: Optional[Callable[[np.ndarray], bool]] = None,
    margin: float = 0.0,
    w_clearance: float = 2.0,
    w_interp: float = 1.0,
    w_manip: float = 0.5,
    target_pos: Optional[np.ndarray] = None,
    n_interp_steps: int = 5,
    midlink_index: Optional[int] = None,
    lateral_threshold_m: float = 0.04,
    vertical_threshold_m: float = 0.06,
    label_detail: str = "lateral",
    midlink_sample_mode: str = "midpoint",
    verbose: bool = False,
    link_pos_grid: Optional[np.ndarray] = None,
    **kwargs,
) -> List[IKGoalInfo]:
    """
    Fast goal-frame mid-link posture-family proxy classifier.

    This classifier does not compute formal homotopy classes. It computes a
    fast, task-relative posture-family proxy by measuring the perpendicular
    offset of a mid-chain landmark relative to the task-space start-goal line.
    The goal is to preserve coarse IK-attractor diversity, not to prove
    topological separation.

    Args:
        q_start:              Start configuration.
        ik_goals:             List of (7,) goal configurations.
        obs_list:             Collision obstacles.  When empty (or all disabled),
                              all candidates receive ``"open"`` (fast path, no FK).
        target_pos:           Task-space goal position (e.g. panda_grasptarget).
                              If None, uses hand FK of the first goal config.
        midlink_sample_mode:  ``"midpoint"`` (default) — one FK at t=0.5 per goal;
                              ``"mean"`` — average over ``n_interp_steps`` samples;
                              ``"auto"`` — midpoint first, fallback to 3-sample when
                              all labels collapse.
        n_interp_steps:       Samples for ``"mean"`` mode (default 5).
        midlink_index:        Link index for the mid-chain landmark (default link5).
        lateral_threshold_m:  Half-width of the lateral center bin (metres).
        label_detail:         ``"lateral"`` (default, 3 bins) or
                              ``"lateral_vertical"`` (up to 9 bins).
        link_pos_grid:        Pre-built FK grid (n_goals, n_steps, 8, 3).
                              When provided, no additional FK is performed.
    """
    q_start = np.asarray(q_start, dtype=float)
    n_goals = len(ik_goals)
    if n_goals == 0:
        return []

    # Fast path: no active obstacles → all "open", skip FK entirely.
    if not _has_active_obstacles(obs_list):
        qs = [np.asarray(g, dtype=float) for g in ik_goals]
        return [
            IKGoalInfo(
                q_goal=q, goal_idx=i, family_label="open",
                mean_elbow_y=0.0,
                static_clearance=float("inf"), interp_min_clearance=float("inf"),
                manipulability=1.0, score_prior=float(i), is_collision_free=True,
                family_method="goal_frame_midlink",
            )
            for i, q in enumerate(qs)
        ]

    mid_idx = midlink_index if midlink_index is not None else _DEFAULT_MIDLINK_IDX
    qs = np.stack([np.asarray(g, dtype=float) for g in ik_goals])  # (n_goals, 7)

    # Task-space goal frame — built once, independent of FK mode.
    x_start_ee = np.array(_panda_link_positions(q_start)[-1], dtype=float)
    x_goal_ee  = (np.asarray(target_pos, dtype=float) if target_pos is not None
                  else np.array(_panda_link_positions(qs[0])[-1], dtype=float))
    e1, e2, e3 = _make_goal_frame(x_start_ee, x_goal_ee)

    def _do_classify(mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run FK → mid_pos (n_goals,3), goal_lp (n_goals,8,3), returns (u_lat, u_vert, goal_lp)."""
        if link_pos_grid is not None:
            mid_step = link_pos_grid.shape[1] // 2
            if mode == "midpoint":
                pbar = link_pos_grid[:, mid_step, mid_idx, :]        # (n_goals, 3)
            else:
                pbar = link_pos_grid[:, :, mid_idx, :].mean(axis=1)  # (n_goals, 3)
            gl = link_pos_grid[:, -1, :, :]                          # (n_goals, 8, 3)
        else:
            if mode == "midpoint":
                q_mid  = q_start[np.newaxis, :] + 0.5 * (qs - q_start[np.newaxis, :])
                q_both = np.vstack([q_mid, qs])          # (2*n_goals, 7)
                lp     = _panda_fk_batch(q_both)         # (2*n_goals, 8, 3)
                pbar   = lp[:n_goals, mid_idx, :]         # (n_goals, 3)
                gl     = lp[n_goals:, :, :]               # (n_goals, 8, 3)
            else:
                t_vals    = np.linspace(0.0, 1.0, n_interp_steps)
                diff      = qs - q_start[np.newaxis, :]
                interp_qs = (q_start[np.newaxis, np.newaxis, :]
                             + t_vals[np.newaxis, :, np.newaxis] * diff[:, np.newaxis, :])
                lp_all    = _panda_fk_batch(interp_qs.reshape(-1, 7))
                pbar      = lp_all[:, mid_idx, :].reshape(n_goals, n_interp_steps, 3).mean(axis=1)
                gl        = lp_all[(n_interp_steps - 1)::n_interp_steps]

        # Perpendicular-plane projection (vectorised) + mean-centering.
        v      = pbar - x_start_ee[np.newaxis, :]
        s      = v @ e1
        p_line = x_start_ee[np.newaxis, :] + s[:, np.newaxis] * e1[np.newaxis, :]
        r      = pbar - p_line
        ul = (r @ e2).astype(float) - float((r @ e2).mean())
        uv = (r @ e3).astype(float) - float((r @ e3).mean())
        return ul, uv, gl

    # --- Mode dispatch ---
    effective_mode = midlink_sample_mode if midlink_sample_mode != "auto" else "midpoint"
    u_lat, u_vert, goal_lp = _do_classify(effective_mode)

    if midlink_sample_mode == "auto" and len(set(
        "lat_pos" if u > lateral_threshold_m else
        "lat_neg" if u < -lateral_threshold_m else
        "lat_center"
        for u in u_lat
    )) == 1 and n_goals > 1:
        # All midpoint labels collapsed → fallback to 3-sample mean
        effective_mode = "3sample"
        _n_save = n_interp_steps
        n_interp_steps = 3  # noqa: F841  (used as closure in _do_classify)
        u_lat, u_vert, goal_lp = _do_classify("mean")
        n_interp_steps = _n_save  # noqa: F841

    # --- Lateral (and optional vertical) binning ---
    lat_bins: List[str] = [
        "lat_pos" if u > lateral_threshold_m else
        "lat_neg" if u < -lateral_threshold_m else
        "lat_center"
        for u in u_lat
    ]

    if label_detail == "lateral_vertical":
        vert_bins: List[str] = [
            "above" if u > vertical_threshold_m else
            "below" if u < -vertical_threshold_m else
            "vert_center"
            for u in u_vert
        ]
        labels_arr = [f"mid_{lb}_{vb}" for lb, vb in zip(lat_bins, vert_bins)]
        if len(set(labels_arr)) > max(3, n_goals // 2):
            labels_arr = [f"mid_{lb}" for lb in lat_bins]
        vb_list: Optional[List[str]] = vert_bins
    else:
        labels_arr = [f"mid_{lb}" for lb in lat_bins]
        vb_list = None

    # Fast vectorised static clearance at goal config.
    obs_R_invs = _precompute_obs_rotations(obs_list)
    static_cls = _batch_clearance_from_lp(goal_lp, obs_list, obs_R_invs) - margin
    col_free   = static_cls > 0.0

    if verbose:
        sample_tag = (f"midpoint→3sample" if effective_mode == "3sample"
                      else effective_mode)
        for idx in range(n_goals):
            logging.info(
                "candidate %d: family=%s u_lat=%.3f u_vert=%.3f",
                idx, labels_arr[idx], float(u_lat[idx]), float(u_vert[idx]),
            )
        logging.info(
            "unique_families=%s  sample_mode=%s",
            sorted(set(labels_arr)), sample_tag,
        )

    eps_sc = 1e-3
    infos: List[IKGoalInfo] = []
    for idx in range(n_goals):
        sc    = float(static_cls[idx])
        score = w_clearance / max(sc, eps_sc)
        infos.append(IKGoalInfo(
            q_goal=qs[idx],
            goal_idx=idx,
            family_label=labels_arr[idx],
            mean_elbow_y=0.0,
            static_clearance=sc,
            interp_min_clearance=sc,
            manipulability=1.0,
            score_prior=score,
            is_collision_free=bool(col_free[idx]),
            family_method="goal_frame_midlink",
            lateral_bin=lat_bins[idx],
            vertical_bin=vb_list[idx] if vb_list is not None else None,
        ))
    return infos


# ---------------------------------------------------------------------------
# YZ-cross quadrant classifier
# ---------------------------------------------------------------------------

def classify_ik_goals_yz_cross_quadrant(
    q_start: np.ndarray,
    ik_goals: List[np.ndarray],
    obstacles: List,
    threshold_mode: str = "adaptive",
    landmark_mode: str = "combined",
    lp_goals: Optional[np.ndarray] = None,
) -> Tuple[List["IKGoalInfo"], dict]:
    """
    YZ-cross quadrant classifier for frontal_yz_cross scenarios.

    Classifies each IK goal by the position of key arm links relative to
    the cross center:
      - Y-axis (lateral): midlink Y → "pos_y", "neg_y", or "ctr_y"
      - Z-axis (vertical): wrist Z  → "above", "below", or "mid_z"

    Label format (landmark_mode="combined"): "mid_{y_side}_wrist_{z_side}"
    Label format (landmark_mode="wrist"):    "wrist_{z_side}"
    Label format (landmark_mode="midlink"):  "mid_{y_side}"

    Returns:
        (list_of_IKGoalInfo, diagnostics_dict)

    diagnostics_dict keys: cross_center, has_above_family, has_below_family,
                           landmark_mode, selected_landmark, tau_y, tau_z,
                           family_counts
    """
    q_start = np.asarray(q_start, dtype=float)
    n_goals = len(ik_goals)

    _empty_diag: dict = {
        "cross_center": [0.0, 0.0, 0.0],
        "has_above_family": False, "has_below_family": False,
        "landmark_mode": landmark_mode, "selected_landmark": landmark_mode,
        "tau_y": 0.01, "tau_z": 0.01, "family_counts": {},
    }
    if n_goals == 0:
        return [], _empty_diag

    # Detect cross geometry from obstacle names
    hb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_horizontal_bar"), None)
    vb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_vertical_bar"), None)

    cross_center = np.array(hb.position, dtype=float) if hb is not None else np.zeros(3)

    if threshold_mode == "adaptive" and hb is not None and vb is not None:
        bar_half_y = float(np.asarray(hb.size, dtype=float)[1])
        bar_half_z = float(np.asarray(vb.size, dtype=float)[2])
        tau_y = max(0.005, bar_half_y / 20.0)
        tau_z = max(0.005, bar_half_z / 20.0)
    else:
        tau_y = _CENTRE_THRESH
        tau_z = _CENTRE_THRESH

    # Batch FK: (n, 8, 3) — reuse precomputed result when caller supplies it.
    if lp_goals is not None:
        lp = lp_goals
    else:
        qs  = np.stack([np.asarray(g, dtype=float) for g in ik_goals])
        lp  = _panda_fk_batch(qs)

    midlink_pos = lp[:, 4, :]   # link5 — forearm mid-chain
    wrist_pos   = lp[:, 6, :]   # link7 — wrist

    # Batch clearance
    has_obs = _has_active_obstacles(obstacles)
    if has_obs:
        obs_rot    = _precompute_obs_rotations(obstacles)
        static_cls = _batch_clearance_from_lp(lp, obstacles, obs_rot)   # (n,)
    else:
        static_cls = np.full(n_goals, float("inf"))

    cy = float(cross_center[1])
    cz = float(cross_center[2])

    infos: List[IKGoalInfo] = []
    for i, q_raw in enumerate(ik_goals):
        my = float(midlink_pos[i, 1]) - cy
        wz = float(wrist_pos[i,   2]) - cz

        if my > tau_y:
            y_side = "pos_y"
        elif my < -tau_y:
            y_side = "neg_y"
        else:
            y_side = "ctr_y"

        if wz > tau_z:
            z_side = "above"
        elif wz < -tau_z:
            z_side = "below"
        else:
            z_side = "mid_z"

        if landmark_mode == "wrist":
            label = f"wrist_{z_side}"
        elif landmark_mode == "midlink":
            label = f"mid_{y_side}"
        else:
            label = f"mid_{y_side}_wrist_{z_side}"

        sc = float(static_cls[i])
        infos.append(IKGoalInfo(
            q_goal=np.asarray(q_raw, dtype=float),
            goal_idx=i,
            family_label=label,
            mean_elbow_y=float(midlink_pos[i, 1]),
            static_clearance=sc,
            interp_min_clearance=sc,
            manipulability=1.0,
            score_prior=1.0 / max(sc, 1e-3),
            is_collision_free=sc > 0.0,
            family_method="yz_cross_quadrant",
        ))

    labels = [info.family_label for info in infos]
    fc: dict = {}
    for lab in labels:
        fc[lab] = fc.get(lab, 0) + 1

    diag: dict = {
        "cross_center":      cross_center.tolist(),
        "has_above_family":  any("above" in lab for lab in labels),
        "has_below_family":  any("below" in lab for lab in labels),
        "landmark_mode":     landmark_mode,
        "selected_landmark": landmark_mode,
        "tau_y":             tau_y,
        "tau_z":             tau_z,
        "family_counts":     fc,
    }
    return infos, diag


# ---------------------------------------------------------------------------
# Main classification function (dispatcher)
# ---------------------------------------------------------------------------

def classify_ik_goals(
    q_start: np.ndarray,
    ik_goals: List[np.ndarray],
    obs_list: List[Obstacle],
    col_fn: Optional[Callable[[np.ndarray], bool]] = None,
    margin: float = 0.0,
    w_clearance: float = 2.0,
    w_interp: float = 1.0,
    w_manip: float = 0.5,
    family_classifier_mode: str = "goal_frame_midlink",
    verbose: bool = False,
    target_pos: Optional[np.ndarray] = None,
    link_pos_grid: Optional[np.ndarray] = None,
    midlink_sample_mode: str = "midpoint",
    label_detail: str = "lateral",
) -> List[IKGoalInfo]:
    """
    Classify IK goals by posture family and compute per-goal quality scores.

    Args:
        q_start:    Robot start configuration.
        ik_goals:   List of candidate goal configurations.
        obs_list:   Collision-enabled obstacles (for clearance scoring).
        col_fn:     Optional external collision checker (True = free).
        margin:     Extra clearance margin applied to static check.
        w_*:        Weights for prior score computation.
        family_classifier_mode:
                    "goal_frame_midlink" — fast task-relative route-family proxy (default).
                    "elbow_y"            — mean elbow-Y along interpolation.
                    "closest_passage"    — obstacle-relative signature (slow, debug).
                    "auto"               — prefer goal_frame_midlink; fall back to
                                          elbow_y if result is degenerate.
        midlink_sample_mode:
                    "midpoint" (default) — one FK at t=0.5 per goal; fastest.
                    "mean"               — average over n_interp_steps samples.
                    "auto"               — midpoint first; fallback to 3-sample on collapse.
        target_pos:     Task-space goal position (grasptarget). Used by goal_frame_midlink.
        link_pos_grid:  Pre-built FK grid (n_goals, n_steps, 8, 3) for reuse.
        verbose:    Log per-candidate classification details.

    Returns:
        List of IKGoalInfo, one per input goal, in the same order.
    """
    if family_classifier_mode == "goal_frame_midlink":
        return classify_ik_goals_goal_frame_midlink(
            q_start, ik_goals, obs_list,
            col_fn=col_fn, margin=margin,
            w_clearance=w_clearance, w_interp=w_interp, w_manip=w_manip,
            target_pos=target_pos, verbose=verbose,
            link_pos_grid=link_pos_grid,
            midlink_sample_mode=midlink_sample_mode,
            label_detail=label_detail,
        )

    if family_classifier_mode == "closest_passage":
        return classify_ik_goals_closest_passage(
            q_start, ik_goals, obs_list,
            col_fn=col_fn, margin=margin,
            w_clearance=w_clearance, w_interp=w_interp, w_manip=w_manip,
            verbose=verbose, link_pos_grid=link_pos_grid,
        )

    if family_classifier_mode == "auto":
        if obs_list:
            result = classify_ik_goals_goal_frame_midlink(
                q_start, ik_goals, obs_list,
                col_fn=col_fn, margin=margin,
                w_clearance=w_clearance, w_interp=w_interp, w_manip=w_manip,
                target_pos=target_pos, verbose=verbose,
                link_pos_grid=link_pos_grid,
            )
            labels = [info.family_label for info in result]
            if len(set(labels)) > 1:
                return result
            if verbose:
                logging.info(
                    "classify_ik_goals auto: single-label result (%s); falling back to elbow_y.",
                    labels[0] if labels else "n/a",
                )
        elif verbose:
            logging.info("classify_ik_goals auto: no obstacles; falling back to elbow_y.")

    # elbow_y (explicit mode and auto fallback)
    return _classify_ik_goals_elbow_y(
        q_start, ik_goals, obs_list,
        col_fn=col_fn, margin=margin,
        w_clearance=w_clearance, w_interp=w_interp, w_manip=w_manip,
    )


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------

def select_family_representatives(
    goal_infos: List[IKGoalInfo],
    top_k_per_family: int = 1,
    require_collision_free: bool = True,
) -> List[IKGoalInfo]:
    """
    Select representative goals ensuring family diversity.

    For each posture family, picks the top_k_per_family goals by score_prior
    (lower = better). Collision-free goals are preferred over blocked ones
    unless require_collision_free is False.

    Works with any family_label strings — both elbow-Y and closest-passage labels.
    """
    families: Dict[str, List[IKGoalInfo]] = {}
    for info in goal_infos:
        if require_collision_free and not info.is_collision_free:
            continue
        families.setdefault(info.family_label, []).append(info)

    representatives: List[IKGoalInfo] = []
    for label in sorted(families.keys()):
        ranked = sorted(families[label], key=lambda x: x.score_prior)
        representatives.extend(ranked[:top_k_per_family])

    if not representatives:
        candidates = sorted(goal_infos, key=lambda x: x.score_prior)
        representatives = candidates[:max(1, top_k_per_family)]

    return representatives


def rank_goals_within_family(
    goal_infos: List[IKGoalInfo],
    family_label: str,
) -> List[IKGoalInfo]:
    """Return goals of a given family sorted best-first by score_prior."""
    family_goals = [g for g in goal_infos if g.family_label == family_label]
    return sorted(family_goals, key=lambda x: x.score_prior)


def goals_from_infos(infos: List[IKGoalInfo]) -> List[np.ndarray]:
    """Extract q_goal arrays from a list of IKGoalInfo objects."""
    return [info.q_goal for info in infos]


def best_alternative(
    goal_infos: List[IKGoalInfo],
    blacklisted_goal_indices: List[int],
    blacklisted_families: Optional[List[str]] = None,
) -> Optional[IKGoalInfo]:
    """
    Pick the best available goal not in the blacklist.

    Prefers a different family from any blacklisted families.
    Falls back to any non-blacklisted goal if needed.
    """
    bl_idx = set(blacklisted_goal_indices)
    bl_fam = set(blacklisted_families or [])

    candidates = [g for g in goal_infos
                  if g.goal_idx not in bl_idx
                  and g.family_label not in bl_fam
                  and g.is_collision_free]
    if candidates:
        return min(candidates, key=lambda x: x.score_prior)

    candidates = [g for g in goal_infos
                  if g.goal_idx not in bl_idx and g.is_collision_free]
    if candidates:
        return min(candidates, key=lambda x: x.score_prior)

    candidates = [g for g in goal_infos if g.goal_idx not in bl_idx]
    if candidates:
        return min(candidates, key=lambda x: x.score_prior)

    return None


def select_escape_family(
    goal_infos: List[IKGoalInfo],
    failed_families: List[str],
    blacklisted_goal_indices: Optional[List[int]] = None,
) -> Optional[IKGoalInfo]:
    """
    Choose a family candidate whose escape topology differs from all that failed.

    For elbow-Y families: selects by family_escape_score (existing behavior).
    For closest-passage families: prefers candidates with a different lateral bin
    than the failed families, then falls back to score_prior.
    """
    bl_idx = set(blacklisted_goal_indices or [])
    bl_fam = set(failed_families)

    is_closest_passage = any(
        g.family_method in ("closest_passage", "goal_frame_midlink")
        for g in goal_infos
    )

    candidates = [g for g in goal_infos
                  if g.goal_idx not in bl_idx
                  and g.family_label not in bl_fam
                  and g.is_collision_free]

    if candidates:
        if is_closest_passage:
            failed_laterals = {
                g.lateral_bin for g in goal_infos
                if g.family_label in bl_fam and g.lateral_bin
            }

            def _cp_escape_key(x: IKGoalInfo) -> tuple:
                different = bool(x.lateral_bin and x.lateral_bin not in failed_laterals)
                return (0 if different else 1, x.score_prior)

            return min(candidates, key=_cp_escape_key)
        else:
            return min(candidates, key=lambda x: (x.family_escape_score, x.score_prior))

    candidates = [g for g in goal_infos
                  if g.goal_idx not in bl_idx and g.is_collision_free]
    if candidates:
        return min(candidates, key=lambda x: x.score_prior)

    return None
