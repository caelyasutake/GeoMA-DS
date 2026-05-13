"""
Factory for GeoMultiAttractorDS.

Converts a ScenarioSpec (ik_goals + obstacles) into a ready-to-use
GeoMultiAttractorDS without requiring BiRRT.

The factory:
  1. Builds a clearance_fn (q) -> float from the scenario's collision obstacles.
  2. Optionally expands IK coverage to fill missing homotopy classes.
  3. Classifies each IK goal into an IKAttractor (family, clearance, manipulability).
  4. Returns a GeoMultiAttractorDS ready to use as a drop-in DS source.

PathDS and birrt.py are untouched — this is a parallel instantiation path.

Usage::

    from src.solver.ds.factory import build_geo_multi_attractor_ds
    from src.scenarios.scenario_builders import build_frontal_i_barrier_lr

    spec = build_frontal_i_barrier_lr("easy")
    ds   = build_geo_multi_attractor_ds(spec)
    qdot, result = ds.compute(q)

    # With coverage expansion (for cross-barrier or other multi-window scenarios):
    from src.solver.ik.coverage_expansion import CoverageConfig
    ds = build_geo_multi_attractor_ds(spec, coverage_config=CoverageConfig())
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scenarios.scenario_schema import Obstacle, ScenarioSpec
from src.solver.ds.geo_multi_attractor_ds import (
    EscapeWaypointCandidate,
    GeoMultiAttractorDS,
    GeoMultiAttractorDSConfig,
    IKAttractor,
    ShellWaypointCandidate,
    _position_jacobian,
)
from src.solver.planner.collision import (
    _panda_link_positions,
    _panda_fk_batch,
    _batch_clearance_from_lp,
    _precompute_obs_rotations,
    _LINK_RADII,
)
from src.evaluation.path_quality import _link_clearance_to_obstacle

_PANDA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8975, -0.0175, -2.8973])
_PANDA_Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8975,  3.7525,  2.8973])


# ---------------------------------------------------------------------------
# Clearance function builder
# ---------------------------------------------------------------------------

def _make_batch_from_lp_fn(obstacles: List[Obstacle]):
    """
    Return  (lp: (N, 8, 3)) -> (N,)  that computes minimum signed clearances
    from pre-computed link positions using vectorised NumPy (no FK inside).

    Intended as the fast-path companion to _make_clearance_fn.  Pass the
    returned callable to GeoMultiAttractorDS via batch_from_lp_fn so that
    gradient, Jacobian, and attractor scoring share one _panda_fk_batch call
    each rather than issuing 14 + 7 + N_att individual FK evaluations.

    Obstacle rotations are precomputed once in the closure.
    """
    obs_R_inv = _precompute_obs_rotations(obstacles)

    def _fn(lp: np.ndarray) -> np.ndarray:
        return _batch_clearance_from_lp(lp, obstacles, obs_R_inv)

    return _fn


def _make_clearance_fn(obstacles: List[Obstacle]):
    """
    Return a callable  (q: np.ndarray) -> float  giving the minimum signed
    clearance across all link spheres and all collision-enabled obstacles.

    Positive = free; negative = penetrating.
    Uses the same sphere-swept link model as the BiRRT planner.
    """
    link_radii = [float(_LINK_RADII.get(i, 0.08)) for i in range(8)]

    def _clearance_fn(q: np.ndarray) -> float:
        link_pos = _panda_link_positions(np.asarray(q, dtype=float))
        min_cl = float("inf")
        for i, lp in enumerate(link_pos):
            r = link_radii[i]
            for obs in obstacles:
                cl = _link_clearance_to_obstacle(np.array(lp, dtype=float), r, obs)
                if cl < min_cl:
                    min_cl = cl
        return min_cl

    return _clearance_fn


# ---------------------------------------------------------------------------
# IK goal → IKAttractor classifier
# ---------------------------------------------------------------------------

def _classify_family_elbow_y(q_goal: np.ndarray) -> str:
    """Elbow-Y fallback: classify a single goal config by elbow link3 Y position."""
    _ELBOW_IDX    = 3
    _CENTRE_THRESH = 0.04
    elbow_y = float(_panda_link_positions(q_goal)[_ELBOW_IDX][1])
    if elbow_y > _CENTRE_THRESH:
        return "elbow_fwd"
    elif elbow_y < -_CENTRE_THRESH:
        return "elbow_back"
    return "elbow_center"


def _goal_manipulability(q_goal: np.ndarray) -> float:
    """sqrt(det(J Jᵀ)) for the 3×7 position Jacobian at q_goal."""
    J = _position_jacobian(q_goal)
    JJT = J @ J.T
    return float(np.sqrt(max(0.0, np.linalg.det(JJT))))


def ik_goals_to_attractors(
    ik_goals: List[np.ndarray],
    clearance_fn,
    q_start: Optional[np.ndarray] = None,
    obstacles: Optional[List] = None,
    lp_grid: Optional[np.ndarray] = None,
    n_interp: int = 5,
    family_classifier_mode: str = "goal_frame_midlink",
    target_pos: Optional[np.ndarray] = None,
    midlink_sample_mode: str = "midpoint",
    label_detail: str = "lateral",
    precomputed_infos: Optional[List] = None,
) -> List[IKAttractor]:
    """
    Convert a list of raw IK goal configurations to IKAttractor objects.

    When q_start and obstacles are provided, uses the family classifier
    (default: goal_frame_midlink) for the family label.  Falls back to
    elbow-Y when obstacles are absent or q_start is not given.

    Args:
        ik_goals:               List of (7,) joint configs.
        clearance_fn:           (q) -> float — signed clearance function.
        q_start:                Start configuration — required for classification.
        obstacles:              Collision obstacles — required for classification.
        lp_grid:                Pre-built FK grid (n_goals, n_interp, 8, 3) — reused.
        n_interp:               Interpolation steps (default 5).
        family_classifier_mode: Classifier mode passed to classify_ik_goals.
        target_pos:             Task-space goal position for goal_frame_midlink.
        midlink_sample_mode:    FK sampling mode for goal_frame_midlink classifier.

    Returns:
        List of IKAttractor, one per goal, with family/clearance/manipulability set.
    """
    has_obstacles = bool(obstacles)

    if precomputed_infos is not None:
        infos   = precomputed_infos
        families = [info.family_label for info in infos]
    elif q_start is not None and obstacles is not None:
        from src.solver.ik.goal_selection import classify_ik_goals
        infos = classify_ik_goals(
            np.asarray(q_start, dtype=float), ik_goals, obstacles,
            family_classifier_mode=family_classifier_mode,
            target_pos=target_pos,
            link_pos_grid=lp_grid,
            midlink_sample_mode=midlink_sample_mode,
            label_detail=label_detail,
        )
        families = [info.family_label for info in infos]
    else:
        infos   = None
        families = [_classify_family_elbow_y(np.asarray(q, dtype=float)) for q in ik_goals]

    # goal_frame_midlink supplies its own clearance via static_cls; skip
    # the expensive FD-Jacobian manipulability (weight 0.5 vs clearance 3.5).
    skip_manip = (family_classifier_mode == "goal_frame_midlink")

    attractors = []
    for i, q_goal in enumerate(ik_goals):
        q = np.asarray(q_goal, dtype=float)
        if has_obstacles:
            # Reuse static_clearance from classifier when available, avoiding N
            # redundant serial clearance_fn(q) calls (each requires a full FK pass).
            if infos is not None:
                clearance = float(infos[i].static_clearance)
            else:
                clearance = float(clearance_fn(q))
            manip = 1.0 if skip_manip else _goal_manipulability(q)
        else:
            clearance = float("inf")
            manip = 1.0
        attractors.append(IKAttractor(
            q_goal=q,
            family=families[i],
            clearance=clearance,
            manipulability=manip,
        ))
    return attractors


# ---------------------------------------------------------------------------
# Obstacle-type detection
# ---------------------------------------------------------------------------

def _detect_cross_barrier(obstacles: List[Obstacle]) -> bool:
    """True when the obstacle set contains both a center_post and a horiz_bar."""
    names = {o.name for o in obstacles if o.collision_enabled}
    return "center_post" in names and "horiz_bar" in names


def _cross_barrier_geometry(obstacles: List[Obstacle]):
    """
    Extract (x_post, z_mid, x_post_half, z_bar_half) from cross-barrier obstacles.
    Returns None if the expected obstacles are not found.
    """
    cp = next((o for o in obstacles if o.name == "center_post"), None)
    hb = next((o for o in obstacles if o.name == "horiz_bar"), None)
    if cp is None or hb is None:
        return None
    # center_post: position[0]=x_post, position[2]=z_mid, size[0]=x_post_half
    x_post      = float(cp.position[0])
    z_mid       = float(cp.position[2])
    x_post_half = float(cp.size[0])
    z_bar_half  = float(hb.size[2])
    return x_post, z_mid, x_post_half, z_bar_half


# ---------------------------------------------------------------------------
# Static attractor property injection
# ---------------------------------------------------------------------------

def _inject_attractor_static_scores(
    attractors: List[IKAttractor],
    q_start: np.ndarray,
    batch_from_lp_fn,
    window_label_fn=None,
    n_interp: int = 12,
    lp_all: Optional[np.ndarray] = None,
) -> None:
    """
    Compute and inject static properties into each IKAttractor in-place:

        straight_line_min_clearance — minimum clearance along the straight
            joint-space interpolation from q_start to q_goal (12 samples),
            evaluated via one vectorised _panda_fk_batch call.

        window_label — gate window label (cross-barrier only, else "none").

        static_score = 2.0 * goal_clearance
                     + 1.5 * straight_line_min_clearance
                     + 0.5 * manipulability

    The static_score strongly favours attractors whose direct joint-space
    paths from the start have high obstacle clearance (routing around the
    Cross rather than through it), while keeping manipulable goal configs.
    """
    if not attractors:
        return

    q_s = np.asarray(q_start, dtype=float)

    if lp_all is None:
        ts  = np.linspace(0.0, 1.0, n_interp, dtype=float)
        rows = []
        for att in attractors:
            for t in ts:
                rows.append((1.0 - t) * q_s + t * att.q_goal)
        q_all  = np.array(rows, dtype=float)
        lp_all = _panda_fk_batch(q_all)             # (n_att*n_interp, 8, 3)

    cl_all  = batch_from_lp_fn(lp_all)              # (n_att*n_interp,)
    cl_mat  = cl_all.reshape(len(attractors), -1)   # (n_att, n_interp)

    for i, att in enumerate(attractors):
        sl_cl = float(cl_mat[i].min())
        att.straight_line_min_clearance = sl_cl
        att.window_label = (
            window_label_fn(att.q_goal, q_s) if window_label_fn is not None else "none"
        )
        att.static_score = (
            2.0 * att.clearance
            + 1.5 * sl_cl
            + 0.5 * att.manipulability
        )


# ---------------------------------------------------------------------------
# Online IK generation helper
# ---------------------------------------------------------------------------

def _generate_ik_goals_online(
    spec: ScenarioSpec,
    clearance_fn,
    batch_from_lp_fn=None,
    *,
    batch_size: int = 1000,
    num_solutions: int = 8,
    min_clearance_m: float = 0.03,
    pose_tol_mm: float = 5.0,
    ik_filter_mode: str = "safe",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Run HJCD-IK and return collision-filtered, deduplicated IK goals.

    Args:
        batch_from_lp_fn: Optional vectorised clearance function (N,8,3)->( N,).
                          When provided, clearance filtering uses one FK batch
                          call instead of N individual calls.

    Returns (goals, meta) with detailed timing breakdown.
    """
    from src.solver.ik.hjcd_wrapper import solve_batch

    t0 = time.perf_counter()

    env_config = {"obstacles": spec.obstacles_as_hjcd_dict()} if spec.obstacles else None
    pose_tol_m = pose_tol_mm * 1e-3

    # --- HJCD solve (includes dedup/cluster inside solve_batch) ---
    t_solve = time.perf_counter()
    result = solve_batch(
        target_pose=spec.target_pose,
        env_config=env_config,
        batch_size=batch_size,
        num_solutions=num_solutions,
    )
    hjcd_solve_batch_ms = (time.perf_counter() - t_solve) * 1000.0
    sb = getattr(result, "solve_breakdown", {})

    # --- pose-error filter ---
    t_pose = time.perf_counter()
    after_pose = [
        (q, float(pe))
        for q, pe in zip(result.solutions, result.metadata.pos_errors)
        if float(pe) <= pose_tol_m
    ]
    pose_filter_ms = (time.perf_counter() - t_pose) * 1000.0

    # --- joint-limit filter ---
    t_jl = time.perf_counter()
    after_jl = [
        (q, pe) for q, pe in after_pose
        if not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))
    ]
    joint_limit_filter_ms = (time.perf_counter() - t_jl) * 1000.0

    # --- clearance filter (vectorised when batch_from_lp_fn available) ---
    # Skipped when ik_filter_mode="minimal"; GeoMA-DS attractor scoring handles safety.
    t_cl = time.perf_counter()
    if ik_filter_mode == "safe" and bool(spec.obstacles):
        candidates = [q for q, _ in after_jl]
        if batch_from_lp_fn is not None and candidates:
            qs_arr  = np.stack(candidates)
            lp_all  = _panda_fk_batch(qs_arr)          # (N, 8, 3)
            clears  = batch_from_lp_fn(lp_all)          # (N,)
            valid   = [q for q, ok in zip(candidates, clears >= min_clearance_m) if ok]
        else:
            valid = [q for q, _ in after_jl if clearance_fn(q) >= min_clearance_m]
    else:
        valid = [q for q, _ in after_jl]
    clearance_filter_ms = (time.perf_counter() - t_cl) * 1000.0

    ik_generation_ms = (time.perf_counter() - t0) * 1000.0

    meta: Dict[str, Any] = {
        # totals
        "ik_num_raw":              result.metadata.num_raw,
        "ik_num_after_dedup":      result.metadata.num_unique,
        "ik_num_after_pose_filter": len(after_pose),
        "ik_num_after_jl_filter":  len(after_jl),
        "ik_num_after_filter":     len(valid),
        # timing breakdown
        "hjcd_solve_batch_ms":     hjcd_solve_batch_ms,
        "hjcd_ms":                 sb.get("hjcd_ms", 0.0),
        "dedup_ms":                sb.get("dedup_ms", 0.0),
        "pose_filter_ms":          pose_filter_ms,
        "joint_limit_filter_ms":   joint_limit_filter_ms,
        "clearance_filter_ms":     clearance_filter_ms,
        "ik_generation_ms":        ik_generation_ms,
        # config
        "ik_config": {
            "batch_size":      batch_size,
            "num_solutions":   num_solutions,
            "pose_tol_mm":     pose_tol_mm,
            "min_clearance_m": min_clearance_m,
            "ik_filter_mode":  ik_filter_mode,
            "target_pos":      list(spec.target_pose["position"]) if spec.target_pose else None,
            "target_quat":     list(spec.target_pose["quaternion_wxyz"]) if spec.target_pose else None,
        },
    }
    return valid, meta


def warmup_hjcdik(*, batch_size: int = 1000, num_solutions: int = 8) -> float:
    """
    Run a throwaway HJCD-IK call to absorb CUDA context initialization and
    GPU memory allocation at production batch size.

    Returns the warmup wall time in milliseconds.  Call once before timed
    benchmark scenarios so per-scenario planner timing reflects warm IK latency.
    Uses batch_size=1000 by default to match the production call and fully absorb
    GPU buffer allocation; a smaller size only warms CUDA init, not allocation.

    For lightweight per-build rewarm (restoring GPU warm state after CPU-only work),
    use batch_size=64, num_solutions=1 to minimize rewarm overhead.
    """
    from src.solver.ik.hjcd_wrapper import solve_batch

    dummy_target = {"position": [0.307, 0.0, 0.59], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    t0 = time.perf_counter()
    try:
        solve_batch(target_pose=dummy_target, batch_size=batch_size, num_solutions=num_solutions)
    except Exception:
        pass  # solutions may be 0 for this pose; warmup still initialises CUDA
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Goal-shell escape attractor generation
# ---------------------------------------------------------------------------

def _point_clearance_to_obstacles(point: np.ndarray, obstacles: List) -> float:
    """Minimum signed clearance from a 3-D task-space point to all obstacles."""
    min_cl = float("inf")
    for obs in obstacles:
        cl = _link_clearance_to_obstacle(np.asarray(point, dtype=float), 0.0, obs)
        if cl < min_cl:
            min_cl = cl
    return min_cl


def _yz_cross_geometry(obstacles: List) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Return (cross_center, bar_half_y, bar_half_z) from frontal_yz_cross obstacles.
    Detects yz_cross_horizontal_bar and yz_cross_vertical_bar by name.
    Returns None when not a yz_cross scene.
    """
    hb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_horizontal_bar"), None)
    vb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_vertical_bar"), None)
    if hb is None or vb is None:
        return None
    cx         = np.array(hb.position, dtype=float)
    bar_half_y = float(np.asarray(hb.size)[1])   # long Y half-extent of horizontal bar
    bar_half_z = float(np.asarray(vb.size)[2])   # long Z half-extent of vertical bar
    return cx, bar_half_y, bar_half_z


def compute_clearance_recovery_direction(
    ee_pos: np.ndarray,
    obstacles: List,
    fallback_direction: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, dict]:
    """
    Returns (n_away, clearance, metadata) where n_away is the unit vector
    pointing away from the closest collision obstacle.

    For box obstacles the closest surface point is computed analytically;
    the direction is ee_pos - closest_point.  A fallback is used when the
    point is inside the obstacle or the vector is degenerate.
    """
    from src.scenarios.scenario_schema import Obstacle as _Obs

    best_cl     = float("inf")
    best_normal = None
    best_name   = "none"
    best_closest = np.zeros(3)

    for obs in obstacles:
        if not getattr(obs, "collision_enabled", True):
            continue
        obs_pos = np.asarray(obs.position, dtype=float)
        t = getattr(obs, "type", "box").lower()

        if t == "box":
            half = np.asarray(obs.size, dtype=float)
            # Orientation
            ori = np.asarray(getattr(obs, "orientation_wxyz", [1, 0, 0, 0]), dtype=float)
            if abs(ori[0] - 1.0) < 1e-6:
                R_inv = np.eye(3)
            else:
                w, x, y, z = ori
                R = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
                ], dtype=float)
                R_inv = R.T
            local = R_inv @ (ee_pos - obs_pos)
            clamped = np.clip(local, -half, half)
            closest_local = clamped
            vec_local = local - closest_local
            cl = float(np.linalg.norm(vec_local))
            if cl < best_cl:
                best_cl = cl
                # Normal in world frame
                if cl > 1e-8:
                    R = R_inv.T
                    best_normal  = R @ (vec_local / cl)
                else:
                    best_normal  = None
                best_name    = getattr(obs, "name", "box")
                best_closest = obs_pos + R_inv.T @ closest_local

        elif t == "sphere":
            r   = float(obs.size[0])
            vec = ee_pos - obs_pos
            d   = float(np.linalg.norm(vec))
            cl  = d - r
            if cl < best_cl:
                best_cl     = cl
                best_normal = vec / (d + 1e-12)
                best_name   = getattr(obs, "name", "sphere")
                best_closest = obs_pos + r * best_normal

    # Build n_away
    if best_normal is not None and np.linalg.norm(best_normal) > 1e-8:
        n_away = best_normal / (np.linalg.norm(best_normal) + 1e-12)
    elif fallback_direction is not None:
        n_away = np.asarray(fallback_direction, dtype=float)
        nrm = float(np.linalg.norm(n_away))
        n_away = n_away / (nrm + 1e-12)
    else:
        n_away = np.array([0.0, 0.0, 1.0])

    return n_away, best_cl, {"closest_obstacle": best_name, "closest_point": best_closest}


def _make_recovery_builder(
    obstacles: List,
    clearance_fn,
    cfg,
    portal_positions: Optional[List[np.ndarray]] = None,
) -> "Callable":
    """
    Returns a callable that, given the current link positions lp_base (8, 3),
    generates a clearance-recovery IKAttractor or returns None on failure.

    n_away is computed from the closest robot body link (not just EE), then
    blended toward the nearest portal waypoint when available:
        n_recover = normalize(0.6 * n_away + 0.4 * n_portal)

    Distances 0.06–0.14 m are tried until the recovery point clears
    cfg.recovery_target_clearance_m.
    """
    from src.solver.ik.hjcd_wrapper import solve_batch
    from src.solver.planner.collision import _LINK_RADII

    yz_geo = _yz_cross_geometry(obstacles)
    _link_radii = [float(_LINK_RADII.get(i, 0.08)) for i in range(8)]

    def build(lp_base: np.ndarray) -> "Optional[IKAttractor]":
        lp = np.asarray(lp_base, dtype=float)   # (8, 3)
        ee = lp[-1].copy()

        # Fallback: away from cross center in YZ plane
        fallback = None
        if yz_geo is not None:
            cx = np.asarray(yz_geo[0], dtype=float)
            yz_diff = np.array([0.0, ee[1] - cx[1], ee[2] - cx[2]])
            nrm = float(np.linalg.norm(yz_diff))
            if nrm > 1e-6:
                fallback = yz_diff / nrm

        # --- n_away from the closest robot body link (not just EE) ---
        # The trigger clearance is robot minimum body clearance; recovery direction
        # should come from the same pinned link, not the end-effector.
        min_body_cl = float("inf")
        n_away = None
        for i in range(len(lp)):
            r = _link_radii[i]
            n_i, cl_i, _ = compute_clearance_recovery_direction(
                np.array(lp[i], dtype=float), obstacles, fallback_direction=fallback
            )
            cl_body = cl_i - r   # sphere-body clearance at link i
            if cl_body < min_body_cl:
                min_body_cl = cl_body
                n_away = n_i

        # EE-point clearance for logging
        _, cl_now, meta = compute_clearance_recovery_direction(
            ee, obstacles, fallback_direction=fallback
        )

        if n_away is None or float(np.linalg.norm(n_away)) < 1e-8:
            n_away, _, _ = compute_clearance_recovery_direction(
                ee, obstacles, fallback_direction=fallback
            )

        # --- Portal-direction bias ---
        # Blend away-from-obstacle with toward-nearest-portal so recovery
        # moves toward the correct route, not just any clear direction.
        if portal_positions:
            _p_pos = min(portal_positions, key=lambda p: float(np.linalg.norm(p - ee)))
            n_portal = _p_pos - ee
            n_portal_nrm = float(np.linalg.norm(n_portal))
            if n_portal_nrm > 1e-6:
                n_portal = n_portal / n_portal_nrm
                n_blended = 0.6 * n_away + 0.4 * n_portal
                n_blended_nrm = float(np.linalg.norm(n_blended))
                if n_blended_nrm > 1e-6:
                    n_away = n_blended / n_blended_nrm

        # Try progressively larger distances until recovery point is clear
        x_recover = None
        for d in [0.06, 0.08, 0.10, 0.12, 0.14]:
            candidate = ee + d * n_away
            cl_cand   = _point_clearance_to_obstacles(candidate, obstacles)
            if cl_cand >= cfg.recovery_target_clearance_m:
                x_recover = candidate
                break

        if x_recover is None:
            print(
                f"[recovery IK failed: no candidate reached target clearance "
                f"ee_cl={cl_now:.4f} body_cl={min_body_cl:.4f} "
                f"n_away={np.round(n_away, 3).tolist()}]",
                flush=True,
            )
            return None

        print(
            f"[recovery_waypoint pos={np.round(x_recover, 3).tolist()} "
            f"dir={np.round(n_away, 3).tolist()} "
            f"closest_obs={meta['closest_obstacle']} "
            f"ee_cl={cl_now:.4f} "
            f"body_cl={min_body_cl:.4f} "
            f"cl_target={cfg.recovery_target_clearance_m:.3f}]",
            flush=True,
        )

        # Position-only IK: identity orientation with relaxed positional tolerance
        try:
            _r = solve_batch(
                target_pose={"position": x_recover.tolist(),
                             "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                batch_size=cfg.recovery_ik_batch_size,
                num_solutions=cfg.recovery_ik_num_solutions,
            )
            valid = [
                q for q, pe in zip(_r.solutions, _r.metadata.pos_errors)
                if float(pe) <= cfg.recovery_position_tol_m
                and np.all(np.isfinite(q))
            ]
        except Exception as exc:
            print(f"[recovery IK failed: {exc}]", flush=True)
            return None

        if not valid:
            print("[recovery IK failed: no solutions within position tolerance]", flush=True)
            return None

        q_rec = np.asarray(valid[0], dtype=float)
        return IKAttractor(
            q_goal         = q_rec,
            family         = "recovery_clearance",
            clearance      = float(clearance_fn(q_rec)),
            manipulability = _goal_manipulability(q_rec),
            kind           = "recovery",
            target_name    = "clearance_recovery",
            shell_sector   = "recovery",
            shell_waypoint = x_recover.copy(),
        )

    return build


def _build_yz_cross_deterministic_waypoints(
    cross_center: np.ndarray,
    bar_half_y: float,
    bar_half_z: float,
    obstacles: List,
    margin: float = 0.08,
) -> List[ShellWaypointCandidate]:
    """
    Two deterministic escape waypoints for the yz_cross scene.
    Both land below the horizontal bar and to the side of the vertical bar:
      escape_bottom_right:  +Y side, below cross
      escape_bottom_left:   -Y side, below cross
    """
    cx = np.asarray(cross_center, dtype=float)
    entries = [
        (cx + np.array([0.0,  bar_half_y + margin, -(bar_half_z + margin)]), "escape_bottom_right"),
        (cx + np.array([0.0, -(bar_half_y + margin), -(bar_half_z + margin)]), "escape_bottom_left"),
    ]
    candidates: List[ShellWaypointCandidate] = []
    for wp, sector in entries:
        cl  = _point_clearance_to_obstacles(wp, obstacles)
        dv  = wp - cx
        n   = float(np.linalg.norm(dv))
        direction = dv / n if n > 1e-12 else dv.copy()
        candidates.append(ShellWaypointCandidate(
            pos=wp.copy(), sector=sector, clearance=cl,
            score=float(cl), direction=direction,
        ))
    return candidates


def generate_goal_shell_waypoints(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacles: List,
    *,
    shell_radius: float,
    margin: float,
    sample_mode: str,
    cross_center: Optional[np.ndarray] = None,
    attempted_escape_families: Optional[set] = None,
) -> List[ShellWaypointCandidate]:
    """
    Generate task-space waypoint candidates on a spherical shell around goal_pos.

    Goal-shell escape attractors are not formal homotopy classes. They are
    homotopy-inspired task-space approach sectors sampled from a shell around
    the goal. These sectors provide temporary waypoint attractors when final-goal
    IK diversity is insufficient to escape a local minimum.

    Args:
        ee_pos:    Current end-effector position (for scoring).
        goal_pos:  Task-space goal position (shell centre).
        obstacles: Collision obstacles for clearance check.
        shell_radius: Shell radius (m).
        margin:    Minimum clearance for a candidate to be accepted (m).
        sample_mode: "axis_6" | "yz_plane_8" | "sphere_14" | "yz_cross"
        cross_center: Optional cross centre for below-cross bonus scoring.
        attempted_escape_families: Sectors already tried (penalised in score).

    Returns:
        List of ShellWaypointCandidate sorted by score descending.
    """
    if attempted_escape_families is None:
        attempted_escape_families = set()

    if sample_mode == "axis_6":
        dirs_raw = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ]
        sectors = [
            "shell_pos_x", "shell_neg_x",
            "shell_pos_y", "shell_neg_y",
            "shell_pos_z", "shell_neg_z",
        ]
    elif sample_mode == "yz_plane_8":
        dirs_raw = [
            [0,  1,  0], [0, -1,  0],
            [0,  0,  1], [0,  0, -1],
            [0,  1,  1], [0,  1, -1],
            [0, -1,  1], [0, -1, -1],
        ]
        sectors = [
            "shell_pos_y",         "shell_neg_y",
            "shell_pos_z",         "shell_neg_z",
            "shell_pos_y_pos_z",   "shell_pos_y_neg_z",
            "shell_neg_y_pos_z",   "shell_neg_y_neg_z",
        ]
    elif sample_mode == "sphere_14":
        dirs_raw = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        ]
        sectors = [
            "shell_pos_x",         "shell_neg_x",
            "shell_pos_y",         "shell_neg_y",
            "shell_pos_z",         "shell_neg_z",
            "shell_pos_x_pos_y",   "shell_neg_x_pos_y",
            "shell_pos_x_neg_y",   "shell_neg_x_neg_y",
            "shell_pos_y_pos_z",   "shell_neg_y_pos_z",
            "shell_pos_y_neg_z",   "shell_neg_y_neg_z",
        ]
    else:  # "yz_cross" — prioritises below-cross directions
        dirs_raw = [
            [0, +1, -1],
            [0, -1, -1],
            [0, +1,  0],
            [0, -1,  0],
            [0,  0, -1],
            [0, +1, +1],
            [0, -1, +1],
        ]
        sectors = [
            "shell_pos_y_neg_z",
            "shell_neg_y_neg_z",
            "shell_pos_y",
            "shell_neg_y",
            "shell_neg_z",
            "shell_pos_y_pos_z",
            "shell_neg_y_pos_z",
        ]

    # Normalise directions
    dirs_norm = []
    for d in dirs_raw:
        dv = np.array(d, dtype=float)
        n = float(np.linalg.norm(dv))
        dirs_norm.append(dv / n if n > 1e-12 else dv.copy())

    goal_pos = np.asarray(goal_pos, dtype=float)
    ee_pos   = np.asarray(ee_pos,   dtype=float)

    candidates: List[ShellWaypointCandidate] = []
    for d, sector in zip(dirs_norm, sectors):
        wp = goal_pos + shell_radius * d
        cl = _point_clearance_to_obstacles(wp, obstacles)
        if cl < margin:
            continue

        dist_to_goal  = float(np.linalg.norm(wp - goal_pos))
        dist_from_ee  = float(np.linalg.norm(wp - ee_pos))
        progress_bonus = float(np.linalg.norm(ee_pos - goal_pos)) - dist_to_goal
        cl_capped = min(float(cl), 1.0)   # cap so inf (no obstacles) doesn't swamp other terms

        score = 2.0 * cl_capped + 1.0 * progress_bonus - 0.3 * dist_from_ee

        if cross_center is not None and float(wp[2]) < float(cross_center[2]):
            score += 2.0

        if sector in attempted_escape_families:
            score -= 5.0

        candidates.append(ShellWaypointCandidate(
            pos=wp.copy(),
            sector=sector,
            clearance=cl,
            score=score,
            direction=d.copy(),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _build_escape_attractors(
    spec,
    obstacles: List,
    clearance_fn,
    target_pos: np.ndarray,
    cfg: GeoMultiAttractorDSConfig,
    *,
    cross_center: Optional[np.ndarray] = None,
) -> Tuple[List[IKAttractor], Dict[str, Any]]:
    """
    Build goal-shell escape attractors for Phase 1 (pre-build).

    For yz_cross scenes uses 2 deterministic waypoints (below-cross corners);
    for other scenes falls back to generic sphere-shell sampling.

    Each waypoint gets a small HJCD-IK batch (batch=128, solutions=1).
    If strict IK (original orientation, tol=1cm) fails, retries with relaxed
    orientation (identity quaternion, tol=5cm, no clearance check).

    Returns (escape_attractors, diagnostics_dict).
    """
    from src.solver.ik.hjcd_wrapper import solve_batch

    t_total = time.perf_counter()

    # --- Candidate generation ---
    t_cand = time.perf_counter()
    yz_geo = _yz_cross_geometry(obstacles)
    if yz_geo is not None:
        _cx, _bar_half_y, _bar_half_z = yz_geo
        candidates = _build_yz_cross_deterministic_waypoints(
            _cx, _bar_half_y, _bar_half_z, obstacles, margin=0.08,
        )
        candidate_mode = "yz_cross_deterministic"
    else:
        ee_pos_start = np.array(
            _panda_link_positions(np.asarray(spec.q_start, dtype=float))[-1],
            dtype=float,
        )
        candidates = generate_goal_shell_waypoints(
            ee_pos=ee_pos_start,
            goal_pos=target_pos,
            obstacles=obstacles,
            shell_radius=cfg.goal_shell_radius_m,
            margin=cfg.goal_shell_margin_m,
            sample_mode=cfg.goal_shell_sample_mode,
            cross_center=cross_center,
            attempted_escape_families=set(),
        )
        candidate_mode = cfg.goal_shell_sample_mode
    candidate_ms = (time.perf_counter() - t_cand) * 1000.0

    # --- Clearance filter (not penetrating) ---
    t_filter = time.perf_counter()
    valid_cands = [c for c in candidates if c.clearance >= 0.0]
    filter_ms   = (time.perf_counter() - t_filter) * 1000.0

    selected = valid_cands[:cfg.goal_shell_max_waypoints]

    # --- IK per candidate ---
    t_ik_all = time.perf_counter()
    escape_attractors: List[IKAttractor] = []
    diag_sectors    : List[str]  = []
    diag_ik_success : List[bool] = []
    n_fail_pose = n_fail_joint = n_fail_cl = 0
    ik_per_sector: List[Dict[str, Any]] = []

    env_config = {"obstacles": spec.obstacles_as_hjcd_dict()} if spec.obstacles else None
    orig_quat  = spec.target_pose["quaternion_wxyz"]
    n_sols     = cfg.goal_shell_ik_solutions_per_waypoint

    for cand in selected:
        t_ik_cand = time.perf_counter()
        sec_diag: Dict[str, Any] = {
            "sector": cand.sector, "pos": cand.pos.tolist(),
            "success": False, "mode": "none",
        }

        # strict IK: original orientation, 1 cm pose tolerance, clearance check
        try:
            _r = solve_batch(
                target_pose={"position": cand.pos.tolist(), "quaternion_wxyz": orig_quat},
                env_config=env_config,
                batch_size=128,
                num_solutions=n_sols,
            )
            _all = list(zip(_r.solutions, _r.metadata.pos_errors))
        except Exception as exc:
            _all = []
            sec_diag["error"] = str(exc)

        _after_pose = [q for q, pe in _all if float(pe) <= 0.01]
        _fp = len(_all) - len(_after_pose)
        _after_jl = [q for q in _after_pose
                     if not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))]
        _fj = len(_after_pose) - len(_after_jl)
        _after_cl = [q for q in _after_jl if float(clearance_fn(q)) >= 0.02]
        _fc = len(_after_jl) - len(_after_cl)
        n_fail_pose += _fp; n_fail_joint += _fj; n_fail_cl += _fc
        sec_diag.update({"strict_valid": len(_after_cl),
                         "pose_fail": _fp, "jl_fail": _fj, "cl_fail": _fc})
        valid = _after_cl

        # relaxed IK retry: identity orientation, 5 cm tolerance, no clearance check
        if not valid:
            try:
                _r2 = solve_batch(
                    target_pose={"position": cand.pos.tolist(),
                                 "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    env_config=env_config,
                    batch_size=128,
                    num_solutions=n_sols,
                )
                _rel = [
                    q for q, pe in zip(_r2.solutions, _r2.metadata.pos_errors)
                    if float(pe) <= 0.05
                    and not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))
                    and np.all(np.isfinite(q))
                ]
                sec_diag["relaxed_valid"] = len(_rel)
                if _rel:
                    print(
                        f"  [shell_escape] {cand.sector}: strict ik failed;"
                        f" retry relaxed → {len(_rel)} solutions",
                        flush=True,
                    )
                    valid = _rel
                    sec_diag["mode"] = "relaxed"
            except Exception:
                sec_diag["relaxed_valid"] = 0

        sec_diag["ik_ms"] = (time.perf_counter() - t_ik_cand) * 1000.0

        if valid:
            q_esc = np.asarray(valid[0], dtype=float)
            att = IKAttractor(
                q_goal=q_esc,
                family=f"escape_{cand.sector}",
                clearance=float(clearance_fn(q_esc)),
                manipulability=_goal_manipulability(q_esc),
                kind="escape",
                target_name="goal_shell_escape",
                shell_sector=cand.sector,
                shell_waypoint=cand.pos.copy(),
            )
            escape_attractors.append(att)
            diag_sectors.append(cand.sector)
            diag_ik_success.append(True)
            if sec_diag["mode"] == "none":
                sec_diag["mode"] = "strict"
            sec_diag["success"] = True
        else:
            diag_sectors.append(cand.sector)
            diag_ik_success.append(False)
        ik_per_sector.append(sec_diag)

    ik_ms    = (time.perf_counter() - t_ik_all) * 1000.0
    total_ms = (time.perf_counter() - t_total) * 1000.0

    diag: Dict[str, Any] = {
        "enabled":                 True,
        "candidate_mode":          candidate_mode,
        "num_waypoint_candidates": len(candidates),
        "num_valid_clearance":     len(valid_cands),
        "num_waypoints_selected":  len(selected),
        "num_escape_attractors":   len(escape_attractors),
        "num_ik_attempts":         len(selected),
        "num_ik_success":          len(escape_attractors),
        "num_ik_fail_pose":        n_fail_pose,
        "num_ik_fail_joint":       n_fail_joint,
        "num_ik_fail_clearance":   n_fail_cl,
        "sectors":                 diag_sectors,
        "ik_success":              diag_ik_success,
        "waypoints":               [c.pos.tolist() for c in selected],
        "ik_per_sector":           ik_per_sector,
        "shell_candidate_ms":      candidate_ms,
        "shell_filter_ms":         filter_ms,
        "shell_ik_ms":             ik_ms,
        "shell_total_ms":          total_ms,
    }
    return escape_attractors, diag


# ---------------------------------------------------------------------------
# Simple escape waypoint
# ---------------------------------------------------------------------------

def _build_simple_escape_waypoint_attractor(
    spec,
    obstacles: List,
    clearance_fn,
    cfg: GeoMultiAttractorDSConfig,
) -> Tuple[List[IKAttractor], Dict[str, Any]]:
    """
    Build obstacle-corner portal escape waypoint attractors for yz_cross scenarios.

    Generates 4 portals (bottom-right, bottom-left, top-right, top-left) from the
    actual obstacle geometry, scores them, and chooses first by current EE lateral
    side, second by goal lateral side.  Uses position-only IK (10 cm tolerance,
    3 orientation attempts) for each chosen portal.

    Returns (list_of_attractors, diagnostics_dict).
    """
    from src.solver.ik.hjcd_wrapper import solve_batch

    t0 = time.perf_counter()

    yz_geo = _yz_cross_geometry(obstacles)
    if yz_geo is None:
        return [], {"enabled": False, "reason": "no_yz_cross_geometry"}

    cross_center, bar_half_y, bar_half_z = yz_geo
    margin_y = 0.08
    margin_z = 0.08

    cx = np.asarray(cross_center, dtype=float)
    y_right = float(cx[1]) + bar_half_y
    y_left  = float(cx[1]) - bar_half_y
    z_bot   = float(cx[2]) - bar_half_z
    z_top   = float(cx[2]) + bar_half_z

    # 4 obstacle-corner portals
    portals = [
        ("portal_bottom_right", np.array([cx[0], y_right + margin_y, z_bot - margin_z])),
        ("portal_bottom_left",  np.array([cx[0], y_left  - margin_y, z_bot - margin_z])),
        ("portal_top_right",    np.array([cx[0], y_right + margin_y, z_top + margin_z])),
        ("portal_top_left",     np.array([cx[0], y_left  - margin_y, z_top + margin_z])),
    ]

    ee_start = np.array(
        _panda_link_positions(np.asarray(spec.q_start, dtype=float))[-1],
        dtype=float,
    )
    goal_pos = (np.array(spec.target_pose["position"], dtype=float)
                if spec.target_pose else cx.copy())

    # Score portals: clearance + proximity bonus + below-cross bonus
    scored: Dict[str, Tuple[np.ndarray, float, float]] = {}   # sector -> (pos, score, cl)
    for sector, pos in portals:
        cl = _point_clearance_to_obstacles(pos, obstacles)
        if cl < 0.03:
            continue
        cl_capped = min(float(cl), 1.0)
        score = (
            2.0 * cl_capped
            - 0.4 * float(np.linalg.norm(pos - ee_start))
            - 0.2 * float(np.linalg.norm(pos - goal_pos))
        )
        if pos[2] < cx[2] - margin_z:   # below cross: strong preference
            score += 2.0
        scored[sector] = (pos, score, cl)

    if not scored:
        return [], {
            "enabled": True,
            "reason": "all_portals_rejected_clearance",
            "portals": {s: {"cl": float(_point_clearance_to_obstacles(p, obstacles))} for s, p in portals},
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # Choose first by EE lateral side (same side as current arm), second by goal side
    current_side_y = 1.0 if float(ee_start[1]) >= float(cx[1]) else -1.0
    goal_side_y    = 1.0 if float(goal_pos[1])  >= float(cx[1]) else -1.0

    first_sector  = "portal_bottom_right" if current_side_y >= 0 else "portal_bottom_left"
    second_sector = "portal_bottom_right" if goal_side_y    >= 0 else "portal_bottom_left"

    chosen: List[str] = []
    for s in ([first_sector] + ([second_sector] if second_sector != first_sector else [])):
        if s in scored:
            chosen.append(s)
    # Fallback: if primary choices unavailable, use best-scored portal
    if not chosen:
        chosen = [max(scored, key=lambda s: scored[s][1])]

    # Position-only IK: try 3 orientations, 10 cm tolerance
    env_config   = {"obstacles": spec.obstacles_as_hjcd_dict()} if spec.obstacles else None
    orientations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ]

    attractors: List[IKAttractor] = []
    portal_diags: List[Dict[str, Any]] = []

    for sector in chosen:
        escape_pos, portal_score, portal_cl = scored[sector]
        valid: List[np.ndarray] = []
        tried_quats: List[Dict[str, Any]] = []

        for quat in orientations:
            try:
                _r = solve_batch(
                    target_pose={"position": escape_pos.tolist(), "quaternion_wxyz": quat},
                    env_config=env_config,
                    batch_size=cfg.escape_waypoint_batch_size,
                    num_solutions=cfg.escape_waypoint_num_solutions,
                )
                _cands = [
                    q for q, pe in zip(_r.solutions, _r.metadata.pos_errors)
                    if float(pe) <= 0.10
                    and not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))
                    and np.all(np.isfinite(q))
                ]
                tried_quats.append({
                    "quat": quat,
                    "n_returned": len(_r.solutions),
                    "n_valid": len(_cands),
                })
                if _cands:
                    valid = _cands
                    break
            except Exception as exc:
                tried_quats.append({"quat": quat, "error": str(exc)})

        portal_diags.append({
            "sector":          sector,
            "escape_pos":      escape_pos.tolist(),
            "portal_score":    float(portal_score),
            "portal_clearance": float(portal_cl),
            "ik_ok":           len(valid) > 0,
            "tried_quats":     tried_quats,
        })

        if valid:
            q_esc = np.asarray(valid[0], dtype=float)
            att = IKAttractor(
                q_goal=q_esc,
                family=f"escape_{sector}",
                clearance=float(clearance_fn(q_esc)),
                manipulability=_goal_manipulability(q_esc),
                kind="escape",
                target_name="simple_escape_waypoint",
                shell_sector=sector,
                shell_waypoint=escape_pos.copy(),
            )
            attractors.append(att)

    total_ms = (time.perf_counter() - t0) * 1000.0
    diag: Dict[str, Any] = {
        "enabled":           True,
        "cross_center":      cx.tolist(),
        "ee_start":          ee_start.tolist(),
        "goal_pos":          goal_pos.tolist(),
        "current_side_y":    float(current_side_y),
        "goal_side_y":       float(goal_side_y),
        "first_sector":      first_sector,
        "second_sector":     second_sector if second_sector != first_sector else None,
        "chosen_sectors":    chosen,
        "portal_diags":      portal_diags,
        "n_attractors_built": len(attractors),
        "total_ms":          total_ms,
        # backward-compat fields for existing logging in caller
        "escape_pos":  portal_diags[0]["escape_pos"] if portal_diags else [0, 0, 0],
        "sector":      portal_diags[0]["sector"]     if portal_diags else "none",
        "ik_ok":       len(attractors) > 0,
        "tried_quats": portal_diags[0].get("tried_quats", []) if portal_diags else [],
    }
    return attractors, diag


# ---------------------------------------------------------------------------
# Boundary escape waypoints (homotopy-proxy obstacle-corner portals)
# ---------------------------------------------------------------------------

def _yz_cross_full_geometry(obstacles: List):
    """
    Extract full bounding extents of the YZ-cross obstacle from both bars.

    Returns (cross_center, x_plane, y_left, y_right, z_low, z_high) or None.

    y_left / y_right are the outermost Y limits of either bar.
    z_low  / z_high  are the outermost Z limits of either bar.
    These define the axis-aligned bounding box of the full cross.
    """
    hb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_horizontal_bar"), None)
    vb = next((o for o in obstacles if getattr(o, "name", "") == "yz_cross_vertical_bar"), None)
    if hb is None or vb is None:
        return None

    hb_pos  = np.asarray(hb.position, dtype=float)
    hb_size = np.asarray(hb.size,     dtype=float)
    vb_pos  = np.asarray(vb.position, dtype=float)
    vb_size = np.asarray(vb.size,     dtype=float)

    cross_center = hb_pos.copy()   # both bars are centred at the same point
    x_plane      = float(hb_pos[0])

    horizontal_y_min = float(hb_pos[1]) - float(hb_size[1])
    horizontal_y_max = float(hb_pos[1]) + float(hb_size[1])
    vertical_y_min   = float(vb_pos[1]) - float(vb_size[1])
    vertical_y_max   = float(vb_pos[1]) + float(vb_size[1])

    horizontal_z_min = float(hb_pos[2]) - float(hb_size[2])
    horizontal_z_max = float(hb_pos[2]) + float(hb_size[2])
    vertical_z_min   = float(vb_pos[2]) - float(vb_size[2])
    vertical_z_max   = float(vb_pos[2]) + float(vb_size[2])

    y_left  = min(horizontal_y_min, vertical_y_min)
    y_right = max(horizontal_y_max, vertical_y_max)
    z_low   = min(horizontal_z_min, vertical_z_min)
    z_high  = max(horizontal_z_max, vertical_z_max)

    return cross_center, x_plane, y_left, y_right, z_low, z_high


def generate_obstacle_boundary_escape_waypoints(
    obstacles: List,
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    *,
    margin_m: float = 0.08,
    min_waypoint_clearance_m: float = 0.05,
    scene_type: str = "auto",
    n_y: int = 7,
    n_z: int = 7,
    max_candidates: int = 12,
) -> List[EscapeWaypointCandidate]:
    """
    Generate homotopy-proxy escape waypoint candidates by sampling a YZ grid around
    the obstacle and keeping the highest-scoring collision-free candidates.

    Each candidate is tagged with stage_tags indicating which route stage it serves:
      "below_stage_candidate"   — helps arm navigate below the cross center
      "lateral_stage_candidate" — below the vertical bar AND on the goal Y side
      "top_fallback_candidate"  — above the cross center (last resort)

    Candidates are scored by clearance at setup time for initial ranking.
    Runtime scoring (clearance + goal-progress + distance + IK cost) is applied
    in _online_generate_next_escape_attractor during execution.

    Args:
        obstacles:               Collision obstacles.
        ee_pos:                  Current end-effector position (3,).
        goal_pos:                Task-space goal position (3,).
        margin_m:                Extra margin beyond obstacle extents for sample domain.
        min_waypoint_clearance_m: Minimum clearance for a candidate to be kept.
        scene_type:              "auto" or "yz_cross".
        n_y, n_z:                Grid resolution in Y and Z.
        max_candidates:          Maximum candidates to return (top-scoring kept).

    Returns:
        List of EscapeWaypointCandidate sorted by score descending.
        Empty when scene geometry is not recognised.
    """
    ee_pos   = np.asarray(ee_pos,   dtype=float)
    goal_pos = np.asarray(goal_pos, dtype=float)

    # Geometry detection
    _is_yz_cross = (
        scene_type == "yz_cross"
        or (scene_type == "auto"
            and any(getattr(o, "name", "") in ("yz_cross_horizontal_bar",
                                               "yz_cross_vertical_bar")
                    for o in obstacles))
    )
    if not _is_yz_cross:
        return []

    geo = _yz_cross_full_geometry(obstacles)
    if geo is None:
        return []

    cross_center, x_plane, y_left, y_right, z_low, z_high = geo

    # Goal-side sign for stage tagging and metadata
    _goal_side_y_sign = np.sign(float(goal_pos[1]) - float(cross_center[1]))
    if _goal_side_y_sign == 0.0:
        _goal_side_y_sign = 1.0

    # Sample domain: YZ grid covering the cross bounding box plus margin
    y_vals = np.linspace(float(y_left) - margin_m, float(y_right) + margin_m, n_y)
    z_vals = np.linspace(float(z_low)  - margin_m, float(z_high) + margin_m, n_z)

    # Thresholds for stage tagging (relative to cross center)
    _z_stage_margin = 0.05
    _z_below_thr    = float(cross_center[2]) - _z_stage_margin  # below cross center
    _z_top_thr      = float(cross_center[2]) + _z_stage_margin  # above cross center
    _z_lateral_thr  = float(z_low) + 0.02   # truly below the vertical bar (can cross Y freely)

    # Vectorized clearance over the full YZ grid (replaces 49 scalar calls)
    _n_grid = n_y * n_z
    _yy, _zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    _grid_pts = np.column_stack([
        np.full(_n_grid, x_plane), _yy.ravel(), _zz.ravel(),
    ])  # (n_grid, 3)
    _cl_batch = np.full(_n_grid, float("inf"))
    for _obs in obstacles:
        _t = getattr(_obs, "type", "box").lower()
        _p_obs = np.asarray(_obs.position, dtype=float)
        if _t == "box":
            _half = np.asarray(_obs.size, dtype=float)
            _closest = np.clip(_grid_pts, _p_obs - _half, _p_obs + _half)
            _cl_batch = np.minimum(_cl_batch, np.linalg.norm(_grid_pts - _closest, axis=1))
        elif _t == "sphere":
            _cl_batch = np.minimum(
                _cl_batch,
                np.linalg.norm(_grid_pts - _p_obs, axis=1) - float(_obs.size[0]),
            )

    candidates: List[EscapeWaypointCandidate] = []
    _idx = 0
    for _vi in np.where(_cl_batch >= min_waypoint_clearance_m)[0]:
        _iy, _iz = divmod(int(_vi), n_z)
        y  = float(y_vals[_iy])
        z  = float(z_vals[_iz])
        pos = _grid_pts[_vi]
        cl  = float(_cl_batch[_vi])

        y_rel = y - float(cross_center[1])
        z_rel = z - float(cross_center[2])

        # Stage tags: purely geometric, no family names
        tags: List[str] = []
        if z < _z_below_thr:
            tags.append("below_stage_candidate")
        if z >= _z_top_thr:
            tags.append("top_fallback_candidate")
        if (z < _z_lateral_thr
                and np.sign(y - float(cross_center[1])) * _goal_side_y_sign > 0):
            tags.append("lateral_stage_candidate")

        # Setup-time score: prefer high clearance; used for initial pool ranking.
        # Runtime scoring adds goal-progress, distance, and IK-cost terms.
        cl_capped = min(float(cl), 1.0)
        score = (
            3.0 * cl_capped
            - 0.3 * float(np.linalg.norm(pos - ee_pos))
        )

        # Auto-generated label for logging — not used for selection logic
        _y_bin = "ypos" if y_rel > 0.05 else ("yneg" if y_rel < -0.05 else "yctr")
        _z_bin = "zhigh" if z >= _z_top_thr else ("zlow" if z < _z_below_thr else "zmid")
        family = f"escape_sample_{_y_bin}_{_z_bin}_{_idx:02d}"
        _idx += 1

        candidates.append(EscapeWaypointCandidate(
            pos=pos.copy(),
            family=family,
            clearance=float(cl),
            score=score,
            stage_tags=tags,
            y_rel=y_rel,
            z_rel=z_rel,
            metadata={
                "cross_center":     cross_center.tolist(),
                "goal_pos":         goal_pos.tolist(),
                "x_plane":          float(x_plane),
                "y_left":           float(y_left),
                "y_right":          float(y_right),
                "z_low":            float(z_low),
                "z_high":           float(z_high),
                "goal_side_y_sign": float(_goal_side_y_sign),
                "cross_center_y":   float(cross_center[1]),
            },
        ))

    # Keep top-scoring candidates up to max_candidates
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:max_candidates]

    print(
        f"[boundary_escape sampled_candidates={_idx} "
        f"valid={len(candidates)} grid={n_y}x{n_z} "
        f"below={sum(1 for c in candidates if 'below_stage_candidate' in c.stage_tags)} "
        f"lateral={sum(1 for c in candidates if 'lateral_stage_candidate' in c.stage_tags)} "
        f"top={sum(1 for c in candidates if 'top_fallback_candidate' in c.stage_tags)}]",
        flush=True,
    )
    return candidates


def _build_boundary_escape_attractors(
    spec,
    obstacles: List,
    clearance_fn,
    cfg: GeoMultiAttractorDSConfig,
) -> Tuple[List[IKAttractor], Dict[str, Any]]:
    """
    Build homotopy-proxy boundary escape attractors for yz_cross scenarios.

    Generates up to boundary_escape_max_waypoints portal candidates from
    obstacle boundary geometry, filters by clearance, scores, and runs
    position-only IK with num_solutions=1 per candidate.

    Returns (list_of_attractors, diagnostics_dict).
    """
    from src.solver.ik.hjcd_wrapper import solve_batch

    t0 = time.perf_counter()

    ee_start = np.array(
        _panda_link_positions(np.asarray(spec.q_start, dtype=float))[-1],
        dtype=float,
    )
    goal_pos = (np.array(spec.target_pose["position"], dtype=float)
                if spec.target_pose else np.zeros(3))

    print("[boundary_escape enabled]", flush=True)

    candidates = generate_obstacle_boundary_escape_waypoints(
        obstacles, ee_start, goal_pos,
        margin_m=cfg.boundary_escape_margin_m,
        min_waypoint_clearance_m=cfg.boundary_escape_min_clearance_m,
    )

    if not candidates:
        return [], {
            "enabled": False,
            "reason": "no_yz_cross_geometry_or_all_rejected",
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        }

    build_mode = getattr(cfg, "boundary_escape_build_mode", "prebuild")

    if build_mode == "on_stall":
        # Online/deferred mode: store ALL sampled candidates so stage-filtered
        # selection can pick from the full below/lateral/top pool at runtime.
        # Truncating to boundary_escape_max_waypoints here would drop below-stage
        # candidates when top-clearance (top_fallback) candidates sort first.
        selected = candidates
        print(
            f"[boundary_escape candidates={len(candidates)} valid_clearance={len(selected)}"
            f" build_mode=on_stall ik_deferred=True]",
            flush=True,
        )
        diag: Dict[str, Any] = {
            "enabled":        True,
            "build_mode":     "on_stall",
            "num_candidates": len(candidates),
            "num_selected":   len(selected),
            "candidates":     selected,
            "total_ms":       (time.perf_counter() - t0) * 1000.0,
        }
        return [], diag

    selected = candidates[:cfg.boundary_escape_max_waypoints]

    print(
        f"[boundary_escape candidates={len(candidates)} valid_clearance={len(selected)}"
        f" ik_attempts={len(selected)}]",
        flush=True,
    )

    env_config   = {"obstacles": spec.obstacles_as_hjcd_dict()} if spec.obstacles else None
    orientations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ]

    attractors: List[IKAttractor] = []
    diag_families: List[str]  = []
    diag_ik_ok:    List[bool] = []
    ik_per_family: List[Dict[str, Any]] = []

    for cand in selected:
        t_ik = time.perf_counter()
        fam_diag: Dict[str, Any] = {
            "family":    cand.family,
            "pos":       cand.pos.tolist(),
            "score":     float(cand.score),
            "clearance": float(cand.clearance),
            "success":   False,
        }

        valid: List[np.ndarray] = []
        tried_quats: List[Dict[str, Any]] = []

        for quat in orientations:
            try:
                _r = solve_batch(
                    target_pose={"position": cand.pos.tolist(), "quaternion_wxyz": quat},
                    env_config=env_config,
                    batch_size=cfg.boundary_escape_ik_batch_size,
                    num_solutions=cfg.boundary_escape_ik_num_solutions,
                )
                _cands = [
                    q for q, pe in zip(_r.solutions, _r.metadata.pos_errors)
                    if float(pe) <= 0.10
                    and not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))
                    and np.all(np.isfinite(q))
                ]
                tried_quats.append({
                    "quat": quat,
                    "n_returned": len(_r.solutions),
                    "n_valid": len(_cands),
                })
                if _cands:
                    valid = _cands
                    break
            except Exception as exc:
                tried_quats.append({"quat": quat, "error": str(exc)})

        fam_diag["ik_ms"]      = (time.perf_counter() - t_ik) * 1000.0
        fam_diag["tried_quats"] = tried_quats
        ik_ok = len(valid) > 0

        if ik_ok:
            q_esc = np.asarray(valid[0], dtype=float)
            att = IKAttractor(
                q_goal         = q_esc,
                family         = cand.family,
                clearance      = float(clearance_fn(q_esc)),
                manipulability = _goal_manipulability(q_esc),
                kind           = "escape",
                target_name    = "boundary_escape_waypoint",
                shell_sector   = cand.family,
                shell_waypoint = cand.pos.copy(),
            )
            attractors.append(att)
            fam_diag["success"] = True
            print(
                f"[boundary_escape {cand.family}: ik_ok"
                f" pos={np.round(cand.pos, 3).tolist()}"
                f" score={cand.score:.2f} cl={cand.clearance:.3f}]",
                flush=True,
            )
        else:
            print(f"[boundary_escape {cand.family}: ik_fail_pose]", flush=True)

        diag_families.append(cand.family)
        diag_ik_ok.append(ik_ok)
        ik_per_family.append(fam_diag)

    total_ms = (time.perf_counter() - t0) * 1000.0

    ok_families = ",".join(f for f, ok in zip(diag_families, diag_ik_ok) if ok)
    print(
        f"[boundary_escape families={ok_families}]",
        flush=True,
    )

    diag: Dict[str, Any] = {
        "enabled":            True,
        "num_candidates":     len(candidates),
        "num_selected":       len(selected),
        "num_ik_attempts":    len(selected),
        "num_ik_success":     len(attractors),
        "families":           diag_families,
        "ik_success":         diag_ik_ok,
        "ik_per_family":      ik_per_family,
        "total_ms":           total_ms,
    }
    return attractors, diag


# prebuild_debug uses precomputed attractors; not valid for timing benchmarks.
def _make_online_ik_fn(cfg: GeoMultiAttractorDSConfig, env_config, q_start=None):
    """Return a callable (pos) -> Optional[np.ndarray] using solve_batch.

    When q_start is provided it is used as the IK seed configuration so that
    HJCD-IK finds solutions reachable from the same side of configuration space
    as the start (matching prebuild_debug behaviour).
    """
    from src.solver.ik.hjcd_wrapper import solve_batch
    _orientations = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ]
    _robot_cfg = {"start": [float(v) for v in q_start]} if q_start is not None else {}

    def _fn(pos, batch_size: int = 128, pos_tol: float = 0.10, q_seed=None,
            minimize_qdist: bool = False, return_all: bool = False):
        _rc = ({"start": [float(v) for v in q_seed]} if q_seed is not None else _robot_cfg)
        _pick_closest = (minimize_qdist or return_all) and q_seed is not None
        _ns = 8 if _pick_closest else 1
        _candidates: list = []
        for quat in _orientations:
            try:
                r = solve_batch(
                    target_pose={"position": pos.tolist(), "quaternion_wxyz": quat},
                    robot_config=_rc,
                    env_config=env_config,
                    batch_size=batch_size,
                    num_solutions=_ns,
                )
                valid = [
                    q for q, pe in zip(r.solutions, r.metadata.pos_errors)
                    if float(pe) <= pos_tol
                    and not (np.any(q < _PANDA_Q_MIN) or np.any(q > _PANDA_Q_MAX))
                    and np.all(np.isfinite(q))
                ]
                if valid:
                    if _pick_closest:
                        _candidates.extend(valid)
                    else:
                        return np.asarray(valid[0], dtype=float)
            except Exception:
                pass
        if _candidates:
            if return_all:
                return [np.asarray(qc, dtype=float) for qc in _candidates]
            _q_seed_arr = np.asarray(q_seed, dtype=float)
            return min(_candidates,
                       key=lambda qc: float(np.linalg.norm(np.asarray(qc) - _q_seed_arr)))
        return None if not return_all else []

    return _fn


# Main factory
# ---------------------------------------------------------------------------

def build_geo_multi_attractor_ds(
    spec: ScenarioSpec,
    config: Optional[GeoMultiAttractorDSConfig] = None,
    coverage_config=None,   # Optional[CoverageConfig] — lazy import avoids circular
    ik_source: str = "online",
    ik_batch_size: int = 1000,
    ik_num_solutions: int = 8,
    ik_filter_mode: str = "safe",
    midlink_label_detail: str = "lateral",
    family_classifier_mode: str = "goal_frame_midlink",
    yz_threshold_mode: str = "adaptive",
    yz_cross_landmark_mode: str = "auto",
    enable_yz_quadrant_expansion: bool = False,
    yz_expansion_batch_size: int = 64,
    yz_expansion_max_batches: int = 2,
    defer_initial_ik: bool = False,
) -> GeoMultiAttractorDS:
    """
    Build a GeoMultiAttractorDS from a ScenarioSpec.

    Args:
        spec:            ScenarioSpec describing the scenario.
        config:          Optional GeoMultiAttractorDSConfig (uses defaults if None).
        coverage_config: Optional CoverageConfig.  When set, the factory runs
                         the biased-seed differential IK sampler to fill any
                         missing homotopy classes before building the DS.
        ik_source:       "online" (default) — run HJCD-IK; time is included in
                         planner_ms.  "precomputed" — use spec.ik_goals; raises
                         ValueError if spec.ik_goals is empty.

    Returns:
        GeoMultiAttractorDS ready to call .compute(q).
    """
    if ik_source not in ("online", "precomputed"):
        raise ValueError(f"ik_source must be 'online' or 'precomputed', got {ik_source!r}")

    if coverage_config is not None:
        import warnings
        warnings.warn(
            "build_geo_multi_attractor_ds: coverage_config is experimental — "
            "expansion takes 16–18s on cross-barrier scenarios with no improvement "
            "in convergence or clearance. Only use for debugging.",
            stacklevel=2,
        )

    t_build_start = time.perf_counter()

    obstacles        = spec.collision_obstacles()
    clearance_fn     = _make_clearance_fn(obstacles)
    batch_from_lp_fn = _make_batch_from_lp_fn(obstacles)
    target_pos       = (np.array(spec.target_pose["position"], dtype=float)
                        if spec.target_pose else None)

    t_closure_ms = (time.perf_counter() - t_build_start) * 1000.0

    # IK goal resolution
    _defer_async_ik = False   # set True when taskspace_goal replaces IK attractors
    if ik_source == "online" and defer_initial_ik:
        # Use q_start as a placeholder IK goal to keep the scoring pipeline intact.
        # The classification result is discarded — attractors are replaced below with
        # a single taskspace_goal attractor driven by task_pos.  No IK is computed.
        ik_goals            = [np.asarray(spec.q_start, dtype=float).copy()]
        ik_gen_ms           = 0.0
        ik_num_raw          = 0
        ik_num_after_filter = 0
        _ik_meta            = {}
        _defer_async_ik     = True
        print("[initial_goal_taskspace ik_ms=0.0]", flush=True)
    elif ik_source == "online":
        ik_goals_raw, _ik_meta = _generate_ik_goals_online(
            spec, clearance_fn, batch_from_lp_fn,
            batch_size=ik_batch_size,
            num_solutions=ik_num_solutions,
            ik_filter_mode=ik_filter_mode,
        )
        ik_gen_ms           = _ik_meta["ik_generation_ms"]
        ik_num_raw          = _ik_meta["ik_num_raw"]
        ik_num_after_filter = _ik_meta["ik_num_after_filter"]
        if not ik_goals_raw:
            raise RuntimeError(
                f"GeoMA-DS: no valid IK goals found for '{spec.name}' "
                f"(raw={ik_num_raw}, valid=0, ik_ms={ik_gen_ms:.1f}). "
                "Try increasing batch_size or relaxing obstacle constraints."
            )
        ik_goals = ik_goals_raw
    else:  # "precomputed"
        if not spec.ik_goals:
            raise ValueError(
                f"ik_source='precomputed' but spec.ik_goals is empty for '{spec.name}'. "
                "Provide pre-computed IK goals in spec.ik_goals."
            )
        ik_goals            = [np.asarray(q, dtype=float) for q in spec.ik_goals]
        ik_gen_ms           = 0.0
        ik_num_raw          = len(ik_goals)
        ik_num_after_filter = len(ik_goals)
        _ik_meta            = {}

    window_label_fn = None  # set below when cross-barrier coverage is active

    # --- Optional coverage expansion ---
    if coverage_config is not None:
        from src.solver.ik.coverage_expansion import (
            CoverageConfig,
            expand_ik_coverage,
            make_window_label_fn,
            print_coverage_report,
        )

        window_label_fn  = None
        target_windows   = list(coverage_config.target_windows)
        is_cross_barrier = _detect_cross_barrier(obstacles)

        # For non-window scenarios (e.g. I-barrier) restrict family expansion to
        # families already present in the existing IK goals.  Without this guard,
        # the sampler would introduce elbow_back/elbow_center attractors whose
        # natural paths route straight through the barrier rather than around it.
        if not target_windows and not is_cross_barrier:
            from src.solver.ik.coverage_expansion import classify_elbow_family
            present_families = list({classify_elbow_family(q) for q in ik_goals})
            coverage_config = CoverageConfig(
                **{k: v for k, v in vars(coverage_config).items()
                   if k != "target_families"},
                target_families=present_families,
            )

        # Auto-detect cross barrier and inject window classes
        if is_cross_barrier:
            geo = _cross_barrier_geometry(obstacles)
            if geo is not None:
                x_post, z_mid, x_post_half, z_bar_half = geo
                window_label_fn = make_window_label_fn(
                    x_post=x_post, z_mid=z_mid,
                    x_post_half=x_post_half, z_bar_half=z_bar_half,
                )
                # Request all 4 windows if not already specified
                if not target_windows:
                    target_windows = [
                        "upper-left", "upper-right",
                        "lower-left", "lower-right",
                    ]
                coverage_config = CoverageConfig(
                    **{k: v for k, v in vars(coverage_config).items()
                       if k != "target_windows"},
                    target_windows=target_windows,
                )

        # Target grasptarget position: the red sphere = scenario target_pose["position"].
        # HJCD-IK solves for panda_grasptarget, so expanded attractors must target
        # the same frame.  Do NOT use mean(panda_hand FK) — that is 0.105m off.
        target_grasptarget_pos = np.array(spec.target_pose["position"], dtype=float)

        new_goals, report = expand_ik_coverage(
            existing_goals         = ik_goals,
            target_grasptarget_pos = target_grasptarget_pos,
            clearance_fn           = clearance_fn,
            q_start                = spec.q_start,
            config                 = coverage_config,
            window_label_fn        = window_label_fn,
        )

        if coverage_config.verbose:
            _print_factory_coverage_report(
                existing_goals  = ik_goals,
                new_goals       = new_goals,
                q_start         = spec.q_start,
                clearance_fn    = clearance_fn,
                report          = report,
                window_label_fn = window_label_fn,
            )

        ik_goals = ik_goals + new_goals

    # --- Classify: midpoint FK only (fast); no pre-built grid needed ---
    # Use fewer interpolation steps in fast mode to reduce FK work in scoring phase.
    _N_INTERP_SCORE = 3 if family_classifier_mode == "yz_cross_endpoint_fast" else 5
    # Pre-built FK scoring grid -- set in fast path to fuse classification and scoring FK.
    _lp_flat_prebuilt: Optional[np.ndarray] = None
    t_classify_start = time.perf_counter()

    yz_cross_diagnostics: Optional[dict] = None

    if family_classifier_mode == "yz_cross_quadrant":
        # Call YZ-cross classifier directly to capture diagnostics.
        from src.solver.ik.goal_selection import classify_ik_goals_yz_cross_quadrant
        _yz_infos, yz_cross_diagnostics = classify_ik_goals_yz_cross_quadrant(
            spec.q_start, ik_goals, obstacles,
            threshold_mode=yz_threshold_mode,
            landmark_mode=yz_cross_landmark_mode,
        )

        # Vertical coverage expansion: run extra HJCD batches until has_above AND
        # has_below are both true, filtering candidates to below-cross z-region since
        # HJCD has no seed support and naturally biases toward above-cross postures.
        if enable_yz_quadrant_expansion and ik_source == "online":
            _cross_c   = np.asarray(yz_cross_diagnostics.get("cross_center", [0.0, 0.0, 0.0]))
            _has_above = yz_cross_diagnostics.get("has_above_family", False)
            _has_below = yz_cross_diagnostics.get("has_below_family", False)

            for _attempt in range(yz_expansion_max_batches):
                if _has_above and _has_below:
                    break
                _missing_tag = "below" if not _has_below else "above"
                print(
                    f"\nyz expansion: missing {_missing_tag} family; "
                    f"running extra batch {_attempt + 1}/{yz_expansion_max_batches}",
                    flush=True,
                )
                _extra_raw, _ = _generate_ik_goals_online(
                    spec, clearance_fn, batch_from_lp_fn,
                    batch_size=yz_expansion_batch_size,
                    num_solutions=8, ik_filter_mode="minimal",
                )
                _exp_meta: dict = {
                    "batch":           _attempt + 1,
                    "total_raw":       len(_extra_raw),
                    "below_filtered":  0,
                    "added":           0,
                    "has_below_after": False,
                    "rejected_z":      0,
                }
                if _extra_raw:
                    # Z-filter: keep only where min(midlink_z, wrist_z) < cross_center_z.
                    _lp_extra = _panda_fk_batch(np.stack(_extra_raw))
                    _mid_z    = _lp_extra[:, 4, 2]
                    _wrist_z  = _lp_extra[:, 6, 2]
                    _z_mask   = np.minimum(_mid_z, _wrist_z) < _cross_c[2]
                    _z_ok     = [q for q, ok in zip(_extra_raw, _z_mask) if ok]
                    _exp_meta["below_filtered"] = len(_z_ok)
                    _exp_meta["rejected_z"]     = int((~_z_mask).sum())

                    if _z_ok:
                        _extra_infos, _ = classify_ik_goals_yz_cross_quadrant(
                            spec.q_start, _z_ok, obstacles,
                            threshold_mode=yz_threshold_mode,
                            landmark_mode=yz_cross_landmark_mode,
                        )
                        for _ei in _extra_infos:
                            _needed = (
                                ("below" in _ei.family_label and not _has_below) or
                                ("above" in _ei.family_label and not _has_above)
                            )
                            if _needed:
                                _yz_infos.append(_ei)
                                ik_goals.append(_ei.q_goal)
                                _exp_meta["added"] += 1
                                _vtag = "below" if "below" in _ei.family_label else "above"
                                print(
                                    f"yz expansion: found {_vtag} candidate "
                                    f"family={_ei.family_label}",
                                    flush=True,
                                )
                        _has_above = any("above" in i.family_label for i in _yz_infos)
                        _has_below = any("below" in i.family_label for i in _yz_infos)

                _exp_meta["has_below_after"] = _has_below
                if yz_cross_diagnostics is not None:
                    yz_cross_diagnostics.setdefault("expansion_attempts", []).append(_exp_meta)

            # Propagate final coverage state into diagnostics.
            if yz_cross_diagnostics is not None:
                yz_cross_diagnostics["has_below_family"] = _has_below
                yz_cross_diagnostics["has_above_family"] = _has_above

        _n_interp_classify = 1 if family_classifier_mode == "yz_cross_endpoint_fast" else _N_INTERP_SCORE
        attractors = ik_goals_to_attractors(
            ik_goals, clearance_fn, spec.q_start, obstacles,
            n_interp=_n_interp_classify,
            precomputed_infos=_yz_infos,
        )
    elif family_classifier_mode == "yz_cross_endpoint_fast":
        # Fast path: pre-build the scoring FK grid and pass its t=1 slice (= goal
        # configs) to the classifier, eliminating the duplicate FK batch inside it.
        from src.solver.ik.goal_selection import classify_ik_goals_yz_cross_quadrant
        _lp_goals_fk: Optional[np.ndarray] = None
        if obstacles and ik_goals and not _defer_async_ik:
            _q_s_fk    = np.asarray(spec.q_start, dtype=float)
            _qs_arr_fk = np.stack([np.asarray(g, dtype=float) for g in ik_goals])
            _t_vals_fk = np.linspace(0.0, 1.0, _N_INTERP_SCORE)
            _diff_fk   = _qs_arr_fk - _q_s_fk[np.newaxis, :]
            _interp_fk = (_q_s_fk[np.newaxis, np.newaxis, :]
                          + _t_vals_fk[np.newaxis, :, np.newaxis] * _diff_fk[:, np.newaxis, :])
            _lp_flat_prebuilt = _panda_fk_batch(_interp_fk.reshape(-1, 7))
            _lp_goals_fk = _lp_flat_prebuilt.reshape(
                len(ik_goals), _N_INTERP_SCORE, 8, 3
            )[:, -1, :, :]   # t=1.0 slice = goal-config FK results
        _yz_infos_fast, yz_cross_diagnostics = classify_ik_goals_yz_cross_quadrant(
            spec.q_start, ik_goals, obstacles,
            threshold_mode=yz_threshold_mode,
            landmark_mode=yz_cross_landmark_mode,
            lp_goals=_lp_goals_fk,
        )
        attractors = ik_goals_to_attractors(
            ik_goals, clearance_fn, spec.q_start, obstacles,
            n_interp=1,
            precomputed_infos=_yz_infos_fast,
        )
    else:
        attractors = ik_goals_to_attractors(
            ik_goals, clearance_fn, spec.q_start, obstacles,
            lp_grid=None,
            n_interp=_N_INTERP_SCORE,
            target_pos=target_pos,
            midlink_sample_mode="midpoint",
            label_detail=midlink_label_detail,
        )

    t_classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

    # Deferred path: replace classified result with a pure task-space goal attractor.
    # Classification of q_start is discarded — taskspace_goal family disables backtrack
    # staging and YZ routing that require a real IK solution.
    if _defer_async_ik:
        _q_s_arr = np.asarray(spec.q_start, dtype=float)
        attractors = [IKAttractor(
            q_goal=_q_s_arr.copy(),
            family="taskspace_goal",
            clearance=float(clearance_fn(_q_s_arr)),
            manipulability=0.0,
            kind="goal",
            target_name="goal_taskspace_fallback",
            task_pos=target_pos.copy() if target_pos is not None else None,
        )]

    # --- Inject static properties (straight-line clearance, window label, static score) ---
    t_score_start = time.perf_counter()

    if obstacles:
        if _lp_flat_prebuilt is not None:
            # Reuse the FK grid built during classify phase (fast path FK fusion).
            lp_flat = _lp_flat_prebuilt
        else:
            # Build scoring FK grid shared between clearance and window scoring.
            q_s    = np.asarray(spec.q_start, dtype=float)
            qs_arr = np.stack(ik_goals)
            t_vals = np.linspace(0.0, 1.0, _N_INTERP_SCORE)
            diff   = qs_arr - q_s[np.newaxis, :]
            interp = (q_s[np.newaxis, np.newaxis, :]
                      + t_vals[np.newaxis, :, np.newaxis] * diff[:, np.newaxis, :])
            lp_flat = _panda_fk_batch(interp.reshape(-1, 7))
        _inject_attractor_static_scores(
            attractors, spec.q_start, batch_from_lp_fn, window_label_fn,
            lp_all=lp_flat,
        )
    else:
        lp_flat = None
        for att in attractors:
            att.straight_line_min_clearance = float("inf")
            att.window_label = "none"
            att.static_score = 0.0

    t_score_ms = (time.perf_counter() - t_score_start) * 1000.0

    # --- Boundary escape waypoints (takes precedence over simple escape and goal-shell) ---
    boundary_escape_diag: Dict[str, Any] = {"enabled": False}
    if config is not None and config.enable_boundary_escape_waypoints:
        if config.enable_simple_escape_waypoint or config.enable_goal_shell_escape:
            print(
                "[warn] boundary_escape_waypoints enabled alongside simple/shell escape;"
                " boundary escape takes precedence",
                flush=True,
            )
        t_be = time.perf_counter()
        _be_atts, boundary_escape_diag = _build_boundary_escape_attractors(
            spec, obstacles, clearance_fn, config,
        )
        boundary_escape_diag["build_ms"] = (time.perf_counter() - t_be) * 1000.0

        _n_goal_atts_be = len(attractors)
        for _be_att in _be_atts:
            if obstacles:
                _inject_attractor_static_scores([_be_att], spec.q_start, batch_from_lp_fn)
            else:
                _be_att.straight_line_min_clearance = float("inf")
                _be_att.window_label = "none"
                _be_att.static_score = 0.0
        attractors = attractors + _be_atts

        _n_esc_atts_be = len([a for a in attractors if a.kind == "escape"])

        # --- Backtrack / staging attractor (appended after escape attractors) ---
        if config is not None and config.enable_backtrack_staging:
            _q_bt = spec.q_start.copy()
            if config.backtrack_target_mode == "partial_to_start":
                # Partial backtrack computed from first goal attractor as approximation of q_current
                # (actual q_current not available at build time; will be refined at runtime)
                _beta = float(config.backtrack_partial_beta)
                _q_center = attractors[0].q_goal if attractors else _q_bt
                _q_bt = _q_center + _beta * (spec.q_start - _q_center)
            _bt_cl = float(clearance_fn(_q_bt))
            _bt_att = IKAttractor(
                q_goal=_q_bt,
                family="backtrack_start",
                clearance=_bt_cl,
                manipulability=1.0,
                kind="backtrack",
                target_name="start_configuration",
                static_score=0.0,
            )
            attractors = attractors + [_bt_att]
            _n_bt_atts = len([a for a in attractors if a.kind == "backtrack"])
            print(
                f"[attractors goal={_n_goal_atts_be} escape={_n_esc_atts_be}"
                f" backtrack={_n_bt_atts} total={len(attractors)}]",
                flush=True,
            )
        else:
            print(
                f"[attractors goal={_n_goal_atts_be} escape={_n_esc_atts_be} total={len(attractors)}]",
                flush=True,
            )

    # --- Simple escape waypoint (takes precedence over goal-shell; skipped when boundary escape active) ---
    simple_escape_diag: Dict[str, Any] = {"enabled": False}
    if (config is not None and config.enable_simple_escape_waypoint
            and not config.enable_boundary_escape_waypoints):
        if config.enable_goal_shell_escape:
            print(
                "[warn] both goal-shell escape and simple escape waypoint enabled;"
                " using simple escape waypoint",
                flush=True,
            )
        print("[simple_escape enabled]", flush=True)
        t_se = time.perf_counter()
        _se_atts, simple_escape_diag = _build_simple_escape_waypoint_attractor(
            spec, obstacles, clearance_fn, config,
        )
        simple_escape_diag["build_ms"] = (time.perf_counter() - t_se) * 1000.0

        _n_goal_atts_se = len(attractors)

        if simple_escape_diag.get("enabled", True) and simple_escape_diag.get("portal_diags"):
            for _pd in simple_escape_diag["portal_diags"]:
                _ep = np.array(_pd.get("escape_pos", [0, 0, 0])).round(3).tolist()
                if _pd.get("ik_ok"):
                    print(
                        f"[escape_waypoint pos={_ep}"
                        f" ik_ok=True task=position_only"
                        f" sector={_pd['sector']}"
                        f" portal_score={_pd.get('portal_score', 0):.2f}"
                        f" portal_cl={_pd.get('portal_clearance', 0):.3f}]",
                        flush=True,
                    )
                else:
                    print(f"[escape_waypoint pos={_ep} ik_ok=False sector={_pd['sector']}]", flush=True)
                    for _tq in _pd.get("tried_quats", []):
                        print(
                            f"  quat={_tq.get('quat')}"
                            f" n_returned={_tq.get('n_returned', 0)}"
                            f" n_valid={_tq.get('n_valid', 0)}"
                            + (f" err={_tq['error']}" if "error" in _tq else ""),
                            flush=True,
                        )
        else:
            print(
                f"[escape_waypoint ik_ok=False reason={simple_escape_diag.get('reason', 'unknown')}]",
                flush=True,
            )

        for _se_att in _se_atts:
            if obstacles:
                _inject_attractor_static_scores([_se_att], spec.q_start, batch_from_lp_fn)
            else:
                _se_att.straight_line_min_clearance = float("inf")
                _se_att.window_label = "none"
                _se_att.static_score = 0.0
        attractors = attractors + _se_atts

        _n_esc_atts_se = len([a for a in attractors if a.kind == "escape"])
        _n_rec_atts_se = len([a for a in attractors if a.kind == "recovery"])
        print(
            f"[attractors goal={_n_goal_atts_se} recovery={_n_rec_atts_se} "
            f"escape={_n_esc_atts_se} total={len(attractors)}]",
            flush=True,
        )

    # --- Goal-shell escape attractors (Phase 1: pre-build) ---
    # Skipped when simple or boundary escape is active (mutually exclusive).
    goal_shell_diag: Dict[str, Any] = {"enabled": False}
    if (config is not None and config.enable_goal_shell_escape
            and not config.enable_simple_escape_waypoint
            and not config.enable_boundary_escape_waypoints
            and target_pos is not None):
        print("[goal_shell enabled]", flush=True)
        _gs_cross_c = None
        if yz_cross_diagnostics is not None:
            _cc = yz_cross_diagnostics.get("cross_center")
            if _cc is not None:
                _gs_cross_c = np.asarray(_cc, dtype=float)
        t_esc = time.perf_counter()
        _esc_atts, goal_shell_diag = _build_escape_attractors(
            spec, obstacles, clearance_fn, target_pos, config,
            cross_center=_gs_cross_c,
        )
        goal_shell_diag["build_ms"] = (time.perf_counter() - t_esc) * 1000.0

        _n_cand  = goal_shell_diag["num_waypoint_candidates"]
        _n_vcl   = goal_shell_diag["num_valid_clearance"]
        _n_att   = goal_shell_diag["num_ik_attempts"]
        _n_ok    = goal_shell_diag["num_ik_success"]
        _n_esc   = goal_shell_diag["num_escape_attractors"]
        _n_fp    = goal_shell_diag["num_ik_fail_pose"]
        _n_fj    = goal_shell_diag["num_ik_fail_joint"]
        _n_fc    = goal_shell_diag["num_ik_fail_clearance"]
        _ik_ms   = goal_shell_diag["shell_ik_ms"]
        _tot_ms  = goal_shell_diag["shell_total_ms"]
        _cand_ms = goal_shell_diag["shell_candidate_ms"]

        _n_goal_atts = len(attractors)

        if _esc_atts:
            if obstacles:
                _inject_attractor_static_scores(_esc_atts, spec.q_start, batch_from_lp_fn)
            else:
                for _ea in _esc_atts:
                    _ea.straight_line_min_clearance = float("inf")
                    _ea.window_label = "none"
                    _ea.static_score = 0.0
            attractors = attractors + _esc_atts
            _ok_secs = ",".join(
                s for s, ok in zip(goal_shell_diag["sectors"], goal_shell_diag["ik_success"])
                if ok
            )
            print(
                f"[shell_escape candidates={_n_cand} valid_clearance={_n_vcl}"
                f" ik_attempts={_n_att} ik_success={_n_ok}"
                f" escape_attractors={_n_esc} sectors={_ok_secs}]",
                flush=True,
            )
            print(
                f"  shell_candidate_ms={_cand_ms:.1f}"
                f" shell_ik_ms={_ik_ms:.1f} shell_total_ms={_tot_ms:.1f}",
                flush=True,
            )
        else:
            print(
                f"[shell_escape] no escape attractors built"
                f" candidates={_n_cand} valid_clearance={_n_vcl} ik_attempts={_n_att}",
                flush=True,
            )
            print(
                f"  ik_fail_pose={_n_fp} ik_fail_joint={_n_fj} ik_fail_clearance={_n_fc}"
                f" shell_ik_ms={_ik_ms:.1f} shell_total_ms={_tot_ms:.1f}",
                flush=True,
            )

        print(
            f"[attractors goal={_n_goal_atts} escape={len(_esc_atts)} total={len(attractors)}]",
            flush=True,
        )

    total_ms = (time.perf_counter() - t_build_start) * 1000.0

    if ik_source == "online":
        _ik_bd: Dict[str, Any] = {
            "hjcd_solve_batch_ms":     _ik_meta.get("hjcd_solve_batch_ms", 0.0),
            "hjcd_ms":                 _ik_meta.get("hjcd_ms", 0.0),
            "dedup_ms":                _ik_meta.get("dedup_ms", 0.0),
            "pose_filter_ms":          _ik_meta.get("pose_filter_ms", 0.0),
            "joint_limit_filter_ms":   _ik_meta.get("joint_limit_filter_ms", 0.0),
            "clearance_filter_ms":     _ik_meta.get("clearance_filter_ms", 0.0),
            "ik_num_after_pose_filter": _ik_meta.get("ik_num_after_pose_filter", 0),
            "ik_num_after_jl_filter":  _ik_meta.get("ik_num_after_jl_filter", 0),
        }
    else:
        _ik_bd = {}

    planner_breakdown: Dict[str, Any] = {
        "ik_source":             ik_source,
        "ik_num_calls":          1 if ik_source == "online" else 0,
        "ik_generation_ms":      ik_gen_ms,
        "ik_num_raw":            ik_num_raw,
        "ik_num_after_filter":   ik_num_after_filter,
        **_ik_bd,
        "ik_config":             _ik_meta.get("ik_config", {}) if ik_source == "online" else {},
        "clearance_closure_ms":  t_closure_ms,
        "ik_to_attractors_ms":   t_classify_ms,
        "static_scoring_ms":     t_score_ms,
        "other_ms":              max(0.0, total_ms - t_closure_ms - ik_gen_ms - t_classify_ms - t_score_ms),
        "total_planner_ms":      total_ms,
        "classifier_mode":       family_classifier_mode,
        "sample_mode":           "midpoint",
        "label_detail":          midlink_label_detail,
    }

    ds = GeoMultiAttractorDS(
        attractors, clearance_fn, config=config,
        batch_from_lp_fn=batch_from_lp_fn,
    )
    ds.planner_breakdown = planner_breakdown

    # Attach dynamic recovery builder when clearance recovery is enabled
    if config is not None and config.enable_clearance_recovery and obstacles:
        _portal_positions = [
            a.shell_waypoint for a in attractors
            if a.kind == "escape" and a.shell_waypoint is not None
        ]
        ds._build_recovery_attractor_fn = _make_recovery_builder(
            obstacles, clearance_fn, config,
            portal_positions=_portal_positions or None,
        )

    from collections import Counter as _Counter
    ds.attractor_diagnostics = {
        "num_ik_goals":    ik_num_raw,
        "num_attractors":  len(attractors),
        "families":        [a.family for a in attractors],
        "family_counts":   dict(_Counter(a.family for a in attractors)),
        "classifier_mode": family_classifier_mode,
        "label_detail":    midlink_label_detail,
        "selected_representatives": [
            {
                "idx":                        i,
                "family":                     a.family,
                "static_clearance":           float(a.clearance),
                "manipulability":             float(a.manipulability),
                "static_score":               float(a.static_score),
                "straight_line_min_clearance": float(a.straight_line_min_clearance),
                "window_label":               a.window_label,
            }
            for i, a in enumerate(attractors)
        ],
    }
    if yz_cross_diagnostics is not None:
        ds.attractor_diagnostics["yz_cross"] = yz_cross_diagnostics
    ds.attractor_diagnostics["goal_shell"]      = goal_shell_diag
    ds.attractor_diagnostics["boundary_escape"] = boundary_escape_diag
    # On-stall mode: store candidates on the DS so runtime can build IK on demand
    if boundary_escape_diag.get("build_mode") == "on_stall":
        _cands = boundary_escape_diag.get("candidates", [])
        ds.boundary_escape_candidates    = _cands   # backward compat attribute
        ds._deferred_escape_candidates   = _cands
        ds._q_start = np.asarray(spec.q_start, dtype=float)
        _goal_pos = (np.array(spec.target_pose["position"], dtype=float)
                     if spec.target_pose else np.zeros(3))
        ds._goal_task_pos = _goal_pos
        _env_cfg = {"obstacles": spec.obstacles_as_hjcd_dict()} if spec.obstacles else None
        ds._online_ik_fn = _make_online_ik_fn(config, _env_cfg)
        print(
            f"[online_ik_fn_injected candidates={len(_cands)} "
            f"goal_pos={np.round(_goal_pos, 3).tolist()}]",
            flush=True,
        )
        if getattr(config, "prefill_escape_ik_at_build", False):
            _n_ok = 0
            _n_fail = 0
            _t_pre = time.perf_counter()
            _bs = getattr(config, "online_escape_ik_batch_size", 64)
            for _cand in _cands:
                try:
                    _q = ds._online_ik_fn(_cand.pos, batch_size=_bs)
                except Exception:
                    _q = None
                if _q is not None:
                    _cand.q_esc_prefilled = np.asarray(_q, dtype=float)
                    _n_ok += 1
                else:
                    _n_fail += 1
            _pre_ms = (time.perf_counter() - _t_pre) * 1000.0
            print(
                f"[prefill_escape_ik ok={_n_ok} fail={_n_fail} prefill_ms={_pre_ms:.1f}]",
                flush=True,
            )
    return ds


# ---------------------------------------------------------------------------
# Coverage diagnostics (factory-level, has access to clearance_fn)
# ---------------------------------------------------------------------------

def _print_factory_coverage_report(
    existing_goals:  List[np.ndarray],
    new_goals:       List[np.ndarray],
    q_start:         np.ndarray,
    clearance_fn,
    report:          dict,
    window_label_fn  = None,
) -> None:
    from src.solver.ik.coverage_expansion import classify_elbow_family

    sep  = "=" * 72
    dash = "-" * 72

    print()
    print(sep)
    print("IK Coverage Expansion Report")
    print(sep)
    print(f"  Existing goals      : {report['n_existing']}")
    print(f"  New goals generated : {report['n_new_goals']}")
    print(f"  Total goals         : {report['n_existing'] + report['n_new_goals']}")

    # Family histogram
    all_goals = existing_goals + new_goals
    families  = [classify_elbow_family(q) for q in all_goals]
    fam_hist: dict = {}
    for f in families:
        fam_hist[f] = fam_hist.get(f, 0) + 1

    print()
    print("  Elbow family coverage (all goals):")
    for fam, cnt in sorted(fam_hist.items()):
        tag = " [new]" if fam in set(classify_elbow_family(q) for q in new_goals) - \
              set(classify_elbow_family(q) for q in existing_goals) else ""
        print(f"    {fam:<16} {cnt:>3}{tag}")
    print(f"  Missing families    : {report.get('missing_families', [])}")

    # Window histogram
    if window_label_fn is not None:
        all_windows = [window_label_fn(q, q_start) for q in all_goals]
        win_hist: dict = {}
        for w in all_windows:
            win_hist[w] = win_hist.get(w, 0) + 1
        print()
        print("  Gate window coverage (all goals):")
        for win, cnt in sorted(win_hist.items()):
            print(f"    {win:<16} {cnt:>3}")
        print(f"  Missing windows     : {report.get('missing_windows', [])}")

    # Expansion log
    expansion_log = report.get("expansion_log", [])
    if expansion_log:
        print()
        print("  Expansion log:")
        for entry in expansion_log:
            if isinstance(entry, str):
                print(f"    {entry}")
            else:
                ok = "OK" if entry["found"] > 0 else "MISS"
                print(f"    [{ok}] {entry['kind']}/{entry['class']:<16}  "
                      f"found={entry['found']}  attempts={entry['attempts']}")

    # Per-goal table for new goals with clearance
    if new_goals:
        print()
        print("  New goals:")
        print(f"  {'idx':>4}  {'family':<14}  {'window':<14}  {'clearance':>9}  {'manip':>7}")
        print(f"  {dash}")
        for i, q in enumerate(new_goals):
            fam = classify_elbow_family(q)
            win = window_label_fn(q, q_start) if window_label_fn else "n/a"
            cl  = clearance_fn(q)
            J   = _position_jacobian(q)
            man = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
            print(f"  {len(existing_goals)+i:>4}  {fam:<14}  {win:<14}  {cl:>9.4f}  {man:>7.4f}")

    print()
    final_families = sorted(set(families))
    final_windows  = (
        sorted(set([window_label_fn(q, q_start) for q in all_goals]))
        if window_label_fn else []
    )
    print(f"  Final families      : {final_families}")
    if final_windows:
        print(f"  Final windows       : {final_windows}")
    print(sep)
    print()
