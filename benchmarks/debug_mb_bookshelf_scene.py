"""
Debug visualization for MotionBenchMaker bookshelf scenes.

Reads cached artifacts from eval_mb_failure_breakdown and renders four PNGs
per problem:
  scene_start.png          — robot at q_start + collision spheres + obstacles
  scene_goal.png           — robot at q_goal + closest-pair annotation
  scene_path.png           — EE + closest-sphere trace along BiRRT path
  scene_path_clearance.png — min clearance vs normalized path progress

Usage::

    conda run -n ds-iks python -m benchmarks.debug_mb_bookshelf_scene \\
      --set bookshelf_small_panda --problems 63,84 --seed 0

    # Also run inline BiRRT if no cached path exists:
    conda run -n ds-iks python -m benchmarks.debug_mb_bookshelf_scene \\
      --set bookshelf_small_panda --problems 63,84 --seed 0 --run-birrt-if-missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks._mb_cache import (
    build_metadata, cache_dir, check_fingerprint, ik_fingerprint,
    birrt_fingerprint, write_metadata,
)
from src.scenarios.mb_loader import load_mb_problems
from src.scenarios.scenario_schema import Obstacle, ScenarioSpec
from src.solver.controller.cbf_filter import _clearance as _cbf_clearance
from src.solver.planner.collision import (
    _panda_link_positions, _LINK_RADII, _quat_to_rot,
)
from benchmarks.validate_mb_scene import _draw_box_wireframe

_HJCD_BATCH       = 1000
_HJCD_NSOL        = 4
_PLANNER_MAX_ITER = 2000
_PLANNER_STEP     = 0.1
_DEFAULT_SAVE_DIR = "outputs/debug_bookshelf"
_D_SAFE           = 0.03
_D_BUFFER         = 0.05
_LINK_NAMES       = {0: "link1", 1: "link2", 2: "link3", 3: "link4",
                     4: "link5", 5: "link6", 6: "link7", 7: "hand"}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _find_closest_pair(
    q: np.ndarray, obs_list: List[Obstacle]
) -> Tuple[int, str, float]:
    """Return (link_idx, obs_name, clearance) for the closest (link, obs) pair."""
    link_pos = _panda_link_positions(q)
    best_cl  = float("inf")
    best_li  = 0
    best_obs = ""
    for li, pos in enumerate(link_pos):
        r = float(_LINK_RADII.get(li, 0.08))
        for obs in obs_list:
            if not obs.collision_enabled:
                continue
            cl = _cbf_clearance(np.asarray(pos, dtype=float), r, obs)
            if cl < best_cl:
                best_cl  = cl
                best_li  = li
                best_obs = obs.name
    return best_li, best_obs, best_cl


def _compute_path_clearances(
    path: List[np.ndarray], obs_list: List[Obstacle]
) -> np.ndarray:
    """Min clearance (over all links) at each path waypoint."""
    clears = []
    for q in path:
        link_pos = _panda_link_positions(q)
        step_min = float("inf")
        for li, pos in enumerate(link_pos):
            r = float(_LINK_RADII.get(li, 0.08))
            for obs in obs_list:
                if not obs.collision_enabled:
                    continue
                cl = _cbf_clearance(np.asarray(pos, dtype=float), r, obs)
                step_min = min(step_min, cl)
        clears.append(step_min if step_min < float("inf") else 0.0)
    return np.array(clears)


# ---------------------------------------------------------------------------
# Drawing helpers (matplotlib 3D)
# ---------------------------------------------------------------------------
def _draw_robot(ax, q: np.ndarray, color: str, label: str) -> None:
    pts = np.array(_panda_link_positions(q))
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o",
            color=color, linewidth=2, markersize=5, label=f"robot ({label})")


def _draw_link_spheres(ax, q: np.ndarray, highlight_idx: Optional[int] = None) -> None:
    link_pos = _panda_link_positions(q)
    for li, pos in enumerate(link_pos):
        r   = float(_LINK_RADII.get(li, 0.08))
        col = "red" if li == highlight_idx else "steelblue"
        alp = 0.55 if li == highlight_idx else 0.20
        ax.scatter(*pos, s=(r * 300) ** 2, color=col, alpha=alp, zorder=3)


def _draw_obstacles(ax, obs_list: List[Obstacle]) -> None:
    for obs in obs_list:
        if not obs.collision_enabled:
            continue
        pos = np.array(obs.position, dtype=float)
        if obs.type == "box":
            R = _quat_to_rot(*obs.orientation_wxyz)
            _draw_box_wireframe(ax, pos, np.array(obs.size, dtype=float),
                                R, color="#d62728")
        elif obs.type == "cylinder":
            r, hh = float(obs.size[0]), float(obs.size[1])
            theta = np.linspace(0, 2 * np.pi, 24)
            for z_off in (-hh, hh):
                ax.plot(
                    pos[0] + r * np.cos(theta),
                    pos[1] + r * np.sin(theta),
                    np.full(24, pos[2] + z_off),
                    color="#d62728", linewidth=0.8, alpha=0.6,
                )


def _set_axes(ax, configs: List[np.ndarray], target_pos: np.ndarray) -> None:
    all_pts = np.vstack([np.array(_panda_link_positions(q)) for q in configs])
    all_pts = np.vstack([all_pts, target_pos.reshape(1, 3)])
    ctr  = all_pts.mean(axis=0)
    span = max(float((all_pts.max(0) - all_pts.min(0)).max()) * 0.7, 0.5)
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(max(0.0, ctr[2] - span), ctr[2] + span)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


# ---------------------------------------------------------------------------
# Scene renderers
# ---------------------------------------------------------------------------
def _save_scene_start(spec: ScenarioSpec, out_path: Path) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available"); return

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    q_start  = np.asarray(spec.q_start, dtype=float)
    obs_list = spec.collision_obstacles()
    target   = np.array(spec.target_pose["position"], dtype=float)

    _draw_obstacles(ax, obs_list)
    _draw_robot(ax, q_start, "#1f77b4", "start")
    _draw_link_spheres(ax, q_start)
    ax.scatter(*target, s=200, color="#ff7f0e", marker="*", zorder=6, label="MB target")
    _set_axes(ax, [q_start], target)
    ax.set_title("Scene — q_start"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120); plt.close(fig)
    print(f"  saved: {out_path}")


def _save_scene_goal(spec: ScenarioSpec, q_goal: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available"); return

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    obs_list = spec.collision_obstacles()
    target   = np.array(spec.target_pose["position"], dtype=float)

    closest_li, closest_obs_name, closest_cl = _find_closest_pair(q_goal, obs_list)

    _draw_obstacles(ax, obs_list)
    _draw_robot(ax, q_goal, "#2ca02c", "goal")
    _draw_link_spheres(ax, q_goal, highlight_idx=closest_li)
    ax.scatter(*target, s=200, color="#ff7f0e", marker="*", zorder=6, label="MB target")

    link_name = _LINK_NAMES.get(closest_li, f"link{closest_li}")
    radius    = float(_LINK_RADII.get(closest_li, 0.08))
    ann       = f"{link_name} | r={radius:.3f}m | cl={closest_cl:.4f}m"
    ax.set_title(f"Scene — q_goal   closest: {ann}")
    ax.legend(fontsize=8)
    _set_axes(ax, [np.asarray(spec.q_start, dtype=float), q_goal], target)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120); plt.close(fig)
    print(f"  saved: {out_path}")


def _save_scene_path(
    spec: ScenarioSpec,
    path: List[np.ndarray],
    q_goal: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available"); return

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    q_start  = np.asarray(spec.q_start, dtype=float)
    obs_list = spec.collision_obstacles()
    target   = np.array(spec.target_pose["position"], dtype=float)

    _draw_obstacles(ax, obs_list)
    _draw_robot(ax, q_start, "#1f77b4", "start")
    _draw_robot(ax, q_goal,  "#2ca02c", "goal")

    # EE trace: hand body is index 7
    ee_pts = np.array([_panda_link_positions(q)[7] for q in path])
    ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2],
            color="#aec7e8", linewidth=1.5, label="EE trace")

    # Closest-clearance-sphere trace
    closest_pts = []
    for q in path:
        li, _, _ = _find_closest_pair(q, obs_list)
        closest_pts.append(_panda_link_positions(q)[li])
    closest_pts = np.array(closest_pts)
    ax.plot(closest_pts[:, 0], closest_pts[:, 1], closest_pts[:, 2],
            color="#ff7f0e", linewidth=1.0, linestyle="--", label="closest-sphere trace")

    ax.scatter(*target, s=200, color="#ff7f0e", marker="*", zorder=6)
    _set_axes(ax, [q_start, q_goal], target)
    ax.set_title(f"BiRRT path — {len(path)} waypoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120); plt.close(fig)
    print(f"  saved: {out_path}")


def _save_clearance_plot(
    clearances: np.ndarray, out_path: Path
) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available"); return

    n = len(clearances)
    x = np.linspace(0.0, 1.0, n)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, clearances, color="#1f77b4", linewidth=1.5, label="min clearance")
    ax.axhline(0.0,              color="black",  linewidth=1.0, linestyle="-",
               label="clearance=0")
    ax.axhline(_D_SAFE,          color="red",    linewidth=1.0, linestyle="--",
               label=f"d_safe={_D_SAFE}")
    ax.axhline(_D_SAFE + _D_BUFFER, color="orange", linewidth=1.0, linestyle="--",
               label=f"d_safe+d_buffer={_D_SAFE + _D_BUFFER}")
    ax.fill_between(x, clearances, 0, where=(clearances < 0),
                    color="red", alpha=0.2, label="penetrating")
    ax.set_xlabel("Normalized path progress")
    ax.set_ylabel("Min clearance (m)")
    ax.set_title("Clearance along BiRRT path")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120); plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# BiRRT fallback (--run-birrt-if-missing)
# ---------------------------------------------------------------------------
def _run_birrt_inline(
    spec: ScenarioSpec, q_goals: np.ndarray, seed: int
) -> Optional[List[np.ndarray]]:
    from src.solver.planner.birrt import PlannerConfig, plan
    from src.solver.planner.collision import make_collision_fn
    from src.evaluation.trial_runner import _make_clearance_fn

    collision_fn = make_collision_fn(spec=spec)
    cl_fn        = _make_clearance_fn(spec)
    q_start      = np.asarray(spec.q_start, dtype=float)

    for goal_idx, q_goal in enumerate(q_goals):
        goal_seed = seed + 1009 * goal_idx
        cfg = PlannerConfig(max_iterations=_PLANNER_MAX_ITER,
                            step_size=_PLANNER_STEP, seed=goal_seed)
        pr = plan(q_start, [np.asarray(q_goal, dtype=float)],
                  env=collision_fn, config=cfg, clearance_fn=cl_fn)
        if pr.success and pr.path:
            print(f"  BiRRT success to goal {goal_idx} ({len(pr.path)} waypoints)")
            return pr.path
    return None


# ---------------------------------------------------------------------------
# Per-problem entry point
# ---------------------------------------------------------------------------
def _visualize_problem(
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    seed: int,
    save_dir: str,
    run_birrt_if_missing: bool,
    force_cache: bool,
) -> None:
    cd = cache_dir(save_dir, set_name, prob_idx)
    cd.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {set_name} prob={prob_idx} seed={seed} ===")

    # Load IK goal (use first cached goal, or spec.ik_goals, or q_start)
    q_goals_arr: Optional[np.ndarray] = None
    q_goal: Optional[np.ndarray] = None
    if (cd / "ik_goals.npz").exists():
        ik_data     = np.load(cd / "ik_goals.npz", allow_pickle=True)
        q_goals_arr = ik_data["q_goals"]
        if len(q_goals_arr) > 0:
            q_goal = np.asarray(q_goals_arr[0], dtype=float)
    if q_goal is None and spec.ik_goals:
        q_goal = np.asarray(spec.ik_goals[0], dtype=float)
    if q_goal is None:
        q_goal = np.asarray(spec.q_start, dtype=float)
        print("  [WARN] No IK goals found — using q_start for goal visualization")

    # Render start and goal scenes
    _save_scene_start(spec, cd / "scene_start.png")
    _save_scene_goal(spec, q_goal, cd / "scene_goal.png")

    # Load BiRRT path
    path: Optional[List[np.ndarray]] = None
    if (cd / "birrt_paths.npz").exists():
        b = np.load(cd / "birrt_paths.npz", allow_pickle=True)
        sf    = b["success_flags"]
        pmins = b["path_min_clearances"]
        valid = [i for i, ok in enumerate(sf) if ok]
        if valid:
            best  = max(valid, key=lambda i: float(pmins[i]))
            p_arr = b["paths"][best]
            if len(p_arr) >= 2:
                path = [np.asarray(p_arr[i], dtype=float) for i in range(len(p_arr))]
                print(f"  Loaded BiRRT path: {len(path)} waypoints")

    if path is None:
        if run_birrt_if_missing:
            print("  [BiRRT] Running inline BiRRT (--run-birrt-if-missing)...")
            goals = (q_goals_arr if q_goals_arr is not None and len(q_goals_arr) > 0
                     else np.array([q_goal]))
            path = _run_birrt_inline(spec, goals, seed)
        else:
            print("  [SKIP] No cached BiRRT path. Pass --run-birrt-if-missing to compute.")

    if path is not None:
        _save_scene_path(spec, path, q_goal, cd / "scene_path.png")
        clears = _compute_path_clearances(path, spec.collision_obstacles())
        _save_clearance_plot(clears, cd / "scene_path_clearance.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MB bookshelf scene visualizer")
    parser.add_argument("--set",       default="bookshelf_small_panda")
    parser.add_argument("--problems",  default="63,84")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--save-dir",  default=_DEFAULT_SAVE_DIR)
    parser.add_argument("--max-per-set", type=int, default=100)
    parser.add_argument("--run-birrt-if-missing", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    problem_indices = [int(x) for x in args.problems.split(",")]
    all_probs = load_mb_problems(problem_sets=[args.set],
                                  max_per_set=args.max_per_set, seed=0)
    problems = [(sn, pi, sp, ro) for sn, pi, sp, ro in all_probs
                if pi in problem_indices]
    if not problems:
        print(f"ERROR: problems {problem_indices} not found in {args.set}"); return

    for set_name, prob_idx, spec, raw_obs in problems:
        _visualize_problem(set_name, prob_idx, spec, raw_obs, args.seed,
                           args.save_dir, args.run_birrt_if_missing, args.force_cache)


if __name__ == "__main__":
    main()
