"""
MotionBenchmaker scene validation / preflight debug tool.

Checks coordinate frames, obstacle geometry, start/goal validity,
and clearance for a single MB problem before running ablations.

Usage::

    conda run -n ds-iks python -m benchmarks.validate_mb_scene
    conda run -n ds-iks python -m benchmarks.validate_mb_scene --set bookshelf_small_panda --problem 1
    conda run -n ds-iks python -m benchmarks.validate_mb_scene --set bookshelf_small_panda --all

Output::

    outputs/eval/mb_debug/<set>_<problem>/
        scene.json     -- imported obstacle list (position/size/orientation)
        start.png      -- robot at q_start with obstacles and target
        goal.png       -- robot at first IK goal with obstacles and target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.scenarios.mb_loader import load_mb_problems
from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.planner.collision import (
    _panda_link_positions, _LINK_RADII,
    _sphere_box_signed_dist, _precompute_obs_rotations, _quat_to_rot,
)
from src.evaluation.trial_runner import _make_clearance_fn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EE offset: panda_grasptarget is this far along the hand z-axis past panda_hand.
# HJCD-IK targets panda_grasptarget; MB goals are in panda_hand frame.
_GRASPTARGET_OFFSET_M = 0.105   # metres

# Workspace bounds for assertion (panda_link0 frame)
_WORKSPACE_XMIN, _WORKSPACE_XMAX = -0.9,  1.4
_WORKSPACE_YMIN, _WORKSPACE_YMAX = -1.5,  1.5
_WORKSPACE_ZMIN, _WORKSPACE_ZMAX = -0.5,  1.5


# ---------------------------------------------------------------------------
# FK helpers
# ---------------------------------------------------------------------------

def _fk_ee(q: np.ndarray) -> np.ndarray:
    """Return panda_hand body position for joint config q."""
    return _panda_link_positions(q)[-1]


def _fk_grasptarget(q: np.ndarray) -> np.ndarray:
    """Return panda_grasptarget position (0.105 m past hand along hand z-axis)."""
    positions = _panda_link_positions(q)
    # hand is positions[-1]; we need the EE z-axis from the rotation chain.
    # Recompute final frame orientation via the same FK.
    # Use finite-difference along z-offset to get grasptarget world position.
    # The grasptarget is at [0,0,0.105] in link7 post-joint frame.
    # This is already computed in _panda_link_positions as the hand body.
    # Actually hand body IS the [0,0,0.107] position — grasptarget is at 0.105 from hand.
    # Approximate: grasptarget ≈ hand + EE_z * 0.105 where EE_z is the hand z-axis.
    hand_pos = positions[-1]
    link7_pos = positions[6]
    # EE z-axis ≈ normalized (hand - link7) direction (approx, good enough for diagnostics)
    axis = hand_pos - link7_pos
    n = float(np.linalg.norm(axis))
    if n > 1e-9:
        axis = axis / n
    return hand_pos + axis * _GRASPTARGET_OFFSET_M


# ---------------------------------------------------------------------------
# AABB helper for display
# ---------------------------------------------------------------------------

def _obs_aabb(obs) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame AABB (lo, hi) for an obstacle (conservative for rotated boxes)."""
    pos = np.array(obs.position)
    sz  = np.array(obs.size)
    if obs.type == "box":
        R = _quat_to_rot(*obs.orientation_wxyz)
        # Worst-case AABB of rotated box: for each axis, max projection of half-extents
        half_world = np.abs(R) @ sz
        return pos - half_world, pos + half_world
    elif obs.type == "cylinder":
        r, hh = float(sz[0]), float(sz[1])
        return pos - np.array([r, r, hh]), pos + np.array([r, r, hh])
    else:
        return pos, pos


# ---------------------------------------------------------------------------
# Per-link clearance breakdown
# ---------------------------------------------------------------------------

def _per_link_clearance(q: np.ndarray, spec: ScenarioSpec):
    """Return list of (link_name, radius, min_clearance, worst_obstacle) tuples."""
    obs_list = spec.collision_obstacles()
    obs_R_inv = _precompute_obs_rotations(obs_list)
    positions = _panda_link_positions(q)
    results = []
    for i, lp in enumerate(positions):
        r = float(_LINK_RADII.get(i, 0.08))
        link_name = f"link{i}" if i < 7 else "hand"
        min_cl = float("inf")
        worst_obs = None
        for j, obs in enumerate(obs_list):
            obs_pos = np.array(obs.position)
            t = obs.type.lower()
            if t == "box":
                cl = _sphere_box_signed_dist(
                    np.array(lp, dtype=float), r, obs_pos,
                    np.array(obs.size, dtype=float), obs_R_inv[j],
                )
            elif t == "sphere":
                cl = float(np.linalg.norm(np.array(lp) - obs_pos)) - r - float(obs.size[0])
            elif t == "cylinder":
                cyl_r, cyl_hh = float(obs.size[0]), float(obs.size[1])
                dx, dy = float(lp[0]) - obs_pos[0], float(lp[1]) - obs_pos[1]
                dz = float(lp[2]) - obs_pos[2]
                rxy = float(np.sqrt(dx*dx + dy*dy))
                cz  = float(np.clip(dz, -cyl_hh, cyl_hh))
                if rxy < 1e-9:
                    closest = obs_pos + np.array([0, 0, cz])
                elif rxy <= cyl_r:
                    closest = np.array([obs_pos[0]+dx, obs_pos[1]+dy, obs_pos[2]+cz])
                else:
                    scale = cyl_r / rxy
                    closest = np.array([obs_pos[0]+dx*scale, obs_pos[1]+dy*scale, obs_pos[2]+cz])
                cl = float(np.linalg.norm(np.array(lp) - closest)) - r
            else:
                continue
            if cl < min_cl:
                min_cl = cl
                worst_obs = obs.name
        results.append((link_name, r, min_cl, worst_obs))
    return results


# ---------------------------------------------------------------------------
# Straight-line interpolation clearance
# ---------------------------------------------------------------------------

def _interp_min_clearance(
    q_start: np.ndarray,
    q_end: np.ndarray,
    clearance_fn,
    n_samples: int = 20,
) -> float:
    """Minimum Python clearance along the straight-line interpolation q_start→q_end."""
    mn = float("inf")
    for alpha in np.linspace(0.0, 1.0, n_samples):
        q = (1.0 - alpha) * q_start + alpha * q_end
        cl = clearance_fn(q)
        if cl < mn:
            mn = cl
    return mn


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_box_wireframe(ax, pos: np.ndarray, half: np.ndarray, R: np.ndarray, color, lw=0.8, alpha=0.6) -> None:
    """Draw rotated-box wireframe on a mpl 3D axes (one plot call per edge)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = pos + (signs * half) @ R.T

    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                ax.plot(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    color=color, linewidth=lw, alpha=alpha,
                )


def _visualize_scene(
    spec: ScenarioSpec,
    q_configs: dict,      # label → q array
    target_pos: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Save a 3D scene snapshot to out_path (PNG). Silently skips if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"start": "#1f77b4", "goal": "#2ca02c"}

    # Obstacles
    obs_list = spec.collision_obstacles()
    for obs in obs_list:
        pos = np.array(obs.position)
        if obs.type == "box":
            R = _quat_to_rot(*obs.orientation_wxyz)
            _draw_box_wireframe(ax, pos, np.array(obs.size), R, color="#d62728")
        elif obs.type == "cylinder":
            r, hh = float(obs.size[0]), float(obs.size[1])
            theta = np.linspace(0, 2 * np.pi, 20)
            for z_off in (-hh, hh):
                ax.plot(
                    pos[0] + r * np.cos(theta),
                    pos[1] + r * np.sin(theta),
                    np.full(20, pos[2] + z_off),
                    color="#d62728", linewidth=0.8, alpha=0.6,
                )

    # Robot links
    for label, q in q_configs.items():
        pts = np.array(_panda_link_positions(q))
        c = colors.get(label, "#7f7f7f")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=c,
                linewidth=2, markersize=5, label=f"robot ({label})")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=c, s=80, zorder=5)

    # Target
    ax.scatter(target_pos[0], target_pos[1], target_pos[2],
               color="#ff7f0e", s=150, marker="*", zorder=6, label="MB target")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or "MB Scene")
    ax.legend(fontsize=8)

    # Axis limits: cover robot links + target
    all_pts = np.vstack([np.array(_panda_link_positions(q)) for q in q_configs.values()])
    all_pts = np.vstack([all_pts, target_pos.reshape(1, 3)])
    ctr = all_pts.mean(axis=0)
    span = max(float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) * 0.7, 0.5)
    ax.set_xlim(float(ctr[0] - span), float(ctr[0] + span))
    ax.set_ylim(float(ctr[1] - span), float(ctr[1] + span))
    ax.set_zlim(float(max(0.0, ctr[2] - span)), float(ctr[2] + span))

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_problem(
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run preflight validation for one MB problem.

    Returns a summary dict with pass/fail flags and key values.
    """
    clearance_fn = _make_clearance_fn(spec)
    obs_list = spec.collision_obstacles()

    # ------------------------------------------------------------------
    # q_start analysis
    # ------------------------------------------------------------------
    cl_start = clearance_fn(spec.q_start)
    ee_start = _fk_ee(spec.q_start)
    gt_start = _fk_grasptarget(spec.q_start)

    # ------------------------------------------------------------------
    # Target analysis
    # ------------------------------------------------------------------
    target_pos = np.array(spec.target_pose["position"])
    target_quat = np.array(spec.target_pose["quaternion_wxyz"])
    target_dist_from_start_ee = float(np.linalg.norm(ee_start - target_pos))

    # ------------------------------------------------------------------
    # IK goal analysis (pre-computed from JSON)
    # ------------------------------------------------------------------
    ik_results = []
    for k, q_goal in enumerate(spec.ik_goals):
        cl_goal = clearance_fn(q_goal)
        ee_goal = _fk_ee(q_goal)
        gt_goal = _fk_grasptarget(q_goal)
        err_hand_to_target = float(np.linalg.norm(ee_goal - target_pos))
        err_grasp_to_target = float(np.linalg.norm(gt_goal - target_pos))
        interp_cl = _interp_min_clearance(spec.q_start, q_goal, clearance_fn)
        ik_results.append({
            "idx": k,
            "clearance_at_goal": float(cl_goal),
            "interp_min_clearance": float(interp_cl),
            "fk_hand": ee_goal.tolist(),
            "fk_grasptarget": gt_goal.tolist(),
            "err_hand_to_target_m": err_hand_to_target,
            "err_grasptarget_to_target_m": err_grasp_to_target,
        })

    # ------------------------------------------------------------------
    # Obstacle geometry
    # ------------------------------------------------------------------
    obs_records = []
    for obs in obs_list:
        lo, hi = _obs_aabb(obs)
        in_workspace = (
            _WORKSPACE_XMIN <= obs.position[0] <= _WORKSPACE_XMAX and
            _WORKSPACE_YMIN <= obs.position[1] <= _WORKSPACE_YMAX and
            _WORKSPACE_ZMIN <= obs.position[2] <= _WORKSPACE_ZMAX
        )
        obs_records.append({
            "name": obs.name,
            "type": obs.type,
            "position": list(obs.position),
            "size_half_extents": list(obs.size),
            "orientation_wxyz": list(obs.orientation_wxyz),
            "aabb_lo": lo.tolist(),
            "aabb_hi": hi.tolist(),
            "in_workspace": in_workspace,
        })

    # ------------------------------------------------------------------
    # Per-link clearance at q_start
    # ------------------------------------------------------------------
    link_cl = _per_link_clearance(spec.q_start, spec)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    errors: List[str] = []
    warnings: List[str] = []

    if cl_start < 0.0:
        errors.append(f"START IN COLLISION: clearance={cl_start:.4f} < 0")
    elif cl_start < 0.02:
        warnings.append(f"Start near obstacle: clearance={cl_start:.4f}")

    for ir in ik_results[:5]:
        if ir["clearance_at_goal"] < 0.0:
            errors.append(
                f"IK goal [{ir['idx']}] IN COLLISION: clearance={ir['clearance_at_goal']:.4f}"
            )

    for or_ in obs_records:
        if not or_["in_workspace"]:
            warnings.append(f"Obstacle {or_['name']} centre outside expected workspace bounds")
        dims = np.array(or_["size_half_extents"])
        if np.any(dims <= 0):
            errors.append(f"Obstacle {or_['name']} has non-positive half-extents: {dims}")
        if np.any(dims > 2.0):
            warnings.append(f"Obstacle {or_['name']} has large half-extents: {dims}")

    if target_dist_from_start_ee > 2.0:
        warnings.append(
            f"Target very far from start EE: {target_dist_from_start_ee:.3f} m"
        )

    # Check frame: does panda_hand reach the target, or does panda_grasptarget?
    if ik_results:
        hand_errs = [ir["err_hand_to_target_m"] for ir in ik_results]
        grasp_errs = [ir["err_grasptarget_to_target_m"] for ir in ik_results]
        best_hand  = min(hand_errs)
        best_grasp = min(grasp_errs)
        if best_grasp < best_hand - 0.05:
            warnings.append(
                f"IK goals place grasptarget closer to MB target than hand "
                f"(hand_err={best_hand:.3f}, grasp_err={best_grasp:.3f}); "
                f"possible 0.105m frame offset issue"
            )

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    if verbose:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  VALIDATE MB SCENE: {set_name} #{prob_idx}")
        print(f"{sep}")
        print(f"\n  q_start: {np.round(spec.q_start, 3)}")
        print(f"  EE (panda_hand) at q_start:       {np.round(ee_start, 4)}")
        print(f"  EE (panda_grasptarget) at q_start: {np.round(gt_start, 4)}")
        print(f"  Clearance at q_start: {cl_start:.4f}  {'OK' if cl_start >= 0 else 'IN COLLISION'}")
        print(f"\n  target_pose (frame: panda_hand):")
        print(f"    position:        {np.round(target_pos, 4)}")
        print(f"    quaternion_wxyz: {np.round(target_quat, 4)}")
        print(f"    dist from start EE: {target_dist_from_start_ee:.4f} m")

        print(f"\n  Obstacle list ({len(obs_list)} obstacles):")
        for or_ in obs_records:
            flag = "" if or_["in_workspace"] else "  [OUT OF WORKSPACE]"
            print(f"    {or_['name']} ({or_['type']})")
            print(f"      pos={np.round(or_['position'], 3)}  "
                  f"half={np.round(or_['size_half_extents'], 3)}  "
                  f"q={np.round(or_['orientation_wxyz'], 3)}{flag}")

        print(f"\n  Per-link clearance at q_start (showing links with cl < 0.15):")
        any_shown = False
        for lname, r, cl, worst in link_cl:
            if cl < 0.15:
                flag = "  *** COLLISION" if cl < 0 else ""
                print(f"    [{lname}] r={r:.2f}  cl={cl:.4f}  vs {worst}{flag}")
                any_shown = True
        if not any_shown:
            print("    (all links clear with cl >= 0.15)")

        print(f"\n  IK goals (pre-computed from JSON, n={len(spec.ik_goals)}):")
        for ir in ik_results[:8]:
            hand_err  = ir["err_hand_to_target_m"]
            grasp_err = ir["err_grasptarget_to_target_m"]
            cl_g = ir["clearance_at_goal"]
            icl  = ir["interp_min_clearance"]
            frame_tag = "hand" if hand_err < grasp_err else "GRASPTARGET"
            print(f"    goal[{ir['idx']}]: cl={cl_g:.4f}  interp_cl={icl:.4f}  "
                  f"hand_err={hand_err:.4f}m  grasp_err={grasp_err:.4f}m  [{frame_tag}]")

        if warnings:
            print(f"\n  WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"    ! {w}")
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    *** {e}")

        status = "PASS" if not errors else "FAIL"
        print(f"\n  [{status}]  errors={len(errors)}  warnings={len(warnings)}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Save scene JSON
    # ------------------------------------------------------------------
    summary = {
        "set_name": set_name,
        "prob_idx": prob_idx,
        "world_frame": "panda_link0",
        "q_start": spec.q_start.tolist(),
        "ee_start_hand": ee_start.tolist(),
        "ee_start_grasptarget": gt_start.tolist(),
        "clearance_at_start": float(cl_start),
        "target_pose": {
            "frame": "panda_hand",
            "position": target_pos.tolist(),
            "quaternion_wxyz": target_quat.tolist(),
        },
        "n_ik_goals": len(spec.ik_goals),
        "ik_goals": ik_results,
        "obstacles": obs_records,
        "assertions": {
            "errors": errors,
            "warnings": warnings,
            "pass": len(errors) == 0,
        },
    }

    if out_dir is not None:
        tag = f"{set_name}_{prob_idx:04d}"
        d = out_dir / tag
        d.mkdir(parents=True, exist_ok=True)

        # PNG: robot at q_start
        _visualize_scene(
            spec=spec,
            q_configs={"start": spec.q_start},
            target_pos=target_pos,
            out_path=d / "start.png",
            title=f"{set_name} #{prob_idx} — q_start",
        )

        # PNG: robot at first IK goal (if any)
        if spec.ik_goals:
            _visualize_scene(
                spec=spec,
                q_configs={"start": spec.q_start, "goal": spec.ik_goals[0]},
                target_pos=target_pos,
                out_path=d / "goal.png",
                title=f"{set_name} #{prob_idx} — q_goal[0]",
            )

        scene_path = d / "scene.json"
        with open(scene_path, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"  Scene JSON saved: {scene_path}")
            if (d / "start.png").exists():
                print(f"  start.png saved:  {d / 'start.png'}")
            if (d / "goal.png").exists():
                print(f"  goal.png saved:   {d / 'goal.png'}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MotionBenchmaker scene conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--set",      default="bookshelf_small_panda",
                        dest="problem_set", help="MB problem set name")
    parser.add_argument("--problem",  type=int, default=0,
                        help="Problem index within the set (0-indexed within loaded subset)")
    parser.add_argument("--all",      action="store_true",
                        help="Validate all problems in the set (up to --max-per-set)")
    parser.add_argument("--max-per-set", type=int, default=5)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--out-dir",  type=Path,
                        default=Path("outputs/eval/mb_debug"),
                        help="Directory for scene JSON files")
    args = parser.parse_args()

    problems = load_mb_problems(
        problem_sets=[args.problem_set],
        max_per_set=args.max_per_set,
        seed=args.seed,
    )

    if not problems:
        print(f"No problems loaded for set '{args.problem_set}'")
        sys.exit(1)

    if args.all:
        indices = range(len(problems))
    else:
        if args.problem >= len(problems):
            print(f"Problem index {args.problem} out of range (loaded {len(problems)})")
            sys.exit(1)
        indices = [args.problem]

    all_errors = 0
    for idx in indices:
        set_name, prob_idx, spec, raw_obs = problems[idx]
        summary = validate_problem(
            set_name, prob_idx, spec, raw_obs,
            out_dir=args.out_dir,
        )
        all_errors += len(summary["assertions"]["errors"])

    if args.all:
        print(f"\nTotal errors across {len(list(indices))} problems: {all_errors}")
        if all_errors > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
