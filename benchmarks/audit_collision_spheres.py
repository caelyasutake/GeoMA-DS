"""
Standalone collision sphere audit for MotionBenchMaker problems.

Computes FK at q_start and q_goal, reports clearance from every link sphere
to every obstacle, and flags geometry issues (penetration, inside CBF buffer,
grasptarget offset errors).

Does NOT require IK, BiRRT, or any cached artifacts.

Usage::

    conda run -n ds-iks python -m benchmarks.audit_collision_spheres \\
      --set bookshelf_small_panda --problems 63,84 --config start,goal
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks._mb_cache import cache_dir
from src.scenarios.mb_loader import load_mb_problems
from src.scenarios.scenario_schema import Obstacle, ScenarioSpec
from src.solver.controller.cbf_filter import _clearance as _cbf_clearance
from src.solver.planner.collision import (
    _panda_hand_transform, _panda_link_positions, _LINK_RADII,
)

_D_SAFE             = 0.03
_D_BUFFER           = 0.05
_DEFAULT_SAVE_DIR   = "outputs/debug_bookshelf"
_GRASPTARGET_OFFSET = 0.105

_LINK_NAMES = {
    0: "link1_shoulder", 1: "link2", 2: "link3", 3: "link4",
    4: "link5",          5: "link6", 6: "link7", 7: "hand",
}


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------
def _audit_config(
    q: np.ndarray,
    spec: ScenarioSpec,
    config_label: str,
    mb_target_pos: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Return a list of row-dicts, one per link sphere + extra row for panda_grasptarget.

    Each dict has:
      config, link_idx, link_name, world_x, world_y, world_z, radius,
      closest_obstacle, clearance, target_x, target_y, target_z,
      target_error_m, flags
    """
    obs_list = spec.collision_obstacles()
    link_pos = _panda_link_positions(q)
    rows     = []

    for li in range(len(link_pos)):
        pos  = np.asarray(link_pos[li], dtype=float)
        r    = float(_LINK_RADII.get(li, 0.08))
        name = _LINK_NAMES.get(li, f"link{li}")

        best_cl  = float("inf")
        best_obs = ""
        for obs in obs_list:
            cl = _cbf_clearance(pos, r, obs)
            if cl < best_cl:
                best_cl  = cl
                best_obs = obs.name

        flags = []
        if best_cl < 0.0:
            flags.append("PENETRATING")
        elif best_cl < _D_SAFE:
            flags.append("INSIDE_D_SAFE")
        elif best_cl < _D_SAFE + _D_BUFFER:
            flags.append("INSIDE_BUFFER")

        rows.append({
            "config":           config_label,
            "link_idx":         li,
            "link_name":        name,
            "world_x":          float(pos[0]),
            "world_y":          float(pos[1]),
            "world_z":          float(pos[2]),
            "radius":           r,
            "closest_obstacle": best_obs,
            "clearance":        float(best_cl),
            "target_x":         None,
            "target_y":         None,
            "target_z":         None,
            "target_error_m":   None,
            "flags":            "|".join(flags),
        })

    # Mark hand as HAND_CLOSEST if it has the minimum clearance
    min_cl = min(r["clearance"] for r in rows)
    for row in rows:
        if row["link_name"] == "hand" and row["clearance"] == min_cl:
            existing = row["flags"]
            row["flags"] = ("HAND_CLOSEST|" + existing).strip("|") if existing else "HAND_CLOSEST"

    # Extra row: panda_grasptarget (radius=0, IK target site — NOT an MJCF link)
    hand_pos, R_hand = _panda_hand_transform(q)
    gt_pos   = hand_pos + R_hand @ np.array([0.0, 0.0, _GRASPTARGET_OFFSET])
    best_cl  = float("inf")
    best_obs = ""
    for obs in obs_list:
        cl = _cbf_clearance(gt_pos, 0.0, obs)
        if cl < best_cl:
            best_cl  = cl
            best_obs = obs.name

    gt_flags = []
    if best_cl < 0.0:
        gt_flags.append("PENETRATING")
    elif best_cl < _D_SAFE:
        gt_flags.append("INSIDE_D_SAFE")
    elif best_cl < _D_SAFE + _D_BUFFER:
        gt_flags.append("INSIDE_BUFFER")

    target_err = None
    tx = ty = tz = None
    if mb_target_pos is not None:
        target_err = float(np.linalg.norm(gt_pos - mb_target_pos))
        tx = float(mb_target_pos[0])
        ty = float(mb_target_pos[1])
        tz = float(mb_target_pos[2])
        if target_err > 0.005:
            gt_flags.append("OFFSET_SUSPECT")

    rows.append({
        "config":           config_label,
        "link_idx":         8,
        "link_name":        "panda_grasptarget",
        "world_x":          float(gt_pos[0]),
        "world_y":          float(gt_pos[1]),
        "world_z":          float(gt_pos[2]),
        "radius":           0.0,
        "closest_obstacle": best_obs,
        "clearance":        float(best_cl),
        "target_x":         tx,
        "target_y":         ty,
        "target_z":         tz,
        "target_error_m":   target_err,
        "flags":            "|".join(gt_flags),
    })

    # Sort by clearance ascending (worst first)
    rows.sort(key=lambda r: r["clearance"])
    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _print_rows(rows: List[dict], title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    header = (f"{'link_name':<22} {'r':>6}  {'clearance':>10}  "
              f"{'closest_obs':<18} {'flags'}")
    print(header)
    print("-" * 72)
    for row in rows:
        cl_str   = f"{row['clearance']:>10.4f}"
        name_str = f"{row['link_name']:<22}"
        r_str    = f"{row['radius']:>6.3f}"
        obs_str  = f"{row['closest_obstacle']:<18}"
        flags    = row["flags"]
        print(f"{name_str} {r_str}  {cl_str}  {obs_str} {flags}")
        if row["target_error_m"] is not None:
            print(f"  → target_error={row['target_error_m']:.4f}m  "
                  f"target=[{row['target_x']:.3f},{row['target_y']:.3f},"
                  f"{row['target_z']:.3f}]")


def _save_csv(rows: List[dict], out_path: Path) -> None:
    fields = ["config", "link_idx", "link_name",
              "world_x", "world_y", "world_z", "radius",
              "closest_obstacle", "clearance",
              "target_x", "target_y", "target_z", "target_error_m", "flags"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MB collision sphere audit")
    parser.add_argument("--set",       default="bookshelf_small_panda")
    parser.add_argument("--problems",  default="63,84")
    parser.add_argument("--config",    default="start,goal",
                        help="Comma-separated: start,goal")
    parser.add_argument("--save-dir",  default=_DEFAULT_SAVE_DIR)
    parser.add_argument("--max-per-set", type=int, default=100)
    args = parser.parse_args()

    configs_to_run  = [c.strip() for c in args.config.split(",")]
    problem_indices = [int(x) for x in args.problems.split(",")]

    all_probs = load_mb_problems(problem_sets=[args.set],
                                  max_per_set=args.max_per_set, seed=0)
    problems = [(sn, pi, sp, ro) for sn, pi, sp, ro in all_probs
                if pi in problem_indices]
    if not problems:
        print(f"ERROR: problems {problem_indices} not found in {args.set}"); return

    for set_name, prob_idx, spec, raw_obs in problems:
        print(f"\n### {set_name} prob={prob_idx}")
        cd = cache_dir(args.save_dir, set_name, prob_idx)
        cd.mkdir(parents=True, exist_ok=True)

        target_pos = np.array(spec.target_pose["position"], dtype=float)
        all_rows: List[dict] = []

        if "start" in configs_to_run:
            q_start = np.asarray(spec.q_start, dtype=float)
            rows = _audit_config(q_start, spec, "start", mb_target_pos=None)
            _print_rows(rows, f"q_start  (prob {prob_idx})")
            all_rows.extend(rows)

        if "goal" in configs_to_run:
            q_goal = None
            ik_npz = cd / "ik_goals.npz"
            if ik_npz.exists():
                d = np.load(ik_npz, allow_pickle=True)
                if len(d["q_goals"]) > 0:
                    q_goal = np.asarray(d["q_goals"][0], dtype=float)
            if q_goal is None and spec.ik_goals:
                q_goal = np.asarray(spec.ik_goals[0], dtype=float)
            if q_goal is None:
                print("  [SKIP] No IK goal available for goal audit")
            else:
                rows = _audit_config(q_goal, spec, "goal", mb_target_pos=target_pos)
                _print_rows(rows, f"q_goal  (prob {prob_idx})")
                all_rows.extend(rows)

        if all_rows:
            _save_csv(all_rows, cd / "collision_audit.csv")


if __name__ == "__main__":
    main()
