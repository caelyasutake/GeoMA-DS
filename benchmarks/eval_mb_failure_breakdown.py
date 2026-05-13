"""
Failure breakdown ablation for MotionBenchMaker bookshelf problems.

Variants
--------
ik_only             Run HJCD-IK; report goal clearances; save ik_goals.npz.
birrt_only          Plan BiRRT to each IK goal; save birrt_paths.npz.
pathds_no_cbf       Execute with CBF disabled.
pathds_default_cbf  Execute with d_safe=0.03, d_buffer=0.05.
pathds_goal_aware_cbf  Execute with GoalAwareCBFConfig.
pathds_ee_cbf_disabled Execute with CBF disabled for link indices 6 and 7.

Usage::

    conda run -n ds-iks python -m benchmarks.eval_mb_failure_breakdown \\
      --set bookshelf_small_panda --problems 63,84 --seeds 0,1,2

Cache layout (see benchmarks/_mb_cache.py for fingerprint logic)::

    outputs/debug_bookshelf/{set}_{prob:04d}/
        metadata.json
        ik_goals.npz
        birrt_paths.npz
        failure_breakdown.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks._mb_cache import (
    build_metadata, cache_dir, check_fingerprint, ik_fingerprint,
    birrt_fingerprint, write_metadata, read_metadata,
)
from src.evaluation.baselines import ControlCondition, IKCondition, make_condition
from src.evaluation.metrics import TrialMetrics
from src.evaluation.trial_runner import run_planning_trial, _make_clearance_fn
from src.scenarios.mb_loader import load_mb_problems
from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.controller.cbf_filter import CBFConfig, GoalAwareCBFConfig
from src.solver.ik.hjcd_wrapper import solve_batch
from src.solver.planner.birrt import PlannerConfig, plan
from src.solver.planner.collision import (
    _panda_hand_transform, _panda_link_positions, _LINK_RADII, _quat_to_rot,
    make_collision_fn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HJCD_BATCH         = 1000
_HJCD_NSOL          = 4
_PLANNER_MAX_ITER   = 2000
_PLANNER_STEP       = 0.1
_DEFAULT_SAVE_DIR   = "outputs/debug_bookshelf"

CONDITION = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)
_TRIAL_KWARGS = dict(
    morse_override={"enabled": False},
    supervisor_override={"enabled": False},
)

_LINK_NAMES = {
    0: "link1", 1: "link2", 2: "link3", 3: "link4",
    4: "link5", 5: "link6", 6: "link7", 7: "hand",
}

ALL_VARIANTS = [
    "ik_only", "birrt_only",
    "pathds_no_cbf", "pathds_default_cbf",
    "pathds_goal_aware_cbf", "pathds_ee_cbf_disabled",
]
EXECUTION_VARIANTS = [v for v in ALL_VARIANTS if v.startswith("pathds_")]


# ---------------------------------------------------------------------------
# Target frame helpers
# ---------------------------------------------------------------------------
def _mb_target_to_hjcd(target_pose: dict) -> dict:
    pos  = np.array(target_pose["position"], dtype=float)
    quat = np.array(target_pose["quaternion_wxyz"], dtype=float)
    R    = _quat_to_rot(*quat)
    return {
        "position": (pos + R[:, 2] * 0.105).tolist(),
        "quaternion_wxyz": quat.tolist(),
    }


def _grasptarget_pos(q: np.ndarray) -> np.ndarray:
    hand_pos, R_hand = _panda_hand_transform(q)
    return hand_pos + R_hand @ np.array([0.0, 0.0, 0.105])


# ---------------------------------------------------------------------------
# CBF config factory
# ---------------------------------------------------------------------------
def _make_cbf_config(variant: str) -> CBFConfig:
    base = dict(alpha=8.0)
    if variant == "pathds_no_cbf":
        return CBFConfig(enabled=False)
    if variant == "pathds_default_cbf":
        return CBFConfig(enabled=True, d_safe=0.03, d_buffer=0.05, **base)
    if variant == "pathds_goal_aware_cbf":
        ga = GoalAwareCBFConfig(
            enabled=True, goal_radius_start=0.25, goal_radius_end=0.05,
            min_d_safe_scale=0.5, min_d_buffer_scale=0.3,
            goal_clearance_min=0.02, w_slack_goal=100.0,
        )
        return CBFConfig(enabled=True, d_safe=0.03, d_buffer=0.05, goal_aware=ga, **base)
    if variant == "pathds_ee_cbf_disabled":
        return CBFConfig(enabled=True, d_safe=0.03, d_buffer=0.05,
                         disabled_link_indices=(6, 7), **base)
    raise ValueError(f"Unknown execution variant: {variant!r}")


# ---------------------------------------------------------------------------
# Variant: ik_only
# ---------------------------------------------------------------------------
def _run_ik_only(
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    seed: int,
    save_dir: str,
    force_cache: bool,
) -> dict:
    cd = cache_dir(save_dir, set_name, prob_idx)
    meta_now = build_metadata(set_name, prob_idx, seed, spec, raw_obs,
                               _HJCD_BATCH, _HJCD_NSOL,
                               _PLANNER_MAX_ITER, _PLANNER_STEP)
    fp = ik_fingerprint(meta_now)

    # Reuse cache if valid
    if (cd / "ik_goals.npz").exists() and check_fingerprint(cd, fp, "ik_goals.npz", force_cache):
        data = np.load(cd / "ik_goals.npz", allow_pickle=True)
        return {
            "stage_ok": True,
            "n_ik_solutions": int(len(data["q_goals"])),
            "goal_clearances": data["clearances"].tolist(),
            "error": None,
        }

    # Run HJCD-IK
    hjcd_target = _mb_target_to_hjcd(spec.target_pose)
    try:
        result = solve_batch(hjcd_target, env_config={"obstacles": raw_obs},
                             batch_size=_HJCD_BATCH, num_solutions=_HJCD_NSOL)
    except Exception as e:
        return {"stage_ok": False, "n_ik_solutions": 0,
                "goal_clearances": [], "error": str(e)}

    if not result.solutions:
        return {"stage_ok": False, "n_ik_solutions": 0,
                "goal_clearances": [], "error": "HJCD-IK: 0 solutions"}

    q_goals    = np.array(result.solutions, dtype=np.float64)
    cl_fn      = _make_clearance_fn(spec)
    clearances = np.array([cl_fn(q) for q in q_goals])
    ee_pos     = np.array([_panda_link_positions(q)[7] for q in q_goals])

    cd.mkdir(parents=True, exist_ok=True)
    np.savez(cd / "ik_goals.npz",
             q_goals=q_goals, clearances=clearances, ee_positions=ee_pos)
    write_metadata(cd, meta_now)

    return {
        "stage_ok": True,
        "n_ik_solutions": len(q_goals),
        "goal_clearances": clearances.tolist(),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Variant: birrt_only
# ---------------------------------------------------------------------------
def _run_birrt_only(
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    seed: int,
    save_dir: str,
    force_cache: bool,
) -> dict:
    cd = cache_dir(save_dir, set_name, prob_idx)
    meta_now = build_metadata(set_name, prob_idx, seed, spec, raw_obs,
                               _HJCD_BATCH, _HJCD_NSOL,
                               _PLANNER_MAX_ITER, _PLANNER_STEP)
    ik_fp    = ik_fingerprint(meta_now)
    birrt_fp = birrt_fingerprint(meta_now)

    # IK cache must exist
    if not (cd / "ik_goals.npz").exists():
        return {"stage_ok": False,
                "error": "ik_goals.npz not found — run ik_only first"}
    check_fingerprint(cd, ik_fp, "ik_goals.npz", force_cache)
    ik_data = np.load(cd / "ik_goals.npz", allow_pickle=True)
    q_goals = ik_data["q_goals"]   # (N, 7)

    # Reuse BiRRT cache if valid
    if (cd / "birrt_paths.npz").exists() and check_fingerprint(
            cd, birrt_fp, "birrt_paths.npz", force_cache):
        b = np.load(cd / "birrt_paths.npz", allow_pickle=True)
        pmins = b["path_min_clearances"].tolist()
        n_succ = int(b["success_flags"].sum())
        return {
            "stage_ok": True,
            "n_goals_attempted": len(q_goals),
            "n_goals_succeeded": n_succ,
            "path_min_clearances": [
                None if (v is not None and np.isinf(float(v))) else v for v in pmins
            ],
            "error": None,
        }

    # Plan BiRRT to each IK goal
    collision_fn = make_collision_fn(spec=spec)
    cl_fn        = _make_clearance_fn(spec)
    q_start      = np.asarray(spec.q_start, dtype=float)

    paths_list:    list = []
    success_flags: list = []
    pmins:         list = []

    for goal_idx, q_goal in enumerate(q_goals):
        goal_seed  = seed + 1009 * goal_idx
        plan_cfg   = PlannerConfig(max_iterations=_PLANNER_MAX_ITER,
                                   step_size=_PLANNER_STEP, seed=goal_seed)
        pr = plan(q_start, [np.asarray(q_goal, dtype=float)],
                  env=collision_fn, config=plan_cfg, clearance_fn=cl_fn)

        if pr.success and pr.path:
            path_arr = np.array(pr.path, dtype=np.float32)
            pmin = float(min(cl_fn(q) for q in pr.path))
            paths_list.append(path_arr)
            success_flags.append(True)
            pmins.append(pmin)
        else:
            paths_list.append(np.zeros((0, 7), dtype=np.float32))
            success_flags.append(False)
            pmins.append(-np.inf)

    paths_obj = np.empty(len(paths_list), dtype=object)
    for i, p in enumerate(paths_list):
        paths_obj[i] = p

    np.savez(
        cd / "birrt_paths.npz",
        paths=paths_obj,
        goal_indices=np.arange(len(q_goals), dtype=np.int32),
        success_flags=np.array(success_flags, dtype=bool),
        path_min_clearances=np.array(pmins, dtype=np.float64),
    )
    # Update metadata with planner params
    meta_stored = read_metadata(cd) or {}
    meta_stored.update({"planner_max_iter": _PLANNER_MAX_ITER,
                         "planner_step_size": _PLANNER_STEP})
    write_metadata(cd, meta_stored)

    n_succ = sum(success_flags)
    return {
        "stage_ok": True,
        "n_goals_attempted": len(q_goals),
        "n_goals_succeeded": n_succ,
        "path_min_clearances": [None if np.isinf(v) else float(v) for v in pmins],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Variant: execution (pathds_*)
# ---------------------------------------------------------------------------
def _run_execution_variant(
    variant: str,
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    seed: int,
    save_dir: str,
    force_cache: bool,
) -> dict:
    """Run PathDS execution for one variant using the best cached BiRRT path."""
    from src.solver.controller.cbf_filter import _clearance as _cbf_cl

    cd = cache_dir(save_dir, set_name, prob_idx)

    if not (cd / "ik_goals.npz").exists():
        return {"success": False, "failure_reason": "birrt_failed",
                "error": "ik_goals.npz missing — run ik_only first"}
    if not (cd / "birrt_paths.npz").exists():
        return {"success": False, "failure_reason": "birrt_failed",
                "error": "birrt_paths.npz missing — run birrt_only first"}

    ik_data    = np.load(cd / "ik_goals.npz",    allow_pickle=True)
    birrt_data = np.load(cd / "birrt_paths.npz", allow_pickle=True)

    q_goals       = ik_data["q_goals"]
    clearances_ik = ik_data["clearances"]
    success_flags = birrt_data["success_flags"]
    pmins         = birrt_data["path_min_clearances"]

    # Select safest successful BiRRT goal
    valid_idxs = [i for i, ok in enumerate(success_flags) if ok]
    if not valid_idxs:
        return {"success": False, "failure_reason": "birrt_failed", "error": None,
                "goal_err": None, "min_clearance": None, "collision_count": 0,
                "cbf_active_fraction_total": 0.0,
                "cbf_active_fraction_final_20pct": 0.0,
                "progress_final_20pct": 0.0,
                "closest_link": "unknown", "closest_obstacle": "unknown"}

    best_idx   = max(valid_idxs, key=lambda i: float(pmins[i]))
    q_goal     = np.asarray(q_goals[best_idx], dtype=float)
    trial_spec = dataclasses.replace(spec, ik_goals=[q_goal])

    cbf_cfg = _make_cbf_config(variant)
    q_log: List = []
    trial: TrialMetrics = run_planning_trial(
        trial_spec, condition=CONDITION, seed=seed, trial_id=0,
        cbf_override=cbf_cfg, q_log=q_log,
        **_TRIAL_KWARGS,
    )

    reached = bool(trial.execution and trial.execution.terminal_success)
    goal_err_val = (float(trial.execution.final_goal_err)
                    if trial.execution and trial.execution.final_goal_err is not None
                    else None)
    collision_count = 0
    min_cl: Optional[float] = None
    if trial.barrier:
        collision_count = trial.barrier.collision_count
        if trial.barrier.min_clearance < float("inf"):
            min_cl = float(trial.barrier.min_clearance)
    success = reached and (collision_count == 0)

    cbf_total  = 0.0
    cbf_frac20 = 0.0
    progress20 = 0.0
    if trial.cbf_goal_conflict:
        cbf_total  = trial.cbf_goal_conflict.cbf_active_fraction_total
        cbf_frac20 = trial.cbf_goal_conflict.cbf_active_fraction_final_20pct
        progress20 = trial.cbf_goal_conflict.progress_final_20pct

    # Closest link/obstacle at final state
    closest_link = "unknown"
    closest_obs  = "unknown"
    if q_log:
        q_final  = np.asarray(q_log[-1], dtype=float)
        obs_list = spec.collision_obstacles()
        link_pos = _panda_link_positions(q_final)
        best_cl  = float("inf")
        for li, pos in enumerate(link_pos):
            r = float(_LINK_RADII.get(li, 0.08))
            for obs in obs_list:
                cl = _cbf_cl(np.asarray(pos, dtype=float), r, obs)
                if cl < best_cl:
                    best_cl      = cl
                    closest_link = _LINK_NAMES.get(li, f"link{li}")
                    closest_obs  = obs.name

    # Obstacle geometry check: grasptarget FK vs MB target
    target_pos   = np.array(spec.target_pose["position"], dtype=float)
    gt_pos       = _grasptarget_pos(q_goal)
    target_error = float(np.linalg.norm(gt_pos - target_pos))

    # Failure classification (priority order)
    if success:
        failure_reason = "success"
    elif target_error > 0.005:
        failure_reason = "obstacle_geometry_error"
    elif all(float(c) <= 0.0 for c in clearances_ik):
        failure_reason = "ik_goal_invalid"
    elif not any(success_flags):
        failure_reason = "birrt_failed"
    elif (min_cl is not None and min_cl < 0.0
          and closest_link in ("link7", "hand")):
        failure_reason = "flange_collision"
    elif cbf_frac20 > 0.5 and progress20 < 0.01:
        failure_reason = "cbf_goal_conflict"
    else:
        failure_reason = "pathds_deviation"

    return {
        "success":                          success,
        "goal_err":                         goal_err_val,
        "min_clearance":                    min_cl,
        "collision_count":                  collision_count,
        "cbf_active_fraction_total":        cbf_total,
        "cbf_active_fraction_final_20pct":  cbf_frac20,
        "progress_final_20pct":             progress20,
        "closest_link":                     closest_link,
        "closest_obstacle":                 closest_obs,
        "target_error_m":                   target_error,
        "failure_reason":                   failure_reason,
        "error":                            None,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def _run_variant(
    variant: str,
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    seed: int,
    save_dir: str,
    force_cache: bool,
) -> dict:
    if variant == "ik_only":
        return _run_ik_only(set_name, prob_idx, spec, raw_obs, seed, save_dir, force_cache)
    if variant == "birrt_only":
        return _run_birrt_only(set_name, prob_idx, spec, raw_obs, seed, save_dir, force_cache)
    if variant in EXECUTION_VARIANTS:
        return _run_execution_variant(variant, set_name, prob_idx, spec,
                                       raw_obs, seed, save_dir, force_cache)
    raise ValueError(f"Unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _fmt(val, fmt=".3f"):
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    return format(val, fmt) if isinstance(val, float) else str(val)


def _print_table(variants: List[str], results: Dict[str, list]) -> None:
    cols = [
        ("variant",      "Variant",      28),
        ("success",      "OK",            5),
        ("failure",      "Reason",       22),
        ("goal_err",     "GoalErr",       9),
        ("min_cl",       "MinCl",         8),
        ("cbf20",        "CBF20%",        8),
        ("closest_link", "ClosestLink",  12),
        ("closest_obs",  "ClosestObs",   12),
    ]
    header = "  ".join(f"{name:<{w}}" for _, name, w in cols)
    print(f"\n{header}")
    print("-" * len(header))
    for v in variants:
        rows = results.get(v, [])
        if not rows:
            print(f"  {v:<28}  (no results)")
            continue
        ok_rows = [r for r in rows if not r.get("error")]
        if not ok_rows:
            errs = [r.get("error", "?") for r in rows]
            print(f"  {v:<28}  ERROR: {errs[0]}")
            continue
        r0 = ok_rows[0]
        vals = {
            "variant":      v,
            "success":      str(r0.get("success", r0.get("stage_ok", "?"))),
            "failure":      r0.get("failure_reason", r0.get("error", "")),
            "goal_err":     _fmt(r0.get("goal_err")),
            "min_cl":       _fmt(r0.get("min_clearance")),
            "cbf20":        _fmt(r0.get("cbf_active_fraction_final_20pct")),
            "closest_link": r0.get("closest_link", ""),
            "closest_obs":  r0.get("closest_obstacle", ""),
        }
        row = "  ".join(f"{vals[k]:<{w}}" for k, _, w in cols)
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MB bookshelf failure breakdown")
    parser.add_argument("--set",       default="bookshelf_small_panda")
    parser.add_argument("--problems",  default="63,84")
    parser.add_argument("--seeds",     default="0,1,2")
    parser.add_argument("--variants",  nargs="+", default=ALL_VARIANTS)
    parser.add_argument("--save-dir",  default=_DEFAULT_SAVE_DIR)
    parser.add_argument("--max-per-set", type=int, default=100)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    problem_indices = [int(x) for x in args.problems.split(",")]
    seeds           = [int(x) for x in args.seeds.split(",")]
    variants        = args.variants

    print(f"Loading {args.set} (max {args.max_per_set} per set)...")
    all_probs = load_mb_problems(problem_sets=[args.set],
                                  max_per_set=args.max_per_set, seed=0)
    problems = [(sn, pi, sp, ro) for sn, pi, sp, ro in all_probs
                if pi in problem_indices]
    if not problems:
        print(f"ERROR: problems {problem_indices} not found in {args.set}")
        return

    total = len(variants) * len(problems) * len(seeds)
    print(f"Running {len(variants)} variants × {len(problems)} problems × "
          f"{len(seeds)} seeds = {total} trials")

    results: Dict[str, List[dict]] = {v: [] for v in variants}
    done = 0
    for variant in variants:
        for set_name, prob_idx, spec, raw_obs in problems:
            for seed in seeds:
                done += 1
                print(f"  [{done}/{total}] {variant} | prob={prob_idx} seed={seed}",
                      end=" ", flush=True)
                t0 = time.perf_counter()
                r = _run_variant(variant, set_name, prob_idx, spec, raw_obs,
                                  seed, args.save_dir, args.force_cache)
                r["wall_time_s"] = time.perf_counter() - t0
                results[variant].append(r)
                status = ("OK" if r.get("success") or r.get("stage_ok")
                          else f"FAIL ({r.get('failure_reason') or r.get('error') or '?'})")
                print(status)

    # Save failure_breakdown.json per problem (last seed wins for non-seed fields)
    for set_name, prob_idx, _, _ in problems:
        cd = cache_dir(args.save_dir, set_name, prob_idx)
        cd.mkdir(parents=True, exist_ok=True)
        breakdown: Dict[str, dict] = {}
        for v in variants:
            vr = [r for r in results[v] if r]
            breakdown[v] = vr[-1] if vr else {}
        with open(cd / "failure_breakdown.json", "w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2)

    _print_table(variants, results)


if __name__ == "__main__":
    main()
