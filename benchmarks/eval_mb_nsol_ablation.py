"""
MotionBenchmaker nsol ablation benchmark.

Evaluates whether single-solution HJCD-IK is sufficient on broader
MotionBenchmaker-style problems, and whether multi-solution IK improves
robustness.  Isolates HJCD-IK + PathDS; no BiRRT, no Morse.

Variants
--------
baseline_diffik_ds         — Pre-computed goal_ik, DiffIK (no HJCD-IK)
hjcd_ik_pathds_nsol1       — HJCD-IK nsol=1 + PathDS
hjcd_ik_pathds_nsol2       — HJCD-IK nsol=2 + PathDS
hjcd_ik_pathds_nsol4       — HJCD-IK nsol=4 + PathDS
hjcd_ik_pathds_nsol8       — HJCD-IK nsol=8 + PathDS
adaptive_hjcd_ik_pathds    — nsol=1, upgrades to nsol=8 if clearance < threshold

Usage::

    conda run -n ds-iks python -m benchmarks.eval_mb_nsol_ablation
    conda run -n ds-iks python -m benchmarks.eval_mb_nsol_ablation \\
        --problem-sets box_panda cage_panda --max-per-set 10 --n-seeds 3
    conda run -n ds-iks python -m benchmarks.eval_mb_nsol_ablation \\
        --variants baseline_diffik_ds hjcd_ik_pathds_nsol8 --max-per-set 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.baselines import (
    ControlCondition, IKCondition, make_condition,
)
from src.evaluation.metrics import TrialMetrics, append_jsonl
from src.evaluation.trial_runner import run_planning_trial
from src.scenarios.mb_loader import load_mb_problems
from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.planner.collision import _quat_to_rot


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

# HJCD-IK variants: family-aware classify + PathDS + no BiRRT
CONDITION_HJCD = make_condition(
    IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL
)

# Baseline: nearest single pre-computed IK goal + DiffIK, no PathDS, no BiRRT
CONDITION_DIFFIK = make_condition(
    IKCondition.VANILLA_DS, ControlCondition.VANILLA_DS_DIFFIK_MODULATION
)

# Common trial kwargs: no Morse escape (isolates IK quality), BiRRT enabled for
# path planning (bookshelf/cage problems require it — not testing path planning here).
_TRIAL_KWARGS = dict(
    morse_override={"enabled": False},
    supervisor_override={"enabled": False},
)

# Adaptive variant: upgrade from nsol=1 to nsol=8 when interp clearance is tight
_ADAPTIVE_CLEARANCE_THRESHOLD = 0.05   # metres

# Batch size for all HJCD-IK calls
_HJCD_BATCH = 1000

# Available variants
ALL_VARIANTS = [
    "baseline_diffik_ds",
    "hjcd_ik_pathds_nsol1",
    "hjcd_ik_pathds_nsol2",
    "hjcd_ik_pathds_nsol4",
    "hjcd_ik_pathds_nsol8",
    "adaptive_hjcd_ik_pathds",
]

COLORS = {
    "baseline_diffik_ds":      "#D62728",
    "hjcd_ik_pathds_nsol1":   "#AEC7E8",
    "hjcd_ik_pathds_nsol2":   "#6BAED6",
    "hjcd_ik_pathds_nsol4":   "#2171B5",
    "hjcd_ik_pathds_nsol8":   "#08306B",
    "adaptive_hjcd_ik_pathds": "#2CA02C",
}


# ---------------------------------------------------------------------------
# Per-problem result record
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    set_name:        str
    prob_idx:        int
    variant:         str
    seed:            int
    success:         bool          # goal reached AND no execution collision
    goal_err:        Optional[float]
    min_clearance:   Optional[float]   # execution min clearance (sphere model)
    planned_min_clearance: Optional[float]  # planned-path min clearance (diagnostic)
    collision_count: int           # steps with execution clearance < 0 (sphere model)
    wall_time_s:     float
    setup_ms:        float         # IK solve time (0 for baseline)
    n_ik_goals:      int           # solutions handed to the planner
    n_families:      int           # distinct families (0 for baseline)
    upgraded_nsol:   bool          # True if adaptive variant upgraded to nsol=8
    error:           Optional[str]


# ---------------------------------------------------------------------------
# MB target frame correction
# ---------------------------------------------------------------------------

# HJCD-IK targets panda_grasptarget (0.105 m past panda_hand along hand z-axis).
# MotionBenchmaker goal_pose is in panda_hand frame.  To make panda_hand land at
# the MB goal, we must shift the target by +0.105 m along the EE z-axis (from
# the target orientation) before passing it to solve_batch.
_GRASPTARGET_OFFSET_M = 0.105


def _mb_target_to_hjcd(target_pose: dict) -> dict:
    """
    Convert an MB goal_pose (panda_hand frame) to the HJCD-IK target pose
    (panda_grasptarget frame).

    Shifts position by +0.105 m along the EE z-axis derived from the target
    orientation so that panda_hand lands exactly at the MB goal position.
    """
    pos  = np.array(target_pose["position"], dtype=float)
    quat = np.array(target_pose["quaternion_wxyz"], dtype=float)
    R    = _quat_to_rot(*quat)
    ee_z = R[:, 2]   # EE z-axis in world frame
    corrected_pos = pos + ee_z * _GRASPTARGET_OFFSET_M
    return {
        "position":        corrected_pos.tolist(),
        "quaternion_wxyz": quat.tolist(),
    }


# ---------------------------------------------------------------------------
# IK solve helpers
# ---------------------------------------------------------------------------

def _solve_hjcd(
    target_pose: dict,
    raw_obs: dict,
    nsol: int,
) -> Tuple[List[np.ndarray], float]:
    """
    Call solve_batch for a single target with the given nsol.

    Returns (solutions, elapsed_ms).
    Raises RuntimeError if HJCD-IK returns 0 solutions.
    """
    from src.solver.ik.hjcd_wrapper import solve_batch
    # MB goals are in panda_hand frame; HJCD-IK targets panda_grasptarget.
    hjcd_target = _mb_target_to_hjcd(target_pose)
    t0 = time.perf_counter()
    result = solve_batch(
        hjcd_target,
        env_config={"obstacles": raw_obs},
        batch_size=_HJCD_BATCH,
        num_solutions=nsol,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result.solutions, elapsed_ms


def _solve_adaptive(
    target_pose: dict,
    raw_obs: dict,
    q_start: np.ndarray,
    obs_list,
) -> Tuple[List[np.ndarray], float, bool]:
    """
    Adaptive nsol: try nsol=1 first, upgrade to nsol=8 if clearance is tight.

    Returns (solutions, elapsed_ms, upgraded).
    """
    from src.solver.ik.hjcd_wrapper import solve_batch
    from src.solver.ik.goal_selection import classify_ik_goals

    # Apply panda_hand → panda_grasptarget frame correction
    hjcd_target = _mb_target_to_hjcd(target_pose)

    t0 = time.perf_counter()

    result1 = solve_batch(
        hjcd_target,
        env_config={"obstacles": raw_obs},
        batch_size=_HJCD_BATCH,
        num_solutions=1,
    )
    solutions = result1.solutions
    upgraded = False

    if solutions and obs_list:
        infos = classify_ik_goals(q_start, solutions, obs_list)
        if infos:
            min_cl = min(g.interp_min_clearance for g in infos)
            if min_cl < _ADAPTIVE_CLEARANCE_THRESHOLD:
                result8 = solve_batch(
                    hjcd_target,
                    env_config={"obstacles": raw_obs},
                    batch_size=_HJCD_BATCH,
                    num_solutions=8,
                )
                solutions = result8.solutions
                upgraded = True

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return solutions, elapsed_ms, upgraded


# ---------------------------------------------------------------------------
# Per-problem trial runner
# ---------------------------------------------------------------------------

def _run_problem(
    set_name: str,
    prob_idx: int,
    spec: ScenarioSpec,
    raw_obs: dict,
    variant: str,
    seed: int,
) -> ProblemResult:
    """Run one (problem, variant, seed) trial and return a ProblemResult."""
    wall_t0 = time.perf_counter()
    setup_ms = 0.0
    n_ik_goals = 0
    n_families = 0
    upgraded = False
    error: Optional[str] = None
    trial: Optional[TrialMetrics] = None

    try:
        if variant == "baseline_diffik_ds":
            # Use pre-computed goal_ik from the JSON; DiffIK picks nearest.
            trial_spec = spec
            condition = CONDITION_DIFFIK
            n_ik_goals = len(spec.ik_goals)

        else:
            # Determine nsol
            if variant == "hjcd_ik_pathds_nsol1":
                nsol = 1
            elif variant == "hjcd_ik_pathds_nsol2":
                nsol = 2
            elif variant == "hjcd_ik_pathds_nsol4":
                nsol = 4
            elif variant == "hjcd_ik_pathds_nsol8":
                nsol = 8
            elif variant == "adaptive_hjcd_ik_pathds":
                nsol = None  # handled below
            else:
                raise ValueError(f"Unknown variant: {variant!r}")

            if nsol is not None:
                solutions, setup_ms = _solve_hjcd(spec.target_pose, raw_obs, nsol)
            else:
                obs_list = spec.collision_obstacles()
                solutions, setup_ms, upgraded = _solve_adaptive(
                    spec.target_pose, raw_obs, spec.q_start, obs_list
                )

            if not solutions:
                return ProblemResult(
                    set_name=set_name, prob_idx=prob_idx,
                    variant=variant, seed=seed,
                    success=False, goal_err=None, min_clearance=None,
                    wall_time_s=time.perf_counter() - wall_t0,
                    setup_ms=setup_ms, n_ik_goals=0, n_families=0,
                    upgraded_nsol=upgraded,
                    error="HJCD-IK returned 0 solutions",
                )

            n_ik_goals = len(solutions)
            trial_spec = dataclasses.replace(spec, ik_goals=solutions)
            condition = CONDITION_HJCD

        trial = run_planning_trial(
            trial_spec,
            condition=condition,
            seed=seed,
            trial_id=0,
            **_TRIAL_KWARGS,
        )

        # Extract metrics
        goal_err = (
            trial.execution.final_goal_err
            if trial.execution and trial.execution.final_goal_err is not None
            else None
        )

        # Execution clearance from the Python sphere model (BarrierMetrics).
        # This reflects actual robot motion — not the planned-path straight line.
        exec_min_cl: Optional[float] = None
        collision_count: int = 0
        if trial.barrier:
            bcl = trial.barrier.min_clearance
            if bcl < float("inf"):
                exec_min_cl = float(bcl)
            collision_count = int(trial.barrier.collision_count)

        # Planned-path clearance (straight-line interpolation; diagnostic only).
        planned_min_cl: Optional[float] = None
        if trial.plan and hasattr(trial.plan, "planned_min_clearance"):
            pcl = trial.plan.planned_min_clearance
            if pcl < float("inf"):
                planned_min_cl = float(pcl)

        # Success requires goal reached AND no execution collision (sphere model).
        # The sphere model is conservative, so collision_count > 0 may occasionally
        # be a false positive for very thin obstacles; it is still the best proxy
        # available without a full MuJoCo contact log per step.
        reached_goal = bool(trial.execution and trial.execution.terminal_success)
        success = reached_goal and (collision_count == 0)

        if trial.ik and trial.ik.n_families_available:
            n_families = trial.ik.n_families_available

        error = trial.error

    except Exception as exc:
        import traceback
        error = traceback.format_exc()
        success = False
        goal_err = None
        exec_min_cl = None
        planned_min_cl = None
        collision_count = 0

    wall_time = time.perf_counter() - wall_t0
    return ProblemResult(
        set_name=set_name,
        prob_idx=prob_idx,
        variant=variant,
        seed=seed,
        success=success,
        goal_err=goal_err,
        min_clearance=exec_min_cl,
        planned_min_clearance=planned_min_cl,
        collision_count=collision_count,
        wall_time_s=wall_time,
        setup_ms=setup_ms,
        n_ik_goals=n_ik_goals,
        n_families=n_families,
        upgraded_nsol=upgraded,
        error=error,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggStats:
    n_trials:            int
    n_errors:            int
    success_rate:        float
    collision_rate:      float           # fraction with execution collision_count > 0
    mean_goal_err:       Optional[float]
    mean_exec_min_cl:    Optional[float] # mean execution min clearance
    mean_wall_s:         float
    mean_setup_ms:       float
    mean_n_goals:        float
    mean_n_families:     float
    upgrade_rate:        float   # fraction that upgraded (adaptive only)


def _aggregate(results: List[ProblemResult]) -> AggStats:
    if not results:
        return AggStats(0, 0, 0.0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0)

    n_err  = sum(1 for r in results if r.error)
    ok     = [r for r in results if not r.error]
    if not ok:
        return AggStats(len(results), n_err, 0.0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0)

    succ        = [r for r in ok if r.success]
    collisions  = [r for r in ok if r.collision_count > 0]
    errs        = [r.goal_err for r in ok if r.goal_err is not None and np.isfinite(r.goal_err)]
    exec_cls    = [r.min_clearance for r in ok
                   if r.min_clearance is not None and np.isfinite(r.min_clearance)]
    walls       = [r.wall_time_s for r in ok]
    setups      = [r.setup_ms for r in ok]
    goals       = [r.n_ik_goals for r in ok]
    fams        = [r.n_families for r in ok]
    upg         = [r for r in ok if r.upgraded_nsol]

    return AggStats(
        n_trials=len(results),
        n_errors=n_err,
        success_rate=len(succ) / len(ok),
        collision_rate=len(collisions) / len(ok),
        mean_goal_err=float(np.mean(errs)) if errs else None,
        mean_exec_min_cl=float(np.mean(exec_cls)) if exec_cls else None,
        mean_wall_s=float(np.mean(walls)),
        mean_setup_ms=float(np.mean(setups)),
        mean_n_goals=float(np.mean(goals)),
        mean_n_families=float(np.mean(fams)),
        upgrade_rate=len(upg) / len(ok),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".3f", na="N/A"):
    if val is None:
        return na
    return format(val, fmt)


def _print_table(
    variant_names: List[str],
    all_stats: Dict[str, AggStats],
) -> None:
    cols = [
        ("variant",          "Variant",          "24s"),
        ("n_trials",         "N",                "5d"),
        ("n_errors",         "Err",              "4d"),
        ("success_rate",     "Succ%",            "7.1%"),
        ("collision_rate",   "Coll%",            "7.1%"),
        ("mean_goal_err",    "GoalErr",          "8s"),
        ("mean_exec_min_cl", "ExecCl",           "8s"),
        ("mean_wall_s",      "Wall(s)",          "8.2f"),
        ("mean_setup_ms",    "Setup(ms)",        "10.1f"),
        ("mean_n_goals",     "N_IK",             "6.1f"),
        ("mean_n_fam",       "N_Fam",            "6.1f"),
        ("upgrade_rate",     "Upg%",             "6.1%"),
    ]

    header = "  ".join(f"{name:{fmt.rstrip('dfs%')}s}" for _, name, fmt in cols)
    print(f"\n{header}")
    print("-" * len(header))

    for vname in variant_names:
        s = all_stats.get(vname)
        if s is None:
            continue
        row_vals = {
            "variant":          vname,
            "n_trials":         s.n_trials,
            "n_errors":         s.n_errors,
            "success_rate":     s.success_rate,
            "collision_rate":   s.collision_rate,
            "mean_goal_err":    _fmt(s.mean_goal_err),
            "mean_exec_min_cl": _fmt(s.mean_exec_min_cl),
            "mean_wall_s":      s.mean_wall_s,
            "mean_setup_ms":    s.mean_setup_ms,
            "mean_n_goals":     s.mean_n_goals,
            "mean_n_fam":       s.mean_n_families,
            "upgrade_rate":     s.upgrade_rate,
        }
        parts = []
        for key, _, fmt in cols:
            val = row_vals[key]
            if isinstance(val, str) and fmt.endswith("s"):
                parts.append(f"{val:{fmt}}")
            elif fmt.endswith("%"):
                parts.append(f"{val:{fmt}}")
            elif fmt.endswith("d"):
                parts.append(f"{val:{fmt}}")
            elif fmt.endswith("f"):
                parts.append(f"{val:{fmt}}")
            else:
                parts.append(str(val))
        print("  ".join(parts))


def _print_by_set(
    variant_names: List[str],
    results_by_set: Dict[str, Dict[str, List[ProblemResult]]],
) -> None:
    """Print a success-rate sub-table per problem set."""
    print("\nPer problem-set success rates:")
    sets = sorted(results_by_set)
    header = f"  {'Set':<28}" + "".join(f"  {v[:12]:>12}" for v in variant_names)
    print(header)
    print("  " + "-" * (26 + 14 * len(variant_names)))
    for sname in sets:
        row = f"  {sname:<28}"
        for v in variant_names:
            res = results_by_set[sname].get(v, [])
            ok = [r for r in res if not r.error]
            if not ok:
                row += f"  {'N/A':>12}"
            else:
                sr = sum(1 for r in ok if r.success) / len(ok)
                row += f"  {sr:>11.1%}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MotionBenchmaker nsol ablation: HJCD-IK + PathDS, no BiRRT")
    parser.add_argument(
        "--problem-sets", nargs="*", default=None,
        help="Problem set names (default: all 8)")
    parser.add_argument(
        "--max-per-set", type=int, default=20,
        help="Problems to sample per set (default: 20)")
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Random seeds per problem (default: 3)")
    parser.add_argument(
        "--loader-seed", type=int, default=0,
        help="Seed for subsampling problems (default: 0)")
    parser.add_argument(
        "--variants", nargs="+", default=None,
        choices=ALL_VARIANTS,
        help=f"Variants to run (default: all). Choices: {ALL_VARIANTS}")
    parser.add_argument(
        "--output-dir", default="outputs/eval/mb_nsol_ablation",
        help="Output directory for JSONL results")
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress per-trial output")
    args = parser.parse_args(argv)

    variants = args.variants or ALL_VARIANTS
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load problems
    # ------------------------------------------------------------------
    print("\nLoading MotionBenchmaker problems...")
    problems = load_mb_problems(
        problem_sets=args.problem_sets,
        max_per_set=args.max_per_set,
        seed=args.loader_seed,
    )
    print(f"  Loaded {len(problems)} problems "
          f"(max {args.max_per_set}/set, seed={args.loader_seed})")
    by_set: Dict[str, int] = {}
    for sn, _, _, _ in problems:
        by_set[sn] = by_set.get(sn, 0) + 1
    for sn, cnt in sorted(by_set.items()):
        print(f"    {sn}: {cnt} problems")

    # ------------------------------------------------------------------
    # Run trials
    # ------------------------------------------------------------------
    total_trials = len(problems) * args.n_seeds * len(variants)
    print(f"\nRunning {total_trials} trials "
          f"({len(problems)} problems × {args.n_seeds} seeds × {len(variants)} variants)")
    print(f"Output: {out_dir}\n")

    jsonl_path = out_dir / "per_trial_results.jsonl"
    all_results: Dict[str, List[ProblemResult]] = {v: [] for v in variants}
    # Also track by set for sub-table
    results_by_set: Dict[str, Dict[str, List[ProblemResult]]] = {}

    done = 0
    for set_name, prob_idx, spec, raw_obs in problems:
        for seed in range(args.n_seeds):
            for variant in variants:
                t0 = time.perf_counter()
                r = _run_problem(set_name, prob_idx, spec, raw_obs, variant, seed)
                elapsed = time.perf_counter() - t0
                done += 1

                all_results[variant].append(r)
                results_by_set.setdefault(set_name, {}).setdefault(
                    variant, []
                ).append(r)

                # Write to JSONL
                try:
                    with open(jsonl_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
                except Exception:
                    pass

                if not args.quiet:
                    status = "OK " if not r.error else "ERR"
                    succ   = "succ" if r.success else "fail"
                    print(
                        f"  [{done:5d}/{total_trials}] {set_name}[{prob_idx:3d}] "
                        f"seed={seed} {variant:<30s} "
                        f"{status} {succ}  "
                        f"wall={elapsed:.1f}s  setup={r.setup_ms:.0f}ms"
                    )

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    agg: Dict[str, AggStats] = {v: _aggregate(all_results[v]) for v in variants}

    print(f"\n{'='*80}")
    print(f"  MotionBenchmaker nsol ablation — aggregate results")
    print(f"  problems={len(problems)}  seeds/problem={args.n_seeds}")
    print(f"{'='*80}")
    _print_table(variants, agg)
    _print_by_set(variants, results_by_set)

    # ------------------------------------------------------------------
    # Save aggregate JSON
    # ------------------------------------------------------------------
    agg_path = out_dir / "aggregate_stats.json"
    agg_out = {}
    for v, s in agg.items():
        agg_out[v] = dataclasses.asdict(s)
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(agg_out, fh, indent=2)
    print(f"\nAggregate stats saved to {agg_path}")

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  Recommendation")
    print(f"{'='*80}")

    nsol_variants = [v for v in variants if v.startswith("hjcd_ik_pathds_nsol")]
    if nsol_variants:
        best_v = max(nsol_variants, key=lambda v: agg[v].success_rate)
        worst_v = min(nsol_variants, key=lambda v: agg[v].success_rate)
        best_s = agg[best_v]
        worst_s = agg[worst_v]
        delta = best_s.success_rate - worst_s.success_rate
        cost  = best_s.mean_setup_ms - worst_s.mean_setup_ms

        print(f"  Best nsol variant:  {best_v}  "
              f"success={best_s.success_rate:.1%}  "
              f"setup={best_s.mean_setup_ms:.0f}ms")
        print(f"  Worst nsol variant: {worst_v}  "
              f"success={worst_s.success_rate:.1%}  "
              f"setup={worst_s.mean_setup_ms:.0f}ms")
        print(f"  Δsuccess_rate={delta:+.1%}  Δsetup_ms={cost:+.0f}ms")

    if "adaptive_hjcd_ik_pathds" in variants and "adaptive_hjcd_ik_pathds" in agg:
        a = agg["adaptive_hjcd_ik_pathds"]
        print(f"\n  Adaptive variant: success={a.success_rate:.1%}  "
              f"setup={a.mean_setup_ms:.0f}ms  "
              f"upgrade_rate={a.upgrade_rate:.1%}")

    if "baseline_diffik_ds" in variants and "baseline_diffik_ds" in agg:
        b = agg["baseline_diffik_ds"]
        print(f"\n  Baseline (DiffIK): success={b.success_rate:.1%}  "
              f"setup={b.mean_setup_ms:.0f}ms")

    print(f"\n  Full results: {jsonl_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
