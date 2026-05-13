"""
CBF margin ablation benchmark for MotionBenchmaker bookshelf tasks.

Tests whether reducing CBF safety margins allows the controller to converge
when goals lie inside the CBF activation buffer (clearance < d_safe + d_buffer).
Focused on bookshelf_small_panda problems 63 and 84.

Variants
--------
default_cbf         d_safe=0.03, d_buffer=0.05 (baseline)
small_buffer        d_safe=0.03, d_buffer=0.02 (tighter activation window)
small_safe          d_safe=0.015, d_buffer=0.03 (halved d_safe)
tiny_safe           d_safe=0.01, d_buffer=0.02 (minimum viable margins)
goal_tapered_cbf    d_safe=0.03, d_buffer=0.05 + GoalAwareCBFConfig (taper margins near goal)
goal_slack_cbf      d_safe=0.03, d_buffer=0.05 + GoalAwareCBFConfig (reduce slack weight near goal only)

Usage::

    conda run -n ds-iks python -m benchmarks.eval_mb_cbf_goal_conflict
    conda run -n ds-iks python -m benchmarks.eval_mb_cbf_goal_conflict \\
        --set bookshelf_small_panda --problems 63,84 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.baselines import ControlCondition, IKCondition, make_condition
from src.evaluation.metrics import TrialMetrics
from src.evaluation.trial_runner import run_planning_trial
from src.scenarios.mb_loader import load_mb_problems
from src.solver.controller.cbf_filter import CBFConfig, GoalAwareCBFConfig
from src.solver.planner.collision import _quat_to_rot

_GRASPTARGET_OFFSET_M = 0.105
_HJCD_BATCH = 1000
_HJCD_NSOL = 4

CONDITION = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)

_TRIAL_KWARGS = dict(
    morse_override={"enabled": False},
    supervisor_override={"enabled": False},
)

ALL_VARIANTS = [
    "default_cbf",
    "small_buffer",
    "small_safe",
    "tiny_safe",
    "goal_tapered_cbf",
    "goal_slack_cbf",
]


def _make_cbf_config(variant: str) -> CBFConfig:
    """Return the CBFConfig for a given variant name."""
    base = dict(enabled=True, alpha=8.0)

    if variant == "default_cbf":
        return CBFConfig(**base, d_safe=0.03, d_buffer=0.05)

    elif variant == "small_buffer":
        return CBFConfig(**base, d_safe=0.03, d_buffer=0.02)

    elif variant == "small_safe":
        return CBFConfig(**base, d_safe=0.015, d_buffer=0.03)

    elif variant == "tiny_safe":
        return CBFConfig(**base, d_safe=0.01, d_buffer=0.02)

    elif variant == "goal_tapered_cbf":
        ga = GoalAwareCBFConfig(
            enabled=True,
            goal_radius_start=0.25,
            goal_radius_end=0.05,
            min_d_safe_scale=0.5,
            min_d_buffer_scale=0.3,
            goal_clearance_min=0.02,
            w_slack_goal=1e4,  # same as default qp_weight_slack — only margins taper
        )
        return CBFConfig(**base, d_safe=0.03, d_buffer=0.05, goal_aware=ga)

    elif variant == "goal_slack_cbf":
        ga = GoalAwareCBFConfig(
            enabled=True,
            goal_radius_start=0.25,
            goal_radius_end=0.05,
            min_d_safe_scale=1.0,    # margins unchanged
            min_d_buffer_scale=1.0,  # margins unchanged
            goal_clearance_min=0.02,
            w_slack_goal=100.0,      # reduced slack weight near goal
        )
        return CBFConfig(**base, d_safe=0.03, d_buffer=0.05, goal_aware=ga)

    else:
        raise ValueError(f"Unknown variant: {variant!r}")


def _mb_target_to_hjcd(target_pose: dict) -> dict:
    pos  = np.array(target_pose["position"], dtype=float)
    quat = np.array(target_pose["quaternion_wxyz"], dtype=float)
    R    = _quat_to_rot(*quat)
    return {
        "position": (pos + R[:, 2] * _GRASPTARGET_OFFSET_M).tolist(),
        "quaternion_wxyz": quat.tolist(),
    }


@dataclass
class TrialResult:
    set_name:    str
    prob_idx:    int
    variant:     str
    seed:        int
    success:     bool
    goal_err:    Optional[float]
    min_clearance:   Optional[float]
    final_clearance: Optional[float]
    collision_count: int
    cbf_active_fraction_final_20pct: Optional[float]
    mean_cbf_correction_norm:        Optional[float]
    mean_cbf_angle_deg:              Optional[float]
    failure_reason:  str
    wall_time_s: float
    error:       Optional[str]


def _run_one(
    set_name: str,
    prob_idx: int,
    spec,
    raw_obs: dict,
    variant: str,
    seed: int,
) -> TrialResult:
    wall_t0 = time.perf_counter()
    error: Optional[str] = None
    trial: Optional[TrialMetrics] = None
    cbf_cfg = _make_cbf_config(variant)

    try:
        from src.solver.ik.hjcd_wrapper import solve_batch

        hjcd_target = _mb_target_to_hjcd(spec.target_pose)
        result = solve_batch(hjcd_target, env_config={"obstacles": raw_obs},
                             batch_size=_HJCD_BATCH, num_solutions=_HJCD_NSOL)
        if not result.solutions:
            return TrialResult(
                set_name=set_name, prob_idx=prob_idx, variant=variant, seed=seed,
                success=False, goal_err=None, min_clearance=None, final_clearance=None,
                collision_count=0, cbf_active_fraction_final_20pct=None,
                mean_cbf_correction_norm=None, mean_cbf_angle_deg=None,
                failure_reason="no_ik_solutions",
                wall_time_s=time.perf_counter() - wall_t0, error="HJCD-IK: 0 solutions",
            )

        trial_spec = dataclasses.replace(spec, ik_goals=result.solutions)
        trial = run_planning_trial(
            trial_spec, condition=CONDITION, seed=seed, trial_id=0,
            cbf_override=cbf_cfg,
            **_TRIAL_KWARGS,
        )

        # Extract metrics
        reached = bool(trial.execution and trial.execution.terminal_success)
        goal_err = (trial.execution.final_goal_err
                    if trial.execution and trial.execution.final_goal_err is not None
                    else None)

        min_cl: Optional[float] = None
        collision_count = 0
        if trial.barrier:
            if trial.barrier.min_clearance < float("inf"):
                min_cl = float(trial.barrier.min_clearance)
            collision_count = int(trial.barrier.collision_count)

        success = reached and (collision_count == 0)

        # CBFGoalConflictDiagnostics
        final_cl: Optional[float] = None
        cbf_frac20: Optional[float] = None
        cbf_corr: Optional[float] = None
        cbf_angle: Optional[float] = None
        fail_reason = ""
        if trial.cbf_goal_conflict is not None:
            d = trial.cbf_goal_conflict
            if d.final_clearance < float("inf"):
                final_cl = float(d.final_clearance)
            cbf_frac20 = d.cbf_active_fraction_final_20pct
            cbf_corr   = d.mean_cbf_correction_norm
            cbf_angle  = d.mean_cbf_angle_deg
            fail_reason = d.failure_reason

        error = trial.error

    except Exception:
        import traceback
        error = traceback.format_exc()
        success = False
        goal_err = None
        min_cl = None
        final_cl = None
        collision_count = 0
        cbf_frac20 = None
        cbf_corr = None
        cbf_angle = None
        fail_reason = "exception"

    return TrialResult(
        set_name=set_name, prob_idx=prob_idx, variant=variant, seed=seed,
        success=success, goal_err=goal_err, min_clearance=min_cl,
        final_clearance=final_cl, collision_count=collision_count,
        cbf_active_fraction_final_20pct=cbf_frac20,
        mean_cbf_correction_norm=cbf_corr,
        mean_cbf_angle_deg=cbf_angle,
        failure_reason=fail_reason,
        wall_time_s=time.perf_counter() - wall_t0,
        error=error,
    )


def _fmt(val, fmt=".3f"):
    if val is None:
        return "N/A"
    return format(val, fmt)


def _print_table(variant_names: List[str], results_by_variant: Dict[str, List[TrialResult]]) -> None:
    cols = [
        ("variant",       "Variant",      "22s"),
        ("success_rate",  "Succ%",        "7.1%"),
        ("coll_rate",     "Coll%",        "7.1%"),
        ("goal_err",      "GoalErr",      "9s"),
        ("min_cl",        "MinCl",        "8s"),
        ("final_cl",      "FinalCl",      "9s"),
        ("cbf_frac20",    "CBFFrac20",    "10s"),
        ("cbf_corr",      "CBFCorr",      "8s"),
        ("cbf_angle",     "CBFAngle",     "9s"),
        ("fail_reason",   "FailReason",   "18s"),
    ]

    def _header_width(fmt: str) -> int:
        return int("".join(c for c in fmt if c.isdigit() or c == ".").split(".")[0] or "8")
    header = "  ".join(f"{name:<{_header_width(fmt)}}" for _, name, fmt in cols)
    print(f"\n{header}")
    print("-" * len(header))

    for vname in variant_names:
        rows = results_by_variant.get(vname, [])
        ok = [r for r in rows if not r.error]
        if not ok:
            print(f"  {vname:<22}  (no valid results)")
            continue
        succ = [r for r in ok if r.success]
        colls = [r for r in ok if r.collision_count > 0]
        errs = [r.goal_err for r in ok if r.goal_err is not None]
        min_cls = [r.min_clearance for r in ok if r.min_clearance is not None]
        final_cls = [r.final_clearance for r in ok if r.final_clearance is not None]
        frac20s = [r.cbf_active_fraction_final_20pct for r in ok if r.cbf_active_fraction_final_20pct is not None]
        corrs = [r.mean_cbf_correction_norm for r in ok if r.mean_cbf_correction_norm is not None]
        angles = [r.mean_cbf_angle_deg for r in ok if r.mean_cbf_angle_deg is not None]
        reasons = [r.failure_reason for r in ok if r.failure_reason]
        top_reason = max(set(reasons), key=reasons.count) if reasons else ""

        vals = {
            "variant":      vname,
            "success_rate": len(succ) / len(ok),
            "coll_rate":    len(colls) / len(ok),
            "goal_err":     _fmt(np.mean(errs) if errs else None),
            "min_cl":       _fmt(np.mean(min_cls) if min_cls else None),
            "final_cl":     _fmt(np.mean(final_cls) if final_cls else None),
            "cbf_frac20":   _fmt(np.mean(frac20s) if frac20s else None),
            "cbf_corr":     _fmt(np.mean(corrs) if corrs else None),
            "cbf_angle":    _fmt(np.mean(angles) if angles else None),
            "fail_reason":  top_reason,
        }

        parts = []
        for key, _, fmt in cols:
            val = vals[key]
            if isinstance(val, str) and fmt.endswith("s"):
                parts.append(f"{val:{fmt}}")
            elif fmt.endswith("%"):
                parts.append(f"{val:{fmt}}")
            else:
                parts.append(str(val))
        print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="CBF margin ablation benchmark")
    parser.add_argument("--set", default="bookshelf_small_panda")
    parser.add_argument("--problems", default="63,84",
                        help="Comma-separated problem indices")
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma-separated seeds")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    parser.add_argument("--max-per-set", type=int, default=100)
    args = parser.parse_args()

    problem_indices = [int(x) for x in args.problems.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = args.variants

    print(f"Loading {args.set} (max {args.max_per_set} per set)...")
    all_problems = load_mb_problems(
        problem_sets=[args.set], max_per_set=args.max_per_set, seed=0
    )

    # Filter to requested problem indices
    problems = [(sn, pi, sp, ro) for sn, pi, sp, ro in all_problems
                if pi in problem_indices]
    if not problems:
        print(f"ERROR: no problems found for indices {problem_indices} in {args.set}")
        return

    print(f"Running {len(variants)} variants x {len(problems)} problems x {len(seeds)} seeds "
          f"= {len(variants)*len(problems)*len(seeds)} trials")

    results_by_variant: Dict[str, List[TrialResult]] = {v: [] for v in variants}
    total = len(variants) * len(problems) * len(seeds)
    done = 0

    for variant in variants:
        for set_name, prob_idx, spec, raw_obs in problems:
            for seed in seeds:
                done += 1
                print(f"  [{done}/{total}] {variant} | prob={prob_idx} seed={seed}", end=" ", flush=True)
                r = _run_one(set_name, prob_idx, spec, raw_obs, variant, seed)
                results_by_variant[variant].append(r)
                status = "OK" if r.success else f"FAIL ({r.failure_reason or r.error or 'failed'})"
                print(status)

    _print_table(variants, results_by_variant)


if __name__ == "__main__":
    main()
