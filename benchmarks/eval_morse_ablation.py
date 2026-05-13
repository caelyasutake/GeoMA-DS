"""
Morse Escape Planner Ablation Benchmark.

Compares conditions over shared seeds:
  baseline    — Morse disabled (existing BACKTRACK/ESCAPE_CLEARANCE only)
  +morse      — Morse enabled, negative curvature off
  +morse+nc   — Morse enabled, negative curvature on (Lanczos)
  diagnostic  — Morse forced at step 100 (verifies integration path without trap requirement)

Key questions:
  - Does Morse improve trap recovery rate vs. baseline?
  - Does negative curvature improve escape quality vs. Morse alone?
  - Does Morse change overall success rate or final goal error?
  - Does Morse reduce the number of BiRRT replans?
  - (diagnostic) Does the prefix queue execute correctly and are metrics populated?

Usage::

    conda run -n ds-iks python -m benchmarks.eval_morse_ablation
    conda run -n ds-iks python -m benchmarks.eval_morse_ablation \\
        --scenario frontal_i_barrier_lr_hard --n-trials 20
    conda run -n ds-iks python -m benchmarks.eval_morse_ablation \\
        --scenarios frontal_i_barrier_lr_hard u_shape narrow_passage \\
        --n-trials 30 --output-dir outputs/eval/morse_ablation
    conda run -n ds-iks python -m benchmarks.eval_morse_ablation \\
        --variants baseline +morse diagnostic --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.baselines import IKCondition, ControlCondition, make_condition
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import TrialMetrics, load_jsonl
from src.scenarios.scenario_builders import SCENARIO_REGISTRY


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

CONDITION = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)

_VARIANTS: Dict[str, dict] = {
    "baseline": {
        "enabled": False,
    },
    "+morse": {
        "enabled": True,
    },
    "+morse+nc": {
        "enabled": True,
        "negative_curvature": {"enabled": True},
    },
    # Forces Morse to activate at step 100 regardless of trap detection.
    # Use to verify the integration path works even on easy scenarios.
    "diagnostic": {
        "enabled": True,
        "force_at_step": 100,
    },
    # FastMorseEscapeController + MorseEscapeSupervisor active by default.
    "fast_morse": {
        "enabled": True,
    },
}

COLORS = {
    "baseline":   "#6B6B6B",
    "+morse":     "#2176AE",
    "+morse+nc":  "#E07B39",
    "diagnostic": "#7B2D8B",
    "fast_morse": "#2CA02C",
}

LABELS = {
    "baseline":   "Baseline (no Morse)",
    "+morse":     "+Morse (full rollout)",
    "+morse+nc":  "+Morse + neg. curvature",
    "diagnostic": "Diagnostic (forced step 100)",
    "fast_morse": "Fast Morse (reactive)",
}

# Scenarios ordered from hardest to easiest for activation probability.
# frontal_i_barrier_lr_medium is kept for regression (should not regress success).
DEFAULT_SCENARIOS = [
    "frontal_i_barrier_lr_hard",
    "u_shape",
    "narrow_passage",
    "frontal_i_barrier_lr_medium",
]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _run_variant(
    spec,
    variant_name: str,
    morse_kwargs: dict,
    n_trials: int,
    seed_offset: int,
    out_dir: Path,
) -> List[TrialMetrics]:
    sub_dir = out_dir / variant_name.lstrip("+").replace("+", "_")
    sub_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(output_dir=sub_dir, verbose=True)
    runner.run_spec(
        spec, [CONDITION], n_trials=n_trials, seed_offset=seed_offset,
        trial_kwargs={"morse_override": morse_kwargs},
    )
    return runner.load_results()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _ok(trials: List[TrialMetrics]) -> List[TrialMetrics]:
    return [t for t in trials if t.error is None]


@dataclass
class VariantStats:
    n: int
    success_rate: float
    mean_goal_err: Optional[float]
    mean_wall_time: Optional[float]
    stall_rate: float
    trap_rate: float
    morse_activated_rate: float
    escape_succeeded_rate: float
    prefix_selected_rate: float
    fallback_rate: float
    mean_clearance_gain: Optional[float]
    mean_replans: Optional[float]
    mean_planning_time_s: Optional[float]
    # Fast Morse timing
    fast_morse_activation_rate: float = 0.0
    mean_fast_morse_compute_ms: Optional[float] = None
    p95_fast_morse_compute_ms: Optional[float] = None
    max_fast_morse_compute_ms: Optional[float] = None


def _compute_stats(trials: List[TrialMetrics]) -> VariantStats:
    ok = _ok(trials)
    if not ok:
        return VariantStats(
            0, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None,
        )

    succ = [t for t in ok if t.execution and t.execution.terminal_success]
    errs = [t.execution.final_goal_err for t in ok
            if t.execution and t.execution.final_goal_err is not None
            and np.isfinite(t.execution.final_goal_err)]
    wall = [t.wall_time_s for t in ok if t.wall_time_s is not None]

    stall = [t for t in ok if t.execution and t.execution.stall_step is not None]
    trap  = [t for t in ok if t.escape and t.escape.trap_detected]

    morse_act = [t for t in ok if t.escape and t.escape.morse
                 and t.escape.morse.activated]
    morse_esc = [t for t in ok if t.escape and t.escape.morse
                 and t.escape.morse.escape_succeeded]
    prefix_sel = [t for t in ok if t.escape and t.escape.morse
                  and t.escape.morse.execute_mode == "prefix"]
    fallback   = [t for t in ok if t.escape and t.escape.morse
                  and t.escape.morse.fallback_used is not None]

    cl_gain = [t.escape.clearance_gain_during_escape for t in ok
               if t.escape and t.escape.clearance_gain_during_escape != 0.0]

    replans = [t.execution.n_replans for t in ok
               if t.execution and hasattr(t.execution, "n_replans")
               and t.execution.n_replans is not None]
    plan_t  = [t.escape.morse.planning_time_s for t in ok
               if t.escape and t.escape.morse
               and t.escape.morse.planning_time_s is not None]

    fast_act = [t for t in ok if t.escape and t.escape.fast_morse
                and t.escape.fast_morse.activated]
    fm_mean_ms = [t.escape.fast_morse.mean_compute_time_ms for t in ok
                  if t.escape and t.escape.fast_morse
                  and t.escape.fast_morse.mean_compute_time_ms is not None]
    fm_max_ms  = [t.escape.fast_morse.max_compute_time_ms for t in ok
                  if t.escape and t.escape.fast_morse
                  and t.escape.fast_morse.max_compute_time_ms is not None]

    return VariantStats(
        n=len(ok),
        success_rate=len(succ) / len(ok),
        mean_goal_err=float(np.mean(errs)) if errs else None,
        mean_wall_time=float(np.mean(wall)) if wall else None,
        stall_rate=len(stall) / len(ok),
        trap_rate=len(trap) / len(ok),
        morse_activated_rate=len(morse_act) / len(ok),
        escape_succeeded_rate=len(morse_esc) / len(ok),
        prefix_selected_rate=len(prefix_sel) / len(ok),
        fallback_rate=len(fallback) / len(ok),
        mean_clearance_gain=float(np.mean(cl_gain)) if cl_gain else None,
        mean_replans=float(np.mean(replans)) if replans else None,
        mean_planning_time_s=float(np.mean(plan_t)) if plan_t else None,
        fast_morse_activation_rate=len(fast_act) / len(ok),
        mean_fast_morse_compute_ms=float(np.mean(fm_mean_ms)) if fm_mean_ms else None,
        p95_fast_morse_compute_ms=float(np.percentile(fm_mean_ms, 95)) if fm_mean_ms else None,
        max_fast_morse_compute_ms=float(np.max(fm_max_ms)) if fm_max_ms else None,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_group(ax, variant_names, values, ylabel, title, colors, width=0.5):
    x = np.arange(len(variant_names))
    bars = ax.bar(x, [v if v is not None else 0.0 for v in values],
                  width=width, color=[colors[n] for n in variant_names], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in variant_names], fontsize=6, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    for bar, v in zip(bars, values):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    return ax


def plot_results(
    stats: Dict[str, VariantStats],
    scenario: str,
    out_dir: Path,
) -> None:
    variants = list(stats.keys())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    _bar_group(axes[0, 0], variants,
               [stats[v].success_rate for v in variants],
               "Success rate", "Terminal success rate", COLORS)

    _bar_group(axes[0, 1], variants,
               [stats[v].stall_rate for v in variants],
               "Stall rate", "Stall detection rate", COLORS)

    _bar_group(axes[0, 2], variants,
               [stats[v].trap_rate for v in variants],
               "Trap rate", "Trap detection rate", COLORS)

    _bar_group(axes[0, 3], variants,
               [stats[v].morse_activated_rate for v in variants],
               "Morse activated", "Morse activated (incl. forced)", COLORS)

    _bar_group(axes[1, 0], variants,
               [stats[v].escape_succeeded_rate for v in variants],
               "Escape success rate", "Morse escape success\n(fraction of trials)", COLORS)

    _bar_group(axes[1, 1], variants,
               [stats[v].prefix_selected_rate for v in variants],
               "Prefix selected rate", "Prefix mode selected\n(fraction of trials)", COLORS)

    _bar_group(axes[1, 2], variants,
               [stats[v].fallback_rate for v in variants],
               "Fallback rate", "Existing-escape fallback used\n(fraction of trials)", COLORS)

    _bar_group(axes[1, 3], variants,
               [stats[v].mean_planning_time_s for v in variants],
               "Mean Morse planning time (s)", "Morse planning time\n(activated trials only)", COLORS)

    fig.suptitle(f"Morse Escape Ablation — {scenario}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / f"morse_ablation_{scenario}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    scenarios: List[str],
    n_trials: int,
    output_dir: Path,
    seed_offset: int,
    variants: Optional[List[str]] = None,
) -> None:
    variants = variants or list(_VARIANTS.keys())
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, List[TrialMetrics]]] = {}

    for scenario in scenarios:
        if scenario not in SCENARIO_REGISTRY:
            print(f"Unknown scenario '{scenario}'. Available: {list(SCENARIO_REGISTRY)}")
            continue

        spec = SCENARIO_REGISTRY[scenario]()
        scen_dir = output_dir / scenario
        print(f"\n=== Scenario: {scenario} ===")

        scen_results: Dict[str, List[TrialMetrics]] = {}
        for variant in variants:
            print(f"\n--- Running {LABELS[variant]} ---")
            trials = _run_variant(
                spec, variant, _VARIANTS[variant],
                n_trials, seed_offset, scen_dir,
            )
            scen_results[variant] = trials

        all_results[scenario] = scen_results

        stats = {v: _compute_stats(scen_results[v]) for v in variants}

        # Console table
        print(f"\n{'Variant':<32} {'n':>4} {'succ':>6} {'stall':>6} {'trap':>6} "
              f"{'act':>6} {'esc_ok':>7} {'pfx':>6} {'fallbk':>7} {'goal_err':>9} {'wall_s':>7}")
        print("-" * 97)
        for v in variants:
            s = stats[v]
            goal_str = f"{s.mean_goal_err:>9.4f}" if s.mean_goal_err is not None else f"{'N/A':>9}"
            wall_str = f"{s.mean_wall_time:>7.2f}" if s.mean_wall_time is not None else f"{'N/A':>7}"
            print(
                f"{LABELS[v]:<32} {s.n:>4} {s.success_rate:>6.3f} {s.stall_rate:>6.3f} "
                f"{s.trap_rate:>6.3f} {s.morse_activated_rate:>6.3f} "
                f"{s.escape_succeeded_rate:>7.3f} {s.prefix_selected_rate:>6.3f} "
                f"{s.fallback_rate:>7.3f} {goal_str} {wall_str}"
            )

        # Fast Morse timing budget annotation
        print("  Fast Morse timing targets: mean <2 ms, p95 <3.3 ms, max <5 ms")
        for v in [vv for vv in variants if stats[vv].mean_fast_morse_compute_ms is not None]:
            s = stats[v]
            mean_ok = s.mean_fast_morse_compute_ms < 2.0
            p95_ok  = (s.p95_fast_morse_compute_ms or 0) < 3.3
            max_ok  = (s.max_fast_morse_compute_ms  or 0) < 5.0
            flag = "OK" if (mean_ok and p95_ok and max_ok) else "SLOW"
            print(
                f"  {flag} {LABELS[v]}: mean={s.mean_fast_morse_compute_ms:.2f} ms "
                f"p95={s.p95_fast_morse_compute_ms or 0:.2f} ms "
                f"max={s.max_fast_morse_compute_ms or 0:.2f} ms"
            )

        plot_results(stats, scenario, output_dir)

        summary = {
            "scenario": scenario,
            "n_trials_per_variant": n_trials,
            "variants": {
                v: {
                    "n": s.n,
                    "success_rate": s.success_rate,
                    "stall_rate": s.stall_rate,
                    "trap_rate": s.trap_rate,
                    "morse_activated_rate": s.morse_activated_rate,
                    "escape_succeeded_rate": s.escape_succeeded_rate,
                    "prefix_selected_rate": s.prefix_selected_rate,
                    "fallback_rate": s.fallback_rate,
                    "mean_clearance_gain": s.mean_clearance_gain,
                    "mean_goal_err": s.mean_goal_err,
                    "mean_wall_time_s": s.mean_wall_time,
                    "mean_replans": s.mean_replans,
                    "mean_morse_planning_time_s": s.mean_planning_time_s,
                    "fast_morse_activation_rate": s.fast_morse_activation_rate,
                    "mean_fast_morse_compute_ms": s.mean_fast_morse_compute_ms,
                    "p95_fast_morse_compute_ms":  s.p95_fast_morse_compute_ms,
                    "max_fast_morse_compute_ms":  s.max_fast_morse_compute_ms,
                }
                for v, s in stats.items()
            },
        }
        (output_dir / f"summary_{scenario}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs saved to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Morse escape planner ablation benchmark")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                        metavar="SCENARIO")
    parser.add_argument("--n-trials",    type=int,  default=20)
    parser.add_argument("--seed-offset", type=int,  default=0)
    parser.add_argument("--variants",    nargs="+", default=None,
                        choices=list(_VARIANTS.keys()),
                        help="Subset of variants to run (default: all)")
    parser.add_argument("--output-dir",  type=Path,
                        default=Path("outputs/eval/morse_ablation"))
    args = parser.parse_args(argv)
    run(args.scenarios, args.n_trials, args.output_dir, args.seed_offset, args.variants)


if __name__ == "__main__":
    main()
