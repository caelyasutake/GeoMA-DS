"""
U-shape difficulty sweep benchmark.

Sweeps over U-shape geometry parameters (opening_width, depth) and measures
whether multi-IK advantage increases as the passage becomes harder.

Answers:
  * Does multi-IK help more as the U gets tighter?
  * Is there a threshold where single-IK collapses but multi-IK remains viable?

Outputs::

    outputs/eval/u_shape_sweep/
      u_shape_difficulty_results.json
      u_shape_difficulty_plot.png
      per_trial_results.jsonl

Usage::

    python -m benchmarks.eval_u_shape_sweep
    python -m benchmarks.eval_u_shape_sweep --n-trials 5 --output-dir outputs/eval/quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.aggregators import aggregate_results, save_aggregate
from src.evaluation.baselines import (
    ControlCondition, IKCondition, make_condition,
    PLANNING_IK_CONDITIONS,
)
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import load_jsonl, TrialMetrics
from src.evaluation.statistical_tests import compare_matched_trials
from src.scenarios.scenario_builders import u_shape_scenario


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

# Opening width sweep: wide → narrow (harder)
OPENING_WIDTHS = [0.40, 0.30, 0.20, 0.12]

# Depth sweep: shallow → deep
DEPTHS = [0.20, 0.30, 0.40, 0.50]

# IK conditions to compare
_SWEEP_CONDITIONS = [
    make_condition(IKCondition.MULTI_IK_FULL,    ControlCondition.PATH_DS_FULL),
    make_condition(IKCondition.SINGLE_IK_BEST,   ControlCondition.PATH_DS_FULL),
    make_condition(IKCondition.SINGLE_IK_RANDOM, ControlCondition.PATH_DS_FULL),
    make_condition(IKCondition.MULTI_IK_TOP_2,   ControlCondition.PATH_DS_FULL),
]


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

def run_u_shape_sweep(
    n_trials: int = 10,
    output_dir: Path = Path("outputs/eval/u_shape_sweep"),
    seed_offset: int = 0,
    verbose: bool = True,
) -> None:
    """
    Run the U-shape geometry sweep.

    For each (opening_width, depth) combination, run N trials for each IK
    condition and record plan_success, terminal_success, final_goal_error,
    and multi-IK advantage metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    # ---- Opening-width sweep ------------------------------------------------
    if verbose:
        print("\n[u_shape_sweep] === Opening-width sweep ===")

    width_results: List[Dict[str, Any]] = []
    for ow in OPENING_WIDTHS:
        spec = u_shape_scenario(opening_width=ow, depth=0.35)
        sweep_dir = output_dir / f"width_{ow:.2f}"
        runner = ExperimentRunner(output_dir=sweep_dir, verbose=False)

        if verbose:
            print(f"  opening_width={ow:.2f}  depth=0.35  ({n_trials} trials × {len(_SWEEP_CONDITIONS)} conds)")

        # Patch scenario name so runner can identify it uniquely
        spec.name = f"u_shape_w{ow:.2f}"
        # ExperimentRunner looks up spec by name — provide it directly
        runner.run_spec(spec, _SWEEP_CONDITIONS, n_trials=n_trials, seed_offset=seed_offset)

        trials = runner.load_results()
        aggs   = aggregate_results(trials)
        entry  = _make_sweep_entry(
            trials, aggs,
            param_name="opening_width", param_value=ow,
            scenario_name=spec.name,
        )
        width_results.append(entry)
        all_results.append(entry)

    # ---- Depth sweep --------------------------------------------------------
    if verbose:
        print("\n[u_shape_sweep] === Depth sweep ===")

    depth_results: List[Dict[str, Any]] = []
    for dep in DEPTHS:
        spec = u_shape_scenario(opening_width=0.25, depth=dep)
        sweep_dir = output_dir / f"depth_{dep:.2f}"
        runner = ExperimentRunner(output_dir=sweep_dir, verbose=False)

        if verbose:
            print(f"  opening_width=0.25  depth={dep:.2f}  ({n_trials} trials × {len(_SWEEP_CONDITIONS)} conds)")

        spec.name = f"u_shape_d{dep:.2f}"
        runner.run_spec(spec, _SWEEP_CONDITIONS, n_trials=n_trials, seed_offset=seed_offset)

        trials = runner.load_results()
        aggs   = aggregate_results(trials)
        entry  = _make_sweep_entry(
            trials, aggs,
            param_name="depth", param_value=dep,
            scenario_name=spec.name,
        )
        depth_results.append(entry)
        all_results.append(entry)

    # ---- Save results -------------------------------------------------------
    out_json = output_dir / "u_shape_difficulty_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    if verbose:
        print(f"\n[u_shape_sweep] Results → {out_json}")

    # ---- Plots --------------------------------------------------------------
    _plot_sweep(width_results, "opening_width", "Opening width (m)", output_dir, verbose)
    _plot_sweep(depth_results, "depth",         "U depth (m)",        output_dir, verbose)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[u_shape_sweep] Done in {elapsed:.1f}s → {output_dir}")

    _print_interpretation(width_results, depth_results, verbose)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sweep_entry(
    trials: List[TrialMetrics],
    aggs: dict,
    param_name: str,
    param_value: float,
    scenario_name: str,
) -> Dict[str, Any]:
    """Summarise one sweep point into a dict."""
    entry: Dict[str, Any] = {
        "param_name":  param_name,
        "param_value": param_value,
        "scenario":    scenario_name,
        "conditions":  {},
    }

    cond_names = {
        "multi_ik_full":    f"{scenario_name}__multi_ik_full__path_ds_full",
        "single_ik_best":   f"{scenario_name}__single_ik_best__path_ds_full",
        "single_ik_random": f"{scenario_name}__single_ik_random__path_ds_full",
        "multi_ik_top_2":   f"{scenario_name}__multi_ik_top_2__path_ds_full",
    }
    for label, key in cond_names.items():
        agg = aggs.get((scenario_name, key.replace(f"{scenario_name}__", "")), None)
        if agg is not None:
            entry["conditions"][label] = {
                "plan_success_rate":     agg.plan_success_rate,
                "terminal_success_rate": agg.terminal_success_rate,
                "mean_final_goal_error": agg.final_goal_error.mean,
                "n_trials":              agg.n_trials,
            }

    # Matched advantage metrics
    cmp = compare_matched_trials(
        trials,
        cond_a="single_ik_best__path_ds_full",
        cond_b="multi_ik_full__path_ds_full",
    )
    ov = cmp.get("overall", {})
    entry["planning_advantage_rate"]  = ov.get("planning_advantage_rate", 0.0)
    entry["terminal_advantage_rate"]  = ov.get("terminal_advantage_rate", 0.0)
    entry["terminal_success_uplift"]  = ov.get("terminal_success_uplift", 0.0)
    entry["final_goal_error_delta"]   = ov.get("final_goal_error_delta", 0.0)
    return entry


def _plot_sweep(
    results: List[Dict[str, Any]],
    param_name: str,
    xlabel: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Plot success rate vs sweep parameter, by IK condition."""
    if not results:
        return

    xs    = [r["param_value"] for r in results]
    conds = ["multi_ik_full", "single_ik_best", "single_ik_random"]
    colors = {"multi_ik_full": "#43A047", "single_ik_best": "#1976D2", "single_ik_random": "#E53935"}
    styles = {"multi_ik_full": "-o",      "single_ik_best": "--s",     "single_ik_random": ":^"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: plan success rate
    ax = axes[0]
    for c in conds:
        ys = [r["conditions"].get(c, {}).get("plan_success_rate", float("nan")) for r in results]
        ax.plot(xs, ys, styles[c], color=colors[c], label=c.replace("_", " "), linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Plan success rate")
    ax.set_title("Planning success")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: terminal success rate
    ax = axes[1]
    for c in conds:
        ys = [r["conditions"].get(c, {}).get("terminal_success_rate", float("nan")) for r in results]
        ax.plot(xs, ys, styles[c], color=colors[c], label=c.replace("_", " "), linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Terminal success rate")
    ax.set_title("Terminal success")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: terminal advantage rate
    ax = axes[2]
    tar = [r.get("terminal_advantage_rate", 0.0) for r in results]
    par = [r.get("planning_advantage_rate", 0.0) for r in results]
    ax.plot(xs, tar, "-o", color="#43A047", label="terminal adv.", linewidth=1.8)
    ax.plot(xs, par, "--s", color="#1976D2", label="planning adv.", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Advantage rate (multi_full vs single_best)")
    ax.set_title("Multi-IK advantage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"U-shape sweep: {xlabel}", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"u_shape_difficulty_plot_{param_name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    if verbose:
        print(f"[u_shape_sweep] Plot → {out_path}")


def _print_interpretation(
    width_results: List[Dict[str, Any]],
    depth_results: List[Dict[str, Any]],
    verbose: bool,
) -> None:
    """Print a text interpretation of the sweep results."""
    if not verbose:
        return

    print("\n[u_shape_sweep] === Interpretation ===")

    def _interpret_series(results, param_name):
        if not results:
            return
        tars = [r.get("terminal_advantage_rate", 0.0) for r in results]
        vals = [r["param_value"] for r in results]
        max_tar = max(tars)
        min_tar = min(tars)
        print(f"\n  {param_name} sweep:")
        print(f"    terminal_advantage_rate range: {min_tar:.3f} – {max_tar:.3f}")
        if max_tar > min_tar + 0.05:
            print("    -> multi-IK advantage INCREASES with difficulty (good signal)")
        elif max_tar < 0.05:
            print("    -> multi-IK advantage is LOW across all conditions")
        else:
            print("    -> multi-IK advantage is roughly FLAT (no difficulty dependence)")

        # Single-IK collapse threshold
        for r in results:
            si = r["conditions"].get("single_ik_best", {}).get("terminal_success_rate", 1.0)
            mi = r["conditions"].get("multi_ik_full",  {}).get("terminal_success_rate", 1.0)
            if si < 0.5 and mi >= 0.5:
                print(f"    -> Threshold found at {param_name}={r['param_value']:.2f}: "
                      f"single_ik collapses ({si:.2f}) but multi_ik survives ({mi:.2f})")
                break

    _interpret_series(width_results, "opening_width")
    _interpret_series(depth_results, "depth")


def _json_default(obj):
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="U-shape difficulty sweep")
    parser.add_argument("--n-trials",    type=int,  default=10,
                        help="Trials per (condition × sweep point)")
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs/eval/u_shape_sweep"),
                        help="Output directory")
    parser.add_argument("--seed-offset", type=int,  default=0)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args(argv)
    run_u_shape_sweep(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
