"""
Multi-IK Ablation Benchmark (Phase 2 of evaluation plan).

Answers Q2: Does diverse IK selection materially improve performance?

Compares:
  - multi_ik_full          (all safe IK solutions)
  - single_ik_best         (top-ranked single solution)
  - single_ik_random       (randomly chosen solution)
  - multi_ik_top_2/4       (top-k solutions)
  - multi_ik_energy_aware  (energy-sorted full set)

Over scenarios: free_space, narrow_passage, cluttered_tabletop,
                random_obstacle_field

Usage::

    python -m benchmarks.eval_multik_ablation
    python -m benchmarks.eval_multik_ablation --n-trials 10 --output-dir outputs/eval/quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.evaluation.experiment_runner import ExperimentRunner, build_ablation_conditions
from src.evaluation.metrics import load_jsonl, TrialMetrics
from src.evaluation.report_writer import write_report, make_ik_recommendation
from src.evaluation.statistical_tests import (
    compare_conditions,
    compute_multi_ik_advantage,
    compute_terminal_advantage_rate,
    compute_goal_rank_correlations,
    compare_matched_trials,
)


ABLATION_SCENARIOS = [
    "free_space",
    "narrow_passage",
    "cluttered_tabletop",
    "random_obstacle_field",
    "u_shape",
    "left_open_u",
]

_IK_LABEL_MAP = {
    "multi_ik_full":      "multi_full",
    "single_ik_best":     "single_best",
    "single_ik_random":   "single_rnd",
    "multi_ik_top_2":     "top_2",
    "multi_ik_top_4":     "top_4",
    "multi_ik_energy_aware": "energy",
}


def run_ablation(
    n_trials: int = 50,
    output_dir: Path = Path("outputs/eval/ablation"),
    seed_offset: int = 0,
    verbose: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_ablation_conditions(ctrl=ControlCondition.PATH_DS_FULL)
    runner = ExperimentRunner(output_dir=output_dir, verbose=verbose)

    t0 = time.perf_counter()
    for scen in ABLATION_SCENARIOS:
        print(f"\n[ablation] Scenario: {scen} — {len(conditions)} conds × {n_trials} trials")
        runner.run(scen, conditions, n_trials=n_trials, seed_offset=seed_offset)

    elapsed = time.perf_counter() - t0
    print(f"\n[ablation] Finished in {elapsed:.1f}s")

    # Aggregate
    trials = runner.load_results()
    aggs   = aggregate_results(trials)
    save_aggregate(aggs, output_dir)

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_ik_ablation(aggs, trials, plots_dir)
    _plot_terminal_success_uplift(trials, plots_dir)
    _plot_final_goal_error_by_ik(trials, plots_dir)
    _plot_goal_rank_vs_final_error(trials, plots_dir)
    _plot_goal_rank_vs_terminal_success(trials, plots_dir)

    # Report
    write_report(aggs, trials, output_dir)
    print(f"[ablation] Outputs → {output_dir}")


# ---------------------------------------------------------------------------
# Plot 1 (existing) — Success rate and planning time vs IK condition
# ---------------------------------------------------------------------------

def _plot_ik_ablation(aggs, trials, plots_dir: Path) -> None:
    """Success rate and planning time vs IK condition."""
    ik_labels = [_IK_LABEL_MAP.get(c.value, c.value) for c in PLANNING_IK_CONDITIONS]
    scenarios  = sorted({s for (s, _) in aggs if s in ABLATION_SCENARIOS})

    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 8))
    if len(scenarios) == 1:
        axes = axes.reshape(2, 1)

    for col, scen in enumerate(scenarios):
        plan_rates  = []
        term_rates  = []
        time_means  = []
        time_stds   = []

        for ik in PLANNING_IK_CONDITIONS:
            cond_key = (scen, f"{ik.value}__path_ds_full")
            agg = aggs.get(cond_key)
            if agg:
                plan_rates.append(agg.plan_success_rate * 100)
                term_rates.append(agg.terminal_success_rate * 100)
                time_means.append(agg.plan_time_s.mean or 0.0)
                time_stds.append(agg.plan_time_s.std or 0.0)
            else:
                plan_rates.append(0.0)
                term_rates.append(0.0)
                time_means.append(0.0)
                time_stds.append(0.0)

        x = np.arange(len(ik_labels))
        ax0 = axes[0, col]
        ax0.bar(x - 0.2, plan_rates, 0.4, label="Plan Success %",  color="#1976D2", alpha=0.85)
        ax0.bar(x + 0.2, term_rates, 0.4, label="Term. Success %", color="#43A047", alpha=0.85)
        ax0.set_title(scen, fontsize=10)
        ax0.set_xticks(x)
        ax0.set_xticklabels(ik_labels, rotation=30, ha="right", fontsize=7)
        ax0.set_ylim(0, 110)
        ax0.set_ylabel("Success rate (%)")
        ax0.legend(fontsize=7)

        ax1 = axes[1, col]
        ax1.bar(x, time_means, yerr=time_stds, color="#FB8C00", alpha=0.85, capsize=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(ik_labels, rotation=30, ha="right", fontsize=7)
        ax1.set_ylabel("Planning time (s)")

    fig.suptitle("Multi-IK Ablation: Success Rate and Planning Time", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "ik_ablation.png", dpi=120)
    plt.close(fig)
    print("[ablation] Saved ik_ablation.png")


# ---------------------------------------------------------------------------
# Plot 2 (NEW) — Terminal success uplift by scenario
# ---------------------------------------------------------------------------

def _plot_terminal_success_uplift(
    trials: List[TrialMetrics],
    plots_dir: Path,
) -> None:
    """
    Bar plot: terminal success uplift of multi_ik_full vs each single-IK baseline,
    per scenario.
    """
    scenarios = sorted({t.scenario for t in trials if t.scenario in ABLATION_SCENARIOS})
    baselines = {
        "vs single_best":   "single_ik_best__path_ds_full",
        "vs single_random": "single_ik_random__path_ds_full",
    }
    treatment = "multi_ik_full__path_ds_full"

    n_scen = len(scenarios)
    if n_scen == 0:
        return

    x = np.arange(n_scen)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_scen), 5))
    offsets = np.linspace(-width / 2, width / 2, len(baselines))
    colors  = ["#1976D2", "#E53935"]

    for i, (label, bl_cond) in enumerate(baselines.items()):
        uplifts = []
        for scen in scenarios:
            cmp = compare_matched_trials(
                [t for t in trials if t.scenario == scen],
                cond_a=bl_cond, cond_b=treatment,
            )
            upl = cmp["overall"]["terminal_success_uplift"]
            uplifts.append(upl if np.isfinite(upl) else 0.0)

        ax.bar(x + offsets[i], uplifts, width / len(baselines),
               label=label, color=colors[i % len(colors)], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Terminal success uplift (fraction)")
    ax.set_title("Terminal Success Uplift: multi_ik_full vs single-IK baselines")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "terminal_success_uplift.png", dpi=120)
    plt.close(fig)
    print("[ablation] Saved terminal_success_uplift.png")


# ---------------------------------------------------------------------------
# Plot 3 (NEW) — Final goal error by IK condition (boxplot)
# ---------------------------------------------------------------------------

def _plot_final_goal_error_by_ik(
    trials: List[TrialMetrics],
    plots_dir: Path,
) -> None:
    """
    Boxplot: final goal error per IK condition, one subplot per scenario.
    """
    scenarios = sorted({t.scenario for t in trials if t.scenario in ABLATION_SCENARIOS})
    ik_conds  = [c.value for c in PLANNING_IK_CONDITIONS]
    short_labels = [_IK_LABEL_MAP.get(c, c) for c in ik_conds]

    n_scen = len(scenarios)
    if n_scen == 0:
        return

    fig, axes = plt.subplots(1, n_scen, figsize=(5 * n_scen, 5), sharey=False)
    if n_scen == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        data = []
        labels_used = []
        for ik_cond, label in zip(ik_conds, short_labels):
            full_cond = f"{ik_cond}__path_ds_full"
            errs = [
                t.execution.final_goal_err
                for t in trials
                if t.scenario == scen and t.condition == full_cond
                and t.execution and t.execution.final_goal_err is not None
                and np.isfinite(t.execution.final_goal_err)
            ]
            if errs:
                data.append(errs)
                labels_used.append(label)

        if data:
            ax.boxplot(data, labels=labels_used, patch_artist=True,
                       medianprops=dict(color="red", linewidth=2))
            ax.set_title(scen, fontsize=10)
            ax.set_ylabel("Final goal error (rad)")
            ax.set_xticklabels(labels_used, rotation=30, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Final Goal Error by IK Condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "final_goal_error_by_ik_condition.png", dpi=120)
    plt.close(fig)
    print("[ablation] Saved final_goal_error_by_ik_condition.png")


# ---------------------------------------------------------------------------
# Plot 4 (NEW) — Goal-rank vs final error (scatter)
# ---------------------------------------------------------------------------

def _plot_goal_rank_vs_final_error(
    trials: List[TrialMetrics],
    plots_dir: Path,
) -> None:
    """
    Scatter: selected IK goal rank vs final goal error, coloured by scenario.
    Only includes multi-IK trials where both rank and error are available.
    """
    scenarios = sorted({t.scenario for t in trials if t.scenario in ABLATION_SCENARIOS})
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(scenarios)))

    fig, ax = plt.subplots(figsize=(7, 5))

    any_data = False
    for scen, color in zip(scenarios, colors):
        ranks  = []
        errors = []
        for t in trials:
            if t.scenario != scen or "multi_ik" not in t.condition:
                continue
            if t.ik and t.ik.selected_goal_rank is not None:
                if t.execution and t.execution.final_goal_err is not None:
                    if np.isfinite(t.execution.final_goal_err):
                        ranks.append(t.ik.selected_goal_rank)
                        errors.append(t.execution.final_goal_err)
        if ranks:
            ax.scatter(ranks, errors, alpha=0.5, s=30, color=color, label=scen)
            any_data = True

    if any_data:
        ax.set_xlabel("Selected IK goal rank (0 = best-ranked)")
        ax.set_ylabel("Final goal error (rad)")
        ax.set_title("Goal Rank vs Final Goal Error (multi-IK trials)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(plots_dir / "goal_rank_vs_final_error.png", dpi=120)
    plt.close(fig)
    print("[ablation] Saved goal_rank_vs_final_error.png")


# ---------------------------------------------------------------------------
# Plot 5 (NEW) — Goal-rank vs terminal success (grouped bars)
# ---------------------------------------------------------------------------

def _plot_goal_rank_vs_terminal_success(
    trials: List[TrialMetrics],
    plots_dir: Path,
) -> None:
    """
    Grouped bars: terminal success rate per selected IK rank bucket, per scenario.
    """
    scenarios = sorted({t.scenario for t in trials if t.scenario in ABLATION_SCENARIOS})
    if not scenarios:
        return

    n_scen = len(scenarios)
    fig, axes = plt.subplots(1, n_scen, figsize=(5 * n_scen, 4), sharey=True)
    if n_scen == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        rank_term: Dict[int, List[float]] = {}
        for t in trials:
            if t.scenario != scen or "multi_ik" not in t.condition:
                continue
            if t.ik and t.ik.selected_goal_rank is not None and t.execution:
                r = int(t.ik.selected_goal_rank)
                rank_term.setdefault(r, []).append(float(t.execution.terminal_success))

        if not rank_term:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(scen, fontsize=10)
            continue

        ranks   = sorted(rank_term)
        success = [float(np.mean(rank_term[r])) * 100 for r in ranks]
        counts  = [len(rank_term[r]) for r in ranks]

        bars = ax.bar([str(r) for r in ranks], success, color="#43A047", alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={cnt}", ha="center", va="bottom", fontsize=7)

        ax.set_title(scen, fontsize=10)
        ax.set_xlabel("Selected goal rank")
        ax.set_ylabel("Terminal success (%)")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Terminal Success Rate by Selected IK Goal Rank (multi-IK trials)", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "goal_rank_vs_terminal_success.png", dpi=120)
    plt.close(fig)
    print("[ablation] Saved goal_rank_vs_terminal_success.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-IK ablation benchmark")
    parser.add_argument("--n-trials",    type=int,  default=50)
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs/eval/ablation"))
    parser.add_argument("--seed-offset", type=int,  default=0)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args(argv)

    run_ablation(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
