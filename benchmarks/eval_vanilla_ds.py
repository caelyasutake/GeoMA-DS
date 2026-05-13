"""
Vanilla-DS baseline benchmark.

Compares three conditions on the frontal I-barrier scenario:

  1. Vanilla DS    — no BiRRT; straight-line DS directly to nearest IK goal
  2. Single-IK DS  — BiRRT + DS, single best IK goal
  3. Multi-IK DS   — BiRRT + DS + family-aware Multi-IK (full system)

The vanilla-DS condition is the fairest lower bound: the arm uses the exact
same controller and gains but has no obstacle-aware path to follow, so the
DS gradient pulls it straight through the barrier region.

Usage::

    python -m benchmarks.eval_vanilla_ds
    python -m benchmarks.eval_vanilla_ds --difficulty medium --n-trials 20
    python -m benchmarks.eval_vanilla_ds --difficulty all --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import IKCondition, ControlCondition, make_condition
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import TrialMetrics, load_jsonl
from src.scenarios.scenario_builders import build_frontal_i_barrier_lr

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
VANILLA  = make_condition(IKCondition.VANILLA_DS,      ControlCondition.PATH_DS_FULL)
SINGLE   = make_condition(IKCondition.SINGLE_IK_BEST,  ControlCondition.PATH_DS_FULL)
MULTI    = make_condition(IKCondition.MULTI_IK_FULL,   ControlCondition.PATH_DS_FULL)
DIFFIK   = make_condition(IKCondition.VANILLA_DS,      ControlCondition.VANILLA_DS_DIFFIK_MODULATION)
CONDITIONS = [VANILLA, SINGLE, MULTI, DIFFIK]

LABELS = {
    VANILLA.name: "Vanilla DS (no planner)",
    SINGLE.name:  "Single-IK + BiRRT + DS",
    MULTI.name:   "Multi-IK + BiRRT + DS",
    DIFFIK.name:  "Vanilla DS + Diff-IK + Modulation",
}
COLORS = {
    VANILLA.name: "#6B6B6B",
    SINGLE.name:  "#E07B39",
    MULTI.name:   "#2176AE",
    DIFFIK.name:  "#7B2D8B",
}
DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    difficulty: str,
    n_trials: int,
    seed_offset: int,
    out_dir: Path,
) -> List[TrialMetrics]:
    spec   = build_frontal_i_barrier_lr(difficulty)
    runner = ExperimentRunner(output_dir=out_dir, verbose=True)
    runner.run_spec(spec, CONDITIONS, n_trials=n_trials, seed_offset=seed_offset)
    return runner.load_results()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _cond_trials(trials: List[TrialMetrics], cond_name: str, difficulty: str):
    return [t for t in trials
            if t.condition == cond_name
            and t.scenario == f"frontal_i_barrier_lr_{difficulty}"
            and t.error is None]


def build_summary(difficulty: str, trials: List[TrialMetrics], n_trials: int) -> dict:
    summary: dict = {
        "scenario": f"frontal_i_barrier_lr_{difficulty}",
        "n_trials_per_condition": n_trials,
        "conditions": {},
    }
    for cond in CONDITIONS:
        ct = _cond_trials(trials, cond.name, difficulty)
        if not ct:
            summary["conditions"][cond.name] = {}
            continue

        successes  = [t for t in ct if t.execution and t.execution.terminal_success]
        min_errs   = [t.execution.min_goal_error_ever for t in ct
                      if t.execution and t.execution.min_goal_error_ever < float("inf")]
        final_errs = [t.execution.final_goal_err for t in ct
                      if t.execution and t.execution.final_goal_err < float("inf")]
        clearances = [t.barrier.min_clearance for t in ct
                      if t.barrier and t.barrier.min_clearance < float("inf")]
        plan_times = [t.plan.time_s for t in ct if t.plan]
        wall_times = [t.wall_time_s for t in ct]
        ctrl_hz    = [t.ctrl_freq.mean_hz                        for t in ct if t.ctrl_freq and t.ctrl_freq.mean_hz > 0]
        ctrl_ms    = [t.ctrl_freq.mean_ms                        for t in ct if t.ctrl_freq]
        ctrl_p95   = [t.ctrl_freq.p95_ms                         for t in ct if t.ctrl_freq]
        ctrl_cmp   = [t.ctrl_freq.mean_ctrl_compute_ms           for t in ct if t.ctrl_freq]
        ctrl_near  = [t.ctrl_freq.mean_ctrl_ms_near_obstacle     for t in ct if t.ctrl_freq and t.ctrl_freq.n_steps_near_obstacle > 0]
        ctrl_far   = [t.ctrl_freq.mean_ctrl_ms_far_from_obstacle for t in ct if t.ctrl_freq and (t.ctrl_freq.n_steps_executed - t.ctrl_freq.n_steps_near_obstacle) > 0]
        n_exec     = [t.ctrl_freq.n_steps_executed               for t in ct if t.ctrl_freq]
        n_near     = [t.ctrl_freq.n_steps_near_obstacle          for t in ct if t.ctrl_freq]
        n_cbf      = [t.ctrl_freq.n_steps_cbf_active             for t in ct if t.ctrl_freq]
        failure_reasons = [t.execution.failure_reason for t in ct
                           if t.execution and t.execution.failure_reason]
        reason_counts: dict = {}
        for r in failure_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        summary["conditions"][cond.name] = {
            "label": LABELS[cond.name],
            "n_trials": len(ct),
            "success_rate": len(successes) / max(1, len(ct)),
            "mean_min_goal_error_ever": float(np.mean(min_errs)) if min_errs else None,
            "mean_final_goal_error": float(np.mean(final_errs)) if final_errs else None,
            "mean_min_clearance": float(np.mean(clearances)) if clearances else None,
            "mean_plan_time_s": float(np.mean(plan_times)) if plan_times else None,
            "mean_wall_time_s": float(np.mean(wall_times)) if wall_times else None,
            # Timing
            "mean_ctrl_hz":                  float(np.mean(ctrl_hz))   if ctrl_hz   else None,
            "mean_ctrl_ms":                  float(np.mean(ctrl_ms))   if ctrl_ms   else None,
            "p95_ctrl_ms":                   float(np.mean(ctrl_p95))  if ctrl_p95  else None,
            "mean_ctrl_compute_ms":          float(np.mean(ctrl_cmp))  if ctrl_cmp  else None,
            "mean_ctrl_ms_near_obstacle":    float(np.mean(ctrl_near)) if ctrl_near else None,
            "mean_ctrl_ms_far_from_obstacle":float(np.mean(ctrl_far))  if ctrl_far  else None,
            # Step-regime counts
            "mean_n_steps_executed":         float(np.mean(n_exec))    if n_exec    else None,
            "mean_n_steps_near_obstacle":    float(np.mean(n_near))    if n_near    else None,
            "mean_n_steps_cbf_active":       float(np.mean(n_cbf))     if n_cbf     else None,
            "failure_reason_counts": reason_counts,
        }
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_success_rate(trials: List[TrialMetrics], difficulties: List[str], out_dir: Path):
    n = len(difficulties)
    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, n * 2.5), 4))
    for k, cond in enumerate(CONDITIONS):
        rates = []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            succ = [t for t in ct if t.execution and t.execution.terminal_success]
            rates.append(len(succ) / max(1, len(ct)))
        ax.bar(x + k * width, rates, width,
               label=LABELS[cond.name], color=COLORS[cond.name], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate: Vanilla DS vs Single-IK+BiRRT vs Multi-IK+BiRRT")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate.png", dpi=120)
    plt.close(fig)


def plot_min_goal_error(trials: List[TrialMetrics], difficulties: List[str], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    positions, data_all, tick_labels, colors_all = [], [], [], []
    pos = 1
    for diff in difficulties:
        for cond in CONDITIONS:
            ct = _cond_trials(trials, cond.name, diff)
            vals = [t.execution.min_goal_error_ever for t in ct
                    if t.execution and t.execution.min_goal_error_ever < float("inf")]
            if vals:
                data_all.append(vals)
                positions.append(pos)
                tick_labels.append(f"{diff}\n{LABELS[cond.name][:7]}")
                colors_all.append(COLORS[cond.name])
            pos += 1
        pos += 0.5
    if data_all:
        bp = ax.boxplot(data_all, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.axhline(0.05, color="green", linestyle="--", linewidth=0.8, label="goal_radius=0.05")
    ax.set_ylabel("Min goal error ever (rad)")
    ax.set_title("Closest approach to goal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "min_goal_error.png", dpi=120)
    plt.close(fig)


def plot_clearance(trials: List[TrialMetrics], difficulties: List[str], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    positions, data_all, tick_labels, colors_all = [], [], [], []
    pos = 1
    for diff in difficulties:
        for cond in CONDITIONS:
            ct = _cond_trials(trials, cond.name, diff)
            vals = [t.barrier.min_clearance for t in ct
                    if t.barrier and t.barrier.min_clearance < float("inf")]
            if vals:
                data_all.append(vals)
                positions.append(pos)
                tick_labels.append(f"{diff}\n{LABELS[cond.name][:7]}")
                colors_all.append(COLORS[cond.name])
            pos += 1
        pos += 0.5
    if data_all:
        bp = ax.boxplot(data_all, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="collision")
    ax.set_ylabel("Min clearance (m)")
    ax.set_title("Minimum obstacle clearance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "min_clearance.png", dpi=120)
    plt.close(fig)


def plot_plan_time(trials: List[TrialMetrics], difficulties: List[str], out_dir: Path):
    n = len(difficulties)
    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, n * 2.5), 4))
    for k, cond in enumerate(CONDITIONS):
        means = []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            pt = [t.plan.time_s for t in ct if t.plan]
            means.append(float(np.mean(pt)) if pt else 0.0)
        ax.bar(x + k * width, means, width,
               label=LABELS[cond.name], color=COLORS[cond.name], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Mean planning time (s)")
    ax.set_title("Planning overhead (0 = vanilla DS)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plan_time.png", dpi=120)
    plt.close(fig)


def plot_timing(trials: List[TrialMetrics], difficulties: List[str], out_dir: Path):
    """
    Four-panel timing breakdown per difficulty:
      1. Mean total step time (ctrl + sim)
      2. Mean controller-compute-only time
      3. Mean step time when near obstacle  (clearance < 5 cm)
      4. Mean step time when far from obstacle
    Each panel is a grouped bar chart — one group per difficulty, one bar per condition.
    """
    n = len(difficulties)
    x = np.arange(n)
    width = 0.20

    panels = [
        ("mean_ms",                      "Mean step time (ms)\nctrl + sim"),
        ("mean_ctrl_compute_ms",          "Controller compute (ms)\nctrl only, no sim"),
        ("mean_ctrl_ms_near_obstacle",    "Step time near obstacle (ms)\nclearance < 5 cm"),
        ("mean_ctrl_ms_far_from_obstacle","Step time far from obstacle (ms)\nclearance ≥ 5 cm"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(max(14, n * 5), 4))
    for ax, (field, ylabel) in zip(axes, panels):
        for k, cond in enumerate(CONDITIONS):
            means = []
            for diff in difficulties:
                ct = _cond_trials(trials, cond.name, diff)
                vals = [getattr(t.ctrl_freq, field)
                        for t in ct if t.ctrl_freq and getattr(t.ctrl_freq, field, 0) > 0]
                means.append(float(np.mean(vals)) if vals else 0.0)
            ax.bar(x + k * width, means, width,
                   label=LABELS[cond.name], color=COLORS[cond.name], alpha=0.85)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(difficulties)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Per-step controller timing breakdown by difficulty", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "timing.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    difficulties: List[str],
    n_trials: int,
    output_dir: Path,
    seed_offset: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_trials: List[TrialMetrics] = []

    for diff in difficulties:
        diff_dir = output_dir / diff
        diff_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Difficulty: {diff} ===")
        trials = run_benchmark(diff, n_trials, seed_offset, diff_dir)
        all_trials.extend(trials)

        summary = build_summary(diff, trials, n_trials)
        (diff_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  success rates:")
        for cond in CONDITIONS:
            sr = summary["conditions"].get(cond.name, {}).get("success_rate")
            print(f"    {LABELS[cond.name]:35s}: {sr:.2f}" if sr is not None else f"    {LABELS[cond.name]:35s}: n/a")

    # Aggregate plots across all difficulties
    plot_success_rate(all_trials, difficulties, output_dir)
    plot_min_goal_error(all_trials, difficulties, output_dir)
    plot_clearance(all_trials, difficulties, output_dir)
    plot_plan_time(all_trials, difficulties, output_dir)
    plot_timing(all_trials, difficulties, output_dir)
    print(f"\nOutputs saved to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vanilla DS vs BiRRT+DS benchmark")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/eval/vanilla_ds"))
    args = parser.parse_args(argv)

    diffs = DIFFICULTIES if args.difficulty == "all" else [args.difficulty]
    run(diffs, args.n_trials, args.output_dir, args.seed_offset)


if __name__ == "__main__":
    main()
