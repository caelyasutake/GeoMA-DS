"""
Free-space controller timing benchmark.

Runs all four solver conditions on a scenario with NO obstacles.  With no
CBF, modulation, or near-obstacle logic active every executed step is a
pure controller compute step, giving a clean baseline for:

  * per-step controller compute cost (no safety overhead)
  * convergence speed for each solver family
  * planner overhead (BiRRT path-finding even when no obstacles exist)

Compare these numbers against barrier-scenario results to quantify how much
the near-obstacle regime costs each solver.

Usage::

    python -m benchmarks.eval_free_space
    python -m benchmarks.eval_free_space --n-trials 20
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
from src.scenarios.scenario_builders import free_space_scenario

# ---------------------------------------------------------------------------
# Conditions  (same four as eval_vanilla_ds so results are directly comparable)
# ---------------------------------------------------------------------------
VANILLA = make_condition(IKCondition.VANILLA_DS,      ControlCondition.PATH_DS_FULL)
SINGLE  = make_condition(IKCondition.SINGLE_IK_BEST,  ControlCondition.PATH_DS_FULL)
MULTI   = make_condition(IKCondition.MULTI_IK_FULL,   ControlCondition.PATH_DS_FULL)
DIFFIK  = make_condition(IKCondition.VANILLA_DS,      ControlCondition.VANILLA_DS_DIFFIK_MODULATION)
CONDITIONS = [VANILLA, SINGLE, MULTI, DIFFIK]

LABELS = {
    VANILLA.name: "Vanilla DS",
    SINGLE.name:  "Single-IK + BiRRT",
    MULTI.name:   "Multi-IK + BiRRT",
    DIFFIK.name:  "Diff-IK (task-space)",
}
COLORS = {
    VANILLA.name: "#6B6B6B",
    SINGLE.name:  "#E07B39",
    MULTI.name:   "#2176AE",
    DIFFIK.name:  "#7B2D8B",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(n_trials: int, seed_offset: int, out_dir: Path) -> List[TrialMetrics]:
    spec   = free_space_scenario()
    runner = ExperimentRunner(output_dir=out_dir, verbose=True)
    runner.run_spec(spec, CONDITIONS, n_trials=n_trials, seed_offset=seed_offset)
    return runner.load_results()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _cond_trials(trials: List[TrialMetrics], cond_name: str) -> List[TrialMetrics]:
    return [t for t in trials if t.condition == cond_name and t.error is None]


def build_summary(trials: List[TrialMetrics], n_trials: int) -> dict:
    summary: dict = {
        "scenario": "free_space",
        "n_trials_per_condition": n_trials,
        "note": "No obstacles — all steps are pure controller compute with zero CBF/modulation overhead.",
        "conditions": {},
    }
    for cond in CONDITIONS:
        ct = _cond_trials(trials, cond.name)
        if not ct:
            summary["conditions"][cond.name] = {}
            continue

        successes  = [t for t in ct if t.execution and t.execution.terminal_success]
        conv_steps = [t.execution.convergence_step for t in ct
                      if t.execution and t.execution.convergence_step is not None]
        plan_times = [t.plan.time_s for t in ct if t.plan]
        wall_times = [t.wall_time_s for t in ct]

        ctrl_hz    = [t.ctrl_freq.mean_hz              for t in ct if t.ctrl_freq and t.ctrl_freq.mean_hz > 0]
        ctrl_ms    = [t.ctrl_freq.mean_ms              for t in ct if t.ctrl_freq]
        ctrl_p95   = [t.ctrl_freq.p95_ms               for t in ct if t.ctrl_freq]
        ctrl_cmp   = [t.ctrl_freq.mean_ctrl_compute_ms for t in ct if t.ctrl_freq]
        n_exec     = [t.ctrl_freq.n_steps_executed     for t in ct if t.ctrl_freq]
        n_near     = [t.ctrl_freq.n_steps_near_obstacle for t in ct if t.ctrl_freq]
        n_cbf      = [t.ctrl_freq.n_steps_cbf_active   for t in ct if t.ctrl_freq]

        summary["conditions"][cond.name] = {
            "label": LABELS[cond.name],
            "n_trials": len(ct),
            "success_rate": len(successes) / max(1, len(ct)),
            "mean_convergence_steps": float(np.mean(conv_steps)) if conv_steps else None,
            "mean_plan_time_s":       float(np.mean(plan_times)) if plan_times else None,
            "mean_wall_time_s":       float(np.mean(wall_times)) if wall_times else None,
            # Timing — free-space baseline (compare against barrier benchmark)
            "mean_ctrl_hz":           float(np.mean(ctrl_hz))  if ctrl_hz  else None,
            "mean_ctrl_ms":           float(np.mean(ctrl_ms))  if ctrl_ms  else None,
            "p95_ctrl_ms":            float(np.mean(ctrl_p95)) if ctrl_p95 else None,
            "mean_ctrl_compute_ms":   float(np.mean(ctrl_cmp)) if ctrl_cmp else None,
            # Regime sanity checks — should all be zero in free space
            "mean_n_steps_executed":      float(np.mean(n_exec)) if n_exec else None,
            "mean_n_steps_near_obstacle": float(np.mean(n_near)) if n_near else None,
            "mean_n_steps_cbf_active":    float(np.mean(n_cbf))  if n_cbf  else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timing(trials: List[TrialMetrics], out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = [
        ("mean_ms",              "mean_ms",              "Mean step time (ctrl+sim) ms",  "Total step time"),
        ("mean_ctrl_compute_ms", "mean_ctrl_compute_ms", "Controller compute time ms",    "Controller compute only"),
        ("p95_ms",               "p95_ms",               "p95 step time ms",              "p95 step time"),
    ]

    for ax, (field, attr, ylabel, title) in zip(axes, metrics):
        vals  = []
        names = []
        colors = []
        for cond in CONDITIONS:
            ct = _cond_trials(trials, cond.name)
            v  = [getattr(t.ctrl_freq, attr) for t in ct if t.ctrl_freq and getattr(t.ctrl_freq, attr, 0) > 0]
            if v:
                vals.append(float(np.mean(v)))
                names.append(LABELS[cond.name])
                colors.append(COLORS[cond.name])
        if vals:
            ax.bar(names, vals, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Free-space timing baseline (no obstacles, zero CBF overhead)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "timing_free_space.png", dpi=130)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'timing_free_space.png'}")


def plot_convergence(trials: List[TrialMetrics], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    positions, data, tick_labels, colors_used = [], [], [], []
    for k, cond in enumerate(CONDITIONS):
        ct = _cond_trials(trials, cond.name)
        v  = [t.execution.convergence_step for t in ct
              if t.execution and t.execution.convergence_step is not None]
        if v:
            data.append(v)
            positions.append(k + 1)
            tick_labels.append(LABELS[cond.name])
            colors_used.append(COLORS[cond.name])
    if data:
        bp = ax.boxplot(data, positions=positions, patch_artist=True,
                        labels=tick_labels, widths=0.5)
        for patch, color in zip(bp["boxes"], colors_used):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_ylabel("Steps to convergence")
    ax.set_title("Convergence speed in free space (lower = faster)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_free_space.png", dpi=130)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'convergence_free_space.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(n_trials: int, output_dir: Path, seed_offset: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Free-space timing benchmark ({n_trials} trials / condition) ===")

    trials  = run_benchmark(n_trials, seed_offset, output_dir)
    summary = build_summary(trials, n_trials)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nResults (free space — zero CBF overhead):")
    for cond in CONDITIONS:
        s = summary["conditions"].get(cond.name, {})
        hz  = s.get("mean_ctrl_hz")
        ms  = s.get("mean_ctrl_ms")
        cmp = s.get("mean_ctrl_compute_ms")
        sr  = s.get("success_rate")
        print(f"  {LABELS[cond.name]:30s}: "
              f"success={sr:.2f}  "
              f"{ms:.3f} ms/step (ctrl+sim)  "
              f"{cmp:.3f} ms/step (ctrl only)  "
              f"→ {hz:.0f} Hz" if all(v is not None for v in [hz, ms, cmp, sr]) else
              f"  {LABELS[cond.name]:30s}: n/a")

    plot_timing(trials, output_dir)
    plot_convergence(trials, output_dir)
    print(f"\nOutputs saved to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Free-space controller timing baseline benchmark")
    parser.add_argument("--n-trials",    type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir",  type=Path,
                        default=Path("outputs/eval/free_space"))
    args = parser.parse_args(argv)
    run(args.n_trials, args.output_dir, args.seed_offset)


if __name__ == "__main__":
    main()
