"""
Frontal I-barrier benchmark: Multi-IK + DS vs Single-IK + DS.

Runs ExperimentRunner across two conditions for one or all difficulty variants,
writes JSONL + summary.json, and generates 7 comparison plots.

Usage::

    python -m benchmarks.eval_c_barrier
    python -m benchmarks.eval_c_barrier --difficulty hard --n-trials 30
    python -m benchmarks.eval_c_barrier --difficulty all --n-trials 10 --output-dir /tmp/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import IKCondition, ControlCondition, make_condition
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import TrialMetrics, load_jsonl
from src.evaluation.ik_family_analysis import analyse_ik_families
from src.scenarios.scenario_builders import build_frontal_i_barrier_lr
from src.solver.planner.collision import make_collision_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_IK  = make_condition(IKCondition.MULTI_IK_FULL,  ControlCondition.PATH_DS_FULL)
SINGLE_IK = make_condition(IKCondition.SINGLE_IK_BEST, ControlCondition.PATH_DS_FULL)
CONDITIONS = [MULTI_IK, SINGLE_IK]

COND_LABELS = {
    MULTI_IK.name:  "Multi-IK + DS",
    SINGLE_IK.name: "Single-IK + DS",
}
COLORS = {
    MULTI_IK.name:  "#2176AE",
    SINGLE_IK.name: "#E07B39",
}
DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# IK family pre-run characterisation
# ---------------------------------------------------------------------------

def run_ik_family_analysis(difficulty: str, out_dir: Path) -> dict:
    spec   = build_frontal_i_barrier_lr(difficulty)
    col_fn = make_collision_fn(spec=spec)
    report = analyse_ik_families(spec, col_fn)
    print(report)
    report.save(out_dir / "ik_family_report.json")
    return report.to_dict()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

FAST_PLANNER_OVERRIDE = {
    "max_iterations":    3_000,
    "shortcut_iterations": 50,
    "gaussian_bias":     0.20,
}
FAST_N_EXEC_STEPS = 600


def run_benchmark(
    difficulty: str,
    n_trials: int,
    seed_offset: int,
    out_dir: Path,
    fast: bool = False,
) -> List[TrialMetrics]:
    spec   = build_frontal_i_barrier_lr(difficulty)
    runner = ExperimentRunner(output_dir=out_dir, verbose=True)
    trial_kwargs = {}
    if fast:
        trial_kwargs["n_exec_steps"]    = FAST_N_EXEC_STEPS
        trial_kwargs["planner_override"] = FAST_PLANNER_OVERRIDE
    runner.run_spec(spec, CONDITIONS, n_trials=n_trials, seed_offset=seed_offset,
                    trial_kwargs=trial_kwargs)
    return runner.load_results()


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(
    difficulty: str,
    trials: List[TrialMetrics],
    n_trials: int,
    ik_family: dict,
) -> dict:
    summary: dict = {
        "scenario": f"frontal_i_barrier_lr_{difficulty}",
        "n_trials_per_condition": n_trials,
        "conditions": {},
        "ik_family_report": ik_family,
    }
    for cond in CONDITIONS:
        cname = cond.name
        ctrials = [t for t in trials if t.condition == cname and t.error is None]
        if not ctrials:
            summary["conditions"][cname] = {}
            continue

        successes = [t for t in ctrials if t.execution and t.execution.terminal_success]
        clearances = [t.barrier.min_clearance for t in ctrials
                      if t.barrier and t.barrier.min_clearance < float("inf")]
        grazes = [t.barrier.near_graze_count for t in ctrials if t.barrier]
        effic  = [t.barrier.path_efficiency for t in ctrials
                  if t.barrier and t.barrier.path_efficiency > 0]
        conv   = [t.execution.convergence_step for t in successes
                  if t.execution and t.execution.convergence_step is not None]

        plan_times = [t.plan.time_s for t in ctrials if t.plan]
        wall_times  = [t.wall_time_s for t in ctrials]
        ctrl_hz    = [t.ctrl_freq.mean_hz                        for t in ctrials if t.ctrl_freq and t.ctrl_freq.mean_hz > 0]
        ctrl_ms    = [t.ctrl_freq.mean_ms                        for t in ctrials if t.ctrl_freq]
        ctrl_p95   = [t.ctrl_freq.p95_ms                         for t in ctrials if t.ctrl_freq]
        ctrl_cmp   = [t.ctrl_freq.mean_ctrl_compute_ms           for t in ctrials if t.ctrl_freq]
        ctrl_near  = [t.ctrl_freq.mean_ctrl_ms_near_obstacle     for t in ctrials if t.ctrl_freq and t.ctrl_freq.n_steps_near_obstacle > 0]
        ctrl_far   = [t.ctrl_freq.mean_ctrl_ms_far_from_obstacle for t in ctrials if t.ctrl_freq and (t.ctrl_freq.n_steps_executed - t.ctrl_freq.n_steps_near_obstacle) > 0]
        n_exec     = [t.ctrl_freq.n_steps_executed               for t in ctrials if t.ctrl_freq]
        n_near     = [t.ctrl_freq.n_steps_near_obstacle          for t in ctrials if t.ctrl_freq]
        n_cbf      = [t.ctrl_freq.n_steps_cbf_active             for t in ctrials if t.ctrl_freq]
        plan_nodes  = [t.plan.nodes_explored for t in ctrials if t.plan]
        plan_iters  = [t.plan.iterations for t in ctrials if t.plan]
        plan_succ   = [t for t in ctrials if t.plan and t.plan.success]

        # Failure reason breakdown
        failure_reasons = [t.execution.failure_reason for t in ctrials
                           if t.execution and t.execution.failure_reason]
        reason_counts = {}
        for r in failure_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        # New success diagnostic fields
        min_errs = [t.execution.min_goal_error_ever for t in ctrials
                    if t.execution and t.execution.min_goal_error_ever < float("inf")]
        path_prog = [t.execution.path_progress_fraction for t in ctrials
                     if t.execution]
        pt_verdicts = [t.execution.path_tracking_verdict for t in ctrials
                       if t.execution and t.execution.path_tracking_verdict]
        mik_effects = [t.execution.multiik_effect for t in ctrials
                       if t.execution and t.execution.multiik_effect]
        mean_dev = [t.execution.mean_path_deviation_q for t in ctrials if t.execution]
        max_dev  = [t.execution.max_path_deviation_q  for t in ctrials if t.execution]

        summary["conditions"][cname] = {
            "success_rate": len(successes) / max(1, len(ctrials)),
            "mean_min_clearance": float(np.mean(clearances)) if clearances else None,
            "mean_near_graze_count": float(np.mean(grazes)) if grazes else None,
            "mean_time_to_goal_steps": float(np.mean(conv)) if conv else None,
            "timeout_rate": 1.0 - len(successes) / max(1, len(ctrials)),
            "mean_path_efficiency": float(np.mean(effic)) if effic else None,
            # Planning overhead
            "mean_plan_time_s": float(np.mean(plan_times)) if plan_times else None,
            "median_plan_time_s": float(np.median(plan_times)) if plan_times else None,
            "mean_wall_time_s": float(np.mean(wall_times)) if wall_times else None,
            "mean_ctrl_hz":                  float(np.mean(ctrl_hz))   if ctrl_hz   else None,
            "mean_ctrl_ms":                  float(np.mean(ctrl_ms))   if ctrl_ms   else None,
            "p95_ctrl_ms":                   float(np.mean(ctrl_p95))  if ctrl_p95  else None,
            "mean_ctrl_compute_ms":          float(np.mean(ctrl_cmp))  if ctrl_cmp  else None,
            "mean_ctrl_ms_near_obstacle":    float(np.mean(ctrl_near)) if ctrl_near else None,
            "mean_ctrl_ms_far_from_obstacle":float(np.mean(ctrl_far))  if ctrl_far  else None,
            "mean_n_steps_executed":         float(np.mean(n_exec))    if n_exec    else None,
            "mean_n_steps_near_obstacle":    float(np.mean(n_near))    if n_near    else None,
            "mean_n_steps_cbf_active":       float(np.mean(n_cbf))     if n_cbf     else None,
            "plan_success_rate": len(plan_succ) / max(1, len(ctrials)),
            "mean_nodes_explored": float(np.mean(plan_nodes)) if plan_nodes else None,
            "mean_planner_iters": float(np.mean(plan_iters)) if plan_iters else None,
            # Failure reason breakdown
            "failure_reason_counts": reason_counts,
            # Success diagnostics
            "mean_min_goal_error_ever": float(np.mean(min_errs)) if min_errs else None,
            "mean_path_progress_fraction": float(np.mean(path_prog)) if path_prog else None,
            "mean_path_deviation_q": float(np.mean(mean_dev)) if mean_dev else None,
            "mean_max_path_deviation_q": float(np.mean(max_dev)) if max_dev else None,
            # Verdict distributions
            "path_tracking_verdicts": {v: pt_verdicts.count(v) for v in set(pt_verdicts)},
            "multiik_effect_counts": {v: mik_effects.count(v) for v in set(mik_effects)},
        }
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _cond_trials(trials, cond_name, difficulty):
    return [t for t in trials
            if t.condition == cond_name
            and t.scenario == f"frontal_i_barrier_lr_{difficulty}"
            and t.error is None]


def plot_success_rate(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(difficulties))
    width = 0.35
    for k, cond in enumerate(CONDITIONS):
        rates = []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            succ = [t for t in ct if t.execution and t.execution.terminal_success]
            rates.append(len(succ) / max(1, len(ct)))
        ax.bar(x + k * width, rates, width, label=COND_LABELS[cond.name],
               color=COLORS[cond.name])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate by method and difficulty")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate.png", dpi=120)
    plt.close(fig)


def plot_min_clearance(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = []
    data_all  = []
    tick_labels = []
    colors_all  = []
    pos = 1
    for diff in difficulties:
        for cond in CONDITIONS:
            ct = _cond_trials(trials, cond.name, diff)
            vals = [t.barrier.min_clearance for t in ct
                    if t.barrier and t.barrier.min_clearance < float("inf")]
            if vals:
                data_all.append(vals)
                positions.append(pos)
                tick_labels.append(f"{diff}\n{COND_LABELS[cond.name][:5]}")
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
    ax.set_title("Minimum obstacle clearance per trial")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "min_clearance.png", dpi=120)
    plt.close(fig)


def plot_time_to_goal(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(difficulties))
    width = 0.35
    for k, cond in enumerate(CONDITIONS):
        means, errs = [], []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            conv = [t.execution.convergence_step for t in ct
                    if t.execution and t.execution.convergence_step is not None]
            means.append(float(np.mean(conv)) if conv else 0.0)
            errs.append(float(np.std(conv)) if conv else 0.0)
        ax.bar(x + k * width, means, width, yerr=errs,
               label=COND_LABELS[cond.name], color=COLORS[cond.name],
               alpha=0.85, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Convergence step (mean ± std, successes only)")
    ax.set_title("Time to goal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_to_goal.png", dpi=120)
    plt.close(fig)


def plot_final_goal_error(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    positions, data_all, tick_labels, colors_all = [], [], [], []
    pos = 1
    for diff in difficulties:
        for cond in CONDITIONS:
            ct  = _cond_trials(trials, cond.name, diff)
            err = [t.execution.final_goal_err for t in ct
                   if t.execution and t.execution.final_goal_err < float("inf")]
            if err:
                data_all.append(err)
                positions.append(pos)
                tick_labels.append(f"{diff}\n{COND_LABELS[cond.name][:5]}")
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
    ax.set_ylabel("Final goal error (rad)")
    ax.set_title("Final goal error — all trials")
    fig.tight_layout()
    fig.savefig(out_dir / "final_goal_error.png", dpi=120)
    plt.close(fig)


def plot_near_graze_count(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(difficulties))
    width = 0.35
    for k, cond in enumerate(CONDITIONS):
        means, errs = [], []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            gc = [t.barrier.near_graze_count for t in ct if t.barrier]
            means.append(float(np.mean(gc)) if gc else 0.0)
            errs.append(float(np.std(gc)) if gc else 0.0)
        ax.bar(x + k * width, means, width, yerr=errs,
               label=COND_LABELS[cond.name], color=COLORS[cond.name],
               alpha=0.85, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Near-graze steps (mean ± std)")
    ax.set_title("Near-graze events (clearance < 15 mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "near_graze_count.png", dpi=120)
    plt.close(fig)


def plot_path_efficiency(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond in CONDITIONS:
        pl_lens, ex_lens = [], []
        for diff in difficulties:
            ct = _cond_trials(trials, cond.name, diff)
            for t in ct:
                if t.plan and t.barrier and t.plan.path_length > 0:
                    pl_lens.append(t.plan.path_length)
                    ex_lens.append(t.barrier.joint_path_length)
        if pl_lens:
            ax.scatter(pl_lens, ex_lens, label=COND_LABELS[cond.name],
                       color=COLORS[cond.name], alpha=0.6, s=20)
    # Diagonal (perfect efficiency)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.01)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="planned = executed")
    ax.set_xlabel("Planned joint-space path length (rad)")
    ax.set_ylabel("Executed joint-space path length (rad)")
    ax.set_title("Planned vs executed path length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "path_efficiency.png", dpi=120)
    plt.close(fig)


def plot_ik_survival(trials, difficulties, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(difficulties))
    width = 0.5
    for diff_i, diff in enumerate(difficulties):
        ct = _cond_trials(trials, MULTI_IK.name, diff)
        n_raw  = [t.ik.n_raw  for t in ct if t.ik]
        n_safe = [t.ik.n_safe for t in ct if t.ik]
        if not n_raw:
            continue
        mean_raw  = float(np.mean(n_raw))
        mean_safe = float(np.mean(n_safe))
        mean_blk  = mean_raw - mean_safe
        ax.bar(diff_i, mean_safe, width, color="#2176AE", alpha=0.85,
               label="Safe" if diff_i == 0 else "")
        ax.bar(diff_i, mean_blk, width, bottom=mean_safe, color="#C0392B",
               alpha=0.6, label="Blocked" if diff_i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Mean IK goals per trial")
    ax.set_title("IK goal survival — Multi-IK condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ik_survival.png", dpi=120)
    plt.close(fig)


def generate_all_plots(trials, difficulties, out_dir: Path):
    plot_success_rate(trials, difficulties, out_dir)
    plot_min_clearance(trials, difficulties, out_dir)
    plot_time_to_goal(trials, difficulties, out_dir)
    plot_final_goal_error(trials, difficulties, out_dir)
    plot_near_graze_count(trials, difficulties, out_dir)
    plot_path_efficiency(trials, difficulties, out_dir)
    plot_ik_survival(trials, difficulties, out_dir)
    print(f"  7 plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Frontal I-barrier benchmark: Multi-IK vs Single-IK",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "all"], default="medium",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/eval/c_barrier"),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced planner budget (3k iters, 50 shortcut) and 600 exec steps. "
             "Good for quick sanity-checks; not publication quality.",
    )
    parser.add_argument(
        "--skip-ik-analysis", action="store_true",
        help="Skip the IK family pre-characterisation step (saves ~30s per difficulty).",
    )
    args = parser.parse_args()

    diffs = DIFFICULTIES if args.difficulty == "all" else [args.difficulty]
    all_trials: List[TrialMetrics] = []

    if args.fast:
        print(f"[fast mode] n_exec_steps={FAST_N_EXEC_STEPS}, "
              f"max_planner_iters={FAST_PLANNER_OVERRIDE['max_iterations']}, "
              f"shortcut_iters={FAST_PLANNER_OVERRIDE['shortcut_iterations']}")

    for diff in diffs:
        out_dir = args.output_dir / diff
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Difficulty: {diff} ===")

        if args.skip_ik_analysis:
            ik_family = {}
        else:
            ik_family = run_ik_family_analysis(diff, out_dir)

        trials = run_benchmark(diff, args.n_trials, args.seed_offset, out_dir,
                               fast=args.fast)
        all_trials.extend(trials)

        summary = build_summary(diff, trials, args.n_trials, ik_family)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  summary.json written")

        generate_all_plots(trials, [diff], out_dir)

    if args.difficulty == "all" and len(diffs) > 1:
        combined_dir = args.output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(all_trials, DIFFICULTIES, combined_dir)
        print(f"\nCombined plots saved to {combined_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
