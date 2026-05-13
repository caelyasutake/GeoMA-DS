"""
Robustness Benchmark (Phase 6 of evaluation plan).

Stress-tests the solver across:
  - obstacle density variation (n_obstacles sweep)
  - passage width variation (narrow_passage wall thickness)
  - IK set size variation (limiting n_goals available)
  - state noise injection
  - external perturbation magnitude sweep

Usage::

    python -m benchmarks.eval_robustness
    python -m benchmarks.eval_robustness --n-trials 10 --output-dir outputs/eval/quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.aggregators import aggregate_results, MetricStats
from src.evaluation.baselines import ControlCondition, IKCondition, make_condition
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import TrialMetrics, load_jsonl
from src.evaluation.trial_runner import run_planning_trial
from src.scenarios.scenario_builders import (
    narrow_passage_scenario, random_obstacle_field_scenario, free_space_scenario,
)


def _rate(trials: List[TrialMetrics], key: str) -> float:
    vals = []
    for t in trials:
        if key == "plan_success" and t.plan:
            vals.append(float(t.plan.success))
        elif key == "term_success" and t.execution:
            vals.append(float(t.execution.terminal_success))
    return float(np.mean(vals)) if vals else 0.0


def run_obstacle_density_sweep(
    output_dir: Path,
    n_trials: int = 20,
    obstacle_counts: List[int] = None,
    verbose: bool = True,
) -> None:
    """Success vs number of random obstacles."""
    if obstacle_counts is None:
        obstacle_counts = [0, 1, 2, 3, 4, 6, 8]

    output_dir = Path(output_dir) / "obstacle_density"
    output_dir.mkdir(parents=True, exist_ok=True)

    cond = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)
    plan_rates = []
    term_rates = []

    for n_obs in obstacle_counts:
        spec = random_obstacle_field_scenario(seed=0, n_obstacles=n_obs)
        trials_n = []
        for seed in range(n_trials):
            t = run_planning_trial(spec, cond, seed=seed, trial_id=seed)
            trials_n.append(t)
        plan_rates.append(_rate(trials_n, "plan_success"))
        term_rates.append(_rate(trials_n, "term_success"))
        if verbose:
            print(f"  n_obs={n_obs}: plan={plan_rates[-1]:.2f} term={term_rates[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(obstacle_counts, [r * 100 for r in plan_rates],  "o-", label="Plan success %")
    ax.plot(obstacle_counts, [r * 100 for r in term_rates],  "s--", label="Term. success %")
    ax.set_xlabel("Number of obstacles"); ax.set_ylabel("Success rate (%)")
    ax.set_title("Robustness vs Obstacle Density"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    plots_dir = output_dir.parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "robustness_obstacle_density.png", dpi=120)
    plt.close(fig)
    print("[robustness] Saved robustness_obstacle_density.png")


def run_noise_sweep(
    output_dir: Path,
    n_trials: int = 20,
    noise_levels: List[float] = None,
    verbose: bool = True,
) -> None:
    """Terminal success vs state noise std."""
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    output_dir = Path(output_dir) / "noise_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    cond = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)
    spec = free_space_scenario()
    term_rates = []

    for noise in noise_levels:
        trials_n = []
        for seed in range(n_trials):
            t = run_planning_trial(spec, cond, seed=seed, trial_id=seed, noise_std=noise)
            trials_n.append(t)
        r = _rate(trials_n, "term_success")
        term_rates.append(r)
        if verbose:
            print(f"  noise={noise:.4f}: term={r:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, [r * 100 for r in term_rates], "o-", color="#1976D2")
    ax.set_xlabel("State noise std (rad)"); ax.set_ylabel("Terminal success (%)")
    ax.set_title("Robustness vs State Noise"); ax.grid(alpha=0.3)
    fig.tight_layout()
    plots_dir = output_dir.parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "robustness_noise.png", dpi=120)
    plt.close(fig)
    print("[robustness] Saved robustness_noise.png")


def run_perturbation_sweep(
    output_dir: Path,
    n_trials: int = 20,
    magnitudes: List[float] = None,
    verbose: bool = True,
) -> None:
    """Terminal success vs perturbation magnitude."""
    if magnitudes is None:
        magnitudes = [0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0]

    output_dir = Path(output_dir) / "perturb_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    cond = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)
    spec = free_space_scenario()
    term_rates = []

    for mag in magnitudes:
        trials_n = []
        for seed in range(n_trials):
            t = run_planning_trial(
                spec, cond, seed=seed, trial_id=seed,
                perturb_magnitude=mag, perturb_start=100, perturb_duration=30,
            )
            trials_n.append(t)
        r = _rate(trials_n, "term_success")
        term_rates.append(r)
        if verbose:
            print(f"  perturb={mag:.1f} Nm: term={r:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(magnitudes, [r * 100 for r in term_rates], "o-", color="#E53935")
    ax.set_xlabel("Perturbation magnitude (Nm)"); ax.set_ylabel("Terminal success (%)")
    ax.set_title("Robustness vs Perturbation"); ax.grid(alpha=0.3)
    fig.tight_layout()
    plots_dir = output_dir.parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "robustness_perturbation.png", dpi=120)
    plt.close(fig)
    print("[robustness] Saved robustness_perturbation.png")


def run_ik_size_sweep(
    output_dir: Path,
    n_trials: int = 20,
    ik_sizes: List[int] = None,
    verbose: bool = True,
) -> None:
    """Success vs IK goal set size."""
    if ik_sizes is None:
        ik_sizes = [1, 2, 3, 4, 5]

    output_dir = Path(output_dir) / "ik_size_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.baselines import select_ik_goals
    from src.scenarios.scenario_builders import narrow_passage_scenario

    spec_full = free_space_scenario()
    full_goals = spec_full.ik_goals

    cond_base = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)

    plan_rates = []
    for k in ik_sizes:
        trials_k = []
        for seed in range(n_trials):
            import dataclasses, copy
            spec_k = copy.deepcopy(spec_full)
            spec_k = dataclasses.replace(spec_k, ik_goals=full_goals[:k])
            t = run_planning_trial(spec_k, cond_base, seed=seed, trial_id=seed)
            trials_k.append(t)
        r = _rate(trials_k, "plan_success")
        plan_rates.append(r)
        if verbose:
            print(f"  k={k}: plan={r:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ik_sizes, [r * 100 for r in plan_rates], "o-", color="#43A047")
    ax.set_xlabel("IK goal set size (k)"); ax.set_ylabel("Planning success (%)")
    ax.set_title("Success vs IK Set Size"); ax.grid(alpha=0.3)
    fig.tight_layout()
    plots_dir = output_dir.parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "robustness_ik_size.png", dpi=120)
    plt.close(fig)
    print("[robustness] Saved robustness_ik_size.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robustness sweep benchmark")
    parser.add_argument("--n-trials",     type=int,  default=20)
    parser.add_argument("--output-dir",   type=Path, default=Path("outputs/eval/robustness"))
    parser.add_argument("--no-density",   action="store_true", help="Skip obstacle density sweep")
    parser.add_argument("--no-noise",     action="store_true", help="Skip noise sweep")
    parser.add_argument("--no-perturb",   action="store_true", help="Skip perturbation sweep")
    parser.add_argument("--no-ik-size",   action="store_true", help="Skip IK size sweep")
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args(argv)

    out  = Path(args.output_dir)
    verb = not args.quiet
    n    = args.n_trials

    t0 = time.perf_counter()

    if not args.no_density:
        print("\n[robustness] Obstacle density sweep")
        run_obstacle_density_sweep(out, n_trials=n, verbose=verb)

    if not args.no_noise:
        print("\n[robustness] Noise sweep")
        run_noise_sweep(out, n_trials=n, verbose=verb)

    if not args.no_perturb:
        print("\n[robustness] Perturbation sweep")
        run_perturbation_sweep(out, n_trials=n, verbose=verb)

    if not args.no_ik_size:
        print("\n[robustness] IK size sweep")
        run_ik_size_sweep(out, n_trials=n, verbose=verb)

    print(f"\n[robustness] Done in {time.perf_counter() - t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
