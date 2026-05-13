"""
Contact-Task Evaluation Benchmark (Phase 8 of evaluation plan).

Runs the contact-circle scenarios under all contact controller conditions.

Conditions:
  - task_tracking_full          (full contact circle controller)
  - task_tracking_no_force_reg  (tangential only, K_f=0)
  - task_tracking_no_passivity  (no tank / filter)
  - joint_space_path_only       (no task-space circle)

Scenarios:
  - contact_circle              (nominal)
  - contact_circle_perturbation (with perturbation)

Also sweeps omega (angular speed) to characterise tracking vs speed.

Usage::

    python -m benchmarks.eval_contact
    python -m benchmarks.eval_contact --n-trials 10 --slide-steps 800
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

from src.evaluation.aggregators import aggregate_results, save_aggregate
from src.evaluation.baselines import (
    ControlCondition, IKCondition, make_condition,
    CONTACT_CTRL_CONDITIONS,
)
from src.evaluation.experiment_runner import ExperimentRunner, build_contact_conditions
from src.evaluation.report_writer import write_report


OMEGA_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0]   # rad/s


def run_contact_eval(
    n_trials: int = 20,
    output_dir: Path = Path("outputs/eval/contact"),
    seed_offset: int = 0,
    slide_steps: int = 1600,
    omega: float = 1.5,
    verbose: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_contact_conditions(ik=IKCondition.MULTI_IK_FULL)
    runner = ExperimentRunner(output_dir=output_dir, verbose=verbose)

    trial_kwargs = dict(slide_steps=slide_steps, omega=omega)

    t0 = time.perf_counter()
    for scen in ("contact_circle", "contact_circle_perturbation"):
        print(f"\n[contact] Scenario: {scen} — {len(conditions)} conds × {n_trials} trials")
        runner.run(
            scen, conditions, n_trials=n_trials,
            seed_offset=seed_offset, trial_kwargs=trial_kwargs,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n[contact] Finished in {elapsed:.1f}s")

    trials = runner.load_results()
    aggs   = aggregate_results(trials)
    save_aggregate(aggs, output_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_contact_metrics(aggs, trials, plots_dir)

    write_report(aggs, trials, output_dir)
    print(f"[contact] Outputs → {output_dir}")


def run_omega_sweep(
    output_dir: Path = Path("outputs/eval/contact/omega_sweep"),
    n_trials: int = 10,
    seed_offset: int = 0,
    slide_steps: int = 1600,
    verbose: bool = True,
) -> None:
    """Sweep omega to characterise tracking capability vs angular speed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_cond = [make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.TASK_TRACKING_FULL)]

    rmse_by_omega  = []
    arc_by_omega   = []
    cont_by_omega  = []

    for om in OMEGA_SWEEP:
        runner = ExperimentRunner(
            output_dir=output_dir / f"omega_{om:.1f}",
            verbose=verbose,
        )
        runner.run(
            "contact_circle", full_cond, n_trials=n_trials,
            seed_offset=seed_offset,
            trial_kwargs=dict(slide_steps=slide_steps, omega=om),
        )
        trials = runner.load_results()
        aggs   = aggregate_results(trials)
        key    = ("contact_circle", full_cond[0].name)
        agg    = aggs.get(key)
        if agg:
            rmse_by_omega.append(agg.circle_rmse.mean or float("nan"))
            arc_by_omega.append(agg.arc_ratio.mean or 0.0)
            cont_by_omega.append(agg.contact_maintained.mean or 0.0)
        else:
            rmse_by_omega.append(float("nan"))
            arc_by_omega.append(0.0)
            cont_by_omega.append(0.0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, ylabel in zip(
        axes,
        [rmse_by_omega, arc_by_omega, cont_by_omega],
        ["Circle RMSE (m)", "Arc Completion Ratio", "Contact Maintained Fraction"],
    ):
        ax.plot(OMEGA_SWEEP, vals, "o-", color="#1976D2")
        ax.set_xlabel("ω (rad/s)"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Contact Tracking vs Angular Speed (ω sweep)", fontsize=11)
    fig.tight_layout()
    plots_dir = output_dir.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "omega_sweep.png", dpi=120)
    plt.close(fig)
    print("[contact] Saved omega_sweep.png")


def _plot_contact_metrics(aggs, trials, plots_dir: Path) -> None:
    cond_vals = [c.value for c in CONTACT_CTRL_CONDITIONS]
    scenarios = ["contact_circle", "contact_circle_perturbation"]

    fig, axes = plt.subplots(3, len(scenarios), figsize=(7 * len(scenarios), 10))
    if len(scenarios) == 1:
        axes = axes.reshape(3, 1)

    for col, scen in enumerate(scenarios):
        contact_rates  = []
        contact_fracs  = []
        rmse_vals      = []
        arc_vals       = []

        for ctrl in CONTACT_CTRL_CONDITIONS:
            cond_key = (scen, f"multi_ik_full__{ctrl.value}")
            agg = aggs.get(cond_key)
            if agg:
                contact_rates.append(agg.contact_established_rate * 100)
                contact_fracs.append(agg.contact_maintained.mean or 0.0)
                rmse_vals.append(agg.circle_rmse.mean or float("nan"))
                arc_vals.append(agg.arc_ratio.mean or 0.0)
            else:
                contact_rates.append(0.0); contact_fracs.append(0.0)
                rmse_vals.append(float("nan")); arc_vals.append(0.0)

        x      = np.arange(len(cond_vals))
        labels = [c.split("__")[-1] for c in cond_vals]

        axes[0, col].bar(x, contact_rates, color="#E53935", alpha=0.85)
        axes[0, col].set_title(scen, fontsize=10)
        axes[0, col].set_xticks(x); axes[0, col].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        axes[0, col].set_ylabel("Contact established (%)"); axes[0, col].set_ylim(0, 110)

        axes[1, col].bar(x, contact_fracs, color="#1976D2", alpha=0.85)
        axes[1, col].set_xticks(x); axes[1, col].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        axes[1, col].set_ylabel("Contact maintained fraction"); axes[1, col].set_ylim(0, 1)

        rmse_clean = [v if np.isfinite(v) else 0.0 for v in rmse_vals]
        axes[2, col].bar(x, rmse_clean, color="#43A047", alpha=0.85)
        axes[2, col].set_xticks(x); axes[2, col].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        axes[2, col].set_ylabel("Circle RMSE (m)")

    fig.suptitle("Contact Task: Controller Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "contact_metrics.png", dpi=120)
    plt.close(fig)
    print("[contact] Saved contact_metrics.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Contact-task evaluation benchmark")
    parser.add_argument("--n-trials",    type=int,  default=20)
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs/eval/contact"))
    parser.add_argument("--seed-offset", type=int,  default=0)
    parser.add_argument("--slide-steps", type=int,  default=1600)
    parser.add_argument("--omega",       type=float, default=1.5)
    parser.add_argument("--omega-sweep", action="store_true", help="Run omega sweep too")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args(argv)

    run_contact_eval(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed_offset=args.seed_offset,
        slide_steps=args.slide_steps,
        omega=args.omega,
        verbose=not args.quiet,
    )

    if args.omega_sweep:
        run_omega_sweep(
            output_dir=args.output_dir / "omega_sweep",
            n_trials=max(5, args.n_trials // 4),
            slide_steps=args.slide_steps,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
