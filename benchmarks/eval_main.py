"""
Main evaluation entry point — runs all phases sequentially.

Usage::

    # Full evaluation (slow: ~50 trials × many conditions)
    python -m benchmarks.eval_main

    # Quick smoke-test with 3 trials
    python -m benchmarks.eval_main --n-trials 3 --output-dir outputs/eval/smoke

    # Run only specific phases
    python -m benchmarks.eval_main --phases ablation baselines

    # Run only the U-shape difficulty sweep
    python -m benchmarks.eval_main --phases u_shape_sweep

    # Run only the left-opening barrier benchmark
    python -m benchmarks.eval_main --phases left_open_u_sweep

Available phases:
    ablation           — Multi-IK vs single-IK ablation (Q2)
    baselines          — Controller baseline comparison (Q3)
    contact            — Contact-circle task evaluation (Phase 8)
    robustness         — Robustness sweeps (Phase 6)
    u_shape_sweep      — U-shape difficulty sweep (opening width × depth)
    left_open_u_sweep  — Left-to-right barrier multi-IK advantage sweep
    report             — Re-generate report from existing JSONL files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from src.evaluation.aggregators import aggregate_results, save_aggregate
from src.evaluation.experiment_runner import (
    ExperimentRunner,
    SCENARIO_BUILDERS, CONTACT_SCENARIOS,
    build_ablation_conditions, build_baseline_conditions, build_contact_conditions,
)
from src.evaluation.baselines import IKCondition, ControlCondition, make_condition
from src.evaluation.metrics import load_jsonl
from src.evaluation.report_writer import write_report


ALL_PHASES = ["ablation", "baselines", "contact", "robustness", "u_shape_sweep", "left_open_u_sweep"]


def phase_ablation(args) -> None:
    from benchmarks.eval_multik_ablation import run_ablation
    run_ablation(
        n_trials=args.n_trials,
        output_dir=args.output_dir / "ablation",
        seed_offset=args.seed_offset,
        verbose=args.verbose,
    )


def phase_baselines(args) -> None:
    from benchmarks.eval_baselines import run_baselines
    run_baselines(
        n_trials=args.n_trials,
        output_dir=args.output_dir / "baselines",
        seed_offset=args.seed_offset,
        verbose=args.verbose,
    )


def phase_contact(args) -> None:
    from benchmarks.eval_contact import run_contact_eval
    run_contact_eval(
        n_trials=max(1, args.n_trials // 2),   # contact trials are slower
        output_dir=args.output_dir / "contact",
        seed_offset=args.seed_offset,
        slide_steps=args.slide_steps,
        omega=args.omega,
        verbose=args.verbose,
    )


def phase_robustness(args) -> None:
    from benchmarks.eval_robustness import (
        run_obstacle_density_sweep, run_noise_sweep,
        run_perturbation_sweep, run_ik_size_sweep,
    )
    rob_dir = args.output_dir / "robustness"
    n = max(5, args.n_trials // 2)
    run_obstacle_density_sweep(rob_dir, n_trials=n, verbose=args.verbose)
    run_noise_sweep(rob_dir, n_trials=n, verbose=args.verbose)
    run_perturbation_sweep(rob_dir, n_trials=n, verbose=args.verbose)
    run_ik_size_sweep(rob_dir, n_trials=n, verbose=args.verbose)


def phase_u_shape_sweep(args) -> None:
    """Sweep U-shape geometry parameters and measure multi-IK advantage."""
    from benchmarks.eval_u_shape_sweep import run_u_shape_sweep
    run_u_shape_sweep(
        n_trials=max(3, args.n_trials // 5),
        output_dir=args.output_dir / "u_shape_sweep",
        seed_offset=args.seed_offset,
        verbose=args.verbose,
    )


def phase_left_open_u_sweep(args) -> None:
    """Evaluate multi-IK advantage on the left-to-right barrier scenario."""
    from src.evaluation.experiment_runner import ExperimentRunner, build_ablation_conditions
    from src.evaluation.baselines import ControlCondition
    from src.evaluation.aggregators import aggregate_results, save_aggregate

    out = args.output_dir / "left_open_u_sweep"
    out.mkdir(parents=True, exist_ok=True)

    conditions = build_ablation_conditions(ctrl=ControlCondition.PATH_DS_FULL)
    runner = ExperimentRunner(output_dir=out, verbose=args.verbose)

    n = max(3, args.n_trials // 3)
    print(f"[left_open_u_sweep] {len(conditions)} conds × {n} trials")
    runner.run("left_open_u", conditions, n_trials=n,
               seed_offset=args.seed_offset)

    trials = runner.load_results()
    aggs   = aggregate_results(trials)
    save_aggregate(aggs, out)
    print(f"[left_open_u_sweep] Outputs → {out}")


def phase_report(args) -> None:
    """Aggregate all sub-directories and write a combined report."""
    all_trials = []
    for sub in ["ablation", "baselines", "contact"]:
        jsonl = args.output_dir / sub / "per_trial_results.jsonl"
        if jsonl.exists():
            all_trials.extend(load_jsonl(jsonl))

    if not all_trials:
        print("[report] No trial data found — run evaluation phases first.")
        return

    aggs = aggregate_results(all_trials)
    combined_dir = args.output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    save_aggregate(aggs, combined_dir)
    write_report(aggs, all_trials, combined_dir)
    print(f"[report] Combined report → {combined_dir}/reports/")


PHASE_FNS = {
    "ablation":          phase_ablation,
    "baselines":         phase_baselines,
    "contact":           phase_contact,
    "robustness":        phase_robustness,
    "u_shape_sweep":     phase_u_shape_sweep,
    "left_open_u_sweep": phase_left_open_u_sweep,
    "report":            phase_report,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-IK-DS full evaluation suite"
    )
    parser.add_argument(
        "--phases", nargs="+",
        choices=list(PHASE_FNS), default=ALL_PHASES,
        help="Which evaluation phases to run (default: all)",
    )
    parser.add_argument("--n-trials",    type=int,   default=50,
                        help="Trials per (condition × scenario)")
    parser.add_argument("--output-dir",  type=Path,  default=Path("outputs/eval"),
                        help="Root output directory")
    parser.add_argument("--seed-offset", type=int,   default=0,
                        help="Base RNG seed")
    parser.add_argument("--slide-steps", type=int,   default=1600,
                        help="Contact slide steps")
    parser.add_argument("--omega",       type=float, default=1.5,
                        help="Contact circle angular velocity (rad/s)")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args(argv)
    args.verbose = not args.quiet

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "phases":      args.phases,
        "n_trials":    args.n_trials,
        "output_dir":  str(args.output_dir),
        "seed_offset": args.seed_offset,
        "slide_steps": args.slide_steps,
        "omega":       args.omega,
    }
    with open(args.output_dir / "eval_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    t0 = time.perf_counter()
    for phase in args.phases:
        print(f"\n{'='*60}")
        print(f"  Phase: {phase.upper()}")
        print(f"{'='*60}")
        PHASE_FNS[phase](args)

    # Always run combined report at the end
    if "report" not in args.phases:
        phase_report(args)

    total = time.perf_counter() - t0
    print(f"\n[eval_main] All phases complete in {total:.1f}s → {args.output_dir}")


if __name__ == "__main__":
    main()
