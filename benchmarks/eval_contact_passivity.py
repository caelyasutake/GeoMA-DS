"""
Contact-passivity approach-comparison benchmark.

Compares four approach methods on contact establishment quality against a
horizontal wall. All conditions use the same task_space_step contact
controller (TASK_TRACKING_FULL) so differences reflect the approach only.

Conditions:
  1. Vanilla DS       — straight PathDS, no planner
  2. Single-IK+BiRRT  — BiRRT + PathDS to computed approach config
  3. Multi-IK+BiRRT   — BiRRT (multi-goal) + PathDS
  4. DiffIK           — task-space DS via Jacobian diff-IK

Metrics compared:
  * time_to_first_contact_steps  — how quickly each method reaches the wall
  * impact_velocity_norm         — EE speed at moment of contact
  * peak_force_on_impact         — force spike on first contact
  * contact_maintained_fraction  — fraction of slide phase in contact
  * normal_force_error_rmse      — force regulation quality
  * contact_chatter_index        — stability of contact (lower = better)
  * recovered_after_perturbation — resilience after disturbance
  * integrated_positive_power    — energy injected through passivity filter
  * min_tank_energy              — how depleted the energy tank gets

Usage::

    python -m benchmarks.eval_contact_passivity
    python -m benchmarks.eval_contact_passivity --n-trials 20
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

from src.evaluation.baselines import (
    IKCondition, ControlCondition, make_condition,
)
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.metrics import TrialMetrics, load_jsonl
from src.evaluation.trial_runner import run_contact_passivity_trial
from src.scenarios.scenario_builders import contact_passivity_wall_scenario


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

VANILLA = make_condition(IKCondition.VANILLA_DS,      ControlCondition.TASK_TRACKING_FULL)
SINGLE  = make_condition(IKCondition.SINGLE_IK_BEST,  ControlCondition.TASK_TRACKING_FULL)
MULTI   = make_condition(IKCondition.MULTI_IK_FULL,   ControlCondition.TASK_TRACKING_FULL)
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

def run_benchmark(
    n_trials: int,
    seed_offset: int,
    out_dir: Path,
) -> List[TrialMetrics]:
    spec = contact_passivity_wall_scenario()
    all_trials: List[TrialMetrics] = []

    for k, cond in enumerate(CONDITIONS):
        print(f"  [{k+1}/{len(CONDITIONS)}] {LABELS[cond.name]} ...")
        for trial_id in range(n_trials):
            seed = seed_offset + trial_id
            tm   = run_contact_passivity_trial(
                spec, cond, seed, trial_id=trial_id,
            )
            all_trials.append(tm)
            status = "ok" if tm.error is None else "ERR"
            succ = (tm.contact and tm.contact.contact_established) if tm.error is None else False
            print(f"    trial {trial_id}: {status}  contact={'yes' if succ else 'no'}")

    # Persist to JSONL
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trials.jsonl"
    with jsonl_path.open("w") as fh:
        for tm in all_trials:
            fh.write(json.dumps(tm.to_dict()) + "\n")

    return all_trials


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _cond_trials(trials: List[TrialMetrics], cond_name: str) -> List[TrialMetrics]:
    return [t for t in trials if t.condition == cond_name and t.error is None]


def build_summary(trials: List[TrialMetrics], n_trials: int) -> dict:
    summary: dict = {
        "scenario": "contact_passivity_wall",
        "n_trials_per_condition": n_trials,
        "conditions": {},
    }
    for cond in CONDITIONS:
        ct = _cond_trials(trials, cond.name)
        if not ct:
            summary["conditions"][cond.name] = {}
            continue

        contacts = [t for t in ct if t.contact and t.contact.contact_established]
        n_contact = len(contacts)

        def _contact_field(field: str):
            return [getattr(t.contact, field) for t in contacts if t.contact]

        ttc      = _contact_field("time_to_first_contact_steps")
        vel      = _contact_field("impact_velocity_norm")
        peak_f   = _contact_field("peak_force_on_impact")
        maint    = _contact_field("contact_maintained_fraction")
        nf_rmse  = [v for v in _contact_field("normal_force_error_rmse") if v < float("inf")]
        chatter  = _contact_field("contact_chatter_index")
        recov    = _contact_field("recovered_after_perturbation")
        int_pow  = _contact_field("integrated_positive_power")
        tank_min = [t.passivity.min_tank_energy for t in ct if t.passivity]

        summary["conditions"][cond.name] = {
            "label": LABELS[cond.name],
            "n_trials": len(ct),
            "contact_rate": n_contact / max(1, len(ct)),
            "mean_time_to_contact_steps": float(np.mean(ttc)) if ttc else None,
            "mean_impact_velocity_m_s":   float(np.mean(vel))  if vel  else None,
            "mean_peak_force_N":          float(np.mean(peak_f)) if peak_f else None,
            "mean_contact_maintained":    float(np.mean(maint)) if maint else None,
            "mean_normal_force_rmse_N":   float(np.mean(nf_rmse)) if nf_rmse else None,
            "mean_chatter_index":         float(np.mean(chatter)) if chatter else None,
            "recovery_rate":              float(np.mean(recov)) if recov else None,
            "mean_integrated_power_J":    float(np.mean(int_pow)) if int_pow else None,
            "mean_min_tank_energy":       float(np.mean(tank_min)) if tank_min else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _boxplot(ax, data_by_cond: dict, ylabel: str, title: str, hline=None):
    labels  = [LABELS[c.name] for c in CONDITIONS if data_by_cond.get(c.name)]
    data    = [data_by_cond[c.name] for c in CONDITIONS if data_by_cond.get(c.name)]
    colors  = [COLORS[c.name]      for c in CONDITIONS if data_by_cond.get(c.name)]
    if not data:
        return
    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    if hline is not None:
        ax.axhline(hline[0], color=hline[1], linestyle="--", linewidth=0.8, label=hline[2])
        ax.legend(fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=7)


def plot_all(trials: List[TrialMetrics], out_dir: Path):
    def _get(field: str, cond_name: str):
        ct = _cond_trials(trials, cond_name)
        vals = []
        for t in ct:
            if t.contact:
                v = getattr(t.contact, field, None)
                if v is not None and v < float("inf"):
                    vals.append(float(v))
        return vals

    def _pass(field: str, cond_name: str):
        ct = _cond_trials(trials, cond_name)
        return [getattr(t.passivity, field, None) for t in ct
                if t.passivity and getattr(t.passivity, field, None) is not None]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    metrics = [
        ("time_to_first_contact_steps", "Steps",   "Time to first contact",      None),
        ("impact_velocity_norm",         "m/s",     "Impact velocity",            None),
        ("peak_force_on_impact",         "N",       "Peak force on impact",       None),
        ("contact_maintained_fraction",  "fraction","Contact maintained fraction",
         (0.9, "green", "target ≥ 0.9")),
        ("normal_force_error_rmse",      "N",       "Normal force RMSE",
         (1.0, "orange", "1 N threshold")),
        ("contact_chatter_index",        "fraction","Contact chatter index",
         (0.05, "red", "chatter < 0.05")),
        ("integrated_positive_power",    "J proxy", "Integrated injected power",  None),
    ]

    for ax, (field, ylabel, title, hline) in zip(axes[:len(metrics)], metrics):
        d = {c.name: _get(field, c.name) for c in CONDITIONS}
        _boxplot(ax, d, ylabel, title, hline)

    # Min tank energy from passivity metrics
    d_tank = {c.name: _pass("min_tank_energy", c.name) for c in CONDITIONS}
    _boxplot(axes[7], d_tank, "J", "Min tank energy", (0.0, "red", "tank empty"))

    fig.suptitle("Contact-passivity benchmark: approach method comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_passivity.png", dpi=130)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'contact_passivity.png'}")


def plot_contact_rate(trials: List[TrialMetrics], out_dir: Path):
    names  = [LABELS[c.name] for c in CONDITIONS]
    colors = [COLORS[c.name] for c in CONDITIONS]
    rates  = []
    for cond in CONDITIONS:
        ct = _cond_trials(trials, cond.name)
        n_contact = sum(1 for t in ct if t.contact and t.contact.contact_established)
        rates.append(n_contact / max(1, len(ct)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Contact establishment rate")
    ax.set_title("Fraction of trials with successful contact establishment")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="100 %")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_rate.png", dpi=130)
    plt.close(fig)
    print(f"  Plot saved: {out_dir / 'contact_rate.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(n_trials: int, output_dir: Path, seed_offset: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Contact passivity benchmark ({n_trials} trials / condition) ===")

    trials  = run_benchmark(n_trials, seed_offset, output_dir)
    summary = build_summary(trials, n_trials)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nResults:")
    for cond in CONDITIONS:
        s = summary["conditions"].get(cond.name, {})
        cr = s.get("contact_rate")
        iv = s.get("mean_impact_velocity_m_s")
        pf = s.get("mean_peak_force_N")
        label = LABELS[cond.name]
        cr_str = f"{cr:.2f}" if cr is not None else "n/a"
        iv_str = f"{iv:.3f} m/s" if iv is not None else "n/a"
        pf_str = f"{pf:.2f} N"  if pf is not None else "n/a"
        print(f"  {label:30s}: contact_rate={cr_str}  impact_vel={iv_str}  peak_force={pf_str}")

    plot_all(trials, output_dir)
    plot_contact_rate(trials, output_dir)
    print(f"\nOutputs saved to {output_dir.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Contact-passivity approach-comparison benchmark")
    parser.add_argument("--n-trials",   type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/eval/contact_passivity"))
    args = parser.parse_args(argv)
    run(args.n_trials, args.output_dir, args.seed_offset)


if __name__ == "__main__":
    main()
