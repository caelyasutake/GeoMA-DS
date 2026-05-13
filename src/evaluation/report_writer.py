"""
Automated Markdown report generator.

Reads aggregate_results.json and produces:
  - evaluation_summary.md  — full structured report
  - recommendations.md     — concise actionable conclusions
  - ik_effectiveness.json  — per-scenario IK benefit breakdown
  - planning_vs_execution_table.csv
  - goal_rank_analysis.csv
  - contact_benefit_table.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.aggregators import ConditionAggregate, MetricStats
from src.evaluation.metrics import TrialMetrics
from src.evaluation.statistical_tests import (
    compare_conditions,
    compute_multi_ik_advantage,
    compute_terminal_advantage_rate,
    compute_quality_advantage_rate,
    compute_goal_rank_correlations,
    compare_matched_trials,
    pairwise_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "N/A"
    return f"{v:.{decimals}f}"


def _ms_row(ms: MetricStats, label: str = None) -> str:
    n = ms.name if label is None else label
    if ms.n == 0:
        return f"| {n} | N/A | N/A | N/A | N/A | N/A |"
    return (f"| {n} | {_fmt(ms.mean)} | {_fmt(ms.std)} | "
            f"{_fmt(ms.median)} | {_fmt(ms.p25)} | {_fmt(ms.p75)} |")


def _rate_pct(r: float) -> str:
    return f"{r * 100:.1f}%"


def _sign(v: float) -> str:
    """Return '+' prefixed string for positive floats."""
    if not np.isfinite(v):
        return "N/A"
    return f"{v:+.3f}"


# ---------------------------------------------------------------------------
# Markdown table builder — conditions summary
# ---------------------------------------------------------------------------

def _conditions_summary_table(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    scenario: str,
) -> str:
    rows = [
        a for (s, _), a in aggs.items() if s == scenario
    ]
    if not rows:
        return "_No data for this scenario._\n"

    header = (
        "| Condition | Trials | Plan Succ | Term Succ | "
        "Contact Est | Circle RMSE | Arc Ratio | Min Tank |\n"
        "|-----------|--------|-----------|-----------|"
        "------------|-------------|-----------|----------|\n"
    )
    body = ""
    for a in rows:
        body += (
            f"| {a.condition} "
            f"| {a.n_trials} "
            f"| {_rate_pct(a.plan_success_rate)} "
            f"| {_rate_pct(a.terminal_success_rate)} "
            f"| {_rate_pct(a.contact_established_rate)} "
            f"| {_fmt(a.circle_rmse.mean)} "
            f"| {_fmt(a.arc_ratio.mean)} "
            f"| {_fmt(a.min_tank_energy.mean)} |\n"
        )
    return header + body


# ---------------------------------------------------------------------------
# Main report sections
# ---------------------------------------------------------------------------

def _overview_section(n_total: int, scenarios: List[str], conditions: List[str]) -> str:
    return f"""## Overview

This report summarises the evaluation of the Multi-IK Passive Dynamical System (Multi-IK-DS) solver.

- **Total trials run:** {n_total}
- **Scenarios evaluated:** {len(scenarios)} — {", ".join(scenarios)}
- **Conditions evaluated:** {len(conditions)} — {", ".join(conditions)}

The evaluation answers four primary questions:
1. Is the solver good in absolute terms?
2. Is diverse IK selection materially helping?
3. How does the solver compare against baselines?
4. Does the solver remain safe under harder conditions?

"""


def _absolute_performance_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
) -> str:
    lines = ["## Q1 — Absolute Solver Performance\n\n"]
    full_cond = "multi_ik_full__path_ds_full"
    full_rows = [a for (_, c), a in aggs.items() if c == full_cond]

    if full_rows:
        lines.append("### Full solver (multi_ik_full + path_ds_full) per scenario\n\n")
        lines.append("| Scenario | Plan Succ | Term Succ | Clipped% | Min Tank | "
                     "Circle RMSE | Arc Ratio |\n")
        lines.append("|----------|-----------|-----------|----------|----------|"
                     "------------|----------|\n")
        for a in full_rows:
            lines.append(
                f"| {a.scenario} "
                f"| {_rate_pct(a.plan_success_rate)} "
                f"| {_rate_pct(a.terminal_success_rate)} "
                f"| {_rate_pct(a.clipped_ratio.mean or 0.0)} "
                f"| {_fmt(a.min_tank_energy.mean)} "
                f"| {_fmt(a.circle_rmse.mean)} "
                f"| {_fmt(a.arc_ratio.mean)} |\n"
            )
        lines.append("\n")
    else:
        lines.append("_Full-system results not available._\n\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Table 1 — Planning vs terminal benefit
# ---------------------------------------------------------------------------

def _planning_vs_terminal_table(
    trials: List[TrialMetrics],
    scenarios: List[str],
    baselines: List[str] = None,
) -> str:
    """
    Table 1: Planning vs terminal benefit per scenario and comparison.

    Columns: scenario | comparison | plan_success_uplift | terminal_success_uplift |
             final_goal_error_delta | exec_path_len_delta
    """
    if baselines is None:
        baselines = ["single_ik_best__path_ds_full", "single_ik_random__path_ds_full"]

    treatment = "multi_ik_full__path_ds_full"

    lines = ["### Table 1 — Planning vs Terminal Benefit\n\n"]
    lines.append("| Scenario | Comparison | Plan Uplift | Term Uplift | "
                 "Goal Err Δ | Exec Path Δ |\n")
    lines.append("|----------|------------|-------------|-------------|"
                 "-----------|-------------|\n")

    for scen in sorted(scenarios):
        for baseline in baselines:
            cmp = compare_matched_trials(
                [t for t in trials if t.scenario == scen],
                cond_a=baseline, cond_b=treatment,
            )
            ov = cmp["overall"]
            if ov["n_matched"] == 0:
                continue
            label = baseline.replace("__path_ds_full", "")
            lines.append(
                f"| {scen} | vs {label} "
                f"| {_sign(ov['plan_success_uplift'])} "
                f"| {_sign(ov['terminal_success_uplift'])} "
                f"| {_sign(ov['final_goal_error_delta'])} "
                f"| {_sign(ov['exec_path_len_delta'])} |\n"
            )

    lines.append("\n_Δ: positive = multi_ik_full is better. Uplifts in fraction (0–1)._\n\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Table 2 — Multi-IK effectiveness by scenario
# ---------------------------------------------------------------------------

def _ik_effectiveness_table(
    trials: List[TrialMetrics],
    scenarios: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Table 2: Multi-IK effectiveness.

    Returns (markdown_table, effectiveness_dict)
    """
    treatment = "multi_ik_full__path_ds_full"
    baseline  = "single_ik_best__path_ds_full"

    lines = ["### Table 2 — Multi-IK Effectiveness by Scenario\n\n"]
    lines.append("| Scenario | Plan Adv Rate | Term Adv Rate | Quality Adv Rate | "
                 "Term Uplift | Recommended IK Mode |\n")
    lines.append("|----------|---------------|---------------|-----------------|"
                 "------------|---------------------|\n")

    effectiveness: Dict[str, Any] = {}

    for scen in sorted(scenarios):
        cmp = compare_matched_trials(
            [t for t in trials if t.scenario == scen],
            cond_a=baseline, cond_b=treatment,
        )
        ov = cmp["overall"]
        if ov["n_matched"] == 0:
            continue

        plan_adv   = ov["planning_advantage_rate"]
        term_adv   = ov["terminal_advantage_rate"]
        qual_adv   = ov["quality_advantage_rate"]
        term_upl   = ov["terminal_success_uplift"]
        recommendation = make_ik_recommendation(
            planning_advantage_rate=plan_adv,
            terminal_uplift=term_upl,
            scenario=scen,
        )

        lines.append(
            f"| {scen} "
            f"| {_rate_pct(plan_adv)} "
            f"| {_rate_pct(term_adv)} "
            f"| {_rate_pct(qual_adv)} "
            f"| {_sign(term_upl)} "
            f"| {recommendation['mode']} |\n"
        )

        effectiveness[scen] = {
            "planning_advantage_rate": plan_adv,
            "terminal_advantage_rate": term_adv,
            "quality_advantage_rate":  qual_adv,
            "terminal_success_uplift": term_upl,
            "final_goal_error_delta":  ov["final_goal_error_delta"],
            "recommendation":          recommendation["text"],
        }

    lines.append("\n")
    return "".join(lines), effectiveness


# ---------------------------------------------------------------------------
# Table 3 — Goal-rank analysis
# ---------------------------------------------------------------------------

def _goal_rank_table(trials: List[TrialMetrics], scenarios: List[str]) -> str:
    """Table 3: Goal-rank sensitivity."""
    lines = ["### Table 3 — Goal-Rank Analysis\n\n"]
    lines.append("| Scenario | Mean Rank | Rank→Term Corr | Rank→Err Corr |\n")
    lines.append("|----------|-----------|----------------|---------------|\n")

    for scen in sorted(scenarios):
        scen_trials = [
            t for t in trials
            if t.scenario == scen and "multi_ik" in t.condition
        ]
        if not scen_trials:
            continue

        ranks  = [t.ik.selected_goal_rank if t.ik else None for t in scen_trials]
        terms  = [bool(t.execution.terminal_success) if t.execution else False
                  for t in scen_trials]
        errors = [t.execution.final_goal_err if t.execution else None
                  for t in scen_trials]

        corrs = compute_goal_rank_correlations(ranks, terms, errors)
        lines.append(
            f"| {scen} "
            f"| {_fmt(corrs['mean_selected_rank'], 2)} "
            f"| {_fmt(corrs['rank_vs_terminal_success_corr'], 3)} "
            f"| {_fmt(corrs['rank_vs_final_error_corr'], 3)} |\n"
        )

    lines.append("\n_Negative correlation = lower rank → better outcome (desired)._\n\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Table 4 — Contact-task benefit
# ---------------------------------------------------------------------------

def _contact_benefit_table(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    trials: List[TrialMetrics],
) -> str:
    """Table 4: Contact-task IK benefit."""
    contact_scens = [s for s in {k[0] for k in aggs} if "contact" in s]
    if not contact_scens:
        return ""

    lines = ["### Table 4 — Contact-Task Benefit\n\n"]
    lines.append("| Scenario | Contact Frac Δ | Circle RMSE Δ | Arc Ratio Δ |\n")
    lines.append("|----------|----------------|---------------|-------------|\n")

    treatment = "multi_ik_full__task_tracking_full"
    baseline  = "multi_ik_full__joint_space_path_only"

    for scen in sorted(contact_scens):
        agg_t = aggs.get((scen, treatment))
        agg_b = aggs.get((scen, baseline))
        if agg_t is None or agg_b is None:
            # Fall back to any two available conditions
            scen_aggs = [(c, a) for (s, c), a in aggs.items() if s == scen]
            if len(scen_aggs) < 2:
                continue
            baseline_name, agg_b = scen_aggs[0]
            treat_name,    agg_t = scen_aggs[-1]
        else:
            baseline_name, treat_name = baseline, treatment

        def _delta_mean(ms_a: MetricStats, ms_b: MetricStats, higher_is_better: bool = True) -> str:
            if ms_a.n == 0 or ms_b.n == 0:
                return "N/A"
            d = ms_b.mean - ms_a.mean if higher_is_better else ms_a.mean - ms_b.mean
            return _sign(d)

        contact_delta = _delta_mean(agg_b.contact_maintained, agg_t.contact_maintained, True)
        rmse_delta    = _delta_mean(agg_b.circle_rmse,        agg_t.circle_rmse,        False)
        arc_delta     = _delta_mean(agg_b.arc_ratio,          agg_t.arc_ratio,           True)

        lines.append(f"| {scen} | {contact_delta} | {rmse_delta} | {arc_delta} |\n")

    lines.append("\n_Δ: positive = diverse IK / full controller is better._\n\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# IK ablation section (Q2) — now uses matched-trial comparison
# ---------------------------------------------------------------------------

def _ik_ablation_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    trials: List[TrialMetrics],
) -> str:
    lines = ["## Q2 — Diverse IK Ablation\n\n"]

    ik_conds = [
        "multi_ik_full__path_ds_full",
        "single_ik_best__path_ds_full",
        "single_ik_random__path_ds_full",
        "multi_ik_top_2__path_ds_full",
        "multi_ik_top_4__path_ds_full",
    ]

    scenarios = sorted({s for s, _ in aggs})

    for scen in scenarios:
        lines.append(f"### {scen}\n\n")
        lines.append(_conditions_summary_table(
            {k: v for k, v in aggs.items() if k[0] == scen and k[1] in ik_conds},
            scen,
        ))
        lines.append("\n")

        # --- Matched-trial planning advantage (legacy) ---
        multi_plan = [
            t.plan.success for t in trials
            if t.scenario == scen and "multi_ik_full" in t.condition and t.plan
        ]
        single_plan = [
            t.plan.success for t in trials
            if t.scenario == scen and "single_ik_best" in t.condition and t.plan
        ]
        if multi_plan and single_plan:
            n = min(len(multi_plan), len(single_plan))
            adv = compute_multi_ik_advantage(single_plan[:n], multi_plan[:n])
            lines.append(
                f"**Planning advantage rate:** {_rate_pct(adv['advantage_rate'])} "
                f"(trials where multi-IK plans succeed but single-IK fails). "
                f"Net planning benefit: {adv['net_benefit']:+.3f}.\n\n"
            )

        # --- Matched terminal advantage ---
        multi_term = [
            t.execution.terminal_success for t in trials
            if t.scenario == scen and "multi_ik_full" in t.condition and t.execution
        ]
        single_term = [
            t.execution.terminal_success for t in trials
            if t.scenario == scen and "single_ik_best" in t.condition and t.execution
        ]
        if multi_term and single_term:
            n = min(len(multi_term), len(single_term))
            tadv = compute_terminal_advantage_rate(single_term[:n], multi_term[:n])
            lines.append(
                f"**Terminal advantage rate:** {_rate_pct(tadv['terminal_advantage_rate'])} "
                f"(trials where multi-IK terminally succeeds but single-IK does not). "
                f"Terminal uplift: {tadv['terminal_uplift']:+.3f}.\n\n"
            )

            # Assertion check
            _validate_recommendation_consistency(
                scenario=scen,
                terminal_uplift=tadv["terminal_uplift"],
                lines=lines,
            )

    # Table 1 and Table 2
    lines.append(_planning_vs_terminal_table(trials, scenarios))
    table2_md, _ = _ik_effectiveness_table(trials, scenarios)
    lines.append(table2_md)

    # Table 3
    lines.append(_goal_rank_table(trials, scenarios))

    return "".join(lines)


# ---------------------------------------------------------------------------
# Baseline comparison section
# ---------------------------------------------------------------------------

def _baseline_comparison_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
) -> str:
    lines = ["## Q3 — Baseline Comparisons\n\n"]

    ctrl_conds = [
        "path_ds_full", "path_ds_no_tank",
        "path_ds_no_filter", "waypoint_pd_controller",
    ]

    for scen in sorted({s for s, _ in aggs}):
        relevant = {
            k: v for k, v in aggs.items()
            if k[0] == scen and any(c in k[1] for c in ctrl_conds)
        }
        if not relevant:
            continue
        lines.append(f"### {scen}\n\n")
        lines.append(_conditions_summary_table(relevant, scen))
        lines.append("\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Contact section
# ---------------------------------------------------------------------------

def _contact_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    trials: List[TrialMetrics],
) -> str:
    lines = ["## Contact-Task Evaluation\n\n"]
    contact_scens = {"contact_circle", "contact_circle_perturbation"}
    for scen in sorted(contact_scens):
        relevant = {k: v for k, v in aggs.items() if k[0] == scen}
        if not relevant:
            continue
        lines.append(f"### {scen}\n\n")
        for (_, cond), a in relevant.items():
            ct = a.contact_maintained
            lines.append(f"**{cond}**: contact established {_rate_pct(a.contact_established_rate)}, "
                         f"maintained fraction {_fmt(ct.mean)} ± {_fmt(ct.std)}, "
                         f"circle RMSE {_fmt(a.circle_rmse.mean)} m, "
                         f"arc ratio {_fmt(a.arc_ratio.mean)}\n\n")

    # Table 4
    lines.append(_contact_benefit_table(aggs, trials))
    return "".join(lines)


# ---------------------------------------------------------------------------
# Passivity section
# ---------------------------------------------------------------------------

def _passivity_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
) -> str:
    lines = ["## Passivity Safety Metrics\n\n"]
    lines.append("| Scenario | Condition | Clipped% | Min Tank | β_R=0 Fraction |\n")
    lines.append("|----------|-----------|----------|----------|----------------|\n")
    for (scen, cond), a in sorted(aggs.items()):
        lines.append(
            f"| {scen} | {cond} "
            f"| {_rate_pct(a.clipped_ratio.mean or 0.0)} "
            f"| {_fmt(a.min_tank_energy.mean)} "
            f"| {_fmt(a.beta_R_zero_frac.mean)} |\n"
        )
    lines.append("\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def _failure_cases_section(trials: List[TrialMetrics]) -> str:
    errors    = [t for t in trials if t.error]
    plan_fail = [t for t in trials if t.plan and not t.plan.success]
    exec_fail = [t for t in trials if t.execution and not t.execution.terminal_success]

    lines = ["## Failure Analysis\n\n"]
    lines.append(f"- **Crashed trials:** {len(errors)}\n")
    lines.append(f"- **Planning failures:** {len(plan_fail)}\n")
    lines.append(f"- **Execution failures:** {len(exec_fail)}\n\n")

    if errors:
        lines.append("### Sample errors\n")
        for e in errors[:3]:
            snippet = (e.error or "")[:300].replace("\n", " ")
            lines.append(f"- `{e.scenario}/{e.condition}` seed={e.seed}: {snippet}…\n")
        lines.append("\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Validation assertions
# ---------------------------------------------------------------------------

def _validate_recommendation_consistency(
    scenario: str,
    terminal_uplift: float,
    planning_uplift: Optional[float] = None,
    final_goal_error_delta: Optional[float] = None,
    threshold_terminal: float = 0.2,
    threshold_quality: float = 0.05,
    lines: Optional[List[str]] = None,
) -> List[str]:
    """
    Assert that the report cannot call diverse IK 'marginal' when uplifts are large.
    Appends warning text to lines if assertions fire.
    """
    if lines is None:
        lines = []

    # Assertion 1: terminal_success_uplift > 0.2 → cannot say marginal
    if np.isfinite(terminal_uplift) and terminal_uplift > threshold_terminal:
        lines.append(
            f"> ⚠️ **Assertion 1 triggered** [{scenario}]: terminal success uplift "
            f"({terminal_uplift:+.3f}) exceeds {threshold_terminal:.2f}. "
            f"The overall conclusion must NOT label diverse IK as marginal.\n\n"
        )

    # Assertion 3: planning_uplift ≈ 0 but quality improvement > threshold
    if (planning_uplift is not None and np.isfinite(planning_uplift) and abs(planning_uplift) < 0.02
            and final_goal_error_delta is not None and np.isfinite(final_goal_error_delta)
            and final_goal_error_delta > threshold_quality):
        lines.append(
            f"> ⚠️ **Assertion 3 triggered** [{scenario}]: planning uplift ≈ 0 but "
            f"goal-error improvement = {final_goal_error_delta:+.3f}. "
            f"Report must mention quality-level benefit.\n\n"
        )

    return lines


# ---------------------------------------------------------------------------
# Benefit summaries (helpers for recommendations)
# ---------------------------------------------------------------------------

def summarize_planning_vs_execution_benefit(
    trials: List[TrialMetrics],
    scenario: Optional[str] = None,
    baseline: str = "single_ik_best__path_ds_full",
    treatment: str = "multi_ik_full__path_ds_full",
) -> Dict[str, Any]:
    """
    Return a structured summary of planning vs execution benefit for a scenario.
    """
    subset = [t for t in trials if (scenario is None or t.scenario == scenario)]
    cmp = compare_matched_trials(subset, cond_a=baseline, cond_b=treatment)
    ov  = cmp["overall"]

    plan_upl = ov["plan_success_uplift"]
    term_upl = ov["terminal_success_uplift"]
    err_delta = ov["final_goal_error_delta"]

    if not np.isfinite(plan_upl):
        planning_level = "no_data"
    elif plan_upl > 0.05:
        planning_level = "beneficial"
    elif plan_upl < -0.05:
        planning_level = "harmful"
    else:
        planning_level = "negligible"

    if not np.isfinite(term_upl):
        execution_level = "no_data"
    elif term_upl > 0.1:
        execution_level = "substantial"
    elif term_upl > 0.02:
        execution_level = "moderate"
    elif term_upl < -0.05:
        execution_level = "harmful"
    else:
        execution_level = "negligible"

    return {
        "scenario":           scenario or "all",
        "n_matched":          ov["n_matched"],
        "plan_success_uplift": plan_upl,
        "terminal_success_uplift": term_upl,
        "final_goal_error_delta": err_delta,
        "planning_level":     planning_level,
        "execution_level":    execution_level,
    }


def summarize_contact_benefit(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    scenario: str,
    baseline_cond: str,
    treatment_cond: str,
) -> Dict[str, Any]:
    """Return contact-level benefit summary for a specific scenario pair."""
    agg_b = aggs.get((scenario, baseline_cond))
    agg_t = aggs.get((scenario, treatment_cond))
    if agg_b is None or agg_t is None:
        return {"scenario": scenario, "data_available": False}

    def _delta(ms_a: MetricStats, ms_b: MetricStats) -> Optional[float]:
        if ms_a.n > 0 and ms_b.n > 0:
            return ms_b.mean - ms_a.mean
        return None

    return {
        "scenario":                    scenario,
        "data_available":              True,
        "contact_maintained_fraction_delta": _delta(agg_b.contact_maintained, agg_t.contact_maintained),
        "circle_rmse_improvement":     _neg(_delta(agg_b.circle_rmse, agg_t.circle_rmse)),
        "arc_ratio_improvement":       _delta(agg_b.arc_ratio, agg_t.arc_ratio),
    }


def _neg(v: Optional[float]) -> Optional[float]:
    return -v if v is not None else None


def make_ik_recommendation(
    planning_advantage_rate: float,
    terminal_uplift: float,
    scenario: str = "",
    is_contact_scenario: bool = False,
) -> Dict[str, str]:
    """
    Derive an IK-mode recommendation per CLAUDE.md rules.

    Rules:
      1. Low planning advantage + large terminal uplift → recommend diverse IK for execution
      2. High planning advantage in constrained scenario → recommend diverse IK as default
      3. Near-zero benefit in free space, positive elsewhere → scenario-dependent
      4. single_ik_best consistently worse in terminal → do NOT label marginal

    Returns dict with 'mode' and 'text'.
    """
    plan_low     = not np.isfinite(planning_advantage_rate) or planning_advantage_rate < 0.1
    plan_high    = np.isfinite(planning_advantage_rate) and planning_advantage_rate >= 0.1
    term_large   = np.isfinite(terminal_uplift) and terminal_uplift > 0.1
    term_small   = not np.isfinite(terminal_uplift) or terminal_uplift <= 0.02
    constrained  = any(kw in scenario for kw in ["narrow", "cluttered", "obstacle"])
    free         = "free_space" in scenario

    # Rule 1: Low planning benefit but substantial execution benefit
    if plan_low and term_large:
        return {
            "mode": "diverse_ik_for_execution",
            "text": (
                "Diverse IK has limited planning benefit in this scenario family, "
                "but substantial execution/terminal-state benefit. "
                "Use diverse IK for execution quality, even if planning feasibility is unaffected."
            ),
        }

    # Rule 2: High planning advantage in constrained scenarios
    if plan_high and constrained:
        return {
            "mode": "diverse_ik_default_constrained",
            "text": "Use diverse IK as default in obstacle-constrained scenarios.",
        }

    # Rule 3: Low benefit in free space
    if free and term_small:
        return {
            "mode": "single_ik_sufficient_free_space",
            "text": (
                "Diverse IK is scenario-dependent: low value in free space, "
                "high value in constrained scenes."
            ),
        }

    # Rule 4: single_ik_best consistently worse in terminal
    if term_large:
        return {
            "mode": "diverse_ik_beneficial",
            "text": (
                "Diverse IK is beneficial — single_ik_best consistently underperforms "
                "multi_ik_full in terminal success."
            ),
        }

    # Default
    if term_small and plan_low:
        return {
            "mode": "marginal_or_scenario_dependent",
            "text": (
                "Diverse IK shows marginal overall benefit. "
                "Consider scenario-specific tuning."
            ),
        }

    return {
        "mode": "diverse_ik_moderate",
        "text": "Diverse IK shows moderate benefit. Monitor scenario-specific metrics.",
    }


# ---------------------------------------------------------------------------
# Recommendations section — all three levels
# ---------------------------------------------------------------------------

def _recommendations_section(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    trials: List[TrialMetrics],
) -> str:
    lines = ["## Recommendations\n\n"]

    # ---------- gather per-level data ----------
    scenarios = sorted({t.scenario for t in trials})

    plan_uplifts  = []
    term_uplifts  = []
    err_deltas    = []
    free_term_upl = []
    constrained_term_upl = []

    for scen in scenarios:
        summ = summarize_planning_vs_execution_benefit(trials, scenario=scen)
        pu = summ["plan_success_uplift"]
        tu = summ["terminal_success_uplift"]
        ed = summ["final_goal_error_delta"]
        if np.isfinite(pu):
            plan_uplifts.append(pu)
        if np.isfinite(tu):
            term_uplifts.append(tu)
        if np.isfinite(ed):
            err_deltas.append(ed)
        if "free" in scen and np.isfinite(tu):
            free_term_upl.append(tu)
        if any(kw in scen for kw in ["narrow", "cluttered", "obstacle"]) and np.isfinite(tu):
            constrained_term_upl.append(tu)

    overall_plan_upl = float(np.mean(plan_uplifts)) if plan_uplifts else float("nan")
    overall_term_upl = float(np.mean(term_uplifts)) if term_uplifts else float("nan")
    overall_err_delta = float(np.mean(err_deltas)) if err_deltas else float("nan")
    mean_free_term    = float(np.mean(free_term_upl)) if free_term_upl else float("nan")
    mean_constr_term  = float(np.mean(constrained_term_upl)) if constrained_term_upl else float("nan")

    # ---------- Planning-level verdict ----------
    lines.append("### 1. Planning-level benefit\n\n")
    if np.isfinite(overall_plan_upl):
        if overall_plan_upl > 0.05:
            lines.append(
                f"Diverse IK improves planning success by {overall_plan_upl:+.3f} on average.\n\n"
            )
        else:
            lines.append(
                f"Planning benefit is negligible ({overall_plan_upl:+.3f} on average). "
                "Both multi-IK and single-IK reach similar planning success rates.\n\n"
            )
    else:
        lines.append("_Planning comparison data not available._\n\n")

    # ---------- Execution-level verdict ----------
    lines.append("### 2. Execution-level benefit\n\n")
    if np.isfinite(overall_term_upl):
        if overall_term_upl > 0.2:
            lines.append(
                f"**Diverse IK provides substantial execution benefit** "
                f"(terminal success uplift = {overall_term_upl:+.3f} on average). "
                "This is the primary driver of diverse IK value in this benchmark.\n\n"
            )
        elif overall_term_upl > 0.05:
            lines.append(
                f"Diverse IK provides moderate execution benefit "
                f"(terminal success uplift = {overall_term_upl:+.3f}).\n\n"
            )
        else:
            lines.append(
                f"Execution-level benefit is small ({overall_term_upl:+.3f}). "
                "Diverse IK may still help in specific constrained scenarios.\n\n"
            )
    else:
        lines.append("_Execution comparison data not available._\n\n")

    if np.isfinite(overall_err_delta) and overall_err_delta > 0.01:
        lines.append(
            f"Final goal error is lower for multi-IK by {overall_err_delta:.3f} on average "
            "(positive = multi-IK is better).\n\n"
        )

    # ---------- Contact-level verdict ----------
    lines.append("### 3. Contact-level benefit\n\n")
    contact_scens = [s for s in {k[0] for k in aggs} if "contact" in s]
    if contact_scens:
        lines.append(
            "Contact-task evaluation results are presented in Table 4 above. "
            "Refer to circle RMSE, arc completion ratio, and contact-maintained fraction "
            "to assess whether diverse IK helps contact-task approach and stability.\n\n"
        )
    else:
        lines.append("_No contact-task scenarios were evaluated._\n\n")

    # ---------- Scenario-dependence verdict ----------
    lines.append("### 4. When is diverse IK essential vs optional?\n\n")
    if np.isfinite(mean_free_term) and np.isfinite(mean_constr_term):
        if mean_constr_term > 2.0 * max(abs(mean_free_term), 0.01):
            lines.append(
                "**Diverse IK is scenario-dependent:** low value in free space "
                f"({mean_free_term:+.3f}), high value in constrained scenes "
                f"({mean_constr_term:+.3f}). "
                "Use diverse IK as default in cluttered or narrow-passage scenarios.\n\n"
            )
        else:
            lines.append(
                f"Diverse IK benefit is consistent across scenario types "
                f"(free space: {mean_free_term:+.3f}, constrained: {mean_constr_term:+.3f}).\n\n"
            )
    else:
        lines.append("_Insufficient data for scenario-type comparison._\n\n")

    # ---------- Overall verdict (guards against wrong label) ----------
    lines.append("### Diverse IK overall verdict\n\n")
    _assert_not_marginal(overall_term_upl, overall_plan_upl, lines)

    # ---------- Tank/filter benefit ----------
    full_rows   = [a for (_, c), a in aggs.items() if c == "multi_ik_full__path_ds_full"]
    notank_rows = [a for (_, c), a in aggs.items() if c == "multi_ik_full__path_ds_no_tank"]
    if full_rows and notank_rows:
        full_clipped   = np.mean([a.clipped_ratio.mean for a in full_rows   if a.clipped_ratio.mean])
        notank_clipped = np.mean([a.clipped_ratio.mean for a in notank_rows if a.clipped_ratio.mean])
        if full_clipped < notank_clipped:
            lines.append("\n### Passivity verdict\n"
                         "The energy tank reduces passivity violations. "
                         "Keep tank + filter for safety-critical deployments.\n\n")

    lines.append("### Next improvements\n")
    lines.append("- Tune contact circle controller (K_f, F_desired) to improve "
                 "contact_maintained_fraction above 0.5.\n")
    lines.append("- Expand IK solution pool via HJCD-IK for harder scenarios.\n")
    lines.append("- Add learned DS for smoother trajectory tracking.\n\n")

    return "".join(lines)


def _assert_not_marginal(
    overall_term_upl: float,
    overall_plan_upl: float,
    lines: List[str],
) -> None:
    """
    Rule: if terminal_success_uplift > 0.2, the conclusion cannot say 'marginal'.
    """
    if np.isfinite(overall_term_upl) and overall_term_upl > 0.2:
        lines.append(
            f"Diverse IK is **not marginal**: terminal success uplift = "
            f"{overall_term_upl:+.3f}. Even if planning benefit is small "
            f"({_sign(overall_plan_upl)}), the execution-level benefit is substantial.\n\n"
        )
    elif np.isfinite(overall_plan_upl) and overall_plan_upl > 0.05:
        lines.append(
            f"Diverse IK is **beneficial** at the planning level "
            f"(uplift = {overall_plan_upl:+.3f}) and execution level "
            f"({_sign(overall_term_upl)}).\n\n"
        )
    elif np.isfinite(overall_term_upl) and overall_term_upl > 0.05:
        lines.append(
            f"Diverse IK has limited planning benefit ({_sign(overall_plan_upl)}) "
            f"but moderate execution benefit ({overall_term_upl:+.3f}).\n\n"
        )
    else:
        lines.append(
            "Diverse IK shows marginal benefit overall in this evaluation. "
            "Scenario-specific analysis (Tables 1–4) may reveal targeted value.\n\n"
        )


# ---------------------------------------------------------------------------
# Save new structured output files
# ---------------------------------------------------------------------------

def _save_ik_effectiveness_json(
    trials: List[TrialMetrics],
    scenarios: List[str],
    output_dir: Path,
) -> None:
    """Save ik_effectiveness.json."""
    _, effectiveness = _ik_effectiveness_table(trials, scenarios)
    with open(output_dir / "ik_effectiveness.json", "w", encoding="utf-8") as f:
        json.dump(effectiveness, f, indent=2, default=str)


def _save_planning_vs_execution_csv(
    trials: List[TrialMetrics],
    scenarios: List[str],
    output_dir: Path,
    baselines: List[str] = None,
) -> None:
    """Save planning_vs_execution_table.csv."""
    if baselines is None:
        baselines = ["single_ik_best__path_ds_full", "single_ik_random__path_ds_full"]
    treatment = "multi_ik_full__path_ds_full"
    rows = []
    for scen in sorted(scenarios):
        for bl in baselines:
            cmp = compare_matched_trials(
                [t for t in trials if t.scenario == scen],
                cond_a=bl, cond_b=treatment,
            )
            ov = cmp["overall"]
            if ov["n_matched"] == 0:
                continue
            rows.append({
                "scenario":                scen,
                "comparison":              bl,
                "n_matched":               ov["n_matched"],
                "plan_success_uplift":     ov["plan_success_uplift"],
                "terminal_success_uplift": ov["terminal_success_uplift"],
                "final_goal_error_delta":  ov["final_goal_error_delta"],
                "exec_path_len_delta":     ov["exec_path_len_delta"],
                "plan_path_len_delta":     ov["plan_path_len_delta"],
                "conv_time_delta":         ov["conv_time_delta"],
                "planning_advantage_rate": ov["planning_advantage_rate"],
                "terminal_advantage_rate": ov["terminal_advantage_rate"],
                "quality_advantage_rate":  ov["quality_advantage_rate"],
            })

    if rows:
        with open(output_dir / "planning_vs_execution_table.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _save_goal_rank_analysis_csv(
    trials: List[TrialMetrics],
    scenarios: List[str],
    output_dir: Path,
) -> None:
    """Save goal_rank_analysis.csv."""
    rows = []
    for scen in sorted(scenarios):
        scen_trials = [
            t for t in trials
            if t.scenario == scen and "multi_ik" in t.condition
        ]
        if not scen_trials:
            continue
        ranks  = [t.ik.selected_goal_rank if t.ik else None for t in scen_trials]
        terms  = [bool(t.execution.terminal_success) if t.execution else False
                  for t in scen_trials]
        errors = [t.execution.final_goal_err if t.execution else None
                  for t in scen_trials]
        corrs = compute_goal_rank_correlations(ranks, terms, errors)
        rows.append({"scenario": scen, **corrs})

    if rows:
        with open(output_dir / "goal_rank_analysis.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _save_contact_benefit_csv(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    output_dir: Path,
) -> None:
    """Save contact_benefit_table.csv."""
    contact_scens = [s for s in {k[0] for k in aggs} if "contact" in s]
    rows = []
    for scen in sorted(contact_scens):
        treatment = "multi_ik_full__task_tracking_full"
        baseline  = "multi_ik_full__joint_space_path_only"
        agg_t = aggs.get((scen, treatment))
        agg_b = aggs.get((scen, baseline))
        if agg_t is None or agg_b is None:
            continue

        def _d(ms_a: MetricStats, ms_b: MetricStats) -> Optional[float]:
            if ms_a.n > 0 and ms_b.n > 0:
                return ms_b.mean - ms_a.mean
            return None

        rows.append({
            "scenario":                      scen,
            "contact_maintained_fraction_delta": _d(agg_b.contact_maintained, agg_t.contact_maintained),
            "circle_rmse_improvement":       _neg(_d(agg_b.circle_rmse, agg_t.circle_rmse)),
            "arc_ratio_improvement":         _d(agg_b.arc_ratio, agg_t.arc_ratio),
            "contact_established_rate_treatment": agg_t.contact_established_rate,
            "contact_established_rate_baseline":  agg_b.contact_established_rate,
        })

    if rows:
        with open(output_dir / "contact_benefit_table.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_report(
    aggs: Dict[Tuple[str, str], ConditionAggregate],
    trials: List[TrialMetrics],
    output_dir: Path,
) -> None:
    """
    Generate evaluation_summary.md, recommendations.md, and CSV/JSON output files.

    Args:
        aggs:       Aggregate stats from aggregators.aggregate_results().
        trials:     Raw trial list (for advantage analysis and failure cases).
        output_dir: Where to save reports.
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenarios  = sorted({s for s, _ in aggs})
    conditions = sorted({c for _, c in aggs})
    n_total    = len(trials)

    # ---- Full evaluation summary ----------------------------------------
    summary = [
        "# Multi-IK-DS Evaluation Summary\n\n",
        _overview_section(n_total, scenarios, conditions),
        _absolute_performance_section(aggs),
        _ik_ablation_section(aggs, trials),
        _baseline_comparison_section(aggs),
        _contact_section(aggs, trials),
        _passivity_section(aggs),
        _failure_cases_section(trials),
        _recommendations_section(aggs, trials),
    ]

    with open(reports_dir / "evaluation_summary.md", "w", encoding="utf-8") as f:
        f.writelines(summary)

    # ---- Recommendations only -------------------------------------------
    rec_only = [
        "# Recommendations\n\n",
        _recommendations_section(aggs, trials),
    ]
    with open(reports_dir / "recommendations.md", "w", encoding="utf-8") as f:
        f.writelines(rec_only)

    # ---- New structured output files ------------------------------------
    _save_ik_effectiveness_json(trials, scenarios, output_dir)
    _save_planning_vs_execution_csv(trials, scenarios, output_dir)
    _save_goal_rank_analysis_csv(trials, scenarios, output_dir)
    _save_contact_benefit_csv(aggs, output_dir)

    print(f"[report] Saved evaluation_summary.md, recommendations.md, "
          f"ik_effectiveness.json, planning_vs_execution_table.csv, "
          f"goal_rank_analysis.csv, contact_benefit_table.csv → {output_dir}")
