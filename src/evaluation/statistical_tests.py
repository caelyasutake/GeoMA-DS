"""
Statistical analysis for evaluation comparisons.

Provides:
  - bootstrap_ci:                  Bootstrap confidence interval for a scalar metric.
  - compare_conditions:            Pairwise comparison with CI on the difference.
  - wilcoxon_test:                 Nonparametric significance test (Wilcoxon signed-rank).
  - effect_size:                   Cohen's d effect size.
  - pairwise_table:                Full comparison table for a set of conditions.
  - compute_multi_ik_advantage:    Planning-level advantage (legacy).
  - compute_terminal_advantage_rate:  Execution-level advantage.
  - compute_quality_advantage_rate:   Quality-level advantage.
  - compute_goal_rank_correlations:   IK rank sensitivity analysis.
  - compare_matched_trials:           Head-to-head matched-trial comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: List[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data:       Scalar observations.
        statistic:  "mean" | "median" | "std" | "success_rate".
        confidence: Coverage probability (e.g. 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed:       RNG seed.

    Returns:
        (lower, upper, point_estimate)
    """
    arr = np.array([v for v in data if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n   = len(arr)

    stat_fn = {
        "mean":         np.mean,
        "median":       np.median,
        "std":          np.std,
        "success_rate": np.mean,   # data should be 0/1 booleans
    }.get(statistic, np.mean)

    point = float(stat_fn(arr))
    boots = np.array([
        stat_fn(rng.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha)))
    return lo, hi, point


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of pairwise condition comparison."""
    metric:          str
    condition_a:     str
    condition_b:     str
    mean_a:          float
    mean_b:          float
    diff_mean:       float        # mean_b - mean_a
    diff_ci_lo:      float
    diff_ci_hi:      float
    effect_size_d:   float        # Cohen's d
    p_value:         Optional[float] = None
    significant:     bool = False  # at alpha=0.05
    n_a:             int = 0
    n_b:             int = 0

    def to_dict(self) -> dict:
        return {
            "metric":        self.metric,
            "condition_a":   self.condition_a,
            "condition_b":   self.condition_b,
            "mean_a":        self.mean_a,
            "mean_b":        self.mean_b,
            "diff_mean":     self.diff_mean,
            "diff_ci_lo":    self.diff_ci_lo,
            "diff_ci_hi":    self.diff_ci_hi,
            "effect_size_d": self.effect_size_d,
            "p_value":       self.p_value,
            "significant":   self.significant,
            "n_a":           self.n_a,
            "n_b":           self.n_b,
        }


def compare_conditions(
    values_a: List[float],
    values_b: List[float],
    condition_a: str = "A",
    condition_b: str = "B",
    metric: str = "metric",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> ComparisonResult:
    """
    Compare two conditions on a scalar metric using bootstrap.

    The bootstrapped quantity is the difference (mean_b − mean_a).
    Significance is declared when the 95% CI of the difference excludes 0.

    Args:
        values_a / values_b: Scalar observations for conditions A and B.
        condition_a / b:     Labels.
        metric:              Metric name (for reporting).
        n_bootstrap:         Bootstrap resamples.
        seed:                RNG seed.

    Returns:
        ComparisonResult with difference CI and Cohen's d.
    """
    a = np.array([v for v in values_a if v is not None and np.isfinite(v)], dtype=float)
    b = np.array([v for v in values_b if v is not None and np.isfinite(v)], dtype=float)

    if len(a) == 0 or len(b) == 0:
        return ComparisonResult(
            metric=metric,
            condition_a=condition_a, condition_b=condition_b,
            mean_a=float("nan"), mean_b=float("nan"),
            diff_mean=float("nan"), diff_ci_lo=float("nan"), diff_ci_hi=float("nan"),
            effect_size_d=float("nan"),
        )

    rng = np.random.default_rng(seed)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # Bootstrap on the difference
    boots = []
    for _ in range(n_bootstrap):
        ba = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        boots.append(float(np.mean(bb)) - float(np.mean(ba)))
    boots_arr = np.array(boots)

    lo = float(np.percentile(boots_arr, 2.5))
    hi = float(np.percentile(boots_arr, 97.5))

    # Cohen's d
    pooled_std = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    d = ((mean_b - mean_a) / pooled_std) if pooled_std > 1e-12 else 0.0

    # Nonparametric p-value (permutation test)
    p = _permutation_p(a, b, n_perm=min(n_bootstrap, 1000), rng=rng)

    return ComparisonResult(
        metric=metric,
        condition_a=condition_a, condition_b=condition_b,
        mean_a=mean_a, mean_b=mean_b,
        diff_mean=mean_b - mean_a,
        diff_ci_lo=lo, diff_ci_hi=hi,
        effect_size_d=d,
        p_value=p,
        significant=(lo > 0 or hi < 0),  # CI excludes 0
        n_a=len(a), n_b=len(b),
    )


def _permutation_p(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 1000,
    rng=None,
) -> float:
    """Two-sided permutation test p-value for difference in means."""
    if rng is None:
        rng = np.random.default_rng(0)
    obs_diff = abs(float(np.mean(b)) - float(np.mean(a)))
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        d = abs(float(np.mean(perm[na:])) - float(np.mean(perm[:na])))
        if d >= obs_diff:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    """Cohen's d effect size."""
    a = np.array([v for v in values_a if v is not None and np.isfinite(v)], dtype=float)
    b = np.array([v for v in values_b if v is not None and np.isfinite(v)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(b) - np.mean(a)) / max(pooled, 1e-12))


# ---------------------------------------------------------------------------
# Pairwise comparison table
# ---------------------------------------------------------------------------

def pairwise_table(
    condition_values: Dict[str, List[float]],
    baseline: str,
    metric: str = "metric",
    n_bootstrap: int = 2000,
) -> List[ComparisonResult]:
    """
    Compare each condition against a baseline condition.

    Args:
        condition_values: {condition_name: [values]}.
        baseline:         Name of baseline condition (condition_a in each comparison).
        metric:           Metric label.
        n_bootstrap:      Bootstrap resamples.

    Returns:
        List of ComparisonResult, one per non-baseline condition.
    """
    baseline_vals = condition_values.get(baseline, [])
    results = []
    for cond, vals in condition_values.items():
        if cond == baseline:
            continue
        results.append(compare_conditions(
            baseline_vals, vals,
            condition_a=baseline, condition_b=cond,
            metric=metric,
            n_bootstrap=n_bootstrap,
        ))
    return results


# ---------------------------------------------------------------------------
# Multi-IK advantage diagnosis — planning level (legacy)
# ---------------------------------------------------------------------------

def compute_multi_ik_advantage(
    single_ik_successes: List[bool],
    multi_ik_successes: List[bool],
) -> Dict[str, float]:
    """
    Compute planning-level advantage for matched trials.

    Returns dict with:
      - advantage_rate:   fraction of trials where multi-IK succeeds but single-IK fails
      - disadvantage_rate: fraction where single-IK succeeds but multi-IK fails
      - net_benefit:      advantage_rate - disadvantage_rate
    """
    if len(single_ik_successes) != len(multi_ik_successes):
        n = min(len(single_ik_successes), len(multi_ik_successes))
        single_ik_successes = single_ik_successes[:n]
        multi_ik_successes  = multi_ik_successes[:n]

    n = len(single_ik_successes)
    if n == 0:
        return {"advantage_rate": 0.0, "disadvantage_rate": 0.0, "net_benefit": 0.0}

    advantage    = sum(1 for s, m in zip(single_ik_successes, multi_ik_successes) if not s and m)
    disadvantage = sum(1 for s, m in zip(single_ik_successes, multi_ik_successes) if s and not m)

    adv_rate    = advantage    / n
    disadv_rate = disadvantage / n
    return {
        "advantage_rate":    adv_rate,
        "disadvantage_rate": disadv_rate,
        "net_benefit":       adv_rate - disadv_rate,
        "n_trials":          n,
    }


# ---------------------------------------------------------------------------
# Multi-IK advantage — execution / terminal level
# ---------------------------------------------------------------------------

def compute_terminal_advantage_rate(
    single_ik_terminal: List[bool],
    multi_ik_terminal: List[bool],
) -> Dict[str, float]:
    """
    Fraction of matched trials where multi-IK terminally succeeds but single-IK does not.

    Args:
        single_ik_terminal: Boolean terminal-success per trial (single-IK baseline).
        multi_ik_terminal:  Boolean terminal-success per trial (multi-IK).

    Returns dict with:
      - terminal_advantage_rate:    multi succeeds, single fails
      - terminal_disadvantage_rate: single succeeds, multi fails
      - terminal_uplift:            mean(multi_terminal) - mean(single_terminal)
      - n_trials
    """
    n = min(len(single_ik_terminal), len(multi_ik_terminal))
    if n == 0:
        return {
            "terminal_advantage_rate": 0.0,
            "terminal_disadvantage_rate": 0.0,
            "terminal_uplift": 0.0,
            "n_trials": 0,
        }

    s = single_ik_terminal[:n]
    m = multi_ik_terminal[:n]

    advantage    = sum(1 for si, mi in zip(s, m) if not si and mi)
    disadvantage = sum(1 for si, mi in zip(s, m) if si and not mi)
    uplift = float(np.mean([float(mi) for mi in m])) - float(np.mean([float(si) for si in s]))

    return {
        "terminal_advantage_rate":    advantage    / n,
        "terminal_disadvantage_rate": disadvantage / n,
        "terminal_uplift":            uplift,
        "n_trials":                   n,
    }


def compute_quality_advantage_rate(
    single_ik_errors: List[Optional[float]],
    multi_ik_errors: List[Optional[float]],
    single_ik_terminal: Optional[List[bool]] = None,
    multi_ik_terminal: Optional[List[bool]] = None,
    margin: float = 0.05,
) -> Dict[str, float]:
    """
    Fraction of matched trials where both conditions succeed but multi-IK has lower
    final goal error by a meaningful margin.

    Args:
        single_ik_errors:   Final goal error per matched trial (single-IK).
        multi_ik_errors:    Final goal error per matched trial (multi-IK).
        single_ik_terminal: Optional boolean masks; if provided, only both-succeed trials count.
        multi_ik_terminal:  Optional boolean masks.
        margin:             Minimum relative improvement to count as quality advantage.

    Returns dict with:
      - quality_advantage_rate:    fraction of matched trials where multi-IK has meaningfully
                                   lower final error
      - mean_final_goal_error_single
      - mean_final_goal_error_multi
      - final_goal_error_delta:    mean(single) - mean(multi)  (positive = multi is better)
      - n_matched
    """
    n = min(len(single_ik_errors), len(multi_ik_errors))
    if n == 0:
        return {
            "quality_advantage_rate": 0.0,
            "mean_final_goal_error_single": float("nan"),
            "mean_final_goal_error_multi": float("nan"),
            "final_goal_error_delta": float("nan"),
            "n_matched": 0,
        }

    se = single_ik_errors[:n]
    me = multi_ik_errors[:n]
    st = single_ik_terminal[:n] if single_ik_terminal is not None else [True] * n
    mt = multi_ik_terminal[:n] if multi_ik_terminal is not None else [True] * n

    quality_adv = 0
    both_succeed_count = 0
    valid_se, valid_me = [], []

    for i in range(n):
        s_ok = bool(st[i])
        m_ok = bool(mt[i])
        s_err = se[i]
        m_err = me[i]

        if s_err is not None and np.isfinite(s_err):
            valid_se.append(s_err)
        if m_err is not None and np.isfinite(m_err):
            valid_me.append(m_err)

        if s_ok and m_ok and s_err is not None and m_err is not None:
            both_succeed_count += 1
            improvement = s_err - m_err
            ref = max(abs(s_err), 1e-9)
            if improvement / ref > margin:
                quality_adv += 1

    mean_se = float(np.mean(valid_se)) if valid_se else float("nan")
    mean_me = float(np.mean(valid_me)) if valid_me else float("nan")
    delta   = (mean_se - mean_me) if (np.isfinite(mean_se) and np.isfinite(mean_me)) else float("nan")

    rate = (quality_adv / both_succeed_count) if both_succeed_count > 0 else 0.0
    return {
        "quality_advantage_rate":        rate,
        "mean_final_goal_error_single":  mean_se,
        "mean_final_goal_error_multi":   mean_me,
        "final_goal_error_delta":        delta,
        "n_matched":                     n,
        "n_both_succeed":                both_succeed_count,
    }


# ---------------------------------------------------------------------------
# Goal-rank sensitivity
# ---------------------------------------------------------------------------

def compute_goal_rank_correlations(
    selected_ranks: List[Optional[int]],
    terminal_successes: List[bool],
    final_errors: List[Optional[float]],
) -> Dict[str, float]:
    """
    Measure whether the rank of the selected IK goal is associated with
    downstream terminal success and final goal error.

    A lower rank (rank=0 is best) should correlate with higher terminal success
    and lower final error if diverse IK is beneficial.

    Args:
        selected_ranks:     Rank of the goal chosen by planner per trial (0 = best ranked).
        terminal_successes: Boolean terminal-success per trial.
        final_errors:       Final goal error per trial.

    Returns dict with:
      - mean_selected_rank
      - rank_vs_terminal_success_corr:  Spearman correlation (negative = lower rank → more success)
      - rank_vs_final_error_corr:       Spearman correlation (negative = lower rank → lower error)
      - n_valid
    """
    ranks  = []
    terms  = []
    errors = []

    for r, t, e in zip(selected_ranks, terminal_successes, final_errors):
        if r is None:
            continue
        ranks.append(float(r))
        terms.append(float(t))
        if e is not None and np.isfinite(e):
            errors.append((float(r), float(e)))

    result: Dict[str, float] = {
        "mean_selected_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "rank_vs_terminal_success_corr": float("nan"),
        "rank_vs_final_error_corr": float("nan"),
        "n_valid": len(ranks),
    }

    if len(ranks) >= 3:
        r_arr = np.array(ranks)
        t_arr = np.array(terms)
        # Spearman: rank the arrays then correlate
        from scipy.stats import spearmanr
        corr_t, _ = spearmanr(r_arr, t_arr)
        result["rank_vs_terminal_success_corr"] = float(corr_t)

    if len(errors) >= 3:
        re_ranks = np.array([x[0] for x in errors])
        re_errs  = np.array([x[1] for x in errors])
        from scipy.stats import spearmanr
        corr_e, _ = spearmanr(re_ranks, re_errs)
        result["rank_vs_final_error_corr"] = float(corr_e)

    return result


# ---------------------------------------------------------------------------
# Matched-trial comparison
# ---------------------------------------------------------------------------

def compare_matched_trials(
    trials: List[Any],   # List[TrialMetrics]
    cond_a: str,
    cond_b: str,
) -> Dict[str, Any]:
    """
    Head-to-head comparison of two conditions on matched (scenario, seed) pairs.

    Matching key: (scenario, seed).  Any trial not matched in both conditions is dropped.

    Args:
        trials: Full list of TrialMetrics.
        cond_a: Condition name for baseline (e.g. "single_ik_best__path_ds_full").
        cond_b: Condition name for treatment (e.g. "multi_ik_full__path_ds_full").

    Returns dict with per-scenario and overall matched comparisons:
      {
        "overall": {
          "n_matched": int,
          "plan_success_uplift":       float,  # cond_b - cond_a
          "terminal_success_uplift":   float,
          "final_goal_error_delta":    float,  # cond_a - cond_b (positive = b better)
          "exec_path_len_delta":       float,  # cond_a - cond_b
          "plan_path_len_delta":       float,  # cond_a - cond_b
          "conv_time_delta":           float,  # cond_a - cond_b
          "planning_advantage_rate":   float,
          "terminal_advantage_rate":   float,
          "quality_advantage_rate":    float,
        },
        "by_scenario": { scenario: {...same keys...} }
      }
    """
    # Index by (scenario, seed)
    idx_a: Dict[Tuple[str, int], Any] = {}
    idx_b: Dict[Tuple[str, int], Any] = {}

    for t in trials:
        key = (t.scenario, t.seed)
        if cond_a in t.condition:
            idx_a[key] = t
        elif cond_b in t.condition:
            idx_b[key] = t

    matched_keys = sorted(set(idx_a) & set(idx_b))

    def _extract(keys, idx_a, idx_b):
        plan_a, plan_b = [], []
        term_a, term_b = [], []
        err_a,  err_b  = [], []
        epath_a, epath_b = [], []
        ppath_a, ppath_b = [], []
        ctime_a, ctime_b = [], []

        for k in keys:
            ta, tb = idx_a[k], idx_b[k]

            if ta.plan is not None and tb.plan is not None:
                plan_a.append(float(ta.plan.success))
                plan_b.append(float(tb.plan.success))
                if ta.plan.success and ta.plan.path_length is not None:
                    ppath_a.append(ta.plan.path_length)
                if tb.plan.success and tb.plan.path_length is not None:
                    ppath_b.append(tb.plan.path_length)

            if ta.execution is not None and tb.execution is not None:
                term_a.append(float(ta.execution.terminal_success))
                term_b.append(float(tb.execution.terminal_success))
                if ta.execution.final_goal_err is not None:
                    err_a.append(ta.execution.final_goal_err)
                if tb.execution.final_goal_err is not None:
                    err_b.append(tb.execution.final_goal_err)
                if ta.execution.exec_path_length is not None:
                    epath_a.append(ta.execution.exec_path_length)
                if tb.execution.exec_path_length is not None:
                    epath_b.append(tb.execution.exec_path_length)
                if ta.execution.convergence_time_s is not None:
                    ctime_a.append(ta.execution.convergence_time_s)
                if tb.execution.convergence_time_s is not None:
                    ctime_b.append(tb.execution.convergence_time_s)

        def _mean(lst):
            arr = [v for v in lst if v is not None and np.isfinite(v)]
            return float(np.mean(arr)) if arr else float("nan")

        def _uplift(a, b):
            ma, mb = _mean(a), _mean(b)
            if np.isfinite(ma) and np.isfinite(mb):
                return mb - ma
            return float("nan")

        def _delta(a, b):
            # positive = b is better (lower)
            ma, mb = _mean(a), _mean(b)
            if np.isfinite(ma) and np.isfinite(mb):
                return ma - mb
            return float("nan")

        n = len(keys)
        plan_adv   = (sum(1 for a, b in zip(plan_a, plan_b) if not a and b) / n) if n else 0.0
        term_adv   = (sum(1 for a, b in zip(term_a, term_b) if not a and b) / n) if n else 0.0
        qual_adv   = 0.0
        both_succ  = [(ea, eb) for (ta_, tb_, ea, eb) in zip(term_a, term_b, err_a, err_b)
                      if ta_ and tb_]
        if both_succ:
            qual_adv = sum(1 for ea, eb in both_succ
                           if (ea - eb) / max(abs(ea), 1e-9) > 0.05) / len(both_succ)

        return {
            "n_matched":                 n,
            "plan_success_uplift":       _uplift(plan_a, plan_b),
            "terminal_success_uplift":   _uplift(term_a, term_b),
            "final_goal_error_delta":    _delta(err_a, err_b),
            "exec_path_len_delta":       _delta(epath_a, epath_b),
            "plan_path_len_delta":       _delta(ppath_a, ppath_b),
            "conv_time_delta":           _delta(ctime_a, ctime_b),
            "planning_advantage_rate":   plan_adv,
            "terminal_advantage_rate":   term_adv,
            "quality_advantage_rate":    qual_adv,
        }

    overall = _extract(matched_keys, idx_a, idx_b)

    # Per-scenario breakdown
    scenarios = sorted({k[0] for k in matched_keys})
    by_scen: Dict[str, Any] = {}
    for scen in scenarios:
        scen_keys = [k for k in matched_keys if k[0] == scen]
        by_scen[scen] = _extract(scen_keys, idx_a, idx_b)

    return {
        "condition_a": cond_a,
        "condition_b": cond_b,
        "overall":     overall,
        "by_scenario": by_scen,
    }
