"""
Shared success evaluation helper.

Used by both the benchmark trial runner and the MuJoCo demo so that both
report success/failure under identical semantics.

Key design
----------
* evaluate_success() is a pure function: it takes pre-collected per-step
  lists (goal errors, clearances) and a SuccessConfig, and returns a
  SuccessDiagnostics.
* SuccessConfig mirrors the benchmark defaults; the demo passes the same
  object so results are directly comparable.
* Every failed run gets exactly one canonical failure_reason string so that
  the summary can count reasons rather than just "success=False".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Canonical failure reasons
# ---------------------------------------------------------------------------
FAILURE_NEVER_REACHED     = "never_reached_goal_region"
FAILURE_DRIFTED_OUT       = "reached_goal_but_drifted_out"
FAILURE_COLLISION         = "reached_goal_but_collision_invalidated"
FAILURE_STALLED           = "stalled_before_goal"
FAILURE_TIMEOUT           = "timeout_after_partial_progress"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SuccessConfig:
    """Configurable success criteria."""

    goal_radius: float = 0.05
    # Robot must stay inside goal_radius for at least this many consecutive
    # steps.  0 = first entry counts immediately.
    sustained_steps: int = 0
    # If True, any step where clearance < 0 (penetration) invalidates success.
    require_no_collision: bool = False
    # Minimum clearance required throughout rollout (0 = not enforced).
    min_clearance_threshold: float = 0.0
    # Fraction of steps with near-zero EE displacement that counts as "stalled".
    # Only used to distinguish FAILURE_STALLED from FAILURE_TIMEOUT.
    stall_fraction_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class SuccessDiagnostics:
    """Per-trial success/failure breakdown."""

    reached_goal_once:       bool  = False
    stayed_in_goal_steps:    int   = 0     # longest consecutive in-goal run
    min_goal_error_ever:     float = float("inf")
    final_goal_error:        float = float("inf")
    had_collision:           bool  = False
    had_unintended_collision: bool = False
    min_clearance_rollout:   float = float("inf")
    invalidated_by_collision: bool = False
    terminal_success:        bool  = False
    failure_reason:          str   = ""    # empty string = success


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------
def evaluate_success(
    goal_errors: List[float],
    clearances: Optional[List[float]],
    config: SuccessConfig,
    stall_mask: Optional[List[bool]] = None,
) -> SuccessDiagnostics:
    """
    Evaluate trial success from per-step data.

    Args:
        goal_errors:  ||q - q_goal|| at each step (rad).
        clearances:   Minimum obstacle clearance at each step (m), or None.
        config:       SuccessConfig thresholds.
        stall_mask:   Per-step boolean — True when EE displacement is below
                      stall threshold.  Optional; only used for stall vs
                      timeout failure reason.

    Returns:
        SuccessDiagnostics with terminal_success and failure_reason populated.
    """
    diag = SuccessDiagnostics()

    if not goal_errors:
        diag.failure_reason = FAILURE_NEVER_REACHED
        return diag

    diag.final_goal_error   = float(goal_errors[-1])
    diag.min_goal_error_ever = float(min(goal_errors))

    # ---- Clearance stats ---------------------------------------------------
    if clearances:
        valid = [c for c in clearances if c < float("inf")]
        if valid:
            diag.min_clearance_rollout   = float(min(valid))
            diag.had_collision           = diag.min_clearance_rollout < 0.0
            diag.had_unintended_collision = diag.had_collision

    # ---- Goal entry analysis -----------------------------------------------
    in_goal = [e < config.goal_radius for e in goal_errors]
    diag.reached_goal_once = any(in_goal)

    # Longest consecutive in-goal run
    max_consec = cur = 0
    for v in in_goal:
        if v:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    diag.stayed_in_goal_steps = max_consec

    # ---- Collision invalidation --------------------------------------------
    if (diag.reached_goal_once
            and diag.had_unintended_collision
            and config.require_no_collision):
        diag.invalidated_by_collision = True

    # ---- Terminal success --------------------------------------------------
    meets_goal      = diag.min_goal_error_ever < config.goal_radius
    meets_sustained = (config.sustained_steps == 0
                       or diag.stayed_in_goal_steps >= config.sustained_steps)
    meets_clearance = (config.min_clearance_threshold <= 0.0
                       or diag.min_clearance_rollout >= config.min_clearance_threshold)
    meets_no_col    = not config.require_no_collision or not diag.had_unintended_collision

    diag.terminal_success = (meets_goal and meets_sustained
                             and meets_clearance and meets_no_col)

    # ---- Failure reason ----------------------------------------------------
    if diag.terminal_success:
        diag.failure_reason = ""
    elif diag.invalidated_by_collision:
        diag.failure_reason = FAILURE_COLLISION
    elif not diag.reached_goal_once:
        # Distinguish stalled from plain timeout using stall_mask
        if stall_mask is not None:
            stall_frac = sum(stall_mask) / max(1, len(stall_mask))
            if stall_frac >= config.stall_fraction_threshold:
                diag.failure_reason = FAILURE_STALLED
            elif diag.min_goal_error_ever < config.goal_radius * 3:
                diag.failure_reason = FAILURE_TIMEOUT
            else:
                diag.failure_reason = FAILURE_NEVER_REACHED
        else:
            if diag.min_goal_error_ever < config.goal_radius * 3:
                diag.failure_reason = FAILURE_TIMEOUT
            else:
                diag.failure_reason = FAILURE_NEVER_REACHED
    elif diag.stayed_in_goal_steps < config.sustained_steps:
        diag.failure_reason = FAILURE_DRIFTED_OUT
    else:
        diag.failure_reason = FAILURE_TIMEOUT

    return diag


# ---------------------------------------------------------------------------
# Path tracking verdict
# ---------------------------------------------------------------------------
def classify_path_tracking(
    mean_deviation_q: float,
    max_deviation_q: float,
    tube_exit_count: int,
    n_steps: int,
    mean_threshold: float = 0.15,
    max_threshold: float = 0.40,
    exit_fraction_threshold: float = 0.20,
) -> str:
    """
    Classify execution path fidelity as one of three verdicts.

    Returns one of:
        "faithful_path_tracking"
        "moderate_path_drift"
        "severe_homotopy_collapse"
    """
    exit_frac = tube_exit_count / max(1, n_steps)
    if (mean_deviation_q <= mean_threshold
            and max_deviation_q <= max_threshold
            and exit_frac <= exit_fraction_threshold):
        return "faithful_path_tracking"
    if (mean_deviation_q <= mean_threshold * 2.5
            and max_deviation_q <= max_threshold * 2.5):
        return "moderate_path_drift"
    return "severe_homotopy_collapse"


# ---------------------------------------------------------------------------
# Multi-IK effect verdict
# ---------------------------------------------------------------------------
def classify_multiik_effect(
    n_families_available: int,
    family_switch_count: int,
    terminal_success: bool,
    single_ik_would_succeed: Optional[bool] = None,
) -> str:
    """
    Classify the practical effect of Multi-IK for this trial.

    Returns one of:
        "multiik_unused"              — only one family was available
        "multiik_helped_goal_selection" — best family chosen, no switch needed
        "multiik_helped_recovery"     — family switch occurred and succeeded
        "multiik_no_execution_benefit" — families available but execution failed
        "multiik_overhead_only"       — more compute, identical outcome
    """
    if n_families_available <= 1:
        return "multiik_unused"
    if family_switch_count > 0 and terminal_success:
        return "multiik_helped_recovery"
    if family_switch_count > 0 and not terminal_success:
        return "multiik_no_execution_benefit"
    if terminal_success and single_ik_would_succeed is False:
        return "multiik_helped_goal_selection"
    if terminal_success:
        return "multiik_helped_goal_selection"
    return "multiik_no_execution_benefit"
