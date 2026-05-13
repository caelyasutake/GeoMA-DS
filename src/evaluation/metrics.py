"""
Unified metric schema for the Multi-IK-DS evaluation framework.

All per-trial metrics are captured in TrialMetrics, which is serialisable
to/from a plain dict for JSONL storage.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Timing metrics
# ---------------------------------------------------------------------------

@dataclass
class TimingMetrics:
    """
    Separates planner/build cost from runtime control-loop cost.

    Three timing domains:
      planner_s  — one-time build phase (IK, classification, static scoring, BiRRT, etc.)
      run_s      — rollout loop (DS eval, scoring, switching, command generation)
      total_s    — planner_s + run_s

    Derived:
      control_hz     = n_steps / run_s
      end_to_end_hz  = n_steps / total_s
      planner_hz     = 1 / planner_s  (inf when planner_s == 0)
    """
    # ---- Primary timing ----
    planner_s:      float = 0.0
    planner_ms:     float = 0.0
    planner_hz:     Optional[float] = None  # builds/s; None when planner_s == 0

    run_s:          float = 0.0
    control_hz:     float = 0.0

    total_s:        float = 0.0
    end_to_end_hz:  float = 0.0

    planner_fraction: float = 0.0   # planner_s / total_s
    runtime_fraction: float = 0.0   # run_s / total_s

    # ---- Step counts ----
    n_steps:  int = 0

    # ---- Per-step statistics ----
    mean_step_ms:   float = 0.0
    median_step_ms: float = 0.0
    p95_step_ms:    float = 0.0
    p99_step_ms:    float = 0.0
    max_step_ms:    float = 0.0

    # ---- Repeated planner calls (online replanning) ----
    num_planner_calls:    int                = 1
    planner_call_times_s: List[float]        = field(default_factory=list)
    mean_planner_call_ms: Optional[float]    = None
    p95_planner_call_ms:  Optional[float]    = None
    max_planner_call_ms:  Optional[float]    = None

    # ---- Source label ----
    planner_source: str = "computed"
    # "computed"         — IK/planning run inside the trial
    # "precomputed_spec" — goals loaded from spec.ik_goals (offline)
    # "cache_hit"        — path/plan reused from a prior run
    # "disabled"         — no planner (e.g. vanilla DS)


def compute_timing_metrics(
    planner_s: float,
    run_s: float,
    step_times_s: Sequence[float],
    planner_call_times_s: Optional[Sequence[float]] = None,
    planner_source: str = "computed",
) -> TimingMetrics:
    """
    Compute all derived timing fields from raw measurements.

    Args:
        planner_s:            Total build/planner phase time (seconds).
        run_s:                Total rollout loop time (seconds).
        step_times_s:         Per-step wall-clock times (seconds).
        planner_call_times_s: Per-call planner times if replanning occurred.
                              Defaults to [planner_s] (single call).
        planner_source:       Where the plan came from.
    """
    n_steps  = len(step_times_s)
    total_s  = planner_s + run_s

    control_hz    = float(n_steps) / run_s      if run_s    > 0 else 0.0
    end_to_end_hz = float(n_steps) / total_s    if total_s  > 0 else 0.0
    planner_hz    = 1.0 / planner_s             if planner_s > 1e-12 else None

    planner_fraction = planner_s / total_s if total_s > 0 else 0.0
    runtime_fraction = run_s     / total_s if total_s > 0 else 0.0

    # Per-step percentiles (convert to ms)
    if step_times_s:
        step_ms = sorted(s * 1000.0 for s in step_times_s)
        n       = len(step_ms)
        mean_ms   = sum(step_ms) / n
        median_ms = _percentile_sorted(step_ms, 50)
        p95_ms    = _percentile_sorted(step_ms, 95)
        p99_ms    = _percentile_sorted(step_ms, 99)
        max_ms    = step_ms[-1]
    else:
        mean_ms = median_ms = p95_ms = p99_ms = max_ms = 0.0

    # Planner call tracking
    call_times = list(planner_call_times_s) if planner_call_times_s else [planner_s]
    num_calls  = len(call_times)
    if call_times:
        call_ms = [t * 1000.0 for t in call_times]
        call_ms_sorted = sorted(call_ms)
        mean_call_ms = sum(call_ms) / len(call_ms)
        p95_call_ms  = _percentile_sorted(call_ms_sorted, 95)
        max_call_ms  = call_ms_sorted[-1]
    else:
        mean_call_ms = p95_call_ms = max_call_ms = None

    return TimingMetrics(
        planner_s=planner_s,
        planner_ms=planner_s * 1000.0,
        planner_hz=planner_hz,
        run_s=run_s,
        control_hz=control_hz,
        total_s=total_s,
        end_to_end_hz=end_to_end_hz,
        planner_fraction=planner_fraction,
        runtime_fraction=runtime_fraction,
        n_steps=n_steps,
        mean_step_ms=mean_ms,
        median_step_ms=median_ms,
        p95_step_ms=p95_ms,
        p99_step_ms=p99_ms,
        max_step_ms=max_ms,
        num_planner_calls=num_calls,
        planner_call_times_s=list(call_times),
        mean_planner_call_ms=mean_call_ms,
        p95_planner_call_ms=p95_call_ms,
        max_planner_call_ms=max_call_ms,
        planner_source=planner_source,
    )


def _percentile_sorted(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = pct / 100.0 * (n - 1)
    lo   = int(math.floor(rank))
    hi   = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# Sub-metric groups
# ---------------------------------------------------------------------------

@dataclass
class IKMetrics:
    """IK-set quality and diversity."""
    n_raw:               int   = 0
    n_safe:              int   = 0
    selected_goal_idx:   int   = 0
    selected_goal_rank:  int   = 0
    used_multi_ik:       bool  = False
    ik_set_diversity:    float = 0.0   # mean pairwise joint-space distance
    avg_pairwise_dist:   float = 0.0
    goal_spread:         float = 0.0   # std of joint configs (per-joint std, mean)

    # Family-aware fields (populated when family-aware selection is used)
    selected_family_label:    str  = ""      # posture family of chosen goal
    n_families_available:     int  = 0       # distinct posture families in safe set
    available_family_labels:  str  = ""      # comma-separated family names
    family_switch_count:      int  = 0       # how many family switches occurred
    replanned_count:          int  = 0       # total replans triggered by stalls
    stalled_family_label:     str  = ""      # first family that stalled (if any)
    final_selected_goal_idx:  int  = -1      # goal index after any replanning


@dataclass
class PlanMetrics:
    """BiRRT planner metrics."""
    success:           bool  = False
    time_s:            float = 0.0
    iterations:        int   = 0
    nodes_explored:    int   = 0
    collision_checks:  int   = 0
    path_length:       float = 0.0    # sum of joint-space segment norms
    n_waypoints:       int   = 0
    goal_idx:          int   = -1     # which IK goal was reached
    min_clearance:     float = 0.0    # min obstacle clearance (approx, from env)

    # Path executability metrics (populated by path_quality analysis)
    planned_min_clearance:      float = float("inf")   # min clearance along planned path
    planned_mean_clearance:     float = float("inf")   # mean clearance along planned path
    planned_near_obs_fraction:  float = 0.0            # fraction of samples below threshold
    interp_min_clearance:       float = float("inf")   # straight-line interp min clearance
    path_risk_score:            float = 0.0            # composite risk (lower = better)

    # IK classification timing (only set when classify_ik_goals was called)
    ik_classify_time_s:         float = 0.0   # classify_ik_goals wall-clock time (0 if not called)

    # Parallel planning instrumentation (populated for MULTI_IK_* conditions only)
    parallel_enabled:           bool  = False
    parallel_backend:           str   = ""             # "process" | "thread" | "sequential"
    parallel_max_workers:       int   = 0
    parallel_candidate_count:   int   = 0
    parallel_success_count:     int   = 0
    parallel_wall_time_s:       float = 0.0   # elapsed wall time for parallel planning stage
    parallel_sum_candidate_s:   float = 0.0   # sum of per-worker planning times
    parallel_speedup_estimate:  float = 0.0   # sum_candidate / parallel_wall


@dataclass
class ExecutionMetrics:
    """DS execution / controller metrics."""
    terminal_success:    bool           = False
    final_goal_err:      float          = float("inf")
    ever_in_goal:        bool           = False
    convergence_step:    Optional[int]  = None
    convergence_time_s:  Optional[float] = None
    exec_path_length:    float          = 0.0
    path_deviation:      float          = 0.0   # mean ||q_exec - q_planned||

    # Stall / recovery tracking
    stall_step:          Optional[int]  = None    # first stall detection step
    stall_reason:        str            = ""
    family_switch_count: int            = 0
    n_replans:           int            = 0

    # Path-tube tracking metrics
    max_path_deviation_q:      float = 0.0   # max joint-space deviation from planned path (rad)
    mean_path_deviation_q:     float = 0.0   # mean joint-space deviation
    time_in_path_tube_fraction: float = 0.0  # fraction of steps in PATH_TUBE_TRACKING mode
    path_mode_active_fraction:  float = 0.0  # alias for above (benchmark reporting)
    anchor_freeze_count:        int   = 0    # steps where anchor was frozen
    anchor_advance_count:       int   = 0    # steps where anchor advanced
    tube_exit_count:            int   = 0    # steps where robot exceeded tube_radius_q

    # Success diagnostics (from success_criteria.evaluate_success)
    failure_reason:          str   = ""             # canonical failure string; "" = success
    min_goal_error_ever:     float = float("inf")   # closest the robot got to goal
    sustained_goal_steps:    int   = 0              # longest consecutive in-goal run
    path_progress_fraction:  float = 0.0            # arc-length fraction along planned path at end
    path_tracking_verdict:   str   = ""             # faithful / moderate_drift / severe_collapse
    multiik_effect:          str   = ""             # classify Multi-IK contribution


@dataclass
class PassivityMetrics:
    """Energy tank + passivity filter metrics."""
    n_violations:         int   = 0
    clipped_ratio:        float = 0.0
    mean_power_before:    float = 0.0
    mean_power_after:     float = 0.0
    min_tank_energy:      float = float("inf")
    final_tank_energy:    float = 0.0
    beta_R_zero_fraction: float = 0.0


@dataclass
class ContactMetrics:
    """Contact-task metrics (for contact_circle scenarios)."""
    contact_established:         bool  = False
    contact_maintained_fraction: float = 0.0
    mean_contact_force:          float = 0.0
    std_contact_force:           float = 0.0
    mean_height_error:           float = 0.0
    circle_tracking_rmse:        float = float("inf")
    circle_radius_rmse:          float = float("inf")
    arc_completion_ratio:        float = 0.0
    final_phase_progress:        float = 0.0

    # Contact establishment quality
    time_to_first_contact_steps: int   = 0     # steps until first contact detected
    impact_velocity_norm:        float = 0.0   # EE speed (m/s) at moment of first contact
    peak_force_on_impact:        float = 0.0   # max |F| in first 100 ms after contact (N)

    # Contact maintenance quality
    n_contact_losses:            int   = 0     # times contact dropped after establishment
    normal_force_error_rmse:     float = float("inf")  # RMSE(F_n - F_desired) during slide (N)

    # Perturbation recovery
    recovered_after_perturbation: bool  = False
    time_to_recover_steps:        int   = 0    # steps from perturb-end to contact re-established

    # Passivity / energy quality during contact
    integrated_positive_power:   float = 0.0   # Σ max(0, power_nom) — energy injected (J proxy)
    high_power_spike_count:      int   = 0     # steps with |power_nom| > 1.0 W proxy
    contact_chatter_index:       float = 0.0   # fraction of contact_flag transitions (0 = stable)


@dataclass
class RobustnessMetrics:
    """Robustness under noise / perturbation."""
    success_under_noise:        bool  = False
    success_under_perturbation: bool  = False
    final_error_variance:       float = 0.0
    worst_case_final_err:       float = float("inf")
    worst_case_clearance:       float = 0.0


@dataclass
class CBFMetrics:
    """Aggregate CBF safety filter metrics over a full rollout."""
    min_clearance_rollout:    float = float("inf")  # min over all steps
    cbf_active_fraction:      float = 0.0   # fraction of steps with n_active > 0
    mean_correction_norm:     float = 0.0   # mean ‖qdot_safe − qdot_nom‖
    max_correction_norm:      float = 0.0   # max ‖qdot_safe − qdot_nom‖
    n_slack_activations:      int   = 0     # steps where QP fell back to projection
    n_near_grazing_events:    int   = 0     # steps where clearance < d_safe + epsilon
    n_contact_override_steps: int   = 0     # steps where contact exemption was used
    n_unintended_collisions:  int   = 0     # steps where min_clearance < 0
    # DS–CBF conflict diagnostics (Section 7)
    high_angle_fraction:         float = 0.0  # fraction of steps with correction_angle > 45°
    center_post_activation_count: int  = 0    # steps where center_post was most critical obstacle


@dataclass
class CBFGoalConflictDiagnostics:
    """
    Diagnostics for CBF/goal conflict on tight-clearance goals.
    Populated for every trial where CBF is enabled.
    """
    # Goal geometry vs CBF margins
    goal_clearance:          float = float("inf")   # clearance at the IK goal config
    activation_clearance:    float = float("inf")   # d_safe + d_buffer
    goal_inside_cbf_buffer:  bool  = False          # goal_clearance < activation_clearance
    goal_inside_d_safe:      bool  = False          # goal_clearance < d_safe
    straight_line_min_clearance: float = float("inf")  # min clearance on q_start→q_goal line
    d_safe:   float = 0.03
    d_buffer: float = 0.05

    # Final state
    final_clearance:  float = float("inf")
    final_goal_error: float = float("inf")   # ||q_final - q_goal|| in joint space

    # CBF activity fractions
    cbf_active_fraction_total:      float = 0.0
    cbf_active_fraction_final_20pct: float = 0.0

    # CBF correction statistics
    mean_cbf_correction_norm: float = 0.0
    max_cbf_correction_norm:  float = 0.0

    # CBF correction angle (angle between nominal and safe velocity)
    mean_cbf_angle_deg: float = 0.0
    max_cbf_angle_deg:  float = 0.0

    # Goal-CBF conflict angle (angle between goal-direction and CBF correction)
    # Goal direction = (q_goal - q) normalized
    # CBF correction direction = (qdot_safe - qdot_nom) normalized
    # This angle > 90° means CBF is pushing AWAY from the goal
    mean_goal_cbf_conflict_angle_deg: float = 0.0
    max_goal_cbf_conflict_angle_deg:  float = 0.0

    # Progress in final 20% of execution (normalized: 1.0 = full goal progress)
    progress_final_20pct: float = 0.0

    # Classification
    failure_reason: str = ""   # "cbf_goal_conflict" | "no_progress" | "collision" | ""


@dataclass
class EscapeMetrics:
    """Metrics for escape-mode activations during a rollout."""
    # Trap detection
    trap_detected:              bool  = False
    trap_step:                  int   = 0
    trap_reason:                str   = ""

    # Escape mode execution
    escape_mode_activations:    int   = 0   # how many times escape mode was entered
    escape_mode_total_steps:    int   = 0   # cumulative steps in any escape mode
    backtrack_steps:            int   = 0
    escape_success:             bool  = False
    resumed_normal_after_escape: bool = False

    # Bridge IK
    used_bridge_ik:             bool  = False
    bridge_goal_count:          int   = 0
    bridge_family_label:        str   = ""

    # CBF during escape
    escape_cbf_active_fraction: float = 0.0
    escape_mean_correction_norm: float = 0.0

    # Clearance and path progress
    clearance_gain_during_escape: float = 0.0
    path_progress_before_escape:  float = 0.0
    path_progress_after_escape:   float = 0.0

    # Recovery
    family_switch_after_escape:  bool  = False
    replan_after_escape:         bool  = False

    # Morse escape planner (populated when MorseEscapePlanner is enabled)
    morse: Optional["MorseEscapeMetrics"] = None

    # Fast Morse controller (populated when FastMorseEscapeController is enabled)
    fast_morse: Optional["FastMorseMetrics"] = None


@dataclass
class MorseEscapeMetrics:
    """Per-trial metrics for the Morse escape planner."""
    activated: bool = False
    activation_step: Optional[int] = None
    n_candidates_generated: int = 0
    n_candidates_valid: int = 0
    selected_source_tag: Optional[str] = None   # "ik_family"|"nullspace"|"tangent"|"clearance_grad"|"random"|"joint_basis"|"negative_curvature"
    selected_score: Optional[float] = None
    execute_mode: Optional[str] = None          # "prefix" | EscapeMode name string
    negative_curvature_used: bool = False
    negative_curvature_eigenvalue: Optional[float] = None
    escape_succeeded: bool = False
    prefix_aborted: bool = False
    fallback_used: Optional[str] = None         # "existing_escape" | "birrt" | None
    planning_time_s: Optional[float] = None
    rollout_eval_time_s: Optional[float] = None


@dataclass
class FastMorseMetrics:
    """Per-trial metrics for the FastMorseEscapeController."""
    activated: bool = False
    activation_step: Optional[int] = None
    active_steps: int = 0
    mean_compute_time_ms: Optional[float] = None
    max_compute_time_ms: Optional[float] = None
    mean_candidate_count: Optional[float] = None
    selected_source_counts: Dict[str, int] = field(default_factory=dict)
    mean_selected_score: Optional[float] = None
    clearance_gain: Optional[float] = None
    goal_progress_gain: Optional[float] = None
    escaped_trap: bool = False
    fallback_used: Optional[str] = None   # "existing_escape" | None


@dataclass
class BarrierMetrics:
    """Per-rollout metrics specific to barrier-navigation scenarios."""

    # Safety / obstacle interaction
    min_clearance:            float = float("inf")
    mean_clearance:           float = float("inf")
    near_graze_count:         int   = 0
    collision_count:          int   = 0

    # Motion quality
    joint_path_length:        float = 0.0
    task_path_length:         float = 0.0
    path_efficiency:          float = 0.0
    stall_steps:              int   = 0
    oscillation_index:        float = 0.0

    # IK / planning
    n_ik_goals_blocked:            int  = 0
    selected_goal_cleared_barrier: bool = False

    # Per-obstacle clearance (barrier scenarios)
    min_clearance_center_post:    float = float("inf")
    min_clearance_top_bar:        float = float("inf")
    min_clearance_bottom_bar:     float = float("inf")
    active_fraction_center_post:  float = 0.0
    active_fraction_top_bar:      float = 0.0
    active_fraction_bottom_bar:   float = 0.0


@dataclass
class HardShieldMetrics:
    """Per-rollout hard non-contact safety shield metrics."""
    hard_shield_active_fraction: float = 0.0   # fraction of steps where shield fired
    hard_shield_forced_stop_count: int = 0     # steps where only zero velocity was safe
    min_predicted_clearance:     float = float("inf")  # min clearance at accepted q_next
    n_rejected_steps:            int   = 0     # steps where scale < 1.0


@dataclass
class ModulationMetrics:
    """Per-rollout task-space modulation obstacle avoidance metrics."""
    min_gamma_rollout:          float = float("inf")  # min gamma across all steps
    modulation_active_fraction: float = 0.0           # fraction of steps with gamma < 2.0
    mean_correction_norm:       float = 0.0           # mean ||qdot_mod - qdot_nom||
    max_correction_norm:        float = 0.0           # max  ||qdot_mod - qdot_nom||
    n_near_surface_steps:       int   = 0             # steps with gamma < 1.1


@dataclass
class CtrlFreqMetrics:
    """
    Wall-clock control-loop timing and regime-breakdown for one trial rollout.

    Two timing boundaries are used consistently across all solver conditions:
      - total step time : _t_ctrl ... after env.step()   (ctrl_step + physics sim)
      - compute time    : _t_ctrl ... after ctrl_step()  (controller only, no sim)

    Regime flags are set per step before the timing window starts so they do
    not depend on the solver's early-exit behaviour.
    """
    # ---- Step counts -------------------------------------------------------
    n_steps_executed:            int   = 0   # total steps timed
    n_steps_near_obstacle:       int   = 0   # clearance < NEAR_OBS_THRESH (0.05 m)
    n_steps_cbf_active:          int   = 0   # CBF fired at least one constraint
    n_steps_modulation_active:   int   = 0   # task-space modulation gamma < 2.0
    n_steps_path_tracking_mode:  int   = 0   # path-tube tracking override active
    n_steps_stall_logic:         int   = 0   # escape / stall-recovery mode active

    # ---- Totals ------------------------------------------------------------
    total_ctrl_time_s:           float = 0.0  # Σ (ctrl_step + env.step)  seconds
    total_ctrl_compute_time_s:   float = 0.0  # Σ ctrl_step-only          seconds

    # ---- Per-step means: all steps -----------------------------------------
    mean_hz:                     float = 0.0  # 1 / mean_step_time  (ctrl + sim)
    mean_ms:                     float = 0.0  # mean (ctrl + sim)   ms
    p95_ms:                      float = 0.0  # 95th-pctile (ctrl + sim)  ms
    max_ms:                      float = 0.0  # max (ctrl + sim)  ms
    mean_ctrl_compute_ms:        float = 0.0  # mean ctrl-only     ms

    # ---- Per-regime means: total step time ---------------------------------
    mean_ctrl_ms_near_obstacle:      float = 0.0  # mean ms when clearance < threshold
    mean_ctrl_ms_far_from_obstacle:  float = 0.0  # mean ms when clearance >= threshold

    # ---- Aliases for backward compatibility with benchmark summaries -------
    n_steps:              int   = 0     # == n_steps_executed
    ctrl_compute_ms:      float = 0.0   # == mean_ctrl_compute_ms
    ctrl_ms_cbf_active:   float = 0.0   # mean ms when CBF was active
    ctrl_ms_cbf_inactive: float = 0.0   # mean ms when CBF was inactive


# ---------------------------------------------------------------------------
# Top-level per-trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialMetrics:
    """All metrics for a single trial run."""

    # Identity
    trial_id:   int   = 0
    seed:       int   = 0
    scenario:   str   = ""
    condition:  str   = ""
    timestamp:  str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Sub-metrics (None if not applicable for this scenario type)
    ik:          Optional[IKMetrics]          = None
    plan:        Optional[PlanMetrics]        = None
    execution:   Optional[ExecutionMetrics]   = None
    passivity:   Optional[PassivityMetrics]   = None
    contact:     Optional[ContactMetrics]     = None
    robustness:  Optional[RobustnessMetrics]  = None
    cbf:         Optional[CBFMetrics]         = None
    modulation:  Optional[ModulationMetrics]  = None
    hard_shield: Optional[HardShieldMetrics]  = None
    barrier:     Optional[BarrierMetrics]     = None
    escape:      Optional[EscapeMetrics]      = None
    ctrl_freq:   Optional[CtrlFreqMetrics]    = None
    cbf_goal_conflict: Optional[CBFGoalConflictDiagnostics] = None

    # Timing breakdown (planner vs runtime vs end-to-end)
    timing:      Optional[TimingMetrics] = None

    # Failure info
    error:       Optional[str] = None

    # Raw wall-clock time for the full trial
    wall_time_s: float = 0.0

    # ---------------------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict_recursive(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialMetrics":
        """Reconstruct from a plain dict (e.g. loaded from JSONL)."""
        tm = cls(
            trial_id=d.get("trial_id", 0),
            seed=d.get("seed", 0),
            scenario=d.get("scenario", ""),
            condition=d.get("condition", ""),
            timestamp=d.get("timestamp", ""),
            error=d.get("error"),
            wall_time_s=d.get("wall_time_s", 0.0),
        )
        if d.get("ik"):
            tm.ik = IKMetrics(**{k: v for k, v in d["ik"].items()
                                  if k in IKMetrics.__dataclass_fields__})
        if d.get("plan"):
            tm.plan = PlanMetrics(**{k: v for k, v in d["plan"].items()
                                     if k in PlanMetrics.__dataclass_fields__})
        if d.get("execution"):
            tm.execution = ExecutionMetrics(**{k: v for k, v in d["execution"].items()
                                               if k in ExecutionMetrics.__dataclass_fields__})
        if d.get("passivity"):
            tm.passivity = PassivityMetrics(**{k: v for k, v in d["passivity"].items()
                                               if k in PassivityMetrics.__dataclass_fields__})
        if d.get("contact"):
            tm.contact = ContactMetrics(**{k: v for k, v in d["contact"].items()
                                           if k in ContactMetrics.__dataclass_fields__})
        if d.get("robustness"):
            tm.robustness = RobustnessMetrics(**{k: v for k, v in d["robustness"].items()
                                                  if k in RobustnessMetrics.__dataclass_fields__})
        if d.get("cbf"):
            tm.cbf = CBFMetrics(**{k: v for k, v in d["cbf"].items()
                                   if k in CBFMetrics.__dataclass_fields__})
        if d.get("modulation"):
            tm.modulation = ModulationMetrics(**{k: v for k, v in d["modulation"].items()
                                                  if k in ModulationMetrics.__dataclass_fields__})
        if d.get("hard_shield"):
            tm.hard_shield = HardShieldMetrics(**{k: v for k, v in d["hard_shield"].items()
                                                   if k in HardShieldMetrics.__dataclass_fields__})
        if d.get("barrier"):
            tm.barrier = BarrierMetrics(**{k: v for k, v in d["barrier"].items()
                                           if k in BarrierMetrics.__dataclass_fields__})
        if d.get("escape"):
            escape_dict = d["escape"].copy()
            morse_dict = escape_dict.pop("morse", None)
            fast_morse_dict = escape_dict.pop("fast_morse", None)
            tm.escape = EscapeMetrics(**{k: v for k, v in escape_dict.items()
                                         if k in EscapeMetrics.__dataclass_fields__})
            if morse_dict:
                tm.escape.morse = MorseEscapeMetrics(**{k: v for k, v in morse_dict.items()
                                                        if k in MorseEscapeMetrics.__dataclass_fields__})
            if fast_morse_dict:
                tm.escape.fast_morse = FastMorseMetrics(**{k: v for k, v in fast_morse_dict.items()
                                                           if k in FastMorseMetrics.__dataclass_fields__})
        if d.get("ctrl_freq"):
            tm.ctrl_freq = CtrlFreqMetrics(**{k: v for k, v in d["ctrl_freq"].items()
                                              if k in CtrlFreqMetrics.__dataclass_fields__})
        if d.get("cbf_goal_conflict"):
            tm.cbf_goal_conflict = CBFGoalConflictDiagnostics(
                **{k: v for k, v in d["cbf_goal_conflict"].items()
                   if k in CBFGoalConflictDiagnostics.__dataclass_fields__}
            )
        if d.get("timing"):
            tm.timing = TimingMetrics(**{k: v for k, v in d["timing"].items()
                                         if k in TimingMetrics.__dataclass_fields__})
        return tm


def _to_dict_recursive(obj: Any) -> Any:
    """Replace inf/nan with None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_dict_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict_recursive(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_jsonl(path, trial: TrialMetrics) -> None:
    """Append one TrialMetrics record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trial.to_dict()) + "\n")


def load_jsonl(path) -> List[TrialMetrics]:
    """Load all TrialMetrics records from a JSONL file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(TrialMetrics.from_dict(json.loads(line)))
    return results
