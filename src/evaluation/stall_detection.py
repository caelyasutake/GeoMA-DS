"""
Execution stall detection for Multi-IK-DS.

Detects when a running DS+CBF execution is stuck in a local minimum near an
obstacle, so the Multi-IK recovery layer can trigger a family switch.

A stall is declared when, over a rolling window of recent steps, the
combination of:
  - negligible goal-distance reduction,
  - high CBF intervention fraction, and
  - persistent near-obstacle activation

exceeds configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StallDetectionConfig:
    """Thresholds for declaring a stall."""
    window_steps:            int   = 80     # rolling window size (steps)
    min_progress:            float = 1e-3   # min total goal-error reduction over window
    cbf_active_threshold:    float = 0.60   # fraction of window with CBF active
    correction_norm_threshold: float = 0.05  # mean CBF correction norm (rad/s) for "heavy"
    near_graze_threshold:    int   = 15     # near-graze events in window to consider stuck
    min_steps_before_stall:  int   = 60     # do not declare stall before this many steps


@dataclass
class StallDiagnostics:
    """Structured reason for a stall event."""
    stalled:              bool  = False
    stall_step:           int   = 0
    stall_reason:         str   = ""   # e.g. "low_progress+high_cbf"
    goal_err_drop:        float = 0.0  # goal error reduction over window
    cbf_active_frac:      float = 0.0
    mean_correction_norm: float = 0.0
    near_graze_count:     int   = 0
    # Set externally after a stall is detected:
    stalled_goal_idx:     int   = -1
    stalled_family_label: str   = ""
    stalled_obstacle_name: str  = ""  # most-active obstacle in the window


# ---------------------------------------------------------------------------
# Rolling history buffer
# ---------------------------------------------------------------------------

@dataclass
class StallHistory:
    """Rolling per-step state for stall detection. Append one entry per step."""
    goal_errors:       List[float] = field(default_factory=list)
    cbf_active:        List[bool]  = field(default_factory=list)
    correction_norms:  List[float] = field(default_factory=list)
    near_grazes:       List[bool]  = field(default_factory=list)
    # Per-obstacle clearance flag: True if clearance < threshold for that step
    obstacle_active:   dict        = field(default_factory=dict)  # obs_name -> List[bool]
    step_count:        int         = 0
    # Extended fields for trap detection
    correction_angles:      List[float] = field(default_factory=list)  # CBF correction angle (deg)
    center_post_critical:   List[bool]  = field(default_factory=list)  # center_post was most critical

    def append(
        self,
        goal_error: float,
        cbf_was_active: bool,
        correction_norm: float,
        near_graze: bool,
        obstacle_near: Optional[dict] = None,   # obs_name -> bool
        correction_angle: float = 0.0,
        center_post_critical: bool = False,
    ) -> None:
        self.goal_errors.append(goal_error)
        self.cbf_active.append(cbf_was_active)
        self.correction_norms.append(correction_norm)
        self.near_grazes.append(near_graze)
        self.correction_angles.append(float(correction_angle))
        self.center_post_critical.append(bool(center_post_critical))
        if obstacle_near:
            for name, active in obstacle_near.items():
                self.obstacle_active.setdefault(name, []).append(active)
        self.step_count += 1


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------

def detect_stall(
    history: StallHistory,
    config: StallDetectionConfig,
) -> StallDiagnostics:
    """
    Evaluate the rolling window and return stall diagnostics.

    Args:
        history: StallHistory updated each execution step.
        config:  Thresholds.

    Returns:
        StallDiagnostics. ``.stalled = True`` when all criteria are met.
    """
    diag = StallDiagnostics(stall_step=history.step_count)

    if history.step_count < config.min_steps_before_stall:
        return diag

    w = config.window_steps
    ge = history.goal_errors[-w:]
    ca = history.cbf_active[-w:]
    cn = history.correction_norms[-w:]
    ng = history.near_grazes[-w:]

    if len(ge) < 2:
        return diag

    goal_drop   = ge[0] - ge[-1]   # positive = making progress
    cbf_frac    = sum(ca) / max(len(ca), 1)
    mean_corr   = sum(cn) / max(len(cn), 1)
    graze_count = sum(ng)

    diag.goal_err_drop        = goal_drop
    diag.cbf_active_frac      = cbf_frac
    diag.mean_correction_norm = mean_corr
    diag.near_graze_count     = graze_count

    # Determine most-active obstacle in this window
    most_active_obs = ""
    most_active_count = 0
    for obs_name, acts in history.obstacle_active.items():
        recent = acts[-w:] if len(acts) >= w else acts
        c = sum(recent)
        if c > most_active_count:
            most_active_count = c
            most_active_obs = obs_name
    diag.stalled_obstacle_name = most_active_obs

    # --- Declare stall if multiple criteria met ---
    reasons = []
    low_progress = goal_drop < config.min_progress
    high_cbf     = cbf_frac  >= config.cbf_active_threshold
    heavy_corr   = mean_corr >= config.correction_norm_threshold
    many_grazes  = graze_count >= config.near_graze_threshold

    if low_progress:
        reasons.append("low_progress")
    if high_cbf:
        reasons.append("high_cbf")
    if heavy_corr:
        reasons.append("heavy_correction")
    if many_grazes:
        reasons.append("near_graze")

    # Require low_progress + at least one CBF-related condition
    stalled = low_progress and (high_cbf or heavy_corr or many_grazes)
    diag.stalled     = stalled
    diag.stall_reason = "+".join(reasons) if stalled else ""

    return diag


# ---------------------------------------------------------------------------
# Trap detection (stricter than stall)
# ---------------------------------------------------------------------------

@dataclass
class TrapDetectionConfig:
    """Thresholds for declaring a trap (center-post-specific local minimum)."""
    window_steps:                    int   = 80
    min_goal_progress:               float = 1e-3
    cbf_active_threshold:            float = 0.60
    correction_norm_threshold:       float = 0.05
    near_graze_threshold:            int   = 15
    center_post_activation_threshold: int  = 20
    high_angle_fraction_threshold:   float = 0.30
    min_steps_before_trap:           int   = 60


@dataclass
class TrapDiagnostics:
    """Structured reason for a trap event (stricter than generic stall)."""
    trapped:              bool  = False
    trap_reason:          str   = ""
    trap_step:            Optional[int] = None
    center_post_dominant: bool  = False
    high_cbf_conflict:    bool  = False
    low_goal_progress:    bool  = False
    repeated_near_graze:  bool  = False
    goal_progress:        float = 0.0
    cbf_active_frac:      float = 0.0
    high_angle_frac:      float = 0.0
    center_post_count:    int   = 0
    near_graze_count:     int   = 0


def detect_trap(
    history: StallHistory,
    config: TrapDetectionConfig,
) -> TrapDiagnostics:
    """
    Evaluate whether the system is in a center-post trap.

    Stricter than detect_stall: requires evidence that the center post is
    specifically the dominant obstacle and that CBF correction angles are
    frequently large, indicating the DS vector field is fighting the geometry.
    """
    diag = TrapDiagnostics(trap_step=history.step_count)

    if history.step_count < config.min_steps_before_trap:
        return diag

    w = config.window_steps
    ge = history.goal_errors[-w:]
    ca = history.cbf_active[-w:]
    ng = history.near_grazes[-w:]
    ang = history.correction_angles[-w:]
    cp = history.center_post_critical[-w:]

    if len(ge) < 2:
        return diag

    goal_progress    = ge[0] - ge[-1]
    cbf_frac         = sum(ca) / max(len(ca), 1)
    graze_count      = sum(ng)
    high_angle_frac  = sum(1 for a in ang if a > 45.0) / max(len(ang), 1)
    cp_count         = sum(cp)

    diag.goal_progress     = goal_progress
    diag.cbf_active_frac   = cbf_frac
    diag.high_angle_frac   = high_angle_frac
    diag.center_post_count = cp_count
    diag.near_graze_count  = graze_count

    low_progress  = goal_progress  < config.min_goal_progress
    high_cbf      = cbf_frac       >= config.cbf_active_threshold
    many_grazes   = graze_count    >= config.near_graze_threshold
    cp_dominant   = cp_count       >= config.center_post_activation_threshold
    high_angle    = high_angle_frac >= config.high_angle_fraction_threshold

    diag.low_goal_progress    = low_progress
    diag.high_cbf_conflict    = high_cbf or high_angle
    diag.center_post_dominant = cp_dominant
    diag.repeated_near_graze  = many_grazes

    trapped = low_progress and cp_dominant and (high_cbf or high_angle)
    diag.trapped = trapped

    if trapped:
        reasons = []
        if low_progress:
            reasons.append("low_progress")
        if cp_dominant:
            reasons.append("center_post_dominant")
        if high_cbf:
            reasons.append("high_cbf")
        if high_angle:
            reasons.append("high_angle")
        if many_grazes:
            reasons.append("near_graze")
        diag.trap_reason = "+".join(reasons)

    return diag
