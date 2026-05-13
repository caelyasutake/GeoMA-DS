"""
GeoMultiAttractorDS — Geometry-Guided Multi-Attractor Dynamical System.

Replaces the BiRRT-path residual of PathDS with differential-geometric
obstacle avoidance.  The robot has a *set* of valid IK attractors
(one per posture family); the controller selects the best attractor at
each timestep and shapes its motion using three components:

    qdot_des = f_attractor + f_tangent + f_null

    f_attractor  — conservative pull toward the active IK goal:
                     f_att = -K_c * (q - q_goal)

    f_tangent    — tangent-plane correction of f_attractor near obstacles.
                   Removes the inward component so the arm slides around
                   instead of driving through:
                     P_tan = I - (∇h ∇hᵀ) / (‖∇h‖² + ε)
                     f_tan = α · w · (P_tan @ f_att - f_att)

    f_null       — nullspace clearance ascent: increases clearance without
                   moving the end-effector:
                     N = I - J⁺ J
                     f_null = α · w · K_null · N @ ∇h

The blend factor α is computed from a two-threshold smoothstep schedule:
    α = 0  when clearance ≥ clearance_enter   (far — pure attractor DS)
    α = 1  when clearance ≤ clearance_full    (close — full geometric shaping)
    Hermite-smoothstepped between the two thresholds for C¹ continuity.

This means far from obstacles the conservative attractor dominates
(Lyapunov-stable), and shaping only engages when the arm actually
approaches an obstacle.

The three components are returned separately in the result so the
impedance controller can apply passivity gating to f_tangent and f_null
while leaving f_attractor ungated.

Attractor switching uses winner-take-all with hysteresis:
    switch only when  score_best > score_active + switch_hysteresis.

No BiRRT dependency is introduced here.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.solver.planner.collision import _panda_link_positions, _panda_fk_batch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShellWaypointCandidate:
    """
    A candidate task-space waypoint on the spherical shell around the final goal.

    Used by generate_goal_shell_waypoints (factory.py) to propose temporary
    escape targets when final-goal IK diversity is insufficient.
    """
    pos:       np.ndarray   # (3,) task-space position
    sector:    str          # human-readable sector label (e.g. "shell_pos_y_neg_z")
    clearance: float        # minimum clearance from all obstacles at this point
    score:     float        # composite score (higher = better)
    direction: np.ndarray   # (3,) unit direction from goal_pos to this waypoint
    metadata:  dict         = field(default_factory=dict)


@dataclass
class EscapeWaypointCandidate:
    """
    A candidate task-space waypoint placed at an obstacle boundary corner.

    Used by generate_obstacle_boundary_escape_waypoints (factory.py) to propose
    homotopy-proxy escape targets.  Each candidate represents a distinct route
    family around the obstacle (e.g. escape_top_right, escape_bottom_left).
    """
    pos:       np.ndarray   # (3,) task-space position
    family:    str          # homotopy route family label (auto-generated; not used for logic)
    clearance: float        # minimum signed clearance from all obstacles
    score:     float        # composite score (higher = better)
    stage_tags:     list    = field(default_factory=list)  # e.g. ["below_stage_candidate", "lateral_stage_candidate"]
    y_rel:          float   = 0.0    # pos[1] - cross_center_y
    z_rel:          float   = 0.0    # pos[2] - cross_center_z
    metadata:       dict    = field(default_factory=dict)
    ik_generated:   bool   = False   # True after IK has been attempted for this candidate
    ik_failed:      bool   = False   # True if IK returned None
    ik_slow:        bool   = False   # True if IK exceeded online_ik_max_ms budget
    last_ik_ms:     float  = 0.0     # latency of the most recent IK call
    ik_cost_ema_ms: float  = 0.0     # exponential moving average of IK latency
    ik_pending:      bool   = False   # True while background HJCD-IK thread is in flight
    q_esc_prefilled: Optional[np.ndarray] = None  # set by background IK thread on success


@dataclass
class IKAttractor:
    """
    A single IK goal configuration treated as a joint-space attractor.

    Produced by classify_ik_goals / select_family_representatives and
    passed to GeoMultiAttractorDS at construction time.
    """
    q_goal:        np.ndarray   # (7,) joint configuration
    family:        str          # homotopy-proxy label (e.g. "obs:pos_y:center_z" or elbow-Y fallback)
    clearance:     float        # static obstacle clearance at q_goal (m)
    manipulability: float       # sqrt(det(J Jᵀ)) proxy
    weight:        float = 1.0  # optional prior weight (unused in scoring by default)
    # Static properties injected by the factory (zero/none when built without factory)
    static_score:                float = 0.0    # combined quality score at construction time
    straight_line_min_clearance: float = 0.0    # min clearance on straight joint-space path from start
    window_label:                str   = "none" # cross-barrier gate window label (e.g. "upper-left")
    # Goal-shell escape attractor metadata (defaults for regular final-goal attractors)
    kind:           str                  = "goal"       # "goal" | "escape"
    target_name:    str                  = "final_goal"
    shell_sector:   Optional[str]        = None         # e.g. "shell_pos_y_neg_z"
    shell_waypoint: Optional[np.ndarray] = None         # task-space position of shell waypoint
    task_pos:       Optional[np.ndarray] = None         # task-space EE target; when set, force is computed via J⁺ instead of joint-space error


@dataclass
class GeoMultiAttractorDSConfig:
    """
    Gain parameters for GeoMultiAttractorDS.

    Blend schedule (smoothstep):
        α = 0   when clearance ≥ clearance_enter   (far — pure attractor)
        α = 1   when clearance ≤ clearance_full    (close — full shaping)
        Hermite-smoothstepped in between for C¹ continuity.

    clearance_enter must be > clearance_full for a meaningful shaping window.
    Setting clearance_enter ≤ 0 (and clearance_full < clearance_enter) effectively
    disables geometric shaping for configurations with positive clearance.
    """
    # --- Attractor gains ---
    K_c: float = 2.0          # Conservative attractor gain

    # --- Geometric shaping gains ---
    K_null: float = 0.5       # Nullspace clearance-ascent gain

    # --- Blend schedule (two-threshold smoothstep) ---
    clearance_enter: float = 0.15   # m — shaping starts below this clearance
    clearance_full:  float = 0.04   # m — full shaping at or below this clearance
    eps_grad: float = 1e-6          # regulariser in tangent projector denominator

    # --- Speed saturation ---
    max_speed: float = float("inf")

    # --- Goal-proximity weight (mirrors PathDS goal_radius) ---
    goal_radius: float = 0.05   # rad — geometric components vanish inside this radius

    # --- Attractor switching ---
    switch_hysteresis: float = 0.20   # score margin required to switch

    # --- Finite-difference step sizes ---
    clearance_grad_eps: float = 1e-3   # joint perturbation for ∇h
    jacobian_eps: float = 1e-4         # joint perturbation for J

    # --- One-step scoring ---
    dt_score: float = 0.01     # prediction dt (seconds) for attractor scoring

    # Scoring weights
    w_goal_progress: float = 2.0
    w_clearance:     float = 3.0
    w_align:         float = 1.0
    w_manip:         float = 0.5
    w_switch:        float = 0.5

    # --- Lookahead safety (discrete-step penetration prevention) ---
    # When dt is supplied to compute(), a one-step clearance check is run on
    # q + dt*qdot_des.  If the predicted clearance falls below
    # lookahead_min_clearance, qdot_des is scaled down linearly so the arm
    # lands just at that threshold.  Adds one clearance_fn call per step
    # only when a collision is predicted — negligible overhead.
    lookahead_safety: bool = True
    lookahead_min_clearance: float = 0.001   # m — minimum acceptable clearance after step

    # --- Per-step timing instrumentation ---
    # When True, compute() populates GeoMultiAttractorDSResult.timing_ms with
    # a dict of phase → milliseconds.  Adds ~0.05 ms overhead per step.
    enable_timing: bool = False

    # --- Horizon scoring (min-clearance lookahead) ---
    # When batch_from_lp_fn is available (factory path), evaluates minimum
    # clearance along the predicted conservative trajectory for each attractor
    # using one vectorised _panda_fk_batch call.  Refreshed every horizon_rate
    # steps and cached between refreshes (~0.2 ms amortised overhead).
    horizon_n_steps: int   = 8     # steps per attractor in the predicted trajectory
    horizon_dt:      float = 0.02  # dt for each horizon step (seconds)
    horizon_rate:    int   = 5     # refresh every N control steps
    w_static:        float = 1.0   # weight of IKAttractor.static_score
    w_horizon_min_cl:    float = 2.0   # weight of min predicted clearance
    w_horizon_collision: float = 5.0   # penalty when predicted min clearance < 0

    # --- Switch safety gate ---
    # Blocks attractor switches when the immediate one-step clearance under
    # the candidate attractor's velocity field is below switch_min_clearance.
    switch_safety_gate:   bool  = True
    switch_min_clearance: float = 0.005   # m — minimum predicted clearance to allow switch

    # --- Trap-aware reselection ---
    # When goal error stagnates (robot stuck) and alpha > 0.3, hysteresis is
    # temporarily reduced by trap_hysteresis_reduction to allow switching.
    trap_detection:            bool  = True
    trap_n_steps:              int   = 50    # history window length (steps)
    trap_min_progress:         float = 0.002 # rad — goal-error decrease required to not trap
    trap_hysteresis_reduction: float = 0.4   # hysteresis multiplier when trapped

    # --- Near-goal switch lockout ---
    # Once the robot is within goal_switch_lock_radius (joint-space radians) of the
    # active goal, attractor switching is frozen.  Prevents near-goal chattering
    # caused by attractors with nearly identical scores oscillating back and forth.
    enable_goal_switch_lock:   bool  = True
    goal_switch_lock_radius:   float = 0.05  # rad — joint-space L2 distance to active goal

    # --- Switch categorisation threshold ---
    # alpha above this value at switch time → "obstacle_region_switch"; else "route_switch".
    alpha_obstacle_threshold:  float = 0.5

    # --- Stall-escape triggered switch (disabled by default) ---
    # If goal progress has stagnated for stall_window_steps and alpha > 0.5,
    # temporarily force consideration of the best non-active family.
    # Enable only after diagnostics confirm a useful non-active family exists.
    enable_stall_escape_switch:      bool  = False
    stall_window_steps:              int   = 25
    stall_goal_progress_threshold:   float = 1e-3   # rad — minimum progress to not stall

    # --- Forced stall switch (diagnostic, disabled by default) ---
    # Cycles or randomizes among attractors when goal progress stalls, ignoring
    # scoring/hysteresis. Use to diagnose whether switching same-family attractors
    # can escape local minima when no below-cross attractor exists.
    enable_forced_stall_switch:            bool  = False
    forced_stall_switch_mode:              str   = "cycle"   # "cycle"|"best_nonactive"|"random"
    forced_stall_window_steps:             int   = 25
    forced_stall_goal_progress_threshold:  float = 1e-3
    forced_stall_alpha_threshold:          float = 0.5
    forced_stall_cooldown_steps:           int   = 20
    forced_stall_max_switches:             int   = 10

    # --- Goal-shell escape attractors (disabled by default) ---
    # Homotopy-inspired task-space approach sectors sampled from a shell around the
    # goal.  Provides temporary waypoint attractors when final-goal IK diversity is
    # insufficient to escape a local minimum.
    enable_goal_shell_escape:               bool  = False
    goal_shell_radius_m:                    float = 0.22
    goal_shell_margin_m:                    float = 0.06
    goal_shell_max_waypoints:               int   = 3
    goal_shell_ik_solutions_per_waypoint:   int   = 1
    goal_shell_sample_mode:                 str   = "yz_cross"
    goal_shell_trigger_alpha:               float = 0.5
    goal_shell_trigger_clearance_m:         float = 0.04
    goal_shell_stall_window_steps:          int   = 25
    goal_shell_stall_progress_threshold:    float = 1e-3
    goal_shell_escape_boost:                float = 5.0
    goal_shell_goal_return_boost:           float = 5.0
    goal_shell_reached_tol_m:               float = 0.04
    goal_shell_clear_alpha_threshold:       float = 0.3
    goal_shell_cooldown_steps:              int   = 30
    goal_shell_max_generations:             int   = 2
    goal_shell_max_escape_switches:         int   = 6
    goal_shell_blacklist_failed_sectors:    bool  = True

    # --- Simple escape waypoint (disabled by default) ---
    # Single scene-aware escape waypoint using position-only IK.
    # Designed for yz_cross where the arm must go below the cross before
    # approaching the bottom-right goal quadrant.
    enable_simple_escape_waypoint:               bool  = False
    escape_waypoint_mode:                        str   = "yz_cross_below"
    escape_waypoint_batch_size:                  int   = 128
    escape_waypoint_num_solutions:               int   = 1
    escape_waypoint_reached_tol_m:               float = 0.04
    escape_waypoint_trigger_alpha:               float = 0.5
    escape_waypoint_stall_window_steps:          int   = 25
    escape_waypoint_stall_progress_threshold:    float = 1e-3
    escape_waypoint_max_generations:             int   = 1
    escape_waypoint_cross_center_z:              float = 0.0   # set externally for yz_cross
    escape_waypoint_goal_error_threshold:        float = 0.10  # min goal error to allow escape switch

    # --- Clearance recovery (disabled by default) ---
    # Before using a portal escape waypoint, first move the arm away from the
    # closest obstacle to recover clearance from ~1 mm to a safer value.
    # Sequence: goal → recovery → portal → goal.
    enable_clearance_recovery:               bool  = False
    recovery_trigger_clearance_m:            float = 0.02   # enter recovery when clearance below this
    recovery_target_clearance_m:             float = 0.06   # exit recovery when clearance above this
    recovery_distance_m:                     float = 0.08   # task-space distance for recovery waypoint
    recovery_reached_tol_m:                  float = 0.03   # task-space tolerance to declare waypoint reached
    recovery_stall_window_steps:             int   = 25     # goal-error history window for stall detection
    recovery_stall_progress_threshold:       float = 1e-3   # min goal-error decrease to not be stalled
    recovery_alpha_threshold:                float = 0.5    # obstacle blending must exceed this
    recovery_max_generations:                int   = 2      # max recovery attempts per trial
    recovery_ik_batch_size:                  int   = 128
    recovery_ik_num_solutions:               int   = 1
    recovery_position_tol_m:                 float = 0.08   # IK position tolerance for recovery waypoint
    recovery_cooldown_steps:                 int   = 20     # min steps between recovery attempts
    recovery_max_steps:                      int   = 40     # force exit recovery after this many steps (timeout)

    # --- Boundary escape waypoints (disabled by default) ---
    # Homotopy-proxy escape attractors at obstacle boundary corners.
    # Each waypoint defines a distinct route family around the obstacle.
    # Replaces the goal-shell and simple-escape-waypoint approaches.
    enable_boundary_escape_waypoints:                bool  = False
    boundary_escape_max_waypoints:                   int   = 7
    boundary_escape_margin_m:                        float = 0.08
    boundary_escape_min_clearance_m:                 float = 0.05
    boundary_escape_ik_batch_size:                   int   = 128
    boundary_escape_ik_num_solutions:                int   = 1
    boundary_escape_stall_window_steps:              int   = 25
    boundary_escape_stall_progress_threshold:        float = 1e-3
    boundary_escape_cross_center_z:                  float = 0.0
    boundary_escape_goal_error_threshold:            float = 0.10
    boundary_escape_reached_tol_m:                   float = 0.04
    boundary_escape_family_cycle_window:             int   = 30
    boundary_escape_family_cycle_progress_threshold: float = 1e-3

    # --- Escape family cycling (distance-to-waypoint progress tracking) ---
    enable_escape_family_cycling:               bool  = True
    escape_family_cycle_window_steps:           int   = 40
    escape_family_progress_threshold_m:         float = 0.01
    escape_family_min_steps_before_cycle:       int   = 25
    escape_family_max_cycles:                   int   = 8
    escape_family_reached_tol_m:                float = 0.05
    escape_family_clearance_success_m:          float = 0.04
    below_cross_max_dist_to_esc_m:              float = 0.10
    escape_family_cycle_policy:                 str   = "score_then_untried"
    # allowed: "score_then_untried" | "fixed_order" | "nearest_untried" | "random_untried"
    escape_family_allow_retry_after_all_failed: bool  = False

    # --- Boundary escape IK build mode ---
    boundary_escape_build_mode: str = "prebuild"
    # allowed: "prebuild" | "on_stall"

    # --- Gradient-based clearance recovery (used as override inside escape mode) ---
    # When the arm is pinned near contact, escape waypoints cannot help.
    # This directly overrides qdot_des with the clearance gradient until clearance improves.
    critical_clearance_m:     float = 0.005   # below this: full gradient, no portal blend
    recovery_release_clearance_m: float = 0.04  # exit gradient recovery above this
    K_clearance_recovery:     float = 1.5     # joint-space velocity scale for gradient push
    max_grad_recovery_steps:  int   = 100     # steps before logging recovery failure

    # --- Backtrack / staging attractor ---
    # After clearance recovery, move the arm toward a safe staging configuration
    # (e.g. q_start) before attempting boundary escape waypoints.
    # Sequence: GOAL → RECOVERY → BACKTRACK → ESCAPE → GOAL
    enable_backtrack_staging:      bool  = False
    backtrack_target_mode:         str   = "partial_to_start"
    # allowed: "q_start" | "partial_to_start"
    backtrack_partial_beta:        float = 0.5   # fraction toward q_start for partial mode
    backtrack_reached_tol_rad:     float = 0.25  # joint-space distance to declare reached
    backtrack_max_steps:           int   = 120   # timeout before forcing escape
    backtrack_min_clearance_m:     float = 0.03  # required clearance for timeout exit
    backtrack_escape_dist_threshold_rad: float = 0.6  # min goal_err to allow escape after backtrack
    backtrack_velocity_scale:      float = 1.5   # scale K_c when backtrack attractor is active

    # --- Online attractor generation ---
    # When "online", escape/backtrack/pre_goal IK runs at runtime (fair timing).
    # When "prebuild_debug", attractors are built at setup time (not valid for benchmarks).
    attractor_generation_mode: str = "online"
    # allowed: "online" | "prebuild_debug"

    # --- Pre-goal waypoint (online mode only) ---
    # After escape exits via below_cross, run IK at the goal EE position to generate
    # a pre_goal attractor that guides the arm to the final goal from below the barrier.
    enable_pre_goal_waypoint:   bool  = False
    pregoal_reached_tol_rad:    float = 0.10   # exit pre-goal when within this distance of target
    pregoal_max_steps:          int   = 300    # timeout steps in pre-goal mode
    pregoal_log_interval_steps: int   = 20     # how often to log [pregoal_active ...]
    pregoal_min_clearance_m:    float = 0.02   # reject pre_goal IK result if clearance below this
    pregoal_intermediate_beta:  float = 0.80   # interpolation fraction toward goal (0=current, 1=goal)
    pregoal_upgrade_on_near_goal: bool = True  # after intermediate pregoal exits, re-IK at full goal
    pregoal_stall_intervals:    int   = 3     # consecutive log-intervals with no progress → stall exit
    pregoal_stall_min_progress: float = 0.05  # min dist_rad decrease per interval to count as progress
    pregoal_beta_sweep:         List[float] = field(default_factory=lambda: [0.20, 0.40, 0.60, 0.80])
    pregoal_max_qdist:          float = 5.0     # reject pre_goal IK if q_dist > this (rad)
    pregoal_early_exit_clearance: float = 0.10  # stop beta sweep early if first good candidate found
    # Progressive re-entry: when pregoal exits near_goal but final goal is unsafe, regenerate
    # a new pregoal at a slightly higher beta before returning to goal attractors.
    pregoal_progressive_max_regen: int   = 2      # max additional pregoal stages after first exit
    pregoal_reentry_h_current:     float = 0.025  # min current clearance to return to final goal
    pregoal_reentry_h_hat:         float = 0.015  # min predicted clearance toward final goal
    pregoal_reentry_dt:            float = 0.03   # step size for one-step goal-direction prediction
    pregoal_reentry_beta_delta:    float = 0.10   # beta increment per progressive pregoal regen

    # --- Online IK batch sizes ---
    # Smaller batches reduce per-event latency; default 64 is a good starting point.
    online_escape_ik_batch_size:  int = 256
    online_pregoal_ik_batch_size: int = 256

    # --- Online escape candidate selection policy ---
    # "score_first": pick highest geometry score (robust, default)
    # "fast_first":  fixed order (top families first — typically faster IK)
    online_escape_candidate_policy: str = "score_first"

    # --- Online IK fail-fast budget ---
    # When > 0: candidates that exceed this latency are marked slow and deprioritized.
    online_ik_max_ms: float = 0.0

    # --- YZ-cross route staging (online mode only) ---
    # Escape proceeds in two mandatory stages instead of treating waypoints as
    # independent alternatives:
    #   "below":   get the EE below the horizontal bar first
    #   "lateral": then cross laterally to the goal Y-side of the vertical bar
    # Only works when enable_boundary_escape_waypoints is True and the scene is
    # a frontal_yz_cross (candidate metadata must include goal_side_y_sign).
    enable_yz_route_staging:           bool  = True
    route_stage_below_z_margin_m:      float = 0.08   # Z must be this far below cross centre
    route_stage_lateral_reached_tol_m: float = 0.08   # task-space tol for lateral waypoint
    route_stage_max_steps:             int   = 120     # steps allowed per stage before giving up


@dataclass
class GeoMultiAttractorDSResult:
    """
    Per-timestep output of GeoMultiAttractorDS.

    Component breakdown is provided so the impedance/passivity controller
    can gate f_tangent and f_null independently while leaving f_attractor
    ungated (it is conservative).

    Scalar norms (raw_attractor_norm, tangent_norm, null_norm) are provided
    as fast diagnostics without needing to call np.linalg.norm on the arrays.
    """
    qdot_des:  np.ndarray   # (7,) total desired joint velocity

    # Component vectors (all shape (7,))
    f_attractor: np.ndarray   # conservative pull toward active goal
    f_tangent:   np.ndarray   # tangent-plane correction (scaled by α·w)
    f_null:      np.ndarray   # nullspace clearance ascent (scaled by α·w)

    # Scalar norms (convenience for plotting / logging without extra linalg calls)
    raw_attractor_norm: float   # ‖f_attractor‖
    tangent_norm:       float   # ‖f_tangent‖
    null_norm:          float   # ‖f_null‖

    # Active attractor
    active_attractor_idx: int
    active_family:        str
    n_switches:           int    # total switches since last reset()

    # Scores
    score_active:      float
    score_best:        float
    best_attractor_idx: int          # index of highest-scoring attractor (before hysteresis)
    all_scores:        List[float]   # per-attractor scores for full landscape visibility

    # Geometry diagnostics
    clearance:                 float
    clearance_grad_norm:       float
    obs_blend_alpha:           float   # α ∈ [0, 1]: 0=far, 1=full shaping
    tangent_projection_fraction: float  # ‖P_tan @ f_att‖ / (‖f_att‖ + ε)
                                        # reflects geometry, independent of α

    # State
    goal_error: float   # ‖q − q_goal_active‖

    # Optional per-phase timing (populated when enable_timing=True)
    timing_ms: Optional[Dict[str, float]] = None

    # Extended scoring diagnostics (new fields; backward-compatible defaults)
    all_scores_combined: List[float] = field(default_factory=list)  # combined = one_step + static + horizon
    horizon_scores:      List[float] = field(default_factory=list)  # per-attractor horizon min-cl score
    static_scores:       List[float] = field(default_factory=list)  # per-attractor static_score values
    # Switch quality
    switch_blocked:       bool             = False  # True if a switch was blocked by safety gate
    switch_blocked_count: int              = 0      # total blocked switches since reset
    switch_event:         Optional[dict]   = None   # non-None when a switch occurred or was blocked
    # Trap state
    trap_detected:          bool = False   # True if goal-error stagnation detected
    trap_forced_reselection: bool = False  # True if hysteresis was reduced due to trap
    # Switch categorisation (cumulative since reset)
    n_switch_near_goal:       int  = 0     # switches while goal_error < goal_switch_lock_radius
    n_switch_obstacle_region: int  = 0     # switches while alpha > alpha_obstacle_threshold
    n_switch_route:           int  = 0     # all other switches (true homotopy change)
    # Near-goal lock status
    goal_switch_locked:   bool = False     # True if switching was suppressed this step
    # Best non-active attractor diagnostics (for frontal_yz_cross analysis)
    best_nonactive_score:   float = 0.0   # highest combined score among non-active attractors
    best_nonactive_idx:     int   = -1    # index of that attractor (-1 when only one attractor)
    best_nonactive_family:  str   = "none"
    score_margin_to_nonactive: float = 0.0  # best_nonactive_score - active_score (>0 means non-active leads)
    # Forced stall switch diagnostics
    n_switch_forced_stall: int  = 0
    n_switch_total:        int  = 0
    # Goal-shell / portal escape diagnostics
    n_escape_generations:  int           = 0     # times escape mode was activated
    n_escape_attractors:   int           = 0     # number of escape attractors in the DS
    n_escape_switches:     int           = 0     # times switched to an escape attractor
    escape_mode_active:    bool          = False
    active_attractor_kind: str           = "goal"
    active_escape_family:  Optional[str] = None
    # Clearance recovery diagnostics
    n_recovery_switches:      int           = 0     # times switched to a recovery attractor
    escape_state_final:       str           = "goal" # "goal" | "recovery" | "portal"
    # Escape family cycling diagnostics
    n_escape_family_cycles:   int           = 0
    n_escape_families_failed: int           = 0
    escape_exhausted:         bool          = False
    active_escape_family_final: Optional[str] = None
    # Gradient clearance recovery diagnostics
    n_grad_recovery_steps:        int   = 0
    escape_blocked_count:         int   = 0
    grad_recovery_best_clearance: float = 0.0
    # Backtrack staging diagnostics
    n_backtrack_switches:         int   = 0
    n_backtrack_steps:            int   = 0
    backtrack_reached:            bool  = False
    backtrack_final_dist_rad:     float = 0.0
    # Online attractor generation timing
    runtime_planner_ms:          float = 0.0
    runtime_planner_ms_max:      float = 0.0   # peak single-event latency
    runtime_planner_event_count: int   = 0     # total online IK events
    escape_ik_ms:       float = 0.0
    escape_ik_calls:    int   = 0
    pregoal_ik_ms:      float = 0.0
    pregoal_ik_calls:   int   = 0
    # Pre-goal and reentry diagnostics
    n_pregoal_switches:    int   = 0
    n_reentry_fail:        int   = 0
    pregoal_accepted_count: int   = 0
    pregoal_rejected_count: int   = 0
    pregoal_last_clearance: float = -1.0
    pregoal_final_dist_rad: float = -1.0
    n_goal_reentry_blocked: int   = 0


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit vector; return v unchanged if near-zero."""
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v.copy()


def _smoothstep(t: float) -> float:
    """
    Hermite smoothstep: 0 at t=0, 1 at t=1, zero first derivative at endpoints.

        S(t) = t² (3 − 2t)
    """
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _blend_alpha(
    clearance: float,
    clearance_enter: float,
    clearance_full: float,
) -> float:
    """
    Obstacle-proximity blend factor α ∈ [0, 1].

    α = 0   when clearance ≥ clearance_enter  (far — no shaping)
    α = 1   when clearance ≤ clearance_full   (close — full shaping)
    Hermite-smoothstepped between for C¹ continuity.

    Degenerate case (clearance_enter ≤ clearance_full): hard threshold.
    Setting clearance_enter = clearance_full = 0.0 disables shaping for
    configurations with positive clearance (only activates on penetration).
    """
    if clearance_enter <= clearance_full:
        return 1.0 if float(clearance) <= float(clearance_full) else 0.0
    t = (float(clearance_enter) - float(clearance)) / (float(clearance_enter) - float(clearance_full))
    return _smoothstep(t)


def _clearance_gradient(
    q: np.ndarray,
    clearance_fn: Callable[[np.ndarray], float],
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Central-difference gradient of clearance h(q) in joint space.

        ∂h/∂q_i ≈ (h(q + ε eᵢ) − h(q − ε eᵢ)) / (2ε)

    Returns ∇h ∈ ℝⁿ.  Points away from the obstacle (positive = safer direction).
    """
    n = q.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        q_p = q.copy(); q_p[i] += eps
        q_m = q.copy(); q_m[i] -= eps
        grad[i] = (clearance_fn(q_p) - clearance_fn(q_m)) / (2.0 * eps)
    return grad


def _position_jacobian(
    q: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Forward-difference Jacobian of the hand (EE) position w.r.t. joint angles.

        J[i, j] ≈ (p_hand(q + ε eⱼ) − p_hand(q)) / ε

    Returns J ∈ ℝ^{3×7}.  Uses _panda_link_positions(q)[-1] (hand body).
    """
    n = q.shape[0]
    p0 = _panda_link_positions(q)[-1]   # (3,)
    J = np.zeros((3, n))
    for j in range(n):
        q_p = q.copy(); q_p[j] += eps
        J[:, j] = (_panda_link_positions(q_p)[-1] - p0) / eps
    return J


def _ee_position_jacobian(q: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """3×7 position Jacobian of the end-effector (last link) via finite differences."""
    ee0 = np.array(_panda_link_positions(q)[-1], dtype=float)
    J = np.zeros((3, 7))
    for j in range(7):
        q_p = q.copy()
        q_p[j] += eps
        J[:, j] = (np.array(_panda_link_positions(q_p)[-1]) - ee0) / eps
    return J


def _nullspace_projector(J: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Nullspace projector:  N = I − J⁺ J

    J⁺ is the regularised right pseudoinverse:  J⁺ = Jᵀ (J Jᵀ + reg I)⁻¹

    Returns N ∈ ℝ^{n×n}.
    """
    m, n = J.shape
    JJT = J @ J.T + reg * np.eye(m)
    J_pinv = J.T @ np.linalg.inv(JJT)   # (n, m)
    return np.eye(n) - J_pinv @ J


def _tangent_projector(grad: np.ndarray, eps_grad: float = 1e-6) -> np.ndarray:
    """
    Tangent-plane projector:  P_tan = I − (∇h ∇hᵀ) / (‖∇h‖² + ε)

    Projects any vector v onto the tangent plane of the obstacle level set
    h(q) = const, removing the component that drives into (or out of) the
    obstacle.  When ‖∇h‖ ≈ 0 (well-clear), P_tan ≈ I (no projection).

    Returns P_tan ∈ ℝ^{n×n}.
    """
    denom = float(np.dot(grad, grad)) + eps_grad
    return np.eye(grad.shape[0]) - np.outer(grad, grad) / denom


def _goal_weight(q: np.ndarray, q_goal: np.ndarray, goal_radius: float) -> float:
    """
    Weight that fades to 0 inside goal_radius (mirrors PathDS _goal_weight).

        w = min(1, ‖q − q_goal‖ / goal_radius)
    """
    dist = float(np.linalg.norm(q - q_goal))
    return min(1.0, dist / goal_radius) if goal_radius > 0 else 1.0


def _attractor_score(
    q: np.ndarray,
    attractor: IKAttractor,
    clearance_fn: Callable[[np.ndarray], float],
    P_tan: np.ndarray,
    f_att: np.ndarray,
    is_active: bool,
    config: GeoMultiAttractorDSConfig,
) -> float:
    """
    One-step score for a single attractor (higher is better).

        score = w_goal_progress · Δgoal_err / dt
              + w_clearance      · c_pred
              + w_align          · alignment
              + w_manip          · manipulability
              - w_switch         · [not active]

    alignment = (P_tan @ f_att) · f̂_att : how much the obstacle-safe
    direction aligns with the goal direction.
    """
    f_att_norm = float(np.linalg.norm(f_att))
    if f_att_norm < 1e-12:
        return -1e9

    f_dir = f_att / f_att_norm
    q_pred = q + config.dt_score * config.K_c * f_dir
    c_pred = float(clearance_fn(q_pred))

    goal_err_now  = float(np.linalg.norm(q - attractor.q_goal))
    goal_err_pred = float(np.linalg.norm(q_pred - attractor.q_goal))
    delta_goal = (goal_err_now - goal_err_pred) / (config.dt_score + 1e-12)

    f_tan_dir = P_tan @ f_att
    alignment = float(np.dot(f_tan_dir, f_att)) / (f_att_norm ** 2 + 1e-12)

    score = (
        config.w_goal_progress * delta_goal
        + config.w_clearance   * c_pred
        + config.w_align       * alignment
        + config.w_manip       * attractor.manipulability
        - config.w_switch      * (0.0 if is_active else 1.0)
    )
    return float(score)


# ---------------------------------------------------------------------------
# Fast-path batch helpers (used by GeoMultiAttractorDS when batch_from_lp_fn
# is available — i.e. when the DS is constructed by the factory)
# ---------------------------------------------------------------------------

def _fast_gradient_and_jacobian(
    q: np.ndarray,
    lp_base: np.ndarray,
    clearance_at_q: float,
    batch_from_lp_fn: Callable,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-difference clearance gradient and EE Jacobian from ONE batch FK call.

    Shares the n forward-perturbed configurations between the gradient and the
    Jacobian, replacing 14 (central-diff gradient) + 7 (Jacobian) = 21 scalar
    FK calls with a single vectorised _panda_fk_batch(n_configs) call.

    Accuracy: O(eps) vs the scalar central-difference O(eps²), but eps=1e-3 is
    more than sufficient for the tangent projector and nullspace computations.

    Args:
        q:               Current joint config  (n,).
        lp_base:         (8, 3) link positions at q — already computed by caller.
        clearance_at_q:  Scalar clearance at q — already computed by caller.
        batch_from_lp_fn: (lp: (N, 8, 3)) -> (N,) vectorised clearance.
        eps:             Forward-difference step size.

    Returns:
        grad_h: (n,) forward-difference gradient ∂h/∂q.
        J:      (3, n) EE position Jacobian (forward difference).
    """
    n = q.shape[0]
    # n forward-perturbed configs: q_pos[i] = q with q_pos[i, i] += eps
    q_pos = np.tile(q, (n, 1))
    q_pos[np.arange(n), np.arange(n)] += eps

    lp_pos = _panda_fk_batch(q_pos)           # (n, 8, 3) — ONE batch call

    cl_pos = batch_from_lp_fn(lp_pos)         # (n,) — vectorised, no FK
    grad_h = (cl_pos - clearance_at_q) / eps  # (n,)

    ee_base = lp_base[-1]                              # (3,)
    J = (lp_pos[:, -1, :] - ee_base).T / eps          # (3, n)

    return grad_h, J


def _batch_score_attractors(
    q: np.ndarray,
    attractors: List[IKAttractor],
    batch_from_lp_fn: Callable,
    P_tan: np.ndarray,
    active_idx: int,
    config: GeoMultiAttractorDSConfig,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Score all attractors with ONE _panda_fk_batch call.

    Replaces N_att calls to clearance_fn(q_pred) with a single batch FK +
    vectorised clearance evaluation.

    Returns:
        scores: list of float, one per attractor (higher = better).
        f_atts: list of (n,) attractor velocity vectors.
    """
    n_att = len(attractors)
    n     = q.shape[0]
    f_atts  : List[np.ndarray] = []
    q_preds : np.ndarray       = np.empty((n_att, n))

    # Pre-compute EE position and Jacobian if any task-space attractor is present.
    _task_att_present = any(a.task_pos is not None for a in attractors)
    _ee_pos_score: Optional[np.ndarray] = None
    _J_pinv_score: Optional[np.ndarray] = None
    if _task_att_present:
        _ee_pos_score = np.array(_panda_link_positions(q)[-1], dtype=float)
        _J_score = _ee_position_jacobian(q)
        _J_pinv_score = np.linalg.pinv(_J_score)

    for i, att in enumerate(attractors):
        if att.task_pos is not None:
            x_err = att.task_pos - _ee_pos_score
            f_att_i = config.K_c * (_J_pinv_score @ x_err)
        else:
            f_att_i = -config.K_c * (q - att.q_goal)
        f_att_norm = float(np.linalg.norm(f_att_i))
        if f_att_norm > 1e-12:
            f_dir      = f_att_i / f_att_norm
            q_preds[i] = q + config.dt_score * config.K_c * f_dir
        else:
            q_preds[i] = q.copy()
        f_atts.append(f_att_i)

    lp_preds = _panda_fk_batch(q_preds)      # (n_att, 8, 3)
    c_preds  = batch_from_lp_fn(lp_preds)    # (n_att,)

    scores: List[float] = []
    for i, att in enumerate(attractors):
        f_att_i    = f_atts[i]
        f_att_norm = float(np.linalg.norm(f_att_i))
        if f_att_norm < 1e-12:
            scores.append(-1e9)
            continue

        f_dir      = f_att_i / f_att_norm
        q_pred_i   = q + config.dt_score * config.K_c * f_dir
        c_pred     = float(c_preds[i])

        if att.task_pos is not None:
            # Task-space progress: EE distance to waypoint
            goal_err_now  = float(np.linalg.norm(_ee_pos_score - att.task_pos))
            goal_err_pred = float(np.linalg.norm(lp_preds[i, -1] - att.task_pos))
        else:
            goal_err_now  = float(np.linalg.norm(q        - att.q_goal))
            goal_err_pred = float(np.linalg.norm(q_pred_i - att.q_goal))
        delta_goal    = (goal_err_now - goal_err_pred) / (config.dt_score + 1e-12)

        f_tan_dir = P_tan @ f_att_i
        alignment = float(np.dot(f_tan_dir, f_att_i)) / (f_att_norm ** 2 + 1e-12)

        score = (
            config.w_goal_progress * delta_goal
            + config.w_clearance   * c_pred
            + config.w_align       * alignment
            + config.w_manip       * att.manipulability
            - config.w_switch      * (0.0 if i == active_idx else 1.0)
        )
        scores.append(float(score))

    return scores, f_atts


def _compute_horizon_scores(
    q: np.ndarray,
    attractors: List["IKAttractor"],
    clearance_at_q: float,
    batch_from_lp_fn: Callable,
    cfg: "GeoMultiAttractorDSConfig",
) -> List[float]:
    """
    Minimum-clearance horizon score for each attractor.

    Generates predicted trajectories for all attractors simultaneously using
    the analytic linear-spring Euler formula (no geometric shaping, no MuJoCo):

        q_k = q_goal + (q_0 − q_goal) · (1 − K_c · dt)^k

    Then evaluates clearance at all predicted positions with ONE vectorised
    _panda_fk_batch call, returning per-attractor scores:

        score_i = w_horizon_min_cl · min_cl_i − w_horizon_collision · [min_cl_i < 0]

    This directly penalises attractors whose conservative trajectories pass
    through the Cross (or any obstacle) while rewarding clear-path attractors.

    Only called when batch_from_lp_fn is available (factory/static-obstacle path).
    """
    n_att  = len(attractors)
    N      = cfg.horizon_n_steps
    decay  = 1.0 - cfg.K_c * cfg.horizon_dt

    q_goals    = np.array([a.q_goal for a in attractors], dtype=float)  # (n_att, 7)
    delta_0    = q[None, :] - q_goals                                    # (n_att, 7)
    decay_pows = np.array([decay ** (k + 1) for k in range(N)], dtype=float)  # (N,)

    # q_traj[i, k] = q_goals[i] + delta_0[i] * decay^(k+1)
    q_traj  = q_goals[:, None, :] + delta_0[:, None, :] * decay_pows[None, :, None]
    # Shape: (n_att, N, 7)

    q_flat  = q_traj.reshape(n_att * N, q.shape[0])   # (n_att*N, 7)
    lp_flat = _panda_fk_batch(q_flat)                  # (n_att*N, 8, 3)
    cl_flat = batch_from_lp_fn(lp_flat)                # (n_att*N,)
    cl_traj = cl_flat.reshape(n_att, N)                # (n_att, N)

    # Include current clearance so the score accounts for the arm's current state
    min_cl = np.minimum(clearance_at_q, cl_traj.min(axis=-1))  # (n_att,)

    scores = []
    for i in range(n_att):
        mc  = float(min_cl[i])
        col = 1.0 if mc < 0.0 else 0.0
        scores.append(cfg.w_horizon_min_cl * mc - cfg.w_horizon_collision * col)
    return scores


# ---------------------------------------------------------------------------
# GeoMultiAttractorDS
# ---------------------------------------------------------------------------

class GeoMultiAttractorDS:
    """
    Multi-attractor Dynamical System with differential-geometric obstacle
    shaping.

    Usage::

        attractors = [IKAttractor(q_goal=..., family=..., clearance=..., manipulability=...)]
        ds = GeoMultiAttractorDS(attractors, clearance_fn, config=GeoMultiAttractorDSConfig())

        q_dot, result = ds.compute(q, qdot=qdot, dt=0.01)

        # result.f_attractor  — conservative, passivity-safe (ungated)
        # result.f_tangent    — geometric correction (gate with β_tangent)
        # result.f_null       — nullspace clearance ascent (gate with β_null)
        # result.obs_blend_alpha — 0 = far from obstacles, 1 = full shaping
    """

    def __init__(
        self,
        attractors: List[IKAttractor],
        clearance_fn: Callable[[np.ndarray], float],
        config: Optional[GeoMultiAttractorDSConfig] = None,
        batch_from_lp_fn: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            attractors:       List of IKAttractor goal configs.
            clearance_fn:     (q) -> float — scalar clearance (always required;
                              used for base clearance and lookahead check).
            config:           Optional config (uses defaults if None).
            batch_from_lp_fn: Optional (lp: (N,8,3)) -> (N,) vectorised
                              clearance from pre-computed link positions.
                              When provided, gradient + Jacobian + attractor
                              scoring each use one _panda_fk_batch call
                              instead of 14+7+N_att scalar FK calls.
                              Supplied automatically by build_geo_multi_attractor_ds;
                              pass None for dynamic clearance_fn (e.g. moving
                              obstacles) where static precomputation cannot be used.
        """
        if not attractors:
            raise ValueError("GeoMultiAttractorDS requires at least one attractor.")
        self.attractors       = attractors
        self.clearance_fn     = clearance_fn
        self.config           = config or GeoMultiAttractorDSConfig()
        self._batch_from_lp_fn = batch_from_lp_fn   # None → scalar FD fallback

        # Mutable state
        self._active_idx: int = self._best_static_idx()
        self._n_switches: int = 0
        # Horizon scoring cache (refreshed every horizon_rate steps)
        self._step_count:            int                    = 0
        self._cached_horizon_scores: Optional[List[float]] = None
        self._horizon_last_update:   int                    = -9999
        # Goal-error history for trap detection
        self._goal_error_history: List[float] = []
        # Switch diagnostics
        self._switch_blocked_count:    int = 0
        self._n_switch_near_goal:      int = 0
        self._n_switch_obstacle_region: int = 0
        self._n_switch_route:          int = 0
        self._n_switch_forced_stall:    int = 0
        self._last_forced_switch_step:  int = -9999
        self._rng: np.random.Generator  = np.random.default_rng()
        # Goal-shell / portal escape state
        self._escape_mode_active:           bool                  = False
        self._active_escape_target_pos:     Optional[np.ndarray] = None
        self._active_escape_family:         Optional[str]        = None
        self._attempted_escape_families:    set                  = set()
        self._last_escape_generation_step:  int                  = -9999
        self._n_escape_generations:         int                  = 0
        self._n_escape_switches:            int                  = 0
        # EE position history for simple escape stall detection.
        # Unlike _goal_error_history this is NOT reset on attractor switches,
        # so it accumulates continuously and reflects actual arm stagnation.
        self._ee_stall_history: List[np.ndarray] = []
        # Clearance recovery state (goal → recovery → portal → goal)
        self._in_recovery_mode:            bool                  = False
        self._active_recovery_target_pos:  Optional[np.ndarray] = None
        self._n_recovery_generations:      int                   = 0
        self._n_recovery_switches:         int                   = 0
        self._last_recovery_step:          int                   = -9999
        self._recovery_start_step:         int                   = -9999
        # Optional runtime recovery builder — set by factory when enable_clearance_recovery
        self._build_recovery_attractor_fn                        = None
        # Boundary escape family cycling state
        self._escape_family_last_switch_step: int   = -9999
        self._escape_family_start_goal_error: float = float("inf")
        # Escape-family progress tracking (distance to active waypoint, not joint-space goal error)
        self._escape_dist_history: Deque[float]  = deque(maxlen=100)
        self._failed_escape_families: set        = set()
        self._n_escape_family_cycles: int        = 0
        self._escape_exhausted: bool             = False
        # YZ-cross route-stage state machine
        self._escape_route_stage: str   = "none"    # "none" | "below" | "lateral"
        self._route_stage_start_step: int         = -1
        self._route_stage_goal_side_y_sign: float = 0.0
        self._route_stage_cross_center_y: float   = 0.0
        self._route_stage_cross_center_z: float   = 0.0
        self._route_stage_cross_z_low: float      = -1.0  # bottom of vertical bar (from metadata)
        # Gradient-based clearance recovery (override qdot_des when pinned near contact)
        self._grad_recovery_active:           bool  = False
        self._grad_recovery_start_clearance:  float = 0.0
        self._grad_recovery_best_clearance:   float = 0.0
        self._grad_recovery_start_step:       int   = -9999
        self._n_grad_recovery_steps:          int   = 0
        self._escape_blocked_count:           int   = 0
        # Backtrack / staging (runs between clearance recovery and escape portal)
        self._backtrack_mode_active:          bool              = False
        self._backtrack_target:               Optional[np.ndarray] = None
        self._backtrack_start_step:           int               = -9999
        self._n_backtrack_steps:              int               = 0
        self._n_backtrack_switches:           int               = 0
        self._backtrack_reached:              bool              = False
        # Online attractor generation
        self._deferred_escape_candidates:     list              = []
        self._online_ik_fn                                      = None
        self._ik_executor                                       = None  # lazy ThreadPoolExecutor(max_workers=1)
        self._q_start:                        Optional[np.ndarray] = None
        self._yz_cross_geometry_cache                           = None
        self._goal_task_pos:                  Optional[np.ndarray] = None
        # Online timing accumulators
        self._escape_ik_ms:                   float             = 0.0
        self._escape_ik_calls:                int               = 0
        self._pregoal_ik_ms:                  float             = 0.0
        self._pregoal_ik_calls:               int               = 0
        self._runtime_planner_ms:             float             = 0.0
        self._runtime_planner_ms_max:         float             = 0.0
        self._runtime_planner_event_count:    int               = 0
        # Pre-goal state
        self._pre_goal_mode_active:           bool              = False
        self._pregoal_start_step:             int               = 0
        self._n_pregoal_switches:             int               = 0
        self._pregoal_generated:              bool              = False
        self._pregoal_is_upgrade:             bool              = False
        self._pregoal_dist_history:           List[float]       = []
        self._pregoal_accepted_count:         int               = 0
        self._pregoal_rejected_count:         int               = 0
        self._pregoal_last_clearance:         float             = -1.0
        self._pregoal_last_task_err:          float             = -1.0
        self._pregoal_final_dist_rad:         float             = -1.0
        self._pregoal_last_beta:              float             = -1.0
        self._pregoal_regen_count:            int               = 0
        self._n_goal_reentry_blocked:         int               = 0
        self._n_reentry_fail:                 int               = 0
        # Task-space stall tracking for taskspace_goal family
        self._taskspace_err_history:          List[float]       = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _best_static_idx(self) -> int:
        """Return the attractor index with the highest static_score.
        Falls back to 0 when no static scores have been set (all zero)."""
        scores = [a.static_score for a in self.attractors]
        return int(np.argmax(scores)) if max(scores) > 0.0 else 0

    def _select_next_escape_family(
        self,
        scores_combined: List[float],
        ee_pos: np.ndarray,
    ) -> Optional[int]:
        """
        Return the index of the next escape attractor to try.

        Respects escape_family_cycle_policy and excludes _failed_escape_families.
        Returns None when all families are exhausted.
        """
        cfg = self.config
        candidates = [
            (i, a) for i, a in enumerate(self.attractors)
            if a.kind == "escape" and a.shell_sector not in self._failed_escape_families
        ]
        # Route-stage: restrict cycling to stage-appropriate candidates using stage_tags.
        # Look up tags from the original deferred candidates by family (shell_sector).
        if (
            cfg.enable_yz_route_staging
            and self._escape_route_stage in ("below", "lateral")
        ):
            _stage_tag = (
                "lateral_stage_candidate"
                if self._escape_route_stage == "lateral"
                else "below_stage_candidate"
            )
            _tags_by_family = {
                c.family: c.stage_tags
                for c in self._deferred_escape_candidates
            }
            _stage_cands = [
                (i, a) for i, a in candidates
                if _stage_tag in _tags_by_family.get(a.shell_sector, [])
            ]
            if _stage_cands:
                candidates = _stage_cands
        if not candidates:
            if cfg.escape_family_allow_retry_after_all_failed:
                self._failed_escape_families.clear()
                candidates = [(i, a) for i, a in enumerate(self.attractors) if a.kind == "escape"]
            if not candidates:
                return None

        policy = cfg.escape_family_cycle_policy

        if policy == "fixed_order":
            _order = [
                "escape_bottom_right", "escape_bottom_left",
                "escape_top_right",    "escape_top_left",
            ]
            for sector in _order:
                for i, a in candidates:
                    if a.shell_sector == sector:
                        return i
            return candidates[0][0]

        if policy == "nearest_untried":
            def _dist_to_wp(item: tuple) -> float:
                _, a = item
                if a.shell_waypoint is None:
                    return float("inf")
                return float(np.linalg.norm(ee_pos - a.shell_waypoint))
            return min(candidates, key=_dist_to_wp)[0]

        if policy == "random_untried":
            return int(self._rng.choice([i for i, _ in candidates]))

        # Default: "score_then_untried" — use existing combined score + bonus for untried
        _UNTRIED_BONUS = 10.0
        best_i, best_s = -1, float("-inf")
        for i, a in candidates:
            s = scores_combined[i]
            if a.shell_sector not in self._attempted_escape_families:
                s += _UNTRIED_BONUS
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    # ------------------------------------------------------------------
    # Online attractor generation helpers
    # ------------------------------------------------------------------

    def _online_generate_escape_attractors(
        self,
        q: np.ndarray,
        scores: list,
        f_atts: list,
        static_sc: list,
        h_scores: list,
        scores_combined: list,
        P_tan: np.ndarray,
    ) -> int:
        """Run IK on each deferred escape candidate and append escape attractors.

        Returns the number of attractors successfully added.
        """
        cfg = self.config
        candidates = self._deferred_escape_candidates
        if not candidates or self._online_ik_fn is None:
            return 0
        n_added = 0
        _batch_t0 = time.perf_counter()
        for cand in candidates:
            cand.ik_generated = True
            if cand.q_esc_prefilled is not None:
                q_esc = cand.q_esc_prefilled
                print(
                    f"[online_escape_prefilled step={self._step_count} family={cand.family} ik_ms=0.0 ik_ok=True]",
                    flush=True,
                )
            else:
                t0 = time.perf_counter()
                try:
                    q_esc = self._online_ik_fn(cand.pos, batch_size=cfg.online_escape_ik_batch_size)
                except Exception:
                    q_esc = None
                ik_ms = (time.perf_counter() - t0) * 1000.0
                cand.last_ik_ms = ik_ms
                cand.ik_cost_ema_ms = ik_ms if cand.ik_cost_ema_ms == 0.0 else 0.3*ik_ms + 0.7*cand.ik_cost_ema_ms
                if cfg.online_ik_max_ms > 0.0 and ik_ms > cfg.online_ik_max_ms:
                    cand.ik_slow = True
                self._escape_ik_ms += ik_ms
                self._escape_ik_calls += 1
                self._runtime_planner_ms += ik_ms
                if q_esc is None:
                    cand.ik_failed = True
                print(
                    f"[online_escape_ik step={self._step_count} family={cand.family} "
                    f"ik_ms={ik_ms:.1f} ik_ok={q_esc is not None}]",
                    flush=True,
                )
            if q_esc is not None:
                q_esc = np.asarray(q_esc, dtype=float)
                att = IKAttractor(
                    q_goal=q_esc,
                    family=cand.family,
                    clearance=float(self.clearance_fn(q_esc)),
                    manipulability=0.0,
                    kind="escape",
                    target_name="boundary_escape_waypoint",
                    shell_sector=cand.family,
                    shell_waypoint=cand.pos.copy(),
                )
                self.attractors.append(att)
                _f = -cfg.K_c * (q - att.q_goal)
                f_atts.append(_f)
                scores.append(float(_attractor_score(
                    q, att, self.clearance_fn, P_tan, _f, is_active=False, config=cfg,
                )))
                static_sc.append(0.0)
                h_scores.append(0.0)
                scores_combined.append(scores[-1] + cfg.w_static * static_sc[-1] + h_scores[-1])
                self._cached_horizon_scores = None
                n_added += 1
        _batch_ms = (time.perf_counter() - _batch_t0) * 1000.0
        self._runtime_planner_ms_max       = max(self._runtime_planner_ms_max, _batch_ms)
        self._runtime_planner_event_count += 1
        return n_added

    def _online_generate_next_escape_attractor(
        self,
        q: np.ndarray,
        scores: list,
        f_atts: list,
        static_sc: list,
        h_scores: list,
        scores_combined: list,
        P_tan: np.ndarray,
    ) -> bool:
        """Generate IK for ONE untried deferred escape candidate (highest score first).

        Marks the candidate ik_generated=True before calling IK so repeated calls
        advance through the candidate list one at a time.  Sets ik_failed=True when
        IK returns None.

        Returns True if an escape attractor was successfully appended.
        """
        cfg = self.config
        untried = [
            c for c in self._deferred_escape_candidates
            if not c.ik_generated and not c.ik_failed
        ]
        if not untried:
            return False

        # Route-stage filtering: restrict candidate pool to stage-appropriate candidates
        # using stage_tags (geometric tags, no family-name dependency).
        if cfg.enable_yz_route_staging and self._escape_route_stage != "none":
            _stage = self._escape_route_stage
            if _stage == "below":
                _tag = "below_stage_candidate"
            else:  # "lateral"
                _tag = "lateral_stage_candidate"
            _stage_filtered = [c for c in untried if _tag in c.stage_tags]
            if _stage_filtered:
                untried = _stage_filtered
            elif _stage in ("below", "lateral"):
                # Stage pool exhausted — don't generate wrong-homotopy candidates.
                # _select_next_escape_family will pick from already-resolved attractors.
                return False

        # Runtime scoring: prefer high clearance within the valid pool.
        # Also factor in goal-side progress (Y direction), distance from ee, and IK cost.
        # ee_pos from FK so we don't need to thread it through the call signature.
        _ee_pos_rt = np.array(_panda_link_positions(q)[-1], dtype=float)
        _goal_pos_rt = np.array(
            next(
                (c.metadata["goal_pos"] for c in self._deferred_escape_candidates
                 if "goal_pos" in c.metadata),
                [0.0, 0.0, 0.0],
            ),
            dtype=float,
        )
        _cross_ctr_y = float(next(
            (c.metadata.get("cross_center_y", 0.0) for c in self._deferred_escape_candidates
             if "cross_center_y" in c.metadata),
            0.0,
        ))
        _goal_y_dir = float(np.sign(_goal_pos_rt[1] - _cross_ctr_y)) or 1.0

        _W_CL   = 4.0
        _W_PROG = 2.0
        _W_DIST = 0.5
        _W_IK   = 0.03

        def _runtime_score(c: "EscapeWaypointCandidate") -> float:
            _prog = (c.pos[1] - _ee_pos_rt[1]) * _goal_y_dir
            _dist = float(np.linalg.norm(c.pos - _ee_pos_rt))
            return (
                _W_CL   * min(c.clearance, 1.0)
                + _W_PROG * _prog
                - _W_DIST * _dist
                - _W_IK   * c.ik_cost_ema_ms
            )

        cand = max(untried, key=_runtime_score)
        _rt_score = _runtime_score(cand)

        print(
            f"[online_escape_select step={self._step_count} "
            f"stage={self._escape_route_stage} "
            f"family={cand.family} "
            f"tags={cand.stage_tags} "
            f"cl={cand.clearance:.3f} rt_score={_rt_score:.3f}]",
            flush=True,
        )
        cand.ik_generated = True
        t0 = time.perf_counter()
        # Immediately compute J⁺ displacement toward escape EE position (~0.1 ms, never fails).
        # HJCD-IK fires asynchronously in background to upgrade att.q_goal when ready.
        J     = _position_jacobian(q)
        ee0   = np.array(_panda_link_positions(q)[-1], dtype=float)
        q_esc = q + np.linalg.pinv(J) @ (cand.pos - ee0)
        ik_ms = (time.perf_counter() - t0) * 1000.0
        cand.last_ik_ms     = ik_ms
        cand.ik_cost_ema_ms = ik_ms if cand.ik_cost_ema_ms == 0.0 else 0.3 * ik_ms + 0.7 * cand.ik_cost_ema_ms
        self._escape_ik_ms             += ik_ms
        self._escape_ik_calls          += 1
        self._runtime_planner_ms       += ik_ms
        self._runtime_planner_ms_max    = max(self._runtime_planner_ms_max, ik_ms)
        self._runtime_planner_event_count += 1
        print(
            f"[online_escape_taskspace step={self._step_count} family={cand.family} "
            f"ik_ms={ik_ms:.2f}]",
            flush=True,
        )
        if q_esc is None:
            return False
        q_esc = np.asarray(q_esc, dtype=float)
        att = IKAttractor(
            q_goal=q_esc,
            family=cand.family,
            clearance=float(self.clearance_fn(q_esc)),
            manipulability=0.0,
            kind="escape",
            target_name="boundary_escape_waypoint",
            shell_sector=cand.family,
            shell_waypoint=cand.pos.copy(),
            task_pos=cand.pos.copy(),
        )
        self.attractors.append(att)
        _f = -cfg.K_c * (q - att.q_goal)
        f_atts.append(_f)
        scores.append(float(_attractor_score(
            q, att, self.clearance_fn, P_tan, _f, is_active=False, config=cfg,
        )))
        static_sc.append(0.0)
        h_scores.append(0.0)
        scores_combined.append(scores[-1] + cfg.w_static * static_sc[-1] + h_scores[-1])
        self._cached_horizon_scores = None

        # Fire background HJCD-IK to upgrade att.q_goal with a true IK solution.
        # The J⁺ approximation above is active immediately; this upgrade arrives async.
        if self._online_ik_fn is not None and not cand.ik_pending:
            cand.ik_pending = True
            _ik_fn = self._online_ik_fn
            _pos   = cand.pos.copy()
            _bs    = cfg.online_escape_ik_batch_size
            def _bg_ik(_c=cand, _a=att, _p=_pos, _fn=_ik_fn, _b=_bs):
                try:
                    result = _fn(_p, batch_size=_b)
                    if result is not None:
                        _q = np.asarray(result, dtype=float)
                        _a.q_goal = _q          # GIL-atomic upgrade of live attractor
                        _c.q_esc_prefilled = _q  # record for diagnostics
                        print(
                            f"[online_escape_ik_upgraded family={_c.family}]",
                            flush=True,
                        )
                except Exception:
                    pass
                finally:
                    _c.ik_pending = False
            if self._ik_executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._ik_executor = ThreadPoolExecutor(max_workers=1)
            self._ik_executor.submit(_bg_ik)

        return True

    def _online_generate_backtrack_attractor(
        self,
        q: np.ndarray,
        scores: list,
        f_atts: list,
        static_sc: list,
        h_scores: list,
        scores_combined: list,
        P_tan: np.ndarray,
    ) -> bool:
        """Create a backtrack attractor from q_start without running IK.

        Returns True if the attractor was appended.
        """
        cfg = self.config
        q_start = self._q_start
        if q_start is None:
            return False
        if cfg.backtrack_target_mode == "q_start":
            q_bt = q_start.copy()
        else:  # "partial_to_start"
            beta = float(cfg.backtrack_partial_beta)
            q_bt = q + beta * (q_start - q)
        att = IKAttractor(
            q_goal=q_bt.copy(),
            family="backtrack",
            clearance=float(self.clearance_fn(q_bt)),
            manipulability=0.0,
            kind="backtrack",
            target_name="online_backtrack",
        )
        self.attractors.append(att)
        _f = -cfg.K_c * (q - att.q_goal)
        f_atts.append(_f)
        scores.append(float(_attractor_score(
            q, att, self.clearance_fn, P_tan, _f, is_active=False, config=cfg,
        )))
        static_sc.append(0.0)
        h_scores.append(0.0)
        scores_combined.append(scores[-1] + cfg.w_static * static_sc[-1] + h_scores[-1])
        self._cached_horizon_scores = None
        q_dist = float(np.linalg.norm(q - q_bt))
        print(
            f"[online_backtrack_generated step={self._step_count} "
            f"mode={cfg.backtrack_target_mode} q_dist={q_dist:.3f}]",
            flush=True,
        )
        return True

    def _online_generate_pregoal_attractor(
        self,
        q: np.ndarray,
        scores: list,
        f_atts: list,
        static_sc: list,
        h_scores: list,
        scores_combined: list,
        P_tan: np.ndarray,
        _beta_override: Optional[float] = None,
    ) -> bool:
        """Run IK at an intermediate task-space position to create a pre_goal attractor.

        Uses an intermediate target between current grasptarget and the goal grasptarget
        (controlled by pregoal_intermediate_beta) so that HJCD-IK finds a configuration
        where arm links are clear of the barrier — avoiding the collision that occurs when
        IKing directly to the final goal forces links through or near the obstacle.

        Seeds HJCD-IK from the current arm configuration so it stays in the same elbow
        family as the post-escape arm.

        HJCD is configured with end_effector_frame="panda_hand"; the position passed is
        the grasptarget (panda_grasptarget) position.  HJCD places the hand such that
        grasptarget lands at the target — so hand FK will be ~0.105 m below the target.
        ik_task_err is computed as ||grasp_approx(q_pg) - p_pre|| (should be ~0).

        Returns True if a clearance-valid attractor was appended.
        """
        cfg = self.config
        if self._online_ik_fn is None or self._goal_task_pos is None:
            return False

        _seed_clearance = float(self.clearance_fn(q))
        _ee_hand    = np.array(_panda_link_positions(q)[-1], dtype=float)
        _ee_grasp   = _ee_hand + np.array([0.0, 0.0, 0.105])
        _goal_grasp = self._goal_task_pos

        _dist_current_to_goal = float(np.linalg.norm(_ee_grasp - _goal_grasp))

        # When _beta_override is given (upgrade path), use a single beta.
        # Otherwise sweep cfg.pregoal_beta_sweep sequentially, early-exit when
        # the first candidate with clearance >= pregoal_early_exit_clearance is found.
        betas = ([float(_beta_override)] if _beta_override is not None
                 else [float(b) for b in cfg.pregoal_beta_sweep])

        best_q_pg:   Optional[np.ndarray] = None
        best_beta:   float = betas[0]
        best_score:  float = -1e9
        best_ik_cl:  float = -1.0
        best_task_err: float = -1.0
        best_qdist:  float = -1.0
        total_ik_ms: float = 0.0
        total_calls: int   = 0

        for beta in betas:
            p_pre = _ee_grasp + beta * (_goal_grasp - _ee_grasp)
            # For the full-goal upgrade (beta=1.0), collect all candidates and pick
            # the one with maximum clearance — the min-q_dist solution is often
            # colliding (arm still near the barrier), while the correct solution
            # is in a different elbow config with larger q_dist but valid clearance.
            _upgrade_beta = (beta >= 0.99)

            t_b = time.perf_counter()
            try:
                if _upgrade_beta:
                    _all_cands = self._online_ik_fn(
                        p_pre,
                        batch_size=cfg.online_pregoal_ik_batch_size,
                        q_seed=q,
                        return_all=True,
                    )
                    _all_cands = _all_cands if _all_cands else []
                    # Among clearance-valid candidates, prefer minimum q_dist
                    # (stays in same homotopy class as post-escape arm).
                    _valid_cl = [
                        np.asarray(_c, dtype=float) for _c in _all_cands
                        if float(self.clearance_fn(np.asarray(_c, dtype=float)))
                           >= cfg.pregoal_min_clearance_m
                    ]
                    if _valid_cl:
                        q_pg_b = min(
                            _valid_cl,
                            key=lambda _c: float(np.linalg.norm(_c - q)),
                        )
                    else:
                        q_pg_b = None
                else:
                    q_pg_b = self._online_ik_fn(
                        p_pre,
                        batch_size=cfg.online_pregoal_ik_batch_size,
                        q_seed=q,
                        minimize_qdist=True,
                    )
            except Exception:
                q_pg_b = None
            ik_ms_b = (time.perf_counter() - t_b) * 1000.0
            total_ik_ms += ik_ms_b
            total_calls += 1

            if q_pg_b is None:
                self._pregoal_rejected_count += 1
                print(
                    f"[pregoal_beta_try step={self._step_count} beta={beta:.2f} "
                    f"ik_ok=False ik_ms={ik_ms_b:.1f}]",
                    flush=True,
                )
                continue

            _ik_hp = np.array(_panda_link_positions(q_pg_b)[-1], dtype=float)
            _ik_gr = _ik_hp + np.array([0.0, 0.0, 0.105])
            _ik_cl_b  = float(self.clearance_fn(q_pg_b))
            _ik_qd_b  = float(np.linalg.norm(q - q_pg_b))
            _err_gr_b = float(np.linalg.norm(_ik_gr - p_pre))

            print(
                f"[pregoal_beta_try step={self._step_count} beta={beta:.2f} "
                f"clearance={_ik_cl_b:.3f} q_dist={_ik_qd_b:.3f} "
                f"task_err={_err_gr_b:.3f} ik_ms={ik_ms_b:.1f}]",
                flush=True,
            )

            if _ik_cl_b < cfg.pregoal_min_clearance_m:
                self._pregoal_rejected_count += 1
                print(
                    f"[pregoal_beta_rejected step={self._step_count} beta={beta:.2f} "
                    f"reason=low_clearance clearance={_ik_cl_b:.3f} "
                    f"threshold={cfg.pregoal_min_clearance_m:.3f}]",
                    flush=True,
                )
                continue

            if _ik_qd_b > cfg.pregoal_max_qdist and not _upgrade_beta:
                self._pregoal_rejected_count += 1
                print(
                    f"[pregoal_beta_rejected step={self._step_count} beta={beta:.2f} "
                    f"reason=max_qdist q_dist={_ik_qd_b:.3f} "
                    f"max_qdist={cfg.pregoal_max_qdist:.3f}]",
                    flush=True,
                )
                continue

            # Valid candidate — score and possibly early-exit.
            cand_score = 3.0 * _ik_cl_b - 0.5 * _ik_qd_b + 0.5 * beta
            if cand_score > best_score:
                best_score  = cand_score
                best_q_pg   = q_pg_b
                best_beta   = beta
                best_ik_cl  = _ik_cl_b
                best_task_err = _err_gr_b
                best_qdist  = _ik_qd_b

            if _ik_cl_b >= cfg.pregoal_early_exit_clearance:
                break   # excellent candidate — no need to try higher betas

        # Account for all IK time as a single event.
        self._pregoal_ik_ms           += total_ik_ms
        self._pregoal_ik_calls        += total_calls
        self._runtime_planner_ms      += total_ik_ms
        self._runtime_planner_ms_max   = max(self._runtime_planner_ms_max, total_ik_ms)
        self._runtime_planner_event_count += 1

        if best_q_pg is None:
            print(
                f"[pregoal_target step={self._step_count} "
                f"betas={betas} dist_to_goal={_dist_current_to_goal:.3f} "
                f"clearance_at_seed={_seed_clearance:.3f} "
                f"result=all_rejected total_ik_ms={total_ik_ms:.1f}]",
                flush=True,
            )
            return False

        print(
            f"[pregoal_target step={self._step_count} "
            f"best_beta={best_beta:.2f} dist_to_goal={_dist_current_to_goal:.3f} "
            f"clearance_at_seed={_seed_clearance:.3f} "
            f"ik_clearance={best_ik_cl:.3f} ik_q_dist={best_qdist:.3f} "
            f"score={best_score:.3f} total_ik_ms={total_ik_ms:.1f} "
            f"n_betas_tried={total_calls}]",
            flush=True,
        )

        self._pregoal_accepted_count += 1
        self._pregoal_last_clearance  = best_ik_cl
        self._pregoal_last_task_err   = best_task_err
        self._pregoal_last_beta       = best_beta

        q_pg = np.asarray(best_q_pg, dtype=float)
        att = IKAttractor(
            q_goal=q_pg,
            family="pre_goal",
            clearance=float(self.clearance_fn(q_pg)),
            manipulability=0.0,
            kind="pre_goal",
            target_name="online_pregoal",
        )
        self.attractors.append(att)
        _f = -cfg.K_c * (q - att.q_goal)
        f_atts.append(_f)
        scores.append(float(_attractor_score(
            q, att, self.clearance_fn, P_tan, _f, is_active=False, config=cfg,
        )))
        static_sc.append(0.0)
        h_scores.append(0.0)
        scores_combined.append(scores[-1] + cfg.w_static * static_sc[-1] + h_scores[-1])
        self._cached_horizon_scores = None
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        q: np.ndarray,
        qdot: Optional[np.ndarray] = None,
        target_pose=None,
        obstacles=None,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, GeoMultiAttractorDSResult]:
        """
        Compute the desired joint velocity qdot_des and diagnostics.

        Fast path (when batch_from_lp_fn is set, as by build_geo_multi_attractor_ds):
          * Gradient + Jacobian: ONE _panda_fk_batch call (7 perturbed configs).
          * Attractor scoring:   ONE _panda_fk_batch call (N_att predicted configs).
          * Gradient skipped entirely when alpha=0 (arm far from obstacles).
          Total: 1 scalar + 2 batch FK calls vs ~30 scalar FK calls previously.

        Scalar fallback (batch_from_lp_fn is None, e.g. dynamic clearance_fn):
          Unchanged from the original implementation.

        Args:
            q:           Current joint configuration (7,).
            qdot:        Current joint velocity (7,) — unused internally.
            target_pose: Ignored (target encoded in attractor set).
            obstacles:   Ignored (geometry captured by clearance_fn closure).
            dt:          Control timestep for lookahead safety check.
        """
        q   = np.asarray(q, dtype=float)
        cfg = self.config
        n   = q.shape[0]

        # ---- Timing setup -----------------------------------------------
        timing: Optional[Dict[str, float]] = {} if cfg.enable_timing else None
        _t_total = time.perf_counter() if timing is not None else 0.0

        # ---- Base clearance (always scalar — also serves lookahead) ------
        _t0 = time.perf_counter() if timing is not None else 0.0
        lp_base   = np.array(_panda_link_positions(q), dtype=float)  # (8, 3)
        clearance = float(self.clearance_fn(q))
        if timing is not None:
            timing["clearance_ms"] = (time.perf_counter() - _t0) * 1000

        # ---- Blend factor -----------------------------------------------
        alpha = _blend_alpha(clearance, cfg.clearance_enter, cfg.clearance_full)

        # ---- Gradient, Jacobian, projectors (skipped when alpha=0) ------
        _t0 = time.perf_counter() if timing is not None else 0.0
        if alpha > 0.0:
            if self._batch_from_lp_fn is not None:
                # Fast path: one batch FK shared by gradient and Jacobian.
                grad_h, J = _fast_gradient_and_jacobian(
                    q, lp_base, clearance, self._batch_from_lp_fn,
                    eps=cfg.clearance_grad_eps,
                )
            else:
                grad_h = _clearance_gradient(q, self.clearance_fn, eps=cfg.clearance_grad_eps)
                J      = _position_jacobian(q, eps=cfg.jacobian_eps)
            grad_h_norm = float(np.linalg.norm(grad_h))
        else:
            grad_h      = np.zeros(n)
            grad_h_norm = 0.0
            J           = np.zeros((3, n))
        if timing is not None:
            timing["grad_jac_ms"] = (time.perf_counter() - _t0) * 1000

        _t0 = time.perf_counter() if timing is not None else 0.0
        if alpha > 0.0:
            P_tan  = _tangent_projector(grad_h, eps_grad=cfg.eps_grad)
            N_proj = _nullspace_projector(J)
        else:
            P_tan  = np.eye(n)
            N_proj = np.eye(n)
        if timing is not None:
            timing["projectors_ms"] = (time.perf_counter() - _t0) * 1000

        # ---- Score attractors (one-step) --------------------------------
        _t0 = time.perf_counter() if timing is not None else 0.0
        if self._batch_from_lp_fn is not None:
            # Fast path: one batch FK for all N_att predicted positions.
            scores, f_atts = _batch_score_attractors(
                q, self.attractors, self._batch_from_lp_fn,
                P_tan, self._active_idx, cfg,
            )
        else:
            scores = []
            f_atts = []
            _fb_task_present = any(a.task_pos is not None for a in self.attractors)
            _fb_ee_pos: Optional[np.ndarray] = None
            _fb_J_pinv: Optional[np.ndarray] = None
            if _fb_task_present:
                _fb_ee_pos = np.array(_panda_link_positions(q)[-1], dtype=float)
                _fb_J = _ee_position_jacobian(q)
                _fb_J_pinv = np.linalg.pinv(_fb_J)
            for i, att in enumerate(self.attractors):
                if att.task_pos is not None:
                    x_err = att.task_pos - _fb_ee_pos
                    f_att_i = cfg.K_c * (_fb_J_pinv @ x_err)
                else:
                    f_att_i = -cfg.K_c * (q - att.q_goal)
                s = _attractor_score(
                    q, att, self.clearance_fn, P_tan, f_att_i,
                    is_active=(i == self._active_idx), config=cfg,
                )
                scores.append(s)
                f_atts.append(f_att_i)
        if timing is not None:
            timing["scoring_ms"] = (time.perf_counter() - _t0) * 1000

        # ---- Horizon scores (cached min-clearance lookahead) ------------
        _t0 = time.perf_counter() if timing is not None else 0.0
        if self._batch_from_lp_fn is not None:
            refresh_due = (
                self._cached_horizon_scores is None
                or (self._step_count - self._horizon_last_update) >= cfg.horizon_rate
            )
            if refresh_due:
                self._cached_horizon_scores = _compute_horizon_scores(
                    q, self.attractors, clearance, self._batch_from_lp_fn, cfg,
                )
                self._horizon_last_update = self._step_count
            h_scores = self._cached_horizon_scores
        else:
            h_scores = [0.0] * len(self.attractors)
        if timing is not None:
            timing["horizon_ms"] = (time.perf_counter() - _t0) * 1000

        # ---- Combined scores (one-step + static prior + horizon) --------
        static_sc      = [a.static_score for a in self.attractors]
        scores_combined = [
            s + cfg.w_static * st + h
            for s, st, h in zip(scores, static_sc, h_scores)
        ]

        # ---- Trap detection (stagnant goal error) -----------------------
        _active_att_td = self.attractors[self._active_idx]
        if _active_att_td.task_pos is not None:
            goal_err_curr = float(np.linalg.norm(np.array(lp_base[-1]) - _active_att_td.task_pos))
        else:
            goal_err_curr = float(np.linalg.norm(q - _active_att_td.q_goal))
        self._goal_error_history.append(goal_err_curr)
        if len(self._goal_error_history) > cfg.trap_n_steps + 10:
            self._goal_error_history = self._goal_error_history[-(cfg.trap_n_steps + 10):]

        # Simple escape stall counter: increments every step when arm is near obstacle
        # and hasn't yet escaped.  Never resets on attractor switches.  Used instead
        # of EE displacement or goal-err history so switching-induced resets don't
        # mask stagnation.
        if cfg.enable_simple_escape_waypoint:
            self._ee_stall_history.append(self._step_count)

        trap_detected        = False
        trap_forced          = False
        effective_hysteresis = cfg.switch_hysteresis
        if cfg.trap_detection and len(self._goal_error_history) >= cfg.trap_n_steps:
            progress = (
                self._goal_error_history[-cfg.trap_n_steps] - self._goal_error_history[-1]
            )
            if progress < cfg.trap_min_progress and alpha > 0.3:
                trap_detected        = True
                trap_forced          = True
                effective_hysteresis = cfg.switch_hysteresis * cfg.trap_hysteresis_reduction

        # ---- Near-goal switch lockout -----------------------------------
        goal_switch_locked = (
            cfg.enable_goal_switch_lock
            and goal_err_curr < cfg.goal_switch_lock_radius
        )

        # ---- Clearance recovery mode management --------------------------------
        if cfg.enable_clearance_recovery and self._build_recovery_attractor_fn is not None:
            if self._in_recovery_mode:
                # Exit: clearance restored, waypoint reached, or timeout
                _steps_in_recovery = self._step_count - self._recovery_start_step
                _near_rec_wp = (
                    self._active_recovery_target_pos is not None
                    and float(np.linalg.norm(lp_base[-1] - self._active_recovery_target_pos))
                        < cfg.recovery_reached_tol_m
                )
                _clearance_restored = clearance >= cfg.recovery_target_clearance_m
                _timed_out = _steps_in_recovery >= cfg.recovery_max_steps
                if _clearance_restored or _near_rec_wp or _timed_out:
                    _exit_reason = (
                        "clearance_restored" if _clearance_restored
                        else ("recovery_reached" if _near_rec_wp else "timeout")
                    )
                    self._in_recovery_mode = False
                    _esc_portal_idxs = [i for i, a in enumerate(self.attractors)
                                        if a.kind == "escape"]
                    if _esc_portal_idxs:
                        self._escape_mode_active = True
                    print(
                        f"[recovery_to_portal step={self._step_count} "
                        f"reason={_exit_reason} "
                        f"cl={clearance:.4f} "
                        f"steps_in_recovery={_steps_in_recovery}]",
                        flush=True,
                    )
                else:
                    _dist_to_wp = (
                        float(np.linalg.norm(lp_base[-1] - self._active_recovery_target_pos))
                        if self._active_recovery_target_pos is not None else -1.0
                    )
                    _esc_count = sum(1 for a in self.attractors if a.kind == "escape")
                    if _steps_in_recovery > 0 and _steps_in_recovery % 20 == 0:
                        print(
                            f"[recovery_active step={self._step_count} "
                            f"steps_in_recovery={_steps_in_recovery} "
                            f"dist_to_wp={_dist_to_wp:.4f} "
                            f"robot_clearance={clearance:.4f} "
                            f"portal_attractors={_esc_count} "
                            f"max_steps={cfg.recovery_max_steps}]",
                            flush=True,
                        )
            elif (
                not self._escape_mode_active
                and (self._step_count - self._last_recovery_step) >= cfg.recovery_cooldown_steps
                and self._n_recovery_generations < cfg.recovery_max_generations
            ):
                _rec_hist_ok = len(self._goal_error_history) >= cfg.recovery_stall_window_steps
                _rec_stalled = (
                    _rec_hist_ok
                    and (self._goal_error_history[-cfg.recovery_stall_window_steps] - goal_err_curr)
                        < cfg.recovery_stall_progress_threshold
                )
                _rec_near_obs = clearance < cfg.recovery_trigger_clearance_m
                _rec_alpha_ok = alpha > cfg.recovery_alpha_threshold
                if _rec_stalled and _rec_near_obs and _rec_alpha_ok:
                    _new_rec_att = self._build_recovery_attractor_fn(lp_base)
                    if _new_rec_att is not None:
                        self.attractors.append(_new_rec_att)
                        _new_f_att = -cfg.K_c * (q - _new_rec_att.q_goal)
                        f_atts.append(_new_f_att)
                        scores.append(float(_attractor_score(
                            q, _new_rec_att, self.clearance_fn, P_tan, _new_f_att,
                            is_active=False, config=cfg,
                        )))
                        static_sc.append(_new_rec_att.static_score)
                        h_scores.append(0.0)
                        scores_combined.append(
                            scores[-1] + cfg.w_static * static_sc[-1] + h_scores[-1]
                        )
                        self._cached_horizon_scores = None
                        self._in_recovery_mode = True
                        self._active_recovery_target_pos = _new_rec_att.shell_waypoint
                        self._n_recovery_generations += 1
                        self._n_recovery_switches += 1
                        self._last_recovery_step  = self._step_count
                        self._recovery_start_step = self._step_count
                        print(
                            f"[recovery_trigger step={self._step_count} "
                            f"cl={clearance:.4f} alpha={alpha:.2f} "
                            f"target_pos={_new_rec_att.shell_waypoint}]",
                            flush=True,
                        )
                    else:
                        print(
                            f"[recovery_trigger_failed step={self._step_count} reason=ik_failed]",
                            flush=True,
                        )
                elif self._step_count % 200 == 0 and self._step_count > 0:
                    print(
                        f"[no_recovery step={self._step_count} "
                        f"stalled={_rec_stalled} near_obs={_rec_near_obs} alpha_ok={_rec_alpha_ok} "
                        f"hist_ok={_rec_hist_ok}]",
                        flush=True,
                    )

        # ---- Escape mode management (goal-shell + simple escape waypoint + boundary escape) ---
        _any_escape_enabled = (cfg.enable_goal_shell_escape or cfg.enable_simple_escape_waypoint
                               or cfg.enable_boundary_escape_waypoints)
        if _any_escape_enabled:
            # Hoist these early so all sub-blocks can reference them safely.
            _online_mode  = cfg.attractor_generation_mode == "online"
            _has_deferred = bool(self._deferred_escape_candidates)
            _use_boundary = cfg.enable_boundary_escape_waypoints
            _use_simple   = cfg.enable_simple_escape_waypoint and not _use_boundary
            _stall_window = (
                cfg.boundary_escape_stall_window_steps if _use_boundary
                else cfg.escape_waypoint_stall_window_steps if _use_simple
                else cfg.goal_shell_stall_window_steps
            )
            _stall_prog   = (
                cfg.boundary_escape_stall_progress_threshold if _use_boundary
                else cfg.escape_waypoint_stall_progress_threshold if _use_simple
                else cfg.goal_shell_stall_progress_threshold
            )
            _trig_alpha   = (
                cfg.escape_waypoint_trigger_alpha if _use_boundary or _use_simple
                else cfg.goal_shell_trigger_alpha
            )
            _reached_tol  = (
                cfg.boundary_escape_reached_tol_m if _use_boundary
                else cfg.escape_waypoint_reached_tol_m if _use_simple
                else cfg.goal_shell_reached_tol_m
            )
            _max_gen      = (
                cfg.escape_waypoint_max_generations if _use_boundary or _use_simple
                else cfg.goal_shell_max_escape_switches
            )
            _cross_center_z = (
                cfg.boundary_escape_cross_center_z if _use_boundary
                else cfg.escape_waypoint_cross_center_z
            )
            _goal_err_thr = (
                cfg.boundary_escape_goal_error_threshold if _use_boundary
                else cfg.escape_waypoint_goal_error_threshold
            )

            # Escape mode: success / failure / family cycling
            if self._escape_mode_active:
                _att_act = self.attractors[self._active_idx]

                # Track task-space distance to the active escape waypoint
                if _att_act.kind == "escape" and _att_act.shell_waypoint is not None:
                    _dist_to_esc = float(np.linalg.norm(lp_base[-1] - _att_act.shell_waypoint))
                else:
                    _dist_to_esc = float("inf")
                self._escape_dist_history.append(_dist_to_esc)

                # --- Success conditions ---
                # Note: lateral stage is NOT required for a successful run. The most
                # common exit is _near_wp (arm reaches the escape waypoint IK position),
                # which returns directly to goal regardless of route stage. _below_cross_raw
                # is an additional exit for boundary escape and requires BOTH the Z threshold
                # AND proximity to the escape waypoint; it is not the same as "ee_z < z_thr".
                # The route_stage_active diagnostic shows esc_exit_z and esc_exit_near
                # separately so the two halves of _below_cross_raw can be distinguished.
                _near_wp      = _dist_to_esc < cfg.escape_family_reached_tol_m
                _clearance_ok = clearance > cfg.escape_family_clearance_success_m
                # _below_cross_raw: dual condition — Z below (cross_center_z - 0.04) AND
                # arm within below_cross_max_dist_to_esc_m of the active escape waypoint.
                _below_z_thr = _cross_center_z - 0.04
                _below_cross_raw = (
                    (_use_boundary or _use_simple)
                    and _cross_center_z > 0.0
                    and float(lp_base[-1][2]) < _below_z_thr
                    and _dist_to_esc < cfg.below_cross_max_dist_to_esc_m
                )
                # _below_z_low: separate trigger for route-stage transition (arm truly below
                # vertical bar; no dist_to_esc constraint since we just need Z depth).
                _rs_z_low = self._route_stage_cross_z_low
                _below_z_low = (
                    cfg.enable_yz_route_staging
                    and _rs_z_low > 0.0
                    and (_use_boundary or _use_simple)
                    and float(lp_base[-1][2]) < _rs_z_low
                )
                _on_goal_y_side = (
                    cfg.enable_yz_route_staging
                    and self._escape_route_stage == "lateral"
                    and self._route_stage_goal_side_y_sign != 0.0
                    and (float(lp_base[-1][1]) - self._route_stage_cross_center_y)
                        * self._route_stage_goal_side_y_sign
                        > cfg.route_stage_lateral_reached_tol_m
                )
                _below_cross = _below_cross_raw
                # For goal-shell (non-boundary), keep original exit conditions
                _goal_shell_exit = (
                    not _use_boundary and not _use_simple
                    and (
                        alpha < cfg.goal_shell_clear_alpha_threshold
                        or clearance >= cfg.goal_shell_trigger_clearance_m * 2
                        or float(np.linalg.norm(lp_base[-1] - (
                            _att_act.shell_waypoint if _att_act.shell_waypoint is not None
                            else lp_base[-1]
                        ))) < _reached_tol
                    )
                )
                # Suppress clearance_ok when:
                # (a) gradient recovery is active — recovery exit handles transition, or
                # (b) using boundary escape — commit fully to reaching a portal; only
                #     near_waypoint or below_cross should exit boundary escape mode.
                _clearance_ok_valid = (
                    _clearance_ok
                    and not self._grad_recovery_active
                    and not _use_boundary
                )
                _success = _near_wp or _clearance_ok_valid or _below_cross or _goal_shell_exit

                # --- Route-stage transition: below → lateral ---
                # When the arm dips below the horizontal bar for the first time,
                # advance the stage and force-cycle to goal-side lateral candidates.
                if (
                    cfg.enable_yz_route_staging
                    and self._escape_route_stage == "below"
                    and _below_z_low
                    and not self._grad_recovery_active
                ):
                    self._escape_route_stage  = "lateral"
                    self._route_stage_start_step = self._step_count
                    print(
                        f"[route_stage_transition step={self._step_count} "
                        f"old=below new=lateral reason=below_z_low "
                        f"ee_z={float(lp_base[-1][2]):.4f} "
                        f"z_low={_rs_z_low:.4f} "
                        f"cross_z={_cross_center_z:.4f}]",
                        flush=True,
                    )
                    # Mark current escape family failed to trigger cycling into lateral stage
                    if _att_act.kind == "escape":
                        _cur_sector = _att_act.shell_sector
                        if _cur_sector not in self._failed_escape_families:
                            self._failed_escape_families.add(_cur_sector)
                            self._attempted_escape_families.add(_cur_sector)
                    # Generate the next (lateral-stage) escape attractor immediately
                    if _online_mode and _has_deferred:
                        self._online_generate_next_escape_attractor(
                            q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                        )

                # Periodic route-stage diagnostic
                if (
                    cfg.enable_yz_route_staging
                    and self._escape_route_stage != "none"
                    and self._step_count % 30 == 0
                ):
                    print(
                        f"[route_stage_active step={self._step_count} "
                        f"stage={self._escape_route_stage} "
                        f"esc_exit_z={float(lp_base[-1][2]) < _below_z_thr} "
                        f"esc_exit_near={_dist_to_esc < cfg.below_cross_max_dist_to_esc_m} "
                        f"below_cross_raw={_below_cross_raw} "
                        f"below_z_low={_below_z_low} "
                        f"on_goal_y_side={_on_goal_y_side} "
                        f"ee_y={float(lp_base[-1][1]):.4f} "
                        f"ee_z={float(lp_base[-1][2]):.4f} "
                        f"dist_to_esc={_dist_to_esc:.4f} "
                        f"z_thr={_below_z_thr:.4f} "
                        f"z_low={_rs_z_low:.4f} "
                        f"goal_y_sign={self._route_stage_goal_side_y_sign:.1f}]",
                        flush=True,
                    )

                if _success:
                    # If pre-goal mode is already active, escape has already handed off;
                    # just clear escape mode silently without re-logging escape_return_to_goal.
                    if self._pre_goal_mode_active:
                        self._escape_mode_active  = False
                        self._escape_route_stage  = "none"
                    else:
                        _esc_reason = (
                            "near_waypoint" if _near_wp
                            else "clearance_restored" if _clearance_ok_valid
                            else "below_cross" if _below_cross
                            else "goal_shell_clear"
                        )
                        self._escape_mode_active = False
                        # Preserve route-stage across escape-return-to-goal so that if
                        # the arm re-enters escape (goal approach fails), the next cycle
                        # continues from the correct stage rather than starting over.
                        # Only full reset (reset()) clears the stage unconditionally.
                        print(
                            f"[escape_return_to_goal step={self._step_count} "
                            f"reason={_esc_reason} "
                            f"dist_to_esc={_dist_to_esc:.4f} "
                            f"cl={clearance:.4f}]",
                            flush=True,
                        )
                        # below_cross can fire while the arm is still far from the
                        # escape waypoint (wrist Z dips below cross height). Prefetch
                        # the next escape family now so the subsequent backtrack->escape
                        # cycle has a fresh alternative instead of retrying the same one.
                        if _below_cross and _online_mode and _has_deferred:
                            self._online_generate_next_escape_attractor(
                                q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                            )
                    # Log which goal attractors are available for return-to-goal.
                    _goal_cands = [(i, a) for i, a in enumerate(self.attractors)
                                   if a.kind in ("goal", "pre_goal")]
                    if _goal_cands and scores_combined:
                        _bg_i, _bg_a = max(
                            _goal_cands,
                            key=lambda ia: scores_combined[ia[0]] if ia[0] < len(scores_combined) else -999.0,
                        )
                        _bg_qdist = float(np.linalg.norm(q - _bg_a.q_goal))
                        _bg_tdist = (
                            float(np.linalg.norm(self._goal_task_pos - _bg_a.task_pos))
                            if (self._goal_task_pos is not None
                                and _bg_a.task_pos is not None)
                            else -1.0
                        )
                        print(
                            f"[return_to_goal step={self._step_count} "
                            f"active_idx={_bg_i} kind={_bg_a.kind} family={_bg_a.family} "
                            f"q_dist={_bg_qdist:.3f} task_dist={_bg_tdist:.3f} "
                            f"clearance={_bg_a.clearance:.3f}]",
                            flush=True,
                        )
                    # Online pre-goal generation: when escape exits via below_cross OR
                    # near_waypoint in boundary/simple escape, generate a pre_goal attractor
                    # from the current arm position (goal side) and force-switch to it.
                    if (
                        (_below_cross or (_near_wp and (_use_boundary or _use_simple)))
                        and _online_mode
                        and cfg.enable_pre_goal_waypoint
                        and not self._pregoal_generated
                    ):
                        _pg_ok = self._online_generate_pregoal_attractor(
                            q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                        )
                        if _pg_ok:
                            self._pregoal_generated    = True
                            self._pre_goal_mode_active = True
                            self._pregoal_start_step   = self._step_count
                            self._pregoal_dist_history = []
                            _pg_idxs = [i for i, a in enumerate(self.attractors)
                                        if a.kind == "pre_goal"]
                            if _pg_idxs:
                                old_idx = self._active_idx
                                self._active_idx = _pg_idxs[-1]
                                self._n_switches += 1
                                self._n_pregoal_switches += 1
                                self._goal_error_history = []
                                print(
                                    f"[escape_to_pregoal step={self._step_count}]",
                                    flush=True,
                                )

                # --- Gradient clearance recovery tracking (while in escape mode) ---
                if self._grad_recovery_active:
                    self._grad_recovery_best_clearance = max(
                        self._grad_recovery_best_clearance, clearance
                    )
                    self._n_grad_recovery_steps += 1
                    if self._n_grad_recovery_steps % 20 == 0:
                        print(
                            f"[clearance_recovery_active step={self._step_count} "
                            f"cl={clearance:.4f} "
                            f"best_cl={self._grad_recovery_best_clearance:.4f} "
                            f"grad_norm={float(np.linalg.norm(grad_h)):.4f} "
                            f"steps={self._n_grad_recovery_steps}]",
                            flush=True,
                        )
                    if (
                        self._n_grad_recovery_steps >= cfg.max_grad_recovery_steps
                        and (self._grad_recovery_best_clearance - self._grad_recovery_start_clearance) < 0.005
                    ):
                        print(
                            f"[clearance_recovery_failed step={self._step_count} "
                            f"start_cl={self._grad_recovery_start_clearance:.4f} "
                            f"best_cl={self._grad_recovery_best_clearance:.4f}]",
                            flush=True,
                        )
                    if clearance >= cfg.recovery_release_clearance_m:
                        self._grad_recovery_active = False
                        print(
                            f"[clearance_recovery_exit step={self._step_count} "
                            f"cl={clearance:.4f} "
                            f"gain={clearance - self._grad_recovery_start_clearance:.4f}]",
                            flush=True,
                        )
                        # Activate backtrack staging if enabled, otherwise stay in escape
                        if cfg.enable_backtrack_staging:
                            _bt_idxs = [i for i, a in enumerate(self.attractors)
                                        if a.kind == "backtrack"]
                            if _bt_idxs:
                                _bt_att = self.attractors[_bt_idxs[0]]
                                # For partial_to_start: compute target from current q
                                if cfg.backtrack_target_mode == "partial_to_start":
                                    _beta = float(cfg.backtrack_partial_beta)
                                    _bt_att.q_goal = (
                                        q + _beta * (_bt_att.q_goal - q)
                                    )
                                self._backtrack_target         = _bt_att.q_goal.copy()
                                self._backtrack_mode_active    = True
                                self._backtrack_start_step     = self._step_count
                                self._n_backtrack_steps        = 0
                                self._backtrack_reached        = False
                                self._n_backtrack_switches    += 1
                                # Deactivate escape while in backtrack
                                self._escape_mode_active = False
                                print(
                                    f"[recovery_to_backtrack step={self._step_count} "
                                    f"cl={clearance:.4f} "
                                    f"target_mode={cfg.backtrack_target_mode}]",
                                    flush=True,
                                )

                # --- Failure detection (boundary mode only) ---
                elif (
                    _use_boundary
                    and _att_act.kind == "escape"
                    and cfg.enable_escape_family_cycling
                    and _att_act.shell_sector not in self._failed_escape_families
                    and self._n_escape_family_cycles < cfg.escape_family_max_cycles
                ):
                    # Clearance-first gating: block family failure when arm is pinned
                    _progress_fail = False
                    _pinned_fail   = False
                    _window_progress_m = float("inf")
                    if clearance < cfg.recovery_trigger_clearance_m:
                        if not self._grad_recovery_active:
                            self._grad_recovery_active = True
                            self._grad_recovery_start_clearance = clearance
                            self._grad_recovery_best_clearance  = clearance
                            self._grad_recovery_start_step      = self._step_count
                            self._n_grad_recovery_steps         = 0
                            self._escape_blocked_count         += 1
                            print(
                                f"[escape_blocked_by_clearance step={self._step_count} "
                                f"cl={clearance:.4f} "
                                f"threshold={cfg.recovery_trigger_clearance_m:.3f} "
                                f"entering_recovery]",
                                flush=True,
                            )
                    else:
                        _steps_in_fam = self._step_count - self._escape_family_last_switch_step
                        _wl = cfg.escape_family_min_steps_before_cycle

                        # Window progress: oldest distance in window minus current distance
                        # (positive means the arm is moving closer to the waypoint)
                        if len(self._escape_dist_history) >= _wl:
                            _oldest = float(list(self._escape_dist_history)[-_wl])
                            _window_progress_m = _oldest - _dist_to_esc
                        else:
                            _window_progress_m = float("inf")

                        _below_stage = self._escape_route_stage == "below"
                        _fail_thr = 1e-4 if _below_stage else cfg.escape_family_progress_threshold_m
                        _progress_fail = (
                            _steps_in_fam >= _wl
                            and _window_progress_m < _fail_thr
                        )
                        _pinned_fail = (
                            clearance <= 0.002
                            and _steps_in_fam >= cfg.escape_family_cycle_window_steps
                        )

                    if not self._grad_recovery_active and (_progress_fail or _pinned_fail):
                        _sector = _att_act.shell_sector
                        _fail_reason = "no_escape_progress" if _progress_fail else "pinned"
                        self._failed_escape_families.add(_sector)
                        self._attempted_escape_families.add(_sector)
                        print(
                            f"[escape_family_failed step={self._step_count} "
                            f"family={_sector} reason={_fail_reason} "
                            f"progress={_window_progress_m:.4f} "
                            f"dist_to_escape={_dist_to_esc:.4f} "
                            f"robot_min_clearance={clearance:.4f} "
                            f"alpha={alpha:.2f}]",
                            flush=True,
                        )

            # Backtrack mode management: runs between clearance recovery and escape portal
            if self._backtrack_mode_active:
                _steps_in_bt = self._step_count - self._backtrack_start_step
                self._n_backtrack_steps += 1
                _bt_idxs = [i for i, a in enumerate(self.attractors) if a.kind == "backtrack"]
                _q_bt = (self.attractors[_bt_idxs[0]].q_goal if _bt_idxs
                         else (self._backtrack_target if self._backtrack_target is not None else q))
                _bt_dist = float(np.linalg.norm(q - _q_bt))

                _bt_reached = _bt_dist < cfg.backtrack_reached_tol_rad
                # In online mode the arm may be pinned against the obstacle and
                # unable to gain clearance — drop the clearance requirement so
                # the timeout fires and the escape pipeline can still run.
                _online_bt    = cfg.attractor_generation_mode == "online"
                _bt_deferred  = bool(self._deferred_escape_candidates)
                _bt_escaped_far_enough = (
                    goal_err_curr >= cfg.backtrack_escape_dist_threshold_rad
                )
                _bt_timeout = (
                    _steps_in_bt >= cfg.backtrack_max_steps
                    and (clearance > cfg.backtrack_min_clearance_m or _online_bt)
                    and _bt_escaped_far_enough
                )
                if _bt_reached or _bt_timeout:
                    _bt_reason = "reached" if _bt_reached else "timeout"
                    self._backtrack_reached    = _bt_reached
                    self._backtrack_mode_active = False
                    _esc_portal_idxs = [i for i, a in enumerate(self.attractors)
                                        if a.kind == "escape"]
                    _has_untried_deferred = _bt_deferred and any(
                        not c.ik_generated and not c.ik_failed
                        for c in self._deferred_escape_candidates
                    )
                    if _online_bt and (_has_untried_deferred or not _esc_portal_idxs):
                        # Initialize route-stage only on the first backtrack-to-escape
                        # event (when stage is still "none"). Subsequent cycles preserve
                        # the current stage rather than resetting to "below".
                        if cfg.enable_yz_route_staging and (_use_boundary or _use_simple):
                            if self._escape_route_stage == "none":
                                _meta_pre = next(
                                    (c.metadata for c in self._deferred_escape_candidates
                                     if c.metadata.get("goal_side_y_sign") is not None),
                                    None,
                                ) or {}
                                _gsy_pre = _meta_pre.get("goal_side_y_sign", 0.0)
                                _ccy_pre = _meta_pre.get("cross_center_y", 0.0)
                                _ccz_pre = float(_cross_center_z) if _cross_center_z > 0 else 0.0
                                _czl_pre = float(_meta_pre.get("z_low", -1.0))
                                if _gsy_pre != 0.0 and _ccz_pre > 0.0:
                                    self._escape_route_stage           = "below"
                                    self._route_stage_start_step       = self._step_count
                                    self._route_stage_goal_side_y_sign = float(_gsy_pre)
                                    self._route_stage_cross_center_y   = float(_ccy_pre)
                                    self._route_stage_cross_center_z   = _ccz_pre
                                    self._route_stage_cross_z_low      = _czl_pre
                                    print(
                                        f"[route_stage_start step={self._step_count} "
                                        f"stage=below "
                                        f"goal_y_sign={_gsy_pre:.1f} "
                                        f"cross_center_y={_ccy_pre:.4f} "
                                        f"cross_center_z={_ccz_pre:.4f} "
                                        f"cross_z_low={_czl_pre:.4f}]",
                                        flush=True,
                                    )
                        # Generate ONE escape attractor per replan event.
                        # Called on every backtrack-to-escape transition when untried
                        # deferred candidates remain so the arm advances through the pool
                        # rather than re-selecting the same (already-tried) candidate.
                        _ok = self._online_generate_next_escape_attractor(
                            q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                        )
                        if not _ok:
                            self._n_reentry_fail += 1
                        _esc_portal_idxs = [i for i, a in enumerate(self.attractors)
                                            if a.kind == "escape"]
                    if _esc_portal_idxs:
                        # Reset escape state so a fresh portal attempt can run
                        self._escape_mode_active             = True
                        self._failed_escape_families.clear()
                        self._n_escape_generations           = 0
                        self._escape_exhausted               = False
                        self._last_escape_generation_step    = self._step_count
                        self._escape_family_last_switch_step = self._step_count
                        self._escape_dist_history.clear()
                    print(
                        f"[backtrack_to_escape step={self._step_count} "
                        f"reason={_bt_reason} "
                        f"dist_rad={_bt_dist:.3f} "
                        f"steps_in_bt={_steps_in_bt} "
                        f"cl={clearance:.4f}]",
                        flush=True,
                    )
                elif _steps_in_bt % 20 == 0 and _steps_in_bt > 0:
                    print(
                        f"[backtrack_active step={self._step_count} "
                        f"steps={_steps_in_bt} "
                        f"dist_rad={_bt_dist:.3f} "
                        f"cl={clearance:.4f}]",
                        flush=True,
                    )

            # Pre-goal mode management: log activity, detect exit condition.
            if self._pre_goal_mode_active:
                _pg_idxs_act = [i for i, a in enumerate(self.attractors)
                                if a.kind == "pre_goal"]
                _pg_q_target = (self.attractors[_pg_idxs_act[0]].q_goal
                                if _pg_idxs_act else None)
                _pg_dist_now = (float(np.linalg.norm(q - _pg_q_target))
                                if _pg_q_target is not None else float("inf"))
                _pg_steps_in = self._step_count - self._pregoal_start_step
                if (_pg_dist_now < cfg.pregoal_reached_tol_rad
                        or _pg_steps_in >= cfg.pregoal_max_steps):
                    _pg_exit_reason = ("near_goal" if _pg_dist_now < cfg.pregoal_reached_tol_rad
                                       else "timeout")
                    # Two-stage upgrade: when intermediate pregoal exits near_goal, try
                    # one more online IK call at the full goal seeded from current q.
                    # Only remove the old pre_goal attractor if the upgrade IK succeeds;
                    # on failure, fall through to normal deactivation so the arm can use
                    # its own momentum plus the regular goal attractors.
                    if (
                        _pg_exit_reason == "near_goal"
                        and cfg.pregoal_upgrade_on_near_goal
                        and not self._pregoal_is_upgrade
                    ):
                        self._pregoal_is_upgrade = True
                        # Save existing pre_goal indices BEFORE the upgrade call appends.
                        _old_pg_idxs = [
                            i for i, a in enumerate(self.attractors) if a.kind == "pre_goal"
                        ]
                        _upgrade_ok = self._online_generate_pregoal_attractor(
                            q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                            _beta_override=1.0,
                        )
                        if _upgrade_ok:
                            # Remove old pre_goal attractors now that the new one is appended.
                            for _ri in sorted(_old_pg_idxs, reverse=True):
                                self.attractors.pop(_ri)
                                scores.pop(_ri)
                                f_atts.pop(_ri)
                                static_sc.pop(_ri)
                                h_scores.pop(_ri)
                                scores_combined.pop(_ri)
                                if self._active_idx > _ri:
                                    self._active_idx -= 1
                                elif self._active_idx == _ri:
                                    self._active_idx = max(0, _ri - 1)
                            # Force-switch to the newly appended full-goal attractor.
                            self._active_idx       = len(self.attractors) - 1
                            self._pregoal_start_step = self._step_count
                            self._n_pregoal_switches += 1
                            print(
                                f"[pregoal_upgrade step={self._step_count} "
                                f"now_targeting=full_goal]",
                                flush=True,
                            )
                            # Keep _pre_goal_mode_active = True; don't deactivate yet.
                        else:
                            # Upgrade to full goal failed — try the next intermediate
                            # beta (last_beta + 0.20) before force-switching to a
                            # goal attractor that may be in the wrong homotopy class.
                            _fallback_beta = min(
                                self._pregoal_last_beta + 0.20, 0.95
                            )
                            _fallback_ok = False
                            if _fallback_beta > self._pregoal_last_beta + 0.01:
                                _fallback_ok = self._online_generate_pregoal_attractor(
                                    q, scores, f_atts, static_sc,
                                    h_scores, scores_combined, P_tan,
                                    _beta_override=_fallback_beta,
                                )
                            if _fallback_ok:
                                # Stay in pregoal mode with the new intermediate target;
                                # reset upgrade flag so the next near_goal exit can
                                # retry the full-goal upgrade from the closer position.
                                self._pregoal_is_upgrade = False
                                self._pregoal_start_step = self._step_count
                                _pg_new = [i for i, a in enumerate(self.attractors)
                                           if a.kind == "pre_goal"]
                                if _pg_new:
                                    self._active_idx = _pg_new[-1]
                                    self._n_pregoal_switches += 1
                                print(
                                    f"[pregoal_upgrade_fallback step={self._step_count} "
                                    f"failed_beta=1.0 fallback_beta={_fallback_beta:.2f}]",
                                    flush=True,
                                )
                            else:
                                # All fallbacks exhausted — deactivate and switch to goal.
                                self._pre_goal_mode_active = False
                                _goal_idxs = [i for i, a in enumerate(self.attractors)
                                              if a.kind == "goal"]
                                if _goal_idxs:
                                    self._active_idx = max(
                                        _goal_idxs,
                                        key=lambda i: self.attractors[i].static_score,
                                    )
                                print(
                                    f"[pregoal_return_to_goal step={self._step_count} "
                                    f"reason={_pg_exit_reason}_upgrade_failed "
                                    f"dist_rad={_pg_dist_now:.3f} "
                                    f"steps_in_pg={_pg_steps_in}]",
                                    flush=True,
                                )
                    else:
                        self._pregoal_final_dist_rad = _pg_dist_now
                        # Re-entry gate: when pregoal exits near_goal, verify that a
                        # one-step move toward the final goal is collision-safe.
                        # If not, try a progressive regen at a higher beta before
                        # returning to the regular goal attractors.
                        _reentry_allowed = True
                        if (
                            _pg_exit_reason == "near_goal"
                            and self._pregoal_regen_count < cfg.pregoal_progressive_max_regen
                        ):
                            _goal_atts = [(i, a) for i, a in enumerate(self.attractors)
                                          if a.kind == "goal"]
                            if _goal_atts and scores_combined:
                                _best_g_idx = max(
                                    [i for i, a in enumerate(self.attractors) if a.kind == "goal"],
                                    key=lambda i: scores_combined[i] if i < len(scores_combined) else -1e9,
                                )
                                _f_to_goal = self.attractors[_best_g_idx].q_goal - q
                                _f_norm = float(np.linalg.norm(_f_to_goal))
                                if _f_norm > 1e-9:
                                    _q_hat = q + cfg.pregoal_reentry_dt * (_f_to_goal / _f_norm)
                                    _h_now = float(self.clearance_fn(q))
                                    _h_hat = float(self.clearance_fn(_q_hat))
                                    _reentry_allowed = (
                                        _h_now >= cfg.pregoal_reentry_h_current
                                        and _h_hat >= cfg.pregoal_reentry_h_hat
                                    )
                                    _reason_str = "ok" if _reentry_allowed else "unsafe_step"
                                    print(
                                        f"[goal_reentry_check step={self._step_count} "
                                        f"h_now={_h_now:.3f} h_hat={_h_hat:.3f} "
                                        f"allow={_reentry_allowed} reason={_reason_str}]",
                                        flush=True,
                                    )
                                    if not _reentry_allowed:
                                        self._n_goal_reentry_blocked += 1
                                        _next_beta = min(
                                            self._pregoal_last_beta + cfg.pregoal_reentry_beta_delta,
                                            0.95,
                                        )
                                        _regen_ok = self._online_generate_pregoal_attractor(
                                            q, scores, f_atts, static_sc,
                                            h_scores, scores_combined, P_tan,
                                            _beta_override=_next_beta,
                                        )
                                        if _regen_ok:
                                            self._pregoal_regen_count  += 1
                                            self._pregoal_start_step   = self._step_count
                                            self._pregoal_dist_history = []
                                            _pg_new = [i for i, a in enumerate(self.attractors)
                                                       if a.kind == "pre_goal"]
                                            if _pg_new:
                                                self._active_idx = _pg_new[-1]
                                                self._n_pregoal_switches += 1
                                            _reentry_allowed = False  # stay in pregoal
                                            print(
                                                f"[pregoal_regen step={self._step_count} "
                                                f"regen_count={self._pregoal_regen_count} "
                                                f"next_beta={_next_beta:.2f}]",
                                                flush=True,
                                            )
                                        else:
                                            _reentry_allowed = True   # regen failed, allow anyway
                                            print(
                                                f"[goal_reentry_check step={self._step_count} "
                                                f"regen_failed=True allowing_reentry=True]",
                                                flush=True,
                                            )
                        if _reentry_allowed:
                            self._pre_goal_mode_active = False
                            _goal_idxs = [i for i, a in enumerate(self.attractors)
                                          if a.kind == "goal"]
                            if _goal_idxs:
                                self._active_idx = max(
                                    _goal_idxs,
                                    key=lambda i: self.attractors[i].static_score,
                                )
                            print(
                                f"[pregoal_return_to_goal step={self._step_count} "
                                f"reason={_pg_exit_reason} "
                                f"dist_rad={_pg_dist_now:.3f} "
                                f"steps_in_pg={_pg_steps_in}]",
                                flush=True,
                            )
                elif _pg_steps_in % cfg.pregoal_log_interval_steps == 0 and _pg_steps_in > 0:
                    self._pregoal_dist_history.append(_pg_dist_now)
                    # Stall detection: if the last N intervals show no progress, exit.
                    _n_stall = cfg.pregoal_stall_intervals
                    _pg_stalled = (
                        len(self._pregoal_dist_history) >= _n_stall
                        and all(
                            self._pregoal_dist_history[-_n_stall + i]
                            - self._pregoal_dist_history[-_n_stall + i + 1]
                            < cfg.pregoal_stall_min_progress
                            for i in range(_n_stall - 1)
                        )
                    )
                    print(
                        f"[pregoal_active step={self._step_count} "
                        f"dist_rad={_pg_dist_now:.3f} "
                        f"steps={_pg_steps_in}"
                        + (" stall=True]" if _pg_stalled else "]"),
                        flush=True,
                    )
                    if _pg_stalled:
                        self._pregoal_final_dist_rad = _pg_dist_now
                        self._pre_goal_mode_active = False
                        _goal_idxs = [i for i, a in enumerate(self.attractors)
                                      if a.kind == "goal"]
                        if _goal_idxs:
                            self._active_idx = max(
                                _goal_idxs,
                                key=lambda i: self.attractors[i].static_score,
                            )
                        print(
                            f"[pregoal_return_to_goal step={self._step_count} "
                            f"reason=stall dist_rad={_pg_dist_now:.3f} "
                            f"steps_in_pg={_pg_steps_in}]",
                            flush=True,
                        )

            # Stall trigger: activate escape mode when progress stalls near obstacle.
            # Guard: skip when pre-goal mode is active — arm is already on its final approach.
            _esc_idxs = [i for i, a in enumerate(self.attractors) if a.kind == "escape"]
            if (
                not self._escape_mode_active
                and not self._pre_goal_mode_active
                and not self._in_recovery_mode
                and not self._backtrack_mode_active
                and not self._grad_recovery_active
                and (_esc_idxs or _has_deferred)
                and self._n_escape_generations < _max_gen
                and (self._step_count - self._last_escape_generation_step)
                    >= cfg.goal_shell_cooldown_steps
            ):
                if _use_boundary or _use_simple:
                    _steps_active = self._step_count
                    _above_cross = (
                        _cross_center_z > 0.0
                        and float(lp_base[-1][2]) > _cross_center_z - 0.10
                    )
                    _active_fam_stall = self.attractors[self._active_idx].family
                    if _active_fam_stall == "taskspace_goal":
                        # Task-space progress window: detect stagnation without requiring
                        # alpha>threshold or arm-above-cross, since q_goal=q_start makes
                        # standard joint-space stall metrics meaningless.
                        _ts_window = 80
                        self._taskspace_err_history.append(goal_err_curr)
                        if len(self._taskspace_err_history) > _ts_window + 10:
                            self._taskspace_err_history = self._taskspace_err_history[-(_ts_window + 10):]
                        _stall = False
                        if len(self._taskspace_err_history) >= _ts_window and goal_err_curr > 0.08:
                            _ts_progress = self._taskspace_err_history[-_ts_window] - goal_err_curr
                            if _ts_progress < 0.005:
                                _stall = True
                                print(
                                    f"[taskspace_goal_stall step={self._step_count} "
                                    f"err={goal_err_curr:.4f} "
                                    f"progress={_ts_progress:.4f}]",
                                    flush=True,
                                )
                        # Diagnostic: log periodically if candidates exist but escape not called
                        if (
                            _has_deferred
                            and self._n_escape_generations == 0
                            and self._step_count % 200 == 0
                            and self._step_count > 0
                        ):
                            _ts_prog_diag = (
                                self._taskspace_err_history[-min(_ts_window, len(self._taskspace_err_history))]
                                - goal_err_curr
                            ) if self._taskspace_err_history else 0.0
                            print(
                                f"[taskspace_escape_not_called step={self._step_count} "
                                f"task_err={goal_err_curr:.4f} "
                                f"progress={_ts_prog_diag:.4f} "
                                f"alpha={alpha:.2f} "
                                f"clearance={clearance:.4f}]",
                                flush=True,
                            )
                    else:
                        _stall = (
                            _steps_active >= _stall_window
                            and alpha > _trig_alpha
                            and goal_err_curr > _goal_err_thr
                            and (_above_cross or _cross_center_z <= 0.0)
                        )
                else:
                    _hist_long_enough = len(self._goal_error_history) >= _stall_window
                    _stall = (
                        _hist_long_enough
                        and (self._goal_error_history[-_stall_window] - goal_err_curr) < _stall_prog
                        and alpha > _trig_alpha
                        and clearance < cfg.goal_shell_trigger_clearance_m
                    )

                if _stall:
                    if _online_mode and _has_deferred and not _esc_idxs:
                        # Online mode: deferred candidates exist, no pre-built escape attractors.
                        # Log availability, then generate backtrack (if staging enabled) or
                        # generate escape attractors directly.
                        print(
                            f"[online_escape_candidates_available n={len(self._deferred_escape_candidates)}]",
                            flush=True,
                        )
                        _active_family = self.attractors[self._active_idx].family
                        if cfg.enable_backtrack_staging and _active_family != "taskspace_goal":
                            _bt_ok = self._online_generate_backtrack_attractor(
                                q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                            )
                            if _bt_ok:
                                _bt_new = [i for i, a in enumerate(self.attractors)
                                           if a.kind == "backtrack"]
                                if _bt_new:
                                    self._backtrack_target      = self.attractors[_bt_new[-1]].q_goal.copy()
                                    self._backtrack_mode_active = True
                                    self._backtrack_start_step  = self._step_count
                                    self._n_backtrack_steps     = 0
                                    self._backtrack_reached     = False
                                    self._n_backtrack_switches += 1
                                    # Force switch immediately — bypass safety gate since
                                    # the switch gate often blocks when near an obstacle.
                                    self._active_idx = _bt_new[-1]
                        else:
                            # No backtrack staging (or active attractor is taskspace_goal):
                            # generate ONE escape attractor immediately.
                            # Initialize route-stage BEFORE IK so the stage filter applies.
                            if cfg.enable_yz_route_staging and (_use_boundary or _use_simple):
                                _meta2 = next(
                                    (c.metadata for c in self._deferred_escape_candidates
                                     if c.metadata.get("goal_side_y_sign") is not None),
                                    None,
                                ) or {}
                                _gsy2 = _meta2.get("goal_side_y_sign", 0.0)
                                _ccy2 = _meta2.get("cross_center_y", 0.0)
                                _ccz2 = float(_cross_center_z) if _cross_center_z > 0 else 0.0
                                _czl2 = float(_meta2.get("z_low", -1.0))
                                if _gsy2 != 0.0 and _ccz2 > 0.0:
                                    self._escape_route_stage          = "below"
                                    self._route_stage_start_step      = self._step_count
                                    self._route_stage_goal_side_y_sign = float(_gsy2)
                                    self._route_stage_cross_center_y  = float(_ccy2)
                                    self._route_stage_cross_center_z  = _ccz2
                                    self._route_stage_cross_z_low     = _czl2
                                    print(
                                        f"[route_stage_start step={self._step_count} "
                                        f"stage=below "
                                        f"goal_y_sign={_gsy2:.1f} "
                                        f"cross_center_y={_ccy2:.4f} "
                                        f"cross_center_z={_ccz2:.4f} "
                                        f"cross_z_low={_czl2:.4f}]",
                                        flush=True,
                                    )
                            _ok = self._online_generate_next_escape_attractor(
                                q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                            )
                            if _ok:
                                self._escape_mode_active             = True
                                self._failed_escape_families.clear()
                                self._escape_exhausted               = False
                                self._escape_family_last_switch_step = self._step_count
                                self._escape_dist_history.clear()
                            else:
                                self._n_reentry_fail += 1
                        self._n_escape_generations          += 1
                        self._last_escape_generation_step    = self._step_count
                        self._escape_family_start_goal_error = goal_err_curr
                    else:
                        self._escape_mode_active             = True
                        self._n_escape_generations          += 1
                        self._last_escape_generation_step    = self._step_count
                        self._escape_family_last_switch_step = self._step_count
                        self._escape_family_start_goal_error = goal_err_curr

            # Kind-based scoring bias: penalize escape when inactive, boost when active
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind == "escape":
                    _bias = (
                        cfg.goal_shell_escape_boost
                        if self._escape_mode_active
                        else -cfg.goal_shell_escape_boost
                    )
                    if (
                        self._escape_mode_active
                        and (cfg.goal_shell_blacklist_failed_sectors or _use_boundary)
                        and self.attractors[_ki].shell_sector
                            in self._attempted_escape_families
                    ):
                        _bias -= cfg.goal_shell_escape_boost
                    # Hard-penalize failed families so they can never win hysteresis
                    if (
                        _use_boundary
                        and self.attractors[_ki].shell_sector in self._failed_escape_families
                    ):
                        _bias -= 1000.0
                    scores_combined[_ki] += _bias

        # ---- Recovery scoring bias: boost recovery attractors in recovery mode --
        if cfg.enable_clearance_recovery:
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind == "recovery":
                    _rbias = (
                        cfg.goal_shell_escape_boost
                        if self._in_recovery_mode
                        else -cfg.goal_shell_escape_boost
                    )
                    scores_combined[_ki] += _rbias

        # ---- Portal hard-gate: during escape mode, commit to active escape attractor --
        # When the active attractor is already an escape kind, lock ALL other attractors
        # out so hysteresis cannot switch between escape families (which resets the family
        # failure timer and prevents family cycling from running).  Only the force-switch
        # (family cycling logic) may change which escape attractor is active.
        # When the active attractor is not yet an escape kind, allow escape attractors to
        # compete so the hysteresis/force-switch can transition into escape mode.
        if self._escape_mode_active:
            _active_is_esc = (self.attractors[self._active_idx].kind == "escape")
            for _ki in range(len(self.attractors)):
                if _ki == self._active_idx:
                    continue
                if _active_is_esc:
                    scores_combined[_ki] -= 100.0
                elif self.attractors[_ki].kind not in ("escape",):
                    scores_combined[_ki] -= 100.0

        # ---- Backtrack hard-gate: commit to backtrack attractor when in backtrack mode,
        # and block it from competing during normal goal pursuit / escape mode.
        if self._backtrack_mode_active:
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind != "backtrack":
                    scores_combined[_ki] -= 100.0
        else:
            # Backtrack attractor must never win via hysteresis during normal operation;
            # it is only activated explicitly by the state machine.
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind == "backtrack":
                    scores_combined[_ki] -= 100.0

        # ---- Pre-goal hard-gate: commit to pre_goal when active, block it otherwise.
        if self._pre_goal_mode_active:
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind != "pre_goal":
                    scores_combined[_ki] -= 100.0
        else:
            for _ki in range(len(self.attractors)):
                if self.attractors[_ki].kind == "pre_goal":
                    scores_combined[_ki] -= 100.0

        # ---- Hysteresis switching with safety gate ----------------------
        if timing is not None:
            _t_sw = time.perf_counter()
        best_idx          = int(np.argmax(scores_combined))
        score_active      = float(scores[self._active_idx])
        score_best        = float(scores[best_idx])
        score_active_comb = float(scores_combined[self._active_idx])
        score_best_comb   = float(scores_combined[best_idx])

        # When the active escape family has failed and escape cycling is pending,
        # skip the regular switch entirely.  The -1000 score penalty on the failed
        # escape makes the goal attractor win hysteresis, which would fire
        # escape_return_to_goal and set _escape_mode_active=False — exactly one step
        # before the force-switch block runs.  Blocking the regular switch here lets
        # the force-switch block cycle to the next escape family as intended.
        _escape_cycle_needed = (
            cfg.enable_boundary_escape_waypoints
            and self._escape_mode_active
            and self.attractors[self._active_idx].kind == "escape"
            and self.attractors[self._active_idx].shell_sector in self._failed_escape_families
        )

        switch_blocked = False
        switch_event   = None
        if (
            not goal_switch_locked
            and best_idx != self._active_idx
            and score_best_comb > score_active_comb + effective_hysteresis
            and not _escape_cycle_needed
        ):
            switch_safe = True
            cl_sw_pred  = None
            if cfg.switch_safety_gate:
                att_j     = self.attractors[best_idx]
                qdot_j    = -cfg.K_c * (q - att_j.q_goal)
                qdot_norm = float(np.linalg.norm(qdot_j))
                q_sw      = q + cfg.dt_score * qdot_j / (qdot_norm + 1e-12)
                cl_sw_pred = float(self.clearance_fn(q_sw))
                switch_safe = cl_sw_pred > cfg.switch_min_clearance

            if switch_safe:
                old_idx          = self._active_idx
                self._active_idx = best_idx
                self._n_switches += 1
                self._goal_error_history = []   # reset trap history for new attractor

                # Categorise the switch
                _old_kind = self.attractors[old_idx].kind
                _new_kind = self.attractors[best_idx].kind
                _any_esc_cfg = (cfg.enable_goal_shell_escape or cfg.enable_simple_escape_waypoint
                                or cfg.enable_boundary_escape_waypoints)
                if _any_esc_cfg and _new_kind == "escape":
                    switch_type = (
                        "boundary_escape_switch"
                        if cfg.enable_boundary_escape_waypoints
                        else "simple_escape_switch"
                        if cfg.enable_simple_escape_waypoint
                        else "goal_shell_escape_switch"
                    )
                    self._n_escape_switches += 1
                    _new_esc = self.attractors[best_idx]
                    self._active_escape_target_pos = _new_esc.shell_waypoint
                    self._active_escape_family     = _new_esc.shell_sector
                    if _new_esc.shell_sector:
                        self._attempted_escape_families.add(_new_esc.shell_sector)
                    # Track when this family became active for family cycling
                    if cfg.enable_boundary_escape_waypoints:
                        self._escape_family_last_switch_step = self._step_count
                        self._escape_family_start_goal_error = goal_err_curr
                    self._n_switch_route += 1
                elif _any_esc_cfg and _old_kind == "escape" and _new_kind == "goal":
                    switch_type = (
                        "boundary_escape_return_to_goal"
                        if cfg.enable_boundary_escape_waypoints
                        else "simple_escape_return_to_goal"
                        if cfg.enable_simple_escape_waypoint
                        else "goal_shell_return_to_goal"
                    )
                    self._escape_mode_active = False
                    self._n_switch_route += 1
                elif _new_kind == "pre_goal":
                    switch_type = "pre_goal_switch"
                    self._n_pregoal_switches += 1
                    self._n_switch_route += 1
                elif goal_err_curr < cfg.goal_switch_lock_radius:
                    switch_type = "near_goal_chatter"
                    self._n_switch_near_goal += 1
                elif alpha > cfg.alpha_obstacle_threshold:
                    switch_type = "obstacle_region_switch"
                    self._n_switch_obstacle_region += 1
                else:
                    switch_type = "route_switch"
                    self._n_switch_route += 1

                switch_event = {
                    "step":               self._step_count,
                    "old_idx":            old_idx,
                    "new_idx":            best_idx,
                    "old_family":         self.attractors[old_idx].family,
                    "new_family":         self.attractors[best_idx].family,
                    "old_window":         self.attractors[old_idx].window_label,
                    "new_window":         self.attractors[best_idx].window_label,
                    "goal_error":         goal_err_curr,
                    "clearance":          clearance,
                    "alpha":              alpha,
                    "active_score":       score_active_comb,
                    "new_score":          score_best_comb,
                    "score_margin":       score_best_comb - score_active_comb,
                    "hysteresis":         effective_hysteresis,
                    "switch_penalty":     cfg.w_switch,
                    "old_score_onestep":  float(scores[old_idx]),
                    "new_score_onestep":  float(scores[best_idx]),
                    "old_static_score":   float(static_sc[old_idx]),
                    "new_static_score":   float(static_sc[best_idx]),
                    "old_horizon_score":  float(h_scores[old_idx]),
                    "new_horizon_score":  float(h_scores[best_idx]),
                    "trap_forced":        trap_forced,
                    "switch_type":        switch_type,
                    "switch_blocked":     False,
                    "shell_sector":       self.attractors[best_idx].shell_sector,
                    "old_kind":           _old_kind,
                    "new_kind":           _new_kind,
                }
            else:
                switch_blocked = True
                self._switch_blocked_count += 1
                switch_event = {
                    "step":               self._step_count,
                    "blocked_new_idx":    best_idx,
                    "blocked_new_family": self.attractors[best_idx].family,
                    "blocked_new_window": self.attractors[best_idx].window_label,
                    "goal_error":         goal_err_curr,
                    "clearance":          clearance,
                    "alpha":              alpha,
                    "pred_clearance":     cl_sw_pred,
                    "switch_blocked":     True,
                    "block_reason":       "clearance_gate",
                }
        if timing is not None:
            timing["switching_ms"] = (time.perf_counter() - _t_sw) * 1000

        # ---- Best non-active attractor diagnostics ----------------------
        best_na_idx   = -1
        best_na_score = float("-inf")
        for _i, _sc in enumerate(scores_combined):
            if _i != self._active_idx and _sc > best_na_score:
                best_na_score = _sc
                best_na_idx   = _i
        if best_na_idx >= 0:
            best_na_family  = self.attractors[best_na_idx].family
            score_margin_na = float(best_na_score) - float(scores_combined[self._active_idx])
        else:
            best_na_family  = "none"
            score_margin_na = 0.0

        # ---- Stall-escape forced switch (optional, disabled by default) --
        if (
            cfg.enable_stall_escape_switch
            and switch_event is None          # no switch already happened
            and not goal_switch_locked
            and trap_detected
            and alpha > 0.5
            and best_na_idx >= 0
            and float(best_na_score) > float(scores_combined[self._active_idx])
        ):
            old_idx          = self._active_idx
            self._active_idx = best_na_idx
            self._n_switches += 1
            self._n_switch_route += 1
            self._goal_error_history = []
            switch_event = {
                "step":          self._step_count,
                "old_idx":       old_idx,
                "new_idx":       best_na_idx,
                "old_family":    self.attractors[old_idx].family,
                "new_family":    self.attractors[best_na_idx].family,
                "goal_error":    goal_err_curr,
                "clearance":     clearance,
                "alpha":         alpha,
                "active_score":  float(scores_combined[old_idx]),
                "new_score":     float(best_na_score),
                "score_margin":  float(best_na_score) - float(scores_combined[old_idx]),
                "switch_type":   "stall_escape",
                "switch_blocked": False,
            }
            # Recompute non-active margin relative to the new active attractor.
            best_na_idx   = -1
            best_na_score = float("-inf")
            for _i, _sc in enumerate(scores_combined):
                if _i != self._active_idx and _sc > best_na_score:
                    best_na_score = _sc
                    best_na_idx   = _i
            best_na_family  = self.attractors[best_na_idx].family if best_na_idx >= 0 else "none"
            score_margin_na = float(best_na_score) - float(scores_combined[self._active_idx]) if best_na_idx >= 0 else 0.0

        # ---- Forced stall switch (diagnostic, disabled by default) ---------
        if (
            cfg.enable_forced_stall_switch
            and switch_event is None
            and not goal_switch_locked
            and len(self.attractors) > 1
            and self._n_switch_forced_stall < cfg.forced_stall_max_switches
        ):
            _fss_win      = cfg.forced_stall_window_steps
            _fss_stalled  = (
                len(self._goal_error_history) >= _fss_win
                and (self._goal_error_history[-_fss_win] - goal_err_curr)
                    < cfg.forced_stall_goal_progress_threshold
            )
            _fss_alpha_ok    = alpha > cfg.forced_stall_alpha_threshold
            _fss_cooldown_ok = (
                self._step_count - self._last_forced_switch_step >= cfg.forced_stall_cooldown_steps
            )
            if _fss_stalled and _fss_alpha_ok and _fss_cooldown_ok:
                _fss_progress = (
                    self._goal_error_history[-_fss_win] - goal_err_curr
                    if len(self._goal_error_history) >= _fss_win else 0.0
                )
                old_idx = self._active_idx
                mode    = cfg.forced_stall_switch_mode
                if mode == "cycle":
                    new_idx = (old_idx + 1) % len(self.attractors)
                elif mode == "best_nonactive":
                    new_idx = best_na_idx if best_na_idx >= 0 else (old_idx + 1) % len(self.attractors)
                else:  # "random"
                    _others = [i for i in range(len(self.attractors)) if i != old_idx]
                    new_idx = int(self._rng.choice(_others))
                self._active_idx              = new_idx
                self._n_switches             += 1
                self._n_switch_forced_stall  += 1
                self._last_forced_switch_step = self._step_count
                self._goal_error_history      = []
                switch_event = {
                    "step":                  self._step_count,
                    "old_attractor_idx":     old_idx,
                    "new_attractor_idx":     new_idx,
                    "old_family":            self.attractors[old_idx].family,
                    "new_family":            self.attractors[new_idx].family,
                    "goal_error":            goal_err_curr,
                    "clearance":             clearance,
                    "alpha":                 alpha,
                    "switch_type":           "forced_stall_switch",
                    "forced_mode":           mode,
                    "goal_progress_window":  _fss_progress,
                    "best_nonactive_idx":    best_na_idx,
                    "best_nonactive_family": best_na_family,
                    "best_nonactive_score":  float(best_na_score) if best_na_idx >= 0 else 0.0,
                    "active_score":          float(scores_combined[old_idx]),
                    "score_margin":          (float(best_na_score) - float(scores_combined[old_idx])
                                              if best_na_idx >= 0 else 0.0),
                    "switch_blocked":        False,
                }

        # ---- Simple / boundary escape force-switch (state-machine, bypass hysteresis) --
        # Fires when:
        #   (a) escape mode active but active attractor is still a non-escape kind, OR
        #   (b) active escape family was marked failed → cycle to next unfailed family.
        # Bypasses switch_safety_gate (arm is stuck — staying put is always worse).
        _active_escape_failed = _escape_cycle_needed or (
            cfg.enable_boundary_escape_waypoints
            and self._escape_mode_active
            and self.attractors[self._active_idx].kind == "escape"
            and self.attractors[self._active_idx].shell_sector in self._failed_escape_families
        )
        if (
            (cfg.enable_simple_escape_waypoint or cfg.enable_boundary_escape_waypoints)
            and self._escape_mode_active
            and (switch_event is None or switch_blocked or _active_escape_failed)
            and (
                self.attractors[self._active_idx].kind != "escape"
                or _active_escape_failed
            )
        ):
            if _active_escape_failed:
                # Online mode: generate next escape candidate before cycling
                if _online_mode and _has_deferred:
                    self._online_generate_next_escape_attractor(
                        q, scores, f_atts, static_sc, h_scores, scores_combined, P_tan,
                    )
                # Family cycling: select next non-failed escape family via policy
                _best_esc = self._select_next_escape_family(scores_combined, lp_base[-1])
                _is_cycle = True
            else:
                # Initial escape activation: pick best escape by combined score
                _esc_idxs_force = [i for i, a in enumerate(self.attractors) if a.kind == "escape"
                                   and a.shell_sector not in self._failed_escape_families]
                _best_esc = (
                    max(_esc_idxs_force, key=lambda i: scores_combined[i])
                    if _esc_idxs_force else None
                )
                _is_cycle = False

            if _best_esc is not None:
                old_idx          = self._active_idx
                _old_family      = self.attractors[old_idx].shell_sector if _is_cycle else None
                self._active_idx = _best_esc
                self._n_switches += 1
                self._n_escape_switches += 1
                self._goal_error_history = []
                switch_blocked   = False
                _new_esc = self.attractors[_best_esc]
                self._active_escape_target_pos = _new_esc.shell_waypoint
                self._active_escape_family     = _new_esc.shell_sector
                if _new_esc.shell_sector:
                    self._attempted_escape_families.add(_new_esc.shell_sector)
                if cfg.enable_boundary_escape_waypoints:
                    self._escape_family_last_switch_step = self._step_count
                    self._escape_family_start_goal_error = goal_err_curr
                    self._escape_dist_history.clear()
                self._n_switch_route += 1
                if _is_cycle:
                    self._n_escape_family_cycles += 1
                    _force_type = "escape_family_cycle"
                    print(
                        f"[escape_family_cycle step={self._step_count} "
                        f"old={_old_family} new={_new_esc.shell_sector} "
                        f"reason=no_escape_progress "
                        f"n_failed={len(self._failed_escape_families)} "
                        f"n_cycles={self._n_escape_family_cycles}]",
                        flush=True,
                    )
                else:
                    _force_type = (
                        "boundary_escape_switch"
                        if cfg.enable_boundary_escape_waypoints
                        else "simple_escape_switch"
                    )
                switch_event = {
                    "step":          self._step_count,
                    "old_idx":       old_idx,
                    "new_idx":       _best_esc,
                    "old_family":    self.attractors[old_idx].family,
                    "new_family":    _new_esc.family,
                    "goal_error":    goal_err_curr,
                    "clearance":     clearance,
                    "alpha":         alpha,
                    "active_score":  float(scores_combined[old_idx]),
                    "new_score":     float(scores_combined[_best_esc]),
                    "score_margin":  float(scores_combined[_best_esc]) - float(scores_combined[old_idx]),
                    "switch_type":   _force_type,
                    "reason":        "escape_family_cycle" if _is_cycle else "stall_force",
                    "escape_family": _new_esc.shell_sector,
                    "switch_blocked": False,
                }
            elif _active_escape_failed:
                # All escape families exhausted
                self._escape_exhausted   = True
                self._escape_mode_active = False
                self._escape_route_stage = "none"
                print(
                    f"[boundary_escape_exhausted step={self._step_count} "
                    f"failed={sorted(self._failed_escape_families)}]",
                    flush=True,
                )
                # Trigger backtrack/staging after escape exhaustion (first attempt only)
                if cfg.enable_backtrack_staging and self._n_backtrack_switches == 0:
                    _bt_idxs = [i for i, a in enumerate(self.attractors) if a.kind == "backtrack"]
                    if _bt_idxs:
                        _bt_att = self.attractors[_bt_idxs[0]]
                        if cfg.backtrack_target_mode == "partial_to_start":
                            _beta = float(cfg.backtrack_partial_beta)
                            _bt_att.q_goal = q + _beta * (_bt_att.q_goal - q)
                        self._backtrack_target         = _bt_att.q_goal.copy()
                        self._backtrack_mode_active    = True
                        self._backtrack_start_step     = self._step_count
                        self._n_backtrack_steps        = 0
                        self._backtrack_reached        = False
                        self._n_backtrack_switches    += 1
                        print(
                            f"[escape_exhausted_to_backtrack step={self._step_count} "
                            f"cl={clearance:.4f} "
                            f"target_mode={cfg.backtrack_target_mode}]",
                            flush=True,
                        )
            # Recompute non-active diagnostics
            best_na_idx   = -1
            best_na_score = float("-inf")
            for _i, _sc in enumerate(scores_combined):
                if _i != self._active_idx and _sc > best_na_score:
                    best_na_score = _sc
                    best_na_idx   = _i
            best_na_family  = self.attractors[best_na_idx].family if best_na_idx >= 0 else "none"
            score_margin_na = (float(best_na_score) - float(scores_combined[self._active_idx])
                               if best_na_idx >= 0 else 0.0)

        # ---- Recovery force-switch (state-machine, bypass hysteresis) ---
        if (
            cfg.enable_clearance_recovery
            and self._in_recovery_mode
            and (switch_event is None or switch_blocked)
            and self.attractors[self._active_idx].kind != "recovery"
        ):
            _rec_idxs_force = [i for i, a in enumerate(self.attractors) if a.kind == "recovery"]
            if _rec_idxs_force:
                _best_rec    = max(_rec_idxs_force, key=lambda i: scores_combined[i])
                old_idx          = self._active_idx
                self._active_idx = _best_rec
                self._n_switches += 1
                self._goal_error_history = []
                switch_blocked = False
                _new_rec_att   = self.attractors[_best_rec]
                switch_event = {
                    "step":          self._step_count,
                    "old_idx":       old_idx,
                    "new_idx":       _best_rec,
                    "old_family":    self.attractors[old_idx].family,
                    "new_family":    _new_rec_att.family,
                    "goal_error":    goal_err_curr,
                    "clearance":     clearance,
                    "alpha":         alpha,
                    "active_score":  float(scores_combined[old_idx]),
                    "new_score":     float(scores_combined[_best_rec]),
                    "score_margin":  float(scores_combined[_best_rec]) - float(scores_combined[old_idx]),
                    "switch_type":   "clearance_recovery_switch",
                    "reason":        "stall_force",
                    "recovery_target_pos": (
                        _new_rec_att.shell_waypoint.tolist()
                        if _new_rec_att.shell_waypoint is not None else None
                    ),
                    "switch_blocked": False,
                }
                # Recompute non-active diagnostics
                best_na_idx   = -1
                best_na_score = float("-inf")
                for _i, _sc in enumerate(scores_combined):
                    if _i != self._active_idx and _sc > best_na_score:
                        best_na_score = _sc
                        best_na_idx   = _i
                best_na_family  = self.attractors[best_na_idx].family if best_na_idx >= 0 else "none"
                score_margin_na = (float(best_na_score) - float(scores_combined[self._active_idx])
                                   if best_na_idx >= 0 else 0.0)

        self._step_count += 1

        # ---- Field components -------------------------------------------
        if timing is not None:
            _t_comp = time.perf_counter()
        active_att = self.attractors[self._active_idx]
        f_att      = f_atts[self._active_idx]
        if active_att.kind == "backtrack" and cfg.backtrack_velocity_scale != 1.0:
            f_att = f_att * cfg.backtrack_velocity_scale

        # Task-space escape: recompute f_att from J⁺ at the CURRENT q (f_atts used scoring-time q)
        if active_att.task_pos is not None:
            _ee_now = np.array(lp_base[-1], dtype=float)
            _J_now  = _ee_position_jacobian(q)
            f_att   = cfg.K_c * (np.linalg.pinv(_J_now) @ (active_att.task_pos - _ee_now))

        if active_att.task_pos is not None:
            _task_dist = float(np.linalg.norm(np.array(lp_base[-1]) - active_att.task_pos))
            w = min(1.0, _task_dist / cfg.goal_radius) if cfg.goal_radius > 0 else 1.0
        else:
            w  = _goal_weight(q, active_att.q_goal, cfg.goal_radius)
        aw = alpha * w

        f_tan_raw = P_tan @ f_att
        f_tangent = aw * (f_tan_raw - f_att)
        f_null    = aw * cfg.K_null * (N_proj @ grad_h)

        qdot_des = f_att + f_tangent + f_null

        # ---- Gradient clearance recovery override -----------------------
        # When the arm is pinned near contact, override qdot_des with a
        # clearance-gradient push so the arm moves away from the obstacle
        # before trying to navigate toward an escape waypoint.
        if self._grad_recovery_active and float(np.linalg.norm(grad_h)) > 1e-9:
            _grad_unit  = grad_h / float(np.linalg.norm(grad_h))
            _qdot_clear = cfg.K_clearance_recovery * _grad_unit
            if clearance < cfg.critical_clearance_m:
                _w_clear, _w_portal = 1.0, 0.0
            elif clearance < cfg.recovery_trigger_clearance_m:
                _w_clear, _w_portal = 0.8, 0.2
            else:
                _w_clear, _w_portal = 0.5, 0.5
            qdot_des = _w_clear * _qdot_clear + _w_portal * qdot_des

        # ---- Speed saturation -------------------------------------------
        speed = float(np.linalg.norm(qdot_des))
        if cfg.max_speed < float("inf") and speed > cfg.max_speed:
            qdot_des = qdot_des * (cfg.max_speed / speed)
        if timing is not None:
            timing["composition_ms"] = (time.perf_counter() - _t_comp) * 1000

        # ---- Lookahead safety -------------------------------------------
        if timing is not None:
            _t_la = time.perf_counter()
        if dt is not None and cfg.lookahead_safety and clearance > 0:
            q_pred  = q + float(dt) * qdot_des
            cl_pred = float(self.clearance_fn(q_pred))
            if cl_pred < cfg.lookahead_min_clearance:
                dcl = cl_pred - clearance
                if dcl < 0:
                    scale    = (cfg.lookahead_min_clearance - clearance) / dcl
                    qdot_des = float(np.clip(scale, 0.0, 1.0)) * qdot_des
        if timing is not None:
            timing["lookahead_ms"] = (time.perf_counter() - _t_la) * 1000

        # ---- Diagnostics ------------------------------------------------
        f_att_norm      = float(np.linalg.norm(f_att))
        tangent_norm    = float(np.linalg.norm(f_tangent))
        null_norm       = float(np.linalg.norm(f_null))
        f_tan_proj_norm = float(np.linalg.norm(f_tan_raw))
        tangent_fraction = (
            f_tan_proj_norm / (f_att_norm + 1e-12) if f_att_norm > 1e-12 else 1.0
        )

        if timing is not None:
            timing["total_ms"] = (time.perf_counter() - _t_total) * 1000

        result = GeoMultiAttractorDSResult(
            qdot_des=qdot_des,
            f_attractor=f_att.copy(),
            f_tangent=f_tangent.copy(),
            f_null=f_null.copy(),
            raw_attractor_norm=f_att_norm,
            tangent_norm=tangent_norm,
            null_norm=null_norm,
            active_attractor_idx=self._active_idx,
            active_family=active_att.family,
            n_switches=self._n_switches,
            score_active=score_active,
            score_best=score_best,
            best_attractor_idx=best_idx,
            all_scores=scores,
            clearance=clearance,
            clearance_grad_norm=grad_h_norm,
            obs_blend_alpha=alpha,
            tangent_projection_fraction=tangent_fraction,
            goal_error=float(np.linalg.norm(q - active_att.q_goal)),
            timing_ms=timing,
            all_scores_combined=list(scores_combined),
            horizon_scores=list(h_scores),
            static_scores=list(static_sc),
            switch_blocked=switch_blocked,
            switch_blocked_count=self._switch_blocked_count,
            switch_event=switch_event,
            trap_detected=trap_detected,
            trap_forced_reselection=trap_forced,
            n_switch_near_goal=self._n_switch_near_goal,
            n_switch_obstacle_region=self._n_switch_obstacle_region,
            n_switch_route=self._n_switch_route,
            goal_switch_locked=goal_switch_locked,
            best_nonactive_score=float(best_na_score) if best_na_idx >= 0 else 0.0,
            best_nonactive_idx=best_na_idx,
            best_nonactive_family=best_na_family,
            score_margin_to_nonactive=score_margin_na,
            n_switch_forced_stall=self._n_switch_forced_stall,
            n_switch_total=self._n_switches,
            n_escape_generations=self._n_escape_generations,
            n_escape_attractors=sum(1 for a in self.attractors if a.kind == "escape"),
            n_escape_switches=self._n_escape_switches,
            escape_mode_active=self._escape_mode_active,
            active_attractor_kind=active_att.kind,
            active_escape_family=self._active_escape_family,
            n_recovery_switches=self._n_recovery_switches,
            escape_state_final=(
                "recovery" if self._in_recovery_mode
                else ("portal" if self._escape_mode_active else "goal")
            ),
            n_escape_family_cycles=self._n_escape_family_cycles,
            n_escape_families_failed=len(self._failed_escape_families),
            escape_exhausted=self._escape_exhausted,
            active_escape_family_final=self._active_escape_family,
            n_grad_recovery_steps        = self._n_grad_recovery_steps,
            escape_blocked_count         = self._escape_blocked_count,
            grad_recovery_best_clearance = self._grad_recovery_best_clearance,
            n_backtrack_switches  = self._n_backtrack_switches,
            n_backtrack_steps     = self._n_backtrack_steps,
            backtrack_reached     = self._backtrack_reached,
            backtrack_final_dist_rad = (
                float(np.linalg.norm(q - self._backtrack_target))
                if self._backtrack_target is not None else 0.0
            ),
            runtime_planner_ms          = self._runtime_planner_ms,
            runtime_planner_ms_max      = self._runtime_planner_ms_max,
            runtime_planner_event_count = self._runtime_planner_event_count,
            escape_ik_ms       = self._escape_ik_ms,
            escape_ik_calls    = self._escape_ik_calls,
            pregoal_ik_ms      = self._pregoal_ik_ms,
            pregoal_ik_calls   = self._pregoal_ik_calls,
            n_pregoal_switches     = self._n_pregoal_switches,
            n_reentry_fail         = self._n_reentry_fail,
            pregoal_accepted_count = self._pregoal_accepted_count,
            pregoal_rejected_count = self._pregoal_rejected_count,
            pregoal_last_clearance = self._pregoal_last_clearance,
            pregoal_final_dist_rad = self._pregoal_final_dist_rad,
            n_goal_reentry_blocked = self._n_goal_reentry_blocked,
        )
        return qdot_des, result

    # --- Individual component accessors (mirrors PathDS API) ---

    def f_c(self, q: np.ndarray) -> np.ndarray:
        """Conservative attractor component for the current active attractor."""
        q = np.asarray(q, dtype=float)
        att = self.attractors[self._active_idx]
        return -self.config.K_c * (q - att.q_goal)

    def V(self, q: np.ndarray) -> float:
        """Lyapunov function for the active attractor: V = ½ K_c ‖q − q*‖²."""
        q = np.asarray(q, dtype=float)
        att = self.attractors[self._active_idx]
        diff = q - att.q_goal
        return float(0.5 * self.config.K_c * np.dot(diff, diff))

    def passivity_metric(self, q: np.ndarray, qdot: np.ndarray) -> float:
        """
        z = qdotᵀ (f_tangent + f_null) — energy-injecting components only.

        f_attractor is excluded because it is provably passive (conservative).
        """
        q    = np.asarray(q,    dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        cfg  = self.config

        clearance = float(self.clearance_fn(q))
        alpha = _blend_alpha(clearance, cfg.clearance_enter, cfg.clearance_full)

        if alpha <= 0.0:
            return 0.0

        grad_h = _clearance_gradient(q, self.clearance_fn, eps=cfg.clearance_grad_eps)
        P_tan  = _tangent_projector(grad_h, eps_grad=cfg.eps_grad)
        J      = _position_jacobian(q, eps=cfg.jacobian_eps)
        N      = _nullspace_projector(J)

        att   = self.attractors[self._active_idx]
        f_att = -cfg.K_c * (q - att.q_goal)
        w     = _goal_weight(q, att.q_goal, cfg.goal_radius)
        aw    = alpha * w

        f_tangent = aw * (P_tan @ f_att - f_att)
        f_null    = aw * cfg.K_null * (N @ grad_h)
        return float(np.dot(qdot, f_tangent + f_null))

    # ------------------------------------------------------------------
    # Moving-target stub
    # ------------------------------------------------------------------

    def update_target_pose(self, target_pose) -> None:
        """
        Update the target pose for all attractors using differential IK.

        Not yet implemented.  The recommended future approach is:
            δq_i = J(q_goal_i)⁺ @ δx_target   (differential IK correction)
        followed by a few Newton steps to reduce residual.
        """
        raise NotImplementedError(
            "update_target_pose is not implemented.  "
            "For moving targets, regenerate attractors with HJCD-IK at a slow rate "
            "(e.g. 2–10 Hz) and pass a new attractor list."
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset switching state (active attractor index and switch counter)."""
        self._active_idx            = self._best_static_idx()
        self._n_switches            = 0
        self._step_count            = 0
        self._cached_horizon_scores = None
        self._horizon_last_update   = -9999
        self._goal_error_history    = []
        self._switch_blocked_count  = 0
        self._escape_mode_active          = False
        self._active_escape_target_pos    = None
        self._active_escape_family        = None
        self._attempted_escape_families   = set()
        self._last_escape_generation_step = -9999
        self._n_escape_generations        = 0
        self._n_escape_switches           = 0
        self._in_recovery_mode            = False
        self._active_recovery_target_pos  = None
        self._n_recovery_generations      = 0
        self._n_recovery_switches         = 0
        self._last_recovery_step          = -9999
        self._recovery_start_step         = -9999
        self._grad_recovery_active           = False
        self._grad_recovery_start_clearance  = 0.0
        self._grad_recovery_best_clearance   = 0.0
        self._grad_recovery_start_step       = -9999
        self._n_grad_recovery_steps          = 0
        self._escape_blocked_count           = 0
        self._backtrack_mode_active          = False
        self._backtrack_target               = None
        self._backtrack_start_step           = -9999
        self._n_backtrack_steps              = 0
        self._n_backtrack_switches           = 0
        self._backtrack_reached              = False
        self._escape_ik_ms                   = 0.0
        self._escape_ik_calls                = 0
        self._pregoal_ik_ms                  = 0.0
        self._pregoal_ik_calls               = 0
        self._runtime_planner_ms             = 0.0
        self._runtime_planner_ms_max         = 0.0
        self._runtime_planner_event_count    = 0
        self._pre_goal_mode_active           = False
        self._pregoal_start_step             = 0
        self._n_pregoal_switches             = 0
        self._pregoal_generated              = False
        self._pregoal_is_upgrade             = False
        self._pregoal_dist_history           = []
        self._n_reentry_fail                 = 0
        self._taskspace_err_history          = []

    @property
    def active_attractor(self) -> IKAttractor:
        return self.attractors[self._active_idx]

    @property
    def x_goal(self) -> np.ndarray:
        """Joint-space goal of the active attractor (PathDS-compatible property)."""
        return self.attractors[self._active_idx].q_goal
