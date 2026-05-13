"""
CBF Safety Filter — joint-velocity-level obstacle avoidance.

Implements a velocity-level Control Barrier Function (CBF) safety filter
that modifies a nominal joint velocity command to maintain a configurable
clearance from obstacles while preserving passivity.

Theory
------
For each link sphere *i* and obstacle *j* define:

    h_ij(q) = d_ij(q) − d_safe

where d_ij is the clearance distance (positive outside, negative when
penetrating).  The safe set is {q : h_ij(q) ≥ 0}.

Velocity-level CBF condition (class-K linear function):

    ḣ_ij = ∇_q h_ij · q̇ ≥ −α · h_ij

Formulated as a QP (SciPy SLSQP, no extra dependencies):

    min   ½ · w_nom · ‖q̇ − q̇_nom‖² + ½ · w_slack · ‖s‖²
    s.t.  ∇h_i · q̇ + s_i ≥ −α · h_i   ∀ active i
          s_i ≥ 0

Gradients ∇h_i are computed via central finite differences on
_panda_link_positions (the same FK used by the BiRRT planner).

Contact modes
-------------
"avoid_all"          Enforce constraints for every obstacle.
"allow_list"         Skip obstacles in allowed_contact_obstacle_names.
"phase_dependent"    Skip allowed obstacles only during phases in
                     allowed_contact_phases; enforce everywhere else.

Usage
-----
    from src.solver.controller.cbf_filter import CBFSafetyFilter, CBFConfig
    from src.scenarios.scenario_builders import left_open_u_scenario

    spec = left_open_u_scenario()
    cfg  = CBFConfig(enabled=True, d_safe=0.03, alpha=8.0)
    filt = CBFSafetyFilter(cfg)

    qdot_safe, diag = filt.filter(q, qdot_nom, spec.collision_obstacles())

Integration with impedance controller
--------------------------------------
Set ``ControllerConfig.cbf = CBFConfig(...)`` and pass
``obstacles=spec.collision_obstacles()`` to ``impedance.step()``.
The filter is applied after the DS computes the nominal velocity
(xdot_d) and before the damping torque is calculated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse FK + geometry from planner — single source of truth for link geometry
from src.solver.planner.collision import (
    _LINK_RADII,
    _panda_link_positions,
    _quat_to_rot,
    _sphere_box_signed_dist,
)
from src.scenarios.scenario_schema import Obstacle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CBFConfig:
    """
    Parameters for the CBF velocity-level safety filter.

    Attributes
    ----------
    enabled:
        Master switch.  When False, ``filter()`` is a no-op.
    d_safe:
        Desired minimum clearance between any link sphere surface and any
        obstacle surface (metres).  Positive = gap must be maintained.
    d_buffer:
        Activation radius beyond d_safe.  Constraints are only built for
        (link, obstacle) pairs with clearance ≤ d_safe + d_buffer.
        Keeps the active constraint count small during free motion.
    alpha:
        Class-K gain α ≥ 0.  Higher values enforce sharper avoidance but
        can require larger velocity corrections.  Typical range: 2–20.
    qp_weight_nominal:
        Weight on ‖q̇ − q̇_nom‖² in the QP objective.
    qp_weight_slack:
        Weight on the slack penalty ‖s‖².  Large values make constraints
        nearly hard; small values allow graceful degradation.
    max_correction_norm:
        If set, clamp ‖q̇_safe − q̇_nom‖ to this bound after the QP.
        Useful to prevent large corrections that saturate actuators.
    link_sphere_radii:
        Override per-link sphere radii (metres).  None uses the defaults
        from collision.py (_LINK_RADII).
    sample_points_per_link:
        Number of sample points per link.  1 = link frame origin only.
        2 = also sample midpoint to next link frame (catches the shaft).
    contact_mode:
        One of "avoid_all", "allow_list", "phase_dependent".
    allowed_contact_obstacle_names:
        Obstacles that may be contacted intentionally (used by allow_list /
        phase_dependent modes).
    allowed_contact_phases:
        Phase strings during which allowed_contact_obstacle_names are
        exempt (phase_dependent mode only).
    penetration_tolerance:
        Tolerance added to d_safe for contact-exempt obstacles
        (phase_dependent, tangential_contact_only).  Allows slight
        penetration growth while still limiting it.
    tangential_contact_only:
        When True and a contact obstacle is exempt, still block outward
        penetration growth beyond penetration_tolerance (not yet implemented
        in this version — reserved for future use).
    n_max_constraints:
        Maximum number of active CBF constraints passed to the QP.
        Sorted by clearance ascending (worst first) then truncated.
    fd_eps:
        Step size for central finite-difference gradient (radians).
    debug:
        When True, print per-step diagnostics to stdout.
    """

    enabled:                        bool  = False
    d_safe:                         float = 0.03
    d_buffer:                       float = 0.05
    alpha:                          float = 8.0
    qp_weight_nominal:              float = 1.0
    qp_weight_slack:                float = 1e4
    max_correction_norm:            Optional[float] = None
    link_sphere_radii:              Optional[Dict[int, float]] = None
    sample_points_per_link:         int   = 1
    contact_mode:                   str   = "avoid_all"
    allowed_contact_obstacle_names: List[str] = field(default_factory=list)
    allowed_contact_phases:         List[str]  = field(default_factory=list)
    penetration_tolerance:          float = 0.002
    tangential_contact_only:        bool  = True
    n_max_constraints:              int   = 20
    fd_eps:                         float = 1e-4
    debug:                          bool  = False
    disabled_link_indices:          tuple = ()
    goal_aware: Optional["GoalAwareCBFConfig"] = None


@dataclass
class GoalAwareCBFConfig:
    """
    Goal-aware tapering of CBF safety margins.

    When enabled, d_safe, d_buffer, and qp_weight_slack are smoothly
    reduced as the robot approaches the goal.  This allows the arm to
    enter tight shelf regions where the goal lies inside the CBF
    activation buffer, without permanently relaxing safety margins.

    Taper interpolation parameter s ∈ [0, 1]:
        s = clamp((goal_err - goal_radius_end) /
                  (goal_radius_start - goal_radius_end), 0, 1)
    At s=1 (far from goal): full conservative margins.
    At s=0 (at goal): margins scaled by min_*_scale.

    Attributes
    ----------
    enabled:
        Master switch for goal-aware tapering.
    goal_radius_start:
        Tapering begins when ||q - q_goal|| < this value (radians).
    goal_radius_end:
        Fully tapered when ||q - q_goal|| < this value (radians).
    min_d_safe_scale:
        Scale factor for d_safe at the goal (0 < scale <= 1).
        E.g. 0.5 -> d_safe_eff = 0.015 when goal_err ~= 0.
    min_d_buffer_scale:
        Scale factor for d_buffer at the goal (0 < scale <= 1).
        E.g. 0.3 -> d_buffer_eff = 0.015 when goal_err ~= 0.
    goal_clearance_min:
        Minimum goal clearance required to enable tapering.
        If goal_clearance <= this value, tapering is disabled (Step 5).
    w_slack_goal:
        QP slack weight near the goal (reduced from qp_weight_slack).
        Allows small temporary CBF violations near a verified
        collision-free goal instead of blocking progress entirely.
    _cached_goal_cl:
        Cache for the goal clearance value (survives per-step
        re-instantiation since the config object is shared across steps).
    _cached_goal_q_key:
        Byte-key of the q_goal array for which _cached_goal_cl was computed.
    """
    enabled:             bool  = False
    goal_radius_start:   float = 0.25
    goal_radius_end:     float = 0.05
    min_d_safe_scale:    float = 0.5
    min_d_buffer_scale:  float = 0.3
    goal_clearance_min:  float = 0.02
    w_slack_goal:        float = 100.0

    # Cache for goal clearance (survives per-step re-instantiation since config is shared)
    _cached_goal_cl:    Optional[float] = field(default=None, repr=False, compare=False)
    _cached_goal_q_key: Optional[bytes] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class CBFDiagnostics:
    """
    Per-step output from the CBF safety filter.

    Attributes
    ----------
    n_active:
        Number of active CBF constraints passed to the QP.
    min_clearance:
        Minimum clearance (over all link/obstacle pairs scanned).
        None when no obstacles were checked.
    max_violation:
        Largest CBF constraint violation |min(0, ∇h·q̇_nom + α·h)|
        before filtering.  0 when no constraints are violated.
    correction_norm:
        ‖q̇_safe − q̇_nom‖ — magnitude of the velocity correction.
    qp_success:
        True when the QP converged successfully; False means the
        fallback projection was used.
    active_pairs:
        List of (link_idx, obs_name, clearance) for active constraints.
    most_critical_obstacle:
        Name of the obstacle with the smallest clearance this step.
        None when no active constraints exist.
    cbf_triggered:
        True when the filter made a non-trivial correction
        (correction_norm > 1e-6).
    cbf_slack_used:
        True when the QP fell back to the greedy projection
        (i.e. qp_success is False).  Indicates a hard-to-satisfy step.
    allowed_contact_override_used:
        True when at least one obstacle was exempted from the CBF
        constraints due to contact_mode / phase settings.
    """

    n_active:        int   = 0
    min_clearance:   Optional[float] = None
    max_violation:   float = 0.0
    correction_norm: float = 0.0
    qp_success:      bool  = True
    active_pairs:    List[Tuple[int, str, float]] = field(default_factory=list)
    # Additional CLAUDE.md-required fields
    most_critical_obstacle:       Optional[str] = None
    cbf_triggered:                bool = False
    cbf_slack_used:               bool = False
    allowed_contact_override_used: bool = False
    # Angle between nominal and corrected velocity (degrees); 0 when no correction
    correction_angle_deg:         float = 0.0


# ---------------------------------------------------------------------------
# Internal constraint record
# ---------------------------------------------------------------------------
@dataclass
class _CBFConstraint:
    """Active CBF constraint for one (link, obstacle) pair."""

    h:         float        # barrier value: h(q) = d − d_safe
    grad_h:    np.ndarray   # ∇_q h, shape (n_joints,)
    obs_name:  str
    link_idx:  int
    clearance: float        # raw clearance before subtracting d_safe


# ---------------------------------------------------------------------------
# Clearance distance helpers
# ---------------------------------------------------------------------------
def _clearance_sphere_vs_box(
    centre: np.ndarray,
    radius: float,
    box_pos: np.ndarray,
    box_half: np.ndarray,
    box_orientation_wxyz: Optional[List[float]] = None,
) -> float:
    """
    Signed clearance: distance from sphere surface to box surface.

    Positive  → sphere fully outside box (gap).
    Zero      → sphere surface touches box surface.
    Negative  → sphere centre is closer to box than radius (interpenetration).

    Note: when sphere centre is inside the box the closest-point distance
    is 0, so clearance = -radius (hard penetration).
    """
    R_inv = None
    if box_orientation_wxyz is not None:
        q = box_orientation_wxyz
        if not np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6):
            R_inv = _quat_to_rot(*q).T
    return _sphere_box_signed_dist(centre, radius, box_pos, box_half, R_inv)


def _clearance_sphere_vs_sphere(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
) -> float:
    """Signed clearance between two spheres (positive = separated)."""
    return float(np.linalg.norm(c1 - c2)) - r1 - r2


def _clearance_sphere_vs_cylinder(
    centre: np.ndarray,
    radius: float,
    cyl_pos: np.ndarray,
    cyl_radius: float,
    cyl_half_height: float,
) -> float:
    """
    Signed clearance: sphere vs axis-aligned cylinder (axis = Z).

    Mirrors the geometry of collision.py's _sphere_vs_cylinder but
    returns a float distance instead of a bool.
    """
    dx = centre[0] - cyl_pos[0]
    dy = centre[1] - cyl_pos[1]
    dz = centre[2] - cyl_pos[2]

    r_xy = float(np.sqrt(dx * dx + dy * dy))
    cz   = float(np.clip(dz, -cyl_half_height, cyl_half_height))

    if r_xy < 1e-9:
        # on the cylinder axis — closest surface point is directly up/down/side
        # pick min of radial and axial
        closest = np.array([cyl_pos[0], cyl_pos[1], cyl_pos[2] + cz])
    elif r_xy <= cyl_radius:
        # inside XY footprint
        closest = np.array([cyl_pos[0] + dx, cyl_pos[1] + dy, cyl_pos[2] + cz])
    else:
        scale   = cyl_radius / r_xy
        closest = np.array([
            cyl_pos[0] + dx * scale,
            cyl_pos[1] + dy * scale,
            cyl_pos[2] + cz,
        ])

    dist = float(np.linalg.norm(centre - closest))
    return dist - radius


def _clearance(
    link_pos: np.ndarray,
    link_radius: float,
    obs: Obstacle,
) -> float:
    """
    Compute the signed clearance between a link sphere and an obstacle.

    Dispatches to the appropriate geometry helper based on obs.type.
    Returns a large positive number (1e6) for unknown obstacle types.
    """
    obs_pos = np.array(obs.position, dtype=float)
    t       = obs.type.lower()

    if t == "box":
        half = np.array(obs.size, dtype=float)
        return _clearance_sphere_vs_box(
            link_pos, link_radius, obs_pos, half, obs.orientation_wxyz
        )

    elif t == "sphere":
        return _clearance_sphere_vs_sphere(
            link_pos, link_radius, obs_pos, float(obs.size[0])
        )

    elif t == "cylinder":
        return _clearance_sphere_vs_cylinder(
            link_pos, link_radius,
            obs_pos, float(obs.size[0]), float(obs.size[1]),
        )

    return 1e6   # unknown type — assume far away


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------
class CBFSafetyFilter:
    """
    Velocity-level CBF safety filter for the Franka Panda controller.

    Parameters
    ----------
    config : CBFConfig
        Filter configuration.

    Examples
    --------
    Standalone use::

        spec   = left_open_u_scenario()
        cfg    = CBFConfig(enabled=True, d_safe=0.03, alpha=8.0)
        filt   = CBFSafetyFilter(cfg)
        qdot_s, diag = filt.filter(q, qdot_nom, spec.collision_obstacles())

    With impedance controller::

        ctrl_cfg.cbf = CBFConfig(enabled=True, d_safe=0.03)
        res = impedance.step(q, qdot, ds, tank, dt,
                             config=ctrl_cfg,
                             obstacles=spec.collision_obstacles())
    """

    def __init__(self, config: CBFConfig) -> None:
        self.config = config
        self._radii: Dict[int, float] = dict(
            config.link_sphere_radii if config.link_sphere_radii else _LINK_RADII
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def filter(
        self,
        q:         np.ndarray,
        qdot_nom:  np.ndarray,
        obstacles: List[Obstacle],
        phase:     str = "",
        q_goal:    Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, CBFDiagnostics]:
        """
        Apply the CBF safety filter to a nominal joint velocity.

        Parameters
        ----------
        q :        Current joint configuration, shape (n_joints,).
        qdot_nom : Nominal joint velocity command, shape (n_joints,).
        obstacles: List of Obstacle objects (collision_enabled ones only
                   affect the filter; visual-only are always ignored).
        phase :    Current task phase string (used by phase_dependent mode).
        q_goal :   Goal joint configuration for goal-aware tapering (Steps 4+5+6).
                   When None (default) or goal_aware is disabled, full margins apply.

        Returns
        -------
        qdot_safe : Filtered joint velocity, shape (n_joints,).
        diag      : CBFDiagnostics with per-step statistics.
        """
        q        = np.asarray(q,       dtype=float)
        qdot_nom = np.asarray(qdot_nom, dtype=float)
        n        = len(q)

        cfg  = self.config
        diag = CBFDiagnostics()

        # ---- Early exit (before any FK / geometry work) --------------------
        if not cfg.enabled or not obstacles:
            return qdot_nom.copy(), diag

        # ---- Goal-aware margin tapering (Steps 4+5+6) ----------------------
        d_safe_eff   = cfg.d_safe
        d_buffer_eff = cfg.d_buffer
        w_slack_eff  = cfg.qp_weight_slack

        ga = cfg.goal_aware
        if ga is not None and ga.enabled and q_goal is not None:
            _q_goal  = np.asarray(q_goal, dtype=float)
            goal_err = float(np.linalg.norm(q - _q_goal))
            if goal_err < ga.goal_radius_start:
                # Step 5: verify goal clearance before tapering.
                # Cache result on the config object (shared across per-step
                # re-instantiations of CBFSafetyFilter).
                _q_goal_key = _q_goal.tobytes()
                if ga._cached_goal_q_key != _q_goal_key:
                    _active_for_goal = [o for o in obstacles if o.collision_enabled]
                    _raw = self._min_clearance(_q_goal, _active_for_goal)
                    ga._cached_goal_cl    = float("inf") if _raw is None else _raw
                    ga._cached_goal_q_key = _q_goal_key
                _goal_cl = ga._cached_goal_cl
                if _goal_cl > ga.goal_clearance_min:
                    span = ga.goal_radius_start - ga.goal_radius_end
                    if span > 1e-9:
                        s = float(np.clip(
                            (goal_err - ga.goal_radius_end) / span, 0.0, 1.0
                        ))
                    else:
                        s = 0.0 if goal_err <= ga.goal_radius_end else 1.0
                    d_safe_eff   = cfg.d_safe   * (ga.min_d_safe_scale   + (1.0 - ga.min_d_safe_scale)   * s)
                    d_buffer_eff = cfg.d_buffer * (ga.min_d_buffer_scale + (1.0 - ga.min_d_buffer_scale) * s)
                    w_slack_eff  = ga.w_slack_goal + (cfg.qp_weight_slack - ga.w_slack_goal) * s

        # Filter to collision-enabled obstacles only
        active_obs = [o for o in obstacles if o.collision_enabled]
        if not active_obs:
            return qdot_nom.copy(), diag

        # ---- Build active constraints -------------------------------------
        constraints, any_exempt = self._build_constraints(q, active_obs, phase, d_safe_eff, d_buffer_eff)
        diag.allowed_contact_override_used = any_exempt

        # Track minimum clearance across all scanned pairs
        if constraints:
            all_clearances = [c.clearance for c in constraints]
            diag.min_clearance           = float(min(all_clearances))
            diag.active_pairs            = [(c.link_idx, c.obs_name, c.clearance) for c in constraints]
            diag.most_critical_obstacle  = constraints[0].obs_name  # sorted worst-first
        else:
            # Still compute minimum clearance for diagnostics (scan all)
            diag.min_clearance = self._min_clearance(q, active_obs)

        if not constraints:
            diag.n_active = 0
            if cfg.debug:
                print(f"[CBF] No active constraints. min_clearance={diag.min_clearance:.4f}")
            return qdot_nom.copy(), diag

        diag.n_active = len(constraints)

        # ---- Pre-filter violation metric ----------------------------------
        for con in constraints:
            viol = max(0.0, -(con.grad_h @ qdot_nom + cfg.alpha * con.h))
            diag.max_violation = max(diag.max_violation, viol)

        if cfg.debug:
            print(f"[CBF] active={diag.n_active} min_h={min(c.h for c in constraints):.4f}"
                  f" max_viol={diag.max_violation:.4f}")

        # ---- Solve QP -----------------------------------------------------
        qdot_safe, success = self._solve_qp(q, qdot_nom, constraints, n, w_slack_eff)
        diag.qp_success    = success
        diag.cbf_slack_used = not success   # fallback means slack was needed

        # ---- Clamp correction norm (optional) -----------------------------
        if cfg.max_correction_norm is not None:
            delta = qdot_safe - qdot_nom
            norm  = float(np.linalg.norm(delta))
            if norm > cfg.max_correction_norm:
                qdot_safe = qdot_nom + delta * (cfg.max_correction_norm / norm)

        diag.correction_norm = float(np.linalg.norm(qdot_safe - qdot_nom))
        diag.cbf_triggered   = diag.correction_norm > 1e-6

        # Angle between nominal and corrected velocity directions
        nom_n  = float(np.linalg.norm(qdot_nom))
        safe_n = float(np.linalg.norm(qdot_safe))
        if nom_n > 1e-9 and safe_n > 1e-9:
            cos_a = float(np.clip(qdot_nom @ qdot_safe / (nom_n * safe_n), -1.0, 1.0))
            diag.correction_angle_deg = float(np.degrees(np.arccos(cos_a)))

        if cfg.debug and diag.cbf_triggered:
            print(f"[CBF] triggered: correction={diag.correction_norm:.4f}"
                  f" most_critical={diag.most_critical_obstacle}"
                  f" slack_used={diag.cbf_slack_used}")

        return qdot_safe, diag

    # ------------------------------------------------------------------
    # Constraint building
    # ------------------------------------------------------------------
    def _build_constraints(
        self,
        q:             np.ndarray,
        obstacles:     List[Obstacle],
        phase:         str,
        d_safe_eff:    Optional[float] = None,
        d_buffer_eff:  Optional[float] = None,
    ) -> Tuple[List[_CBFConstraint], bool]:
        """
        Build the list of active CBF constraints for this step.

        A constraint is active when the link-sphere to obstacle clearance
        is ≤ d_safe + d_buffer AND the obstacle is not exempt.

        Returns
        -------
        constraints :
            Sorted list of active _CBFConstraint (worst clearance first),
            capped to n_max_constraints.
        any_exempt :
            True if at least one obstacle was skipped due to contact exemption.
        """
        cfg         = self.config
        if d_safe_eff is None:
            d_safe_eff = cfg.d_safe
        if d_buffer_eff is None:
            d_buffer_eff = cfg.d_buffer
        threshold   = d_safe_eff + d_buffer_eff
        link_pos    = _panda_link_positions(q)   # (7,) positions
        n_links     = len(link_pos)

        raw: List[_CBFConstraint] = []
        any_exempt = False

        for link_idx in range(n_links):
            if link_idx in cfg.disabled_link_indices:
                continue
            pos    = link_pos[link_idx]
            radius = self._radii.get(link_idx, 0.08)

            # Optionally sample midpoint to next link
            sample_positions = [pos]
            if cfg.sample_points_per_link >= 2 and link_idx + 1 < n_links:
                mid = 0.5 * (pos + link_pos[link_idx + 1])
                # Use the smaller of the two radii for the midpoint
                r_next = self._radii.get(link_idx + 1, 0.08)
                sample_positions.append((mid, min(radius, r_next)))

            for obs in obstacles:
                if self._is_exempt(obs, phase):
                    any_exempt = True
                    continue

                for sample in sample_positions:
                    if isinstance(sample, tuple):
                        s_pos, s_rad = sample
                    else:
                        s_pos, s_rad = sample, radius

                    d = _clearance(s_pos, s_rad, obs)

                    if d <= threshold:
                        grad = self._fd_gradient(q, link_idx, obs, s_rad)
                        h    = d - d_safe_eff
                        raw.append(_CBFConstraint(
                            h=h, grad_h=grad, obs_name=obs.name,
                            link_idx=link_idx, clearance=d,
                        ))

        # Sort by clearance ascending (worst constraint first) and cap count
        raw.sort(key=lambda c: c.clearance)
        return raw[: cfg.n_max_constraints], any_exempt

    # ------------------------------------------------------------------
    # Finite-difference gradient
    # ------------------------------------------------------------------
    def _fd_gradient(
        self,
        q:        np.ndarray,
        link_idx: int,
        obs:      Obstacle,
        radius:   float,
    ) -> np.ndarray:
        """
        Central finite-difference gradient of h(q) = d(q, link, obs) - d_safe.

        Only perturbs joints 0..link_idx (serial chain causality: joints
        after link_idx do not affect that link's world position).
        """
        n     = len(q)
        eps   = self.config.fd_eps
        grad  = np.zeros(n)

        for j in range(min(link_idx + 1, n)):    # joints 0..link_idx, capped to n DOF
            q_p = q.copy(); q_p[j] += eps
            q_m = q.copy(); q_m[j] -= eps

            pos_p = _panda_link_positions(q_p)[link_idx]
            pos_m = _panda_link_positions(q_m)[link_idx]

            d_p   = _clearance(pos_p, radius, obs)
            d_m   = _clearance(pos_m, radius, obs)
            grad[j] = (d_p - d_m) / (2.0 * eps)

        # Joints link_idx+1..n-1 remain zero (no causal influence)
        return grad

    # ------------------------------------------------------------------
    # QP solver (SciPy SLSQP)
    # ------------------------------------------------------------------
    def _solve_qp(
        self,
        q:           np.ndarray,
        qdot_nom:    np.ndarray,
        constraints: List[_CBFConstraint],
        n:           int,
        w_slack_eff: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve the CBF QP.

        Decision variable: x = [qdot (n,), slack (1,)]
        Objective: ½·w_nom·‖qdot−qdot_nom‖² + ½·w_slack·slack²
        Constraints:
            ∇h_i · qdot + α·h_i + slack ≥ 0   ∀ active i   (CBF)
            slack ≥ 0                                          (positivity)

        A single shared slack is used for numerical simplicity.  If any
        constraint is violated the slack absorbs it; the objective then
        penalises the slack at weight w_slack.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fall back to projection if scipy is unavailable
            return self._greedy_projection(qdot_nom, constraints), False

        cfg     = self.config
        w_nom   = cfg.qp_weight_nominal
        w_slack = w_slack_eff
        alpha   = cfg.alpha

        # Build constraint matrix:  A[i] = grad_h_i  (n,),  b[i] = -alpha*h_i
        A = np.stack([c.grad_h for c in constraints])   # (m, n)
        b = np.array([-alpha * c.h for c in constraints])  # (m,)

        # Decision variable: [qdot(n), slack(1)]  → size n+1
        def objective(x: np.ndarray) -> float:
            dq = x[:n] - qdot_nom
            s  = x[n]
            return 0.5 * w_nom * float(dq @ dq) + 0.5 * w_slack * float(s * s)

        def jac_obj(x: np.ndarray) -> np.ndarray:
            g         = np.zeros(n + 1)
            g[:n]     = w_nom * (x[:n] - qdot_nom)
            g[n]      = w_slack * x[n]
            return g

        def con_cbf(x: np.ndarray) -> np.ndarray:
            """A·qdot + slack ≥ b  →  A·x[:n] + x[n] - b ≥ 0"""
            return A @ x[:n] + x[n] - b

        def jac_cbf(x: np.ndarray) -> np.ndarray:
            J         = np.zeros((len(b), n + 1))
            J[:, :n]  = A
            J[:, n]   = 1.0
            return J

        def con_slack(x: np.ndarray) -> float:
            return x[n]   # ≥ 0

        def jac_slack(x: np.ndarray) -> np.ndarray:
            g    = np.zeros(n + 1)
            g[n] = 1.0
            return g

        scipy_constraints = [
            {"type": "ineq", "fun": con_cbf,   "jac": jac_cbf},
            {"type": "ineq", "fun": con_slack,  "jac": jac_slack},
        ]

        x0     = np.append(qdot_nom, 0.0)
        result = minimize(
            objective, x0,
            jac=jac_obj,
            method="SLSQP",
            constraints=scipy_constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        if result.success or result.status == 0:
            return result.x[:n], True

        if cfg.debug:
            print(f"[CBF] QP failed (status={result.status}): {result.message}")

        # Fallback: greedy projection
        return self._greedy_projection(qdot_nom, constraints), False

    # ------------------------------------------------------------------
    # Greedy projection fallback
    # ------------------------------------------------------------------
    def _greedy_projection(
        self,
        qdot_nom:    np.ndarray,
        constraints: List[_CBFConstraint],
    ) -> np.ndarray:
        """
        Fallback when QP fails.

        For each violated constraint, project q̇ onto the constraint
        hyperplane in order of severity (most violated first).
        This is not globally optimal but ensures all hard violations
        are corrected.
        """
        cfg    = self.config
        qdot   = qdot_nom.copy()

        for con in constraints:
            lhs = float(con.grad_h @ qdot) + cfg.alpha * con.h
            if lhs < 0.0:
                # Project qdot onto the constraint hyperplane
                norm_sq = float(con.grad_h @ con.grad_h)
                if norm_sq < 1e-12:
                    continue
                qdot = qdot - (lhs / norm_sq) * con.grad_h

        return qdot

    # ------------------------------------------------------------------
    # Contact exemption
    # ------------------------------------------------------------------
    def _is_exempt(self, obs: Obstacle, phase: str) -> bool:
        """Return True if this obstacle should be exempt from CBF for this phase."""
        mode = self.config.contact_mode

        if mode == "avoid_all":
            return False

        if mode == "allow_list":
            return obs.name in self.config.allowed_contact_obstacle_names

        if mode == "phase_dependent":
            return (
                obs.name in self.config.allowed_contact_obstacle_names
                and phase in self.config.allowed_contact_phases
            )

        return False   # unknown mode → conservative

    # ------------------------------------------------------------------
    # Utility: minimum clearance scan (diagnostics only)
    # ------------------------------------------------------------------
    def _min_clearance(
        self,
        q:         np.ndarray,
        obstacles: List[Obstacle],
    ) -> Optional[float]:
        """Scan all (link, obstacle) pairs and return minimum clearance."""
        link_pos = _panda_link_positions(q)
        min_d    = float("inf")

        for link_idx, pos in enumerate(link_pos):
            radius = self._radii.get(link_idx, 0.08)
            for obs in obstacles:
                if not obs.collision_enabled:
                    continue
                d    = _clearance(pos, radius, obs)
                min_d = min(min_d, d)

        return min_d if min_d < float("inf") else None
