"""
Task-space modulation-matrix obstacle avoidance.

For each obstacle defines a distance-like scalar gamma(x) >= 1 outside,
= 1 at the surface.  The modulation matrix:

    M(x) = E(x) @ diag(lambda_n, lambda_t, lambda_t) @ E(x).T

where E = [n, e1, e2] is an orthonormal basis (outward normal + two tangents)
and the eigenvalues

    lambda_n = 1 - 1 / gamma^rho        (attenuates into-obstacle motion)
    lambda_t = 1 + gain / gamma^rho     (preserves / enhances tangential motion)

smoothly approach identity far from the obstacle.

Usage
-----
    from src.solver.ds.modulation import ModulationConfig, apply_modulation

    cfg   = ModulationConfig(enabled=True)
    qdot_mod, diag = apply_modulation(q, qdot_nom, obstacles, cfg)

The function maps qdot_nom through the task-space modulation and back to
joint space while preserving null-space motion from the original command:

    qdot_mod = J^+ @ (M @ (J @ qdot_nom)) + (I - J^+ @ J) @ qdot_nom
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.solver.planner.collision import _panda_link_positions
from src.scenarios.scenario_schema import Obstacle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModulationConfig:
    """Parameters for the task-space modulation safety layer."""

    enabled:         bool  = False
    safety_margin:   float = 0.03   # extra clearance added to obstacle surface (m)
    rho:             float = 1.0    # exponent: higher = sharper onset near surface
    tangent_gain:    float = 0.5    # lambda_t = 1 + tangent_gain / gamma^rho
    fd_eps:          float = 1e-4   # FD step for Jacobian (rad)
    damping:         float = 1e-3   # damping factor for pseudoinverse regularisation
    min_gamma:       float = 1e-3   # clamp gamma below this (avoids /0 at centre)
    activation_dist: float = 0.5    # only build constraints when gamma < activation_dist
    debug:           bool  = False


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class ModulationDiagnostics:
    """Per-step output from the modulation layer."""
    gamma_min:       float = float("inf")  # min gamma across all active obstacles
    n_active:        int   = 0             # obstacles within activation_dist
    correction_norm: float = 0.0           # ||qdot_mod - qdot_nom||
    active_names:    List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gamma functions (distance-like scalar, = 1 at surface, > 1 outside)
# ---------------------------------------------------------------------------

def _gamma_sphere(x_ee: np.ndarray, obs: Obstacle, safety_margin: float) -> float:
    """
    gamma = ||x - c|| / (r + safety_margin)

    = 1 at the inflated sphere surface.
    """
    c = np.array(obs.position, dtype=float)
    r = float(obs.size[0]) + safety_margin
    if r < 1e-9:
        return float("inf")
    d = float(np.linalg.norm(x_ee - c))
    return max(d / r, 1e-6)


def _gamma_box(x_ee: np.ndarray, obs: Obstacle, safety_margin: float) -> float:
    """
    Star-shaped gamma for an axis-aligned box:

        gamma = ||x - c|| / t_surface

    where t_surface is the ray-to-surface distance from centre c toward x_ee.

    This equals 1 when x_ee is exactly on the inflated box surface.
    """
    c    = np.array(obs.position, dtype=float)
    half = np.array(obs.size, dtype=float) + safety_margin
    delta = x_ee - c
    dist  = float(np.linalg.norm(delta))

    if dist < 1e-9:
        return 1e-6  # at the centre — inside obstacle

    direction = delta / dist

    # Ray from c along direction: c + t * direction.
    # Intersects box face i at t_i = half[i] / |direction[i]|.
    # The closest face intersection is the minimum t_i.
    t_surface = float("inf")
    for i in range(3):
        if abs(direction[i]) > 1e-9:
            t_i = half[i] / abs(direction[i])
            if t_i < t_surface:
                t_surface = t_i

    if t_surface < 1e-9 or t_surface == float("inf"):
        return float("inf")

    return max(dist / t_surface, 1e-6)


def _gamma_cylinder(x_ee: np.ndarray, obs: Obstacle, safety_margin: float) -> float:
    """
    Star-shaped gamma for an axis-aligned cylinder (axis = Z).

        gamma = ||x - c|| / t_surface

    The surface is the closer of the lateral wall (radial) or the end cap (axial).
    """
    c          = np.array(obs.position, dtype=float)
    cyl_r      = float(obs.size[0]) + safety_margin
    cyl_hh     = float(obs.size[1]) + safety_margin
    delta      = x_ee - c
    dist       = float(np.linalg.norm(delta))

    if dist < 1e-9:
        return 1e-6

    direction = delta / dist

    # Radial component (xy-plane)
    r_xy = float(np.sqrt(direction[0] ** 2 + direction[1] ** 2))
    # Axial component (z)
    r_z  = abs(direction[2])

    t_lateral = cyl_r / r_xy if r_xy > 1e-9 else float("inf")
    t_cap     = cyl_hh / r_z  if r_z  > 1e-9 else float("inf")

    t_surface = min(t_lateral, t_cap)
    if t_surface < 1e-9 or t_surface == float("inf"):
        return float("inf")

    return max(dist / t_surface, 1e-6)


def gamma_obstacle(x_ee: np.ndarray, obs: Obstacle, safety_margin: float) -> float:
    """Dispatch to the appropriate gamma function based on obstacle type."""
    t = obs.type.lower()
    if t == "sphere":
        return _gamma_sphere(x_ee, obs, safety_margin)
    elif t == "box":
        return _gamma_box(x_ee, obs, safety_margin)
    elif t == "cylinder":
        return _gamma_cylinder(x_ee, obs, safety_margin)
    return float("inf")  # unknown type — treat as far away


# ---------------------------------------------------------------------------
# Normal and basis
# ---------------------------------------------------------------------------

def outward_normal(x_ee: np.ndarray, obs_center: np.ndarray) -> np.ndarray:
    """Outward unit normal: radial direction from obstacle centre to x_ee."""
    delta = x_ee - obs_center
    dist  = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return delta / dist


def _perpendicular_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two unit vectors orthogonal to n (and each other).
    Uses Gram-Schmidt against a reference vector.
    """
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = ref - np.dot(ref, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


# ---------------------------------------------------------------------------
# Single-obstacle modulation matrix
# ---------------------------------------------------------------------------

def modulation_matrix(
    x_ee: np.ndarray,
    obs: Obstacle,
    config: ModulationConfig,
) -> Optional[np.ndarray]:
    """
    Build the 3×3 modulation matrix for one obstacle.

    Returns None when the obstacle is too far (gamma > activation_dist) or
    the obstacle is not collision-enabled.
    """
    if not obs.collision_enabled:
        return None

    gamma = gamma_obstacle(x_ee, obs, config.safety_margin)
    gamma = max(gamma, config.min_gamma)

    # Eigenvalues
    inv_g_rho = 1.0 / (gamma ** config.rho)
    lambda_n  = 1.0 - inv_g_rho
    lambda_t  = 1.0 + config.tangent_gain * inv_g_rho

    # Orthonormal basis: normal + two tangents
    centre = np.array(obs.position, dtype=float)
    n      = outward_normal(x_ee, centre)
    e1, e2 = _perpendicular_basis(n)

    # E columns: [n, e1, e2]  (3×3)
    E = np.column_stack([n, e1, e2])
    D = np.diag([lambda_n, lambda_t, lambda_t])

    return E @ D @ E.T


# ---------------------------------------------------------------------------
# Combined modulation
# ---------------------------------------------------------------------------

def combined_modulation(
    x_ee: np.ndarray,
    obstacles: List[Obstacle],
    config: ModulationConfig,
) -> Tuple[np.ndarray, ModulationDiagnostics]:
    """
    Compute the combined 3×3 modulation matrix for all active obstacles.

    Uses a weighted average of per-obstacle modulation matrices where the
    weight is proportional to 1/gamma (closer obstacles contribute more).

    Returns (M_combined, diag).  M_combined = I when no obstacles are active.
    """
    diag = ModulationDiagnostics()
    M    = np.eye(3)

    col_obs = [o for o in obstacles if o.collision_enabled]
    if not col_obs:
        return M, diag

    # Compute gamma for all obstacles; filter to active ones
    gammas  = []
    weights = []
    mats    = []
    names   = []

    for obs in col_obs:
        gamma = gamma_obstacle(x_ee, obs, config.safety_margin)
        gamma = max(gamma, config.min_gamma)
        gammas.append(gamma)

        M_i = modulation_matrix(x_ee, obs, config)
        if M_i is None:
            continue

        mats.append(M_i)
        weights.append(1.0 / max(gamma, 1e-9))
        names.append(obs.name)

    if not mats:
        return M, diag

    diag.n_active       = len(mats)
    diag.gamma_min      = float(min(gammas))
    diag.active_names   = names

    # Weighted average
    total_w = sum(weights)
    M_combined = sum(w * m for w, m in zip(weights, mats)) / total_w

    return M_combined, diag


# ---------------------------------------------------------------------------
# EE Jacobian (position only, 3×7)
# ---------------------------------------------------------------------------

def _ee_jacobian_fd(q: np.ndarray, fd_eps: float = 1e-4) -> np.ndarray:
    """
    Central finite-difference Jacobian of panda_hand position w.r.t. q.

    Returns J of shape (3, 7).
    """
    q   = np.asarray(q, dtype=float)
    n   = len(q)
    J   = np.zeros((3, n))

    for j in range(n):
        q_p = q.copy(); q_p[j] += fd_eps
        q_m = q.copy(); q_m[j] -= fd_eps
        pos_p = _panda_link_positions(q_p)[-1]  # panda_hand
        pos_m = _panda_link_positions(q_m)[-1]
        J[:, j] = (pos_p - pos_m) / (2.0 * fd_eps)

    return J


def _damped_pseudoinverse(J: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """
    Damped least-squares pseudoinverse:  J^+ = J.T @ (J J.T + λI)^-1.

    Regularises the inversion when J is near-singular (e.g. at singularities).
    Returns shape (7, 3).
    """
    m, n  = J.shape  # (3, 7)
    JJT   = J @ J.T + damping * np.eye(m)
    return J.T @ np.linalg.inv(JJT)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def apply_modulation(
    q:         np.ndarray,
    qdot_nom:  np.ndarray,
    obstacles: List[Obstacle],
    config:    ModulationConfig,
) -> Tuple[np.ndarray, ModulationDiagnostics]:
    """
    Apply task-space modulation to a nominal joint velocity.

    Pipeline
    --------
    1. FK → EE position x_ee (panda_hand frame).
    2. FD Jacobian J (3×7).
    3. Map to task space: xdot_task = J @ qdot_nom.
    4. Modulate:  xdot_mod = M(x_ee) @ xdot_task.
    5. Map back, preserving null-space:
       qdot_mod = J^+ @ xdot_mod + (I - J^+ @ J) @ qdot_nom

    Returns
    -------
    qdot_mod : np.ndarray  (7,) — modulated joint velocity.
    diag     : ModulationDiagnostics — per-step statistics.
    """
    q        = np.asarray(q,       dtype=float)
    qdot_nom = np.asarray(qdot_nom, dtype=float)

    if not config.enabled or not obstacles:
        return qdot_nom.copy(), ModulationDiagnostics()

    col_obs = [o for o in obstacles if o.collision_enabled]
    if not col_obs:
        return qdot_nom.copy(), ModulationDiagnostics()

    # EE position and Jacobian
    x_ee = _panda_link_positions(q)[-1]  # panda_hand world position
    J    = _ee_jacobian_fd(q, config.fd_eps)

    # Task-space nominal velocity
    xdot_task = J @ qdot_nom  # (3,)

    # Modulation matrix
    M, diag = combined_modulation(x_ee, col_obs, config)

    # Modulated task-space velocity
    xdot_mod = M @ xdot_task  # (3,)

    # Map back to joint space + null-space preservation
    J_pinv    = _damped_pseudoinverse(J, config.damping)   # (7, 3)
    N         = np.eye(len(q)) - J_pinv @ J               # null-space projector (7,7)
    qdot_mod  = J_pinv @ xdot_mod + N @ qdot_nom          # (7,)

    diag.correction_norm = float(np.linalg.norm(qdot_mod - qdot_nom))

    if config.debug and diag.n_active > 0:
        print(f"[Modulation] gamma_min={diag.gamma_min:.4f}  "
              f"n_active={diag.n_active}  "
              f"correction={diag.correction_norm:.4f}")

    return qdot_mod, diag
