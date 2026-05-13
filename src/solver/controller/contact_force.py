"""
Contact Force Controller — Phase 6 (NEW, CRITICAL)

Implements a potential-based contact force tracking law that is passivity-
safe by construction (and further hardened by the passivity filter).

Nominal contact force velocity:

    f_n_nom = k_f · (F_d − F_n) · n̂         (task space)
    ↓ joint space (via Jacobian transpose):
    τ_n_nom = Jᵀ_lin · f_n_nom

This is equivalent to −∇V_n where V_n = (1/2) k_f (F_n − F_d)²,
which is a proper Lyapunov candidate for the force error.

An optional passivity filter is applied to the resulting joint-space
contribution to ensure ẋᵀ τ_n ≤ ε_contact.

Typical usage:

    tau_contact = contact_force_control(
        F_contact, F_desired, n_hat,
        jacobian=J,
        qdot=qdot,
        epsilon=epsilon,
    )

    xdot_d += tau_contact   # add to desired velocity in joint space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.solver.controller.passivity_filter import (
    FilterResult,
    PassivityFilterConfig,
    filter_residual,
    filter_with_bounds,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ContactForceConfig:
    """Parameters for the contact force controller."""

    k_f:         float = 0.5    # proportional force feedback gain (reduced for Phase 6)
    epsilon:     float = 0.2    # upper power bound  (+ε_n): wider band reduces clip rate
    epsilon_low: float = -0.2   # lower power bound  (−ε_n): prevents over-dissipation
    delta:       float = 1e-6   # filter regularisation
    use_filter:  bool  = True   # apply two-sided power-band filter to contact term


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def contact_force_control(
    F_contact: np.ndarray,
    F_desired: float,
    n_hat:     np.ndarray,
    jacobian:  Optional[np.ndarray] = None,
    qdot:      Optional[np.ndarray] = None,
    config:    Optional[ContactForceConfig] = None,
) -> Tuple[np.ndarray, FilterResult]:
    """
    Compute the joint-space contact force contribution.

    The potential-based nominal term:

        F_error   = F_desired − n̂ᵀ F_contact
        f_n_task  = k_f · F_error · n̂        (Cartesian, shape 3)
        τ_n_nom   = Jᵀ[:3] · f_n_task         (joint space, shape n)

    Then optionally apply the passivity filter to τ_n_nom.

    Args:
        F_contact: Measured contact force vector (3,).
        F_desired: Desired normal force magnitude (scalar).
        n_hat:     Contact surface outward normal, unit vector (3,).
        jacobian:  6×n Jacobian at the EE.  If None, returns zero.
        qdot:      Current joint velocity (n,).  Needed for filter.
        config:    ContactForceConfig (safe defaults when None).

    Returns:
        Tuple of:
          * τ_n: joint-space contact contribution, shape (n,)
          * result: FilterResult (clipped=False when filter not applied)
    """
    if config is None:
        config = ContactForceConfig()

    F_contact = np.asarray(F_contact, dtype=float)
    n_hat     = np.asarray(n_hat,     dtype=float)
    n_hat_u   = n_hat / (np.linalg.norm(n_hat) + 1e-12)   # ensure unit

    if jacobian is None:
        # No Jacobian available — cannot map to joint space
        n = len(qdot) if qdot is not None else 7
        zero_result = FilterResult(
            fR_filtered=np.zeros(n),
            power_nom=0.0,
            power_filtered=0.0,
            clipped=False,
            epsilon=config.epsilon,
        )
        return np.zeros(n), zero_result

    jacobian = np.asarray(jacobian, dtype=float)   # shape (6, n)
    n        = jacobian.shape[1]

    # --- Nominal contact force contribution (potential-based) ---------------
    F_normal  = float(n_hat_u @ F_contact)          # current normal force
    F_error   = float(F_desired) - F_normal          # signed error
    f_n_task  = config.k_f * F_error * n_hat_u      # Cartesian task-space (3,)
    J_lin     = jacobian[:3, :]                      # linear Jacobian (3, n)
    tau_n_nom = J_lin.T @ f_n_task                   # joint-space (n,)

    # --- Two-sided power-band filter: −ε_n ≤ ẋᵀ τ_n ≤ +ε_n ----------------
    if config.use_filter and qdot is not None:
        qdot = np.asarray(qdot, dtype=float)
        result = filter_with_bounds(
            qdot, tau_n_nom,
            eps_low=config.epsilon_low,
            eps_high=config.epsilon,
            delta=config.delta,
        )
        tau_n = result.fR_filtered
    else:
        tau_n = tau_n_nom
        result = FilterResult(
            fR_filtered=tau_n_nom.copy(),
            power_nom=float(qdot @ tau_n_nom) if qdot is not None else 0.0,
            power_filtered=float(qdot @ tau_n_nom) if qdot is not None else 0.0,
            clipped=False,
            epsilon=config.epsilon,
        )

    return tau_n, result


# ---------------------------------------------------------------------------
# Utility: measure contact force from SimEnv contact list
# ---------------------------------------------------------------------------
def extract_contact_normal_force(
    contacts: list,
    n_hat: np.ndarray,
) -> np.ndarray:
    """
    Aggregate contact forces from SimEnv.get_contact_forces() into a
    single resultant force vector (3,).

    Args:
        contacts: List of dicts from SimEnv.get_contact_forces().
        n_hat:    Contact normal direction (for sign reference).

    Returns:
        Resultant contact force vector (3,), zeros if no contacts.
    """
    if not contacts:
        return np.zeros(3)
    total = np.zeros(3)
    for c in contacts:
        total += np.asarray(c["force"], dtype=float)
    return total
