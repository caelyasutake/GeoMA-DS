"""
Passivity Filter — Phase 5 (NEW, CRITICAL)

Enforces an instantaneous power constraint on the residual DS component f_R
BEFORE the energy tank can react, providing a proactive first line of defence.

Constraint:
    ẋᵀ f_R  ≤  ε(s)

Where ε(s) is a state-dependent threshold supplied by the caller
(typically ε = ε_min + (ε_max − ε_min) · β_R(s) from the energy tank).

When the constraint is violated the filter projects f_R onto the
constraint hyperplane {f : ẋᵀ f = ε} via the minimum-norm correction:

    f_R* = f_R_nom − [(ẋᵀ f_R_nom − ε) / (‖ẋ‖² + δ)] · ẋ

Where δ > 0 is a small regularisation constant that prevents division by
near-zero when ẋ → 0.  Note that when ‖ẋ‖ → 0 no energy can be injected
regardless of f_R, so skipping the projection is physically correct.

References:
    Kronander & Billard (2016) — Passive Interaction Control with DSs
    Ferraguti et al. (2013)    — Tank-based impedance control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PassivityFilterConfig:
    """Parameters for the passivity filter."""

    epsilon_min: float = 0.0    # strictest power threshold (tank empty)
    epsilon_max: float = 0.0    # most lenient threshold (tank full)
    delta: float = 1e-6         # regularisation to avoid division by zero


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    """Output of one passivity filter application."""

    fR_filtered:   np.ndarray   # projected f_R (shape n)
    power_nom:     float        # ẋᵀ f_R_nom (before projection)
    power_filtered: float       # ẋᵀ f_R_filtered (after projection)
    clipped:       bool         # True when projection was applied
    epsilon:       float        # threshold used


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------
def filter_residual(
    xdot:       np.ndarray,
    fR_nom:     np.ndarray,
    epsilon:    float = 0.0,
    config:     Optional[PassivityFilterConfig] = None,
) -> FilterResult:
    """
    Project ``fR_nom`` so that ``ẋᵀ f_R ≤ epsilon``.

    Args:
        xdot:    Current joint velocity ẋ, shape (n,).
        fR_nom:  Nominal residual DS velocity f_R, shape (n,).
        epsilon: Power threshold.  Positive → some injection allowed;
                 zero → no injection; negative → must be dissipative.
        config:  PassivityFilterConfig (safe defaults when None).

    Returns:
        FilterResult with the (possibly projected) f_R and diagnostic fields.
    """
    if config is None:
        config = PassivityFilterConfig()

    xdot   = np.asarray(xdot,   dtype=float)
    fR_nom = np.asarray(fR_nom, dtype=float)

    power_nom = float(xdot @ fR_nom)

    if power_nom <= epsilon:
        # Constraint already satisfied — return unchanged
        return FilterResult(
            fR_filtered=fR_nom.copy(),
            power_nom=power_nom,
            power_filtered=power_nom,
            clipped=False,
            epsilon=epsilon,
        )

    # Project onto the constraint boundary ẋᵀ f_R = epsilon
    denom      = float(xdot @ xdot) + config.delta
    correction = (power_nom - epsilon) / denom
    fR_star    = fR_nom - correction * xdot

    power_filtered = float(xdot @ fR_star)

    return FilterResult(
        fR_filtered=fR_star,
        power_nom=power_nom,
        power_filtered=power_filtered,
        clipped=True,
        epsilon=epsilon,
    )


# ---------------------------------------------------------------------------
# Convenience: compute epsilon from beta_R
# ---------------------------------------------------------------------------
def compute_epsilon(
    beta_R:      float,
    epsilon_min: float = 0.0,
    epsilon_max: float = 0.0,
) -> float:
    """
    State-dependent power threshold:

        ε(s) = ε_min + (ε_max − ε_min) · β_R(s)

    When the tank is full  (β_R = 1): ε = ε_max  (most lenient)
    When the tank is empty (β_R = 0): ε = ε_min  (most strict)
    """
    return float(epsilon_min + (epsilon_max - epsilon_min) * float(beta_R))


# ---------------------------------------------------------------------------
# Phase 5B: Energy-neutral residual (orthogonalization)
# ---------------------------------------------------------------------------
def orthogonalize_residual(
    xdot:  np.ndarray,
    fR:    np.ndarray,
    eps:   float = 1e-8,
) -> np.ndarray:
    """
    Remove the velocity-aligned component from f_R so that ẋᵀ f_R = 0 exactly.

    Uses a normalised unit vector v = ẋ / ‖ẋ‖ so the projection is exact:

        v          = ẋ / (‖ẋ‖ + ε)     (unit-length approximation)
        f_R_neutral = f_R − (vᵀ f_R) · v

    Since ‖v‖ ≈ 1, we get vᵀ f_R_neutral = vᵀ f_R − (vᵀ f_R)(vᵀ v) = 0.

    When ‖ẋ‖ < 1e-6 the velocity is effectively zero and no energy can be
    injected regardless of f_R, so f_R is returned unchanged.

    Args:
        xdot: Current joint velocity ẋ, shape (n,).
        fR:   Nominal residual DS velocity f_R, shape (n,).
        eps:  Small constant added to ‖ẋ‖ to avoid exact division by zero.

    Returns:
        f_R_neutral: Velocity-orthogonal residual, shape (n,).
    """
    xdot = np.asarray(xdot, dtype=float)
    fR   = np.asarray(fR,   dtype=float)

    norm = float(np.linalg.norm(xdot))
    if norm < 1e-6:
        return fR.copy()

    v = xdot / norm                   # exact unit-length direction
    return fR - float(v @ fR) * v    # ẋᵀ f_R_neutral = 0 exactly


# ---------------------------------------------------------------------------
# Phase 6 refinement: two-sided power-band filter
# ---------------------------------------------------------------------------
def filter_with_bounds(
    xdot:      np.ndarray,
    fR_nom:    np.ndarray,
    eps_low:   float = -np.inf,
    eps_high:  float = 0.0,
    delta:     float = 1e-6,
) -> FilterResult:
    """
    Project f_R_nom so that eps_low ≤ ẋᵀ f_R ≤ eps_high.

    This is the two-sided generalisation of filter_residual.  It prevents
    both energy injection (upper bound) AND over-dissipation (lower bound),
    which is the refined contact-force constraint from Phase 6:

        −ε_n ≤ ẋᵀ f_n ≤ +ε_n

    Algorithm:
        1. Compute power_nom = ẋᵀ f_R_nom.
        2. If power_nom > eps_high  → project DOWN to eps_high.
        3. Elif power_nom < eps_low → project UP   to eps_low.
        4. Otherwise               → return unchanged.

    The projection in each case is the minimum-norm correction along ẋ:

        f_R* = f_R_nom − [(power_nom − target) / (‖ẋ‖² + δ)] · ẋ

    Args:
        xdot:     Current joint velocity ẋ, shape (n,).
        fR_nom:   Nominal f_R, shape (n,).
        eps_low:  Lower power bound (−∞ = no lower bound).
        eps_high: Upper power bound (0 = no injection allowed).
        delta:    Regularisation constant.

    Returns:
        FilterResult with the (possibly projected) f_R and diagnostics.
        The ``epsilon`` field stores eps_high (the upper bound used).
    """
    xdot   = np.asarray(xdot,   dtype=float)
    fR_nom = np.asarray(fR_nom, dtype=float)

    power_nom = float(xdot @ fR_nom)
    denom     = float(xdot @ xdot) + delta

    if power_nom > eps_high:
        correction = (power_nom - eps_high) / denom
        fR_star    = fR_nom - correction * xdot
        power_filtered = float(xdot @ fR_star)
        return FilterResult(
            fR_filtered=fR_star,
            power_nom=power_nom,
            power_filtered=power_filtered,
            clipped=True,
            epsilon=eps_high,
        )

    if power_nom < eps_low:
        correction = (power_nom - eps_low) / denom
        fR_star    = fR_nom - correction * xdot
        power_filtered = float(xdot @ fR_star)
        return FilterResult(
            fR_filtered=fR_star,
            power_nom=power_nom,
            power_filtered=power_filtered,
            clipped=True,
            epsilon=eps_high,
        )

    # Within bounds — return unchanged
    return FilterResult(
        fR_filtered=fR_nom.copy(),
        power_nom=power_nom,
        power_filtered=power_nom,
        clipped=False,
        epsilon=eps_high,
    )
