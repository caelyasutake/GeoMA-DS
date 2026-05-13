"""
Phase 5 — Energy Tank

Enforces passivity of the residual DS component f_R at runtime.

Tank state ODE (Euler-integrated at each control step):

    ṡ = α(s) · ẋᵀ D ẋ  −  βₛ(z, s) · λ · z

Where:
    s               — tank energy (scalar ≥ 0)
    z = ẋᵀ f_R(x)  — passivity metric (positive → f_R injects energy)
    ẋᵀ D ẋ          — power dissipated by damping (always ≥ 0, charges tank)
    α(s)            — charging coefficient; 0 when tank is full
    βₛ(z, s)        — extraction gate; 1 when z > 0 and s > s_min
    λ               — extraction scaling gain

Safety rule:
    β_R = 0   if s ≤ s_min   (disable f_R to protect passivity)
    β_R = 1   otherwise

References:
    Kronander & Billard (2016) "Passive Interaction Control with Dynamical Systems"
    Ferraguti et al. (2013) "A tank-based approach to impedance control"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TankConfig:
    """Parameters for the energy tank."""

    s_init:  float = 1.0    # initial energy level
    s_min:   float = 0.01   # depletion threshold; β_R = 0 at or below this
    s_max:   float = 2.0    # maximum energy (cap); above this α = 0
    lambda_: float = 1.0    # extraction scaling gain

    # Smooth gate: β_R transitions from 0 → 1 over [s_min, s_high].
    # None → auto-computed as s_min + 0.1*(s_max − s_min).
    s_high: float = None    # upper edge of smooth transition region

    # State-dependent passivity filter threshold:
    #   ε(s) = ε_min + (ε_max − ε_min) · β_R(s)
    epsilon_min: float = 0.0   # threshold when tank empty  (strictest)
    epsilon_max: float = 0.0   # threshold when tank full   (most lenient)


# ---------------------------------------------------------------------------
# EnergyTank
# ---------------------------------------------------------------------------
class EnergyTank:
    """
    Scalar energy tank that modulates β_R to guarantee passivity.

    Usage::

        tank = EnergyTank(TankConfig(s_init=1.0))
        beta_R = tank.beta_R          # 0 or 1
        tank.step(z, qdot, D, dt)     # integrate one control step
    """

    def __init__(self, config: TankConfig | None = None) -> None:
        self.config = config or TankConfig()
        self._s = float(self.config.s_init)
        # Cumulative metrics
        self._total_energy_injected: float = 0.0
        self._passivity_violations: int = 0

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    @property
    def energy(self) -> float:
        """Current tank energy s ≥ 0."""
        return self._s

    @property
    def _s_high(self) -> float:
        """Upper edge of the smooth β_R transition region."""
        if self.config.s_high is not None:
            return float(self.config.s_high)
        # Default: lower 10% of the usable range above s_min
        return self.config.s_min + 0.1 * (self.config.s_max - self.config.s_min)

    @property
    def beta_R(self) -> float:
        """
        Smooth passivity gate for the residual DS component.

        β_R(s) = clip( (s − s_min) / (s_high − s_min), 0, 1 )

        Boundary cases:
          s ≤ s_min    → β_R = 0.0  (disable f_R; same as old hard switch)
          s ≥ s_high   → β_R = 1.0  (fully enable f_R)
          s in between → β_R ∈ (0, 1) continuous transition

        The default s_high is s_min + 10 % of (s_max − s_min), so the
        smooth zone is narrow and existing tests that check β_R ∈ {0, 1}
        at typical energies remain valid.
        """
        denom = self._s_high - self.config.s_min
        if denom <= 0.0:
            return 0.0 if self._s <= self.config.s_min else 1.0
        raw = (self._s - self.config.s_min) / denom
        return float(np.clip(raw, 0.0, 1.0))

    @property
    def epsilon(self) -> float:
        """
        State-dependent passivity filter threshold:

            ε(s) = ε_min + (ε_max − ε_min) · β_R(s)

        Returns ε_min when tank is empty (strictest), ε_max when full.
        """
        return float(
            self.config.epsilon_min
            + (self.config.epsilon_max - self.config.epsilon_min) * self.beta_R
        )

    # ------------------------------------------------------------------
    # Tank coefficients
    # ------------------------------------------------------------------
    def alpha(self) -> float:
        """
        Charging coefficient α(s).

        α = 0  when s ≥ s_max  (tank full — stop accepting dissipated energy)
        α = 1  otherwise
        """
        return 0.0 if self._s >= self.config.s_max else 1.0

    def beta_s(self, z: float) -> float:
        """
        Extraction gate βₛ(z, s).

        βₛ = 1  when z > 0 (f_R injects energy) AND s > s_min (tank has energy)
        βₛ = 0  otherwise  (f_R is dissipative, or tank is already depleted)
        """
        if z > 0.0 and self._s > self.config.s_min:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(
        self,
        z: float,
        qdot: np.ndarray,
        D: np.ndarray,
        dt: float,
    ) -> float:
        """
        Integrate the tank ODE one step (Euler).

            ṡ = α(s) · ẋᵀ D ẋ  −  βₛ(z, s) · λ · z

        Args:
            z:     Passivity metric ẋᵀ f_R (scalar).
            qdot:  Current joint velocity ẋ, shape (n_joints,).
            D:     Damping matrix, shape (n_joints, n_joints).
            dt:    Time step (seconds).

        Returns:
            sdot: Instantaneous rate of change of tank energy.
        """
        qdot = np.asarray(qdot, dtype=float)
        D = np.asarray(D, dtype=float)

        # Power dissipated by damping (always non-negative)
        dissipation = float(qdot @ D @ qdot)

        # Tank ODE
        sdot = (
            self.alpha() * dissipation
            - self.beta_s(z) * self.config.lambda_ * float(z)
        )

        # Euler integration with hard bounds
        s_new = float(np.clip(self._s + dt * sdot, 0.0, self.config.s_max))

        # Update cumulative metrics
        if z > 0.0:
            self._passivity_violations += 1
            self._total_energy_injected += float(z) * dt

        self._s = s_new
        return sdot

    # ------------------------------------------------------------------
    # Metrics (for evaluation / CLAUDE.md logging)
    # ------------------------------------------------------------------
    def reset_metrics(self) -> None:
        """Reset cumulative passivity metrics (call at episode start)."""
        self._total_energy_injected = 0.0
        self._passivity_violations = 0

    @property
    def total_energy_injected(self) -> float:
        """∫ max(0, z) dt — cumulative energy injected by f_R."""
        return self._total_energy_injected

    @property
    def passivity_violation_count(self) -> int:
        """Number of steps where z > 0 (f_R injected energy)."""
        return self._passivity_violations

    def reset(self, s: float | None = None) -> None:
        """Reset tank to initial (or specified) energy level."""
        self._s = float(s if s is not None else self.config.s_init)
        self.reset_metrics()
