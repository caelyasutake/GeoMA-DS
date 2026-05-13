"""
Hard non-contact safety shield — step acceptance / line-search filter.

For every control step, predicts the next joint configuration under the
proposed command and verifies that all obstacle clearances remain above
d_hard_min.  If the full command is unsafe, repeatedly scales it down
(binary backtrack) until a safe step is found.  If no non-zero command
is safe, outputs zero velocity so the robot holds position.

This is the final safety layer applied AFTER modulation and BEFORE
simulation.  It guarantees:

    min_clearance(q + dt * qdot_out) >= d_hard_min

for every executed step, regardless of upstream controller output.

Usage
-----
    from src.solver.controller.hard_shield import HardShieldConfig, enforce_hard_clearance

    cfg    = HardShieldConfig(enabled=True, d_hard_min=0.01)
    qdot_s, diag = enforce_hard_clearance(q, qdot_cmd, dt, obstacles, cfg)

Intended solver conditions
--------------------------
- vanilla_ds_modulation   : may stall/freeze rather than contact
- single_ik_best__ds_modulation : same
- (optional) multi_ik_full__ds_modulation : softer requirement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.solver.planner.collision import _panda_link_positions, _LINK_RADII
from src.scenarios.scenario_schema import Obstacle


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class HardShieldConfig:
    """
    Parameters for the hard non-contact safety shield.

    Attributes
    ----------
    enabled:
        Master switch.  When False, this layer is a no-op.
    d_hard_min:
        Minimum required clearance at the predicted next state (metres).
        Any step that would bring any link sphere closer than this to any
        obstacle is rejected or scaled down.  Default 0.01 m (1 cm).
    n_backtrack:
        Number of backtrack steps in the binary line search.  The command
        magnitude is halved at each step, so after n steps the minimum
        accepted scale is 2^(-n).  Default 12 → minimum scale ≈ 0.0002.
    check_all_links:
        When True, clearance is checked for all link spheres (same set as
        CBF).  When False, only the EE link (panda_hand) is checked.
        True is safer but more expensive.
    """
    enabled:        bool  = False
    d_hard_min:     float = 0.01
    n_backtrack:    int   = 12
    check_all_links: bool = True


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class HardShieldDiagnostics:
    """Per-step output from the hard non-contact safety shield."""
    shield_triggered:   bool  = False   # True when any backtracking occurred
    forced_stop:        bool  = False   # True when even alpha=0 was needed
    accepted_scale:     float = 1.0     # alpha that was accepted (0 = full stop)
    predicted_clearance: float = float("inf")  # clearance at accepted q_next
    n_backtracks:       int   = 0       # number of halvings before acceptance


# ---------------------------------------------------------------------------
# Internal: minimum clearance at a predicted configuration
# ---------------------------------------------------------------------------
def _min_clearance_at_q(
    q: np.ndarray,
    obstacles: List[Obstacle],
    check_all_links: bool = True,
) -> float:
    """
    Compute the minimum clearance between any link sphere and any obstacle
    at configuration q.

    Uses the same FK and sphere model as the CBF filter so the two layers
    are consistent.
    """
    from src.solver.controller.cbf_filter import _clearance

    link_pos = _panda_link_positions(q)
    min_d    = float("inf")

    link_range = range(len(link_pos)) if check_all_links else [len(link_pos) - 1]

    for link_idx in link_range:
        pos    = link_pos[link_idx]
        radius = _LINK_RADII.get(link_idx, 0.08)
        for obs in obstacles:
            if not obs.collision_enabled:
                continue
            d = _clearance(pos, radius, obs)
            if d < min_d:
                min_d = d

    return min_d if min_d < float("inf") else float("inf")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def enforce_hard_clearance(
    q:         np.ndarray,
    qdot_cmd:  np.ndarray,
    dt:        float,
    obstacles: List[Obstacle],
    config:    HardShieldConfig,
) -> tuple[np.ndarray, HardShieldDiagnostics]:
    """
    Apply the hard non-contact safety shield.

    Algorithm
    ---------
    1. Try alpha = 1.0: predict q_next = q + dt * qdot_cmd, check clearance.
    2. If unsafe, halve alpha and retry (up to n_backtrack times).
    3. Accept the largest alpha that satisfies clearance >= d_hard_min.
    4. If no alpha > 0 is safe, return zero velocity.

    Parameters
    ----------
    q         : Current joint configuration (7,).
    qdot_cmd  : Proposed joint velocity command (7,).
    dt        : Control timestep (seconds).
    obstacles : Collision-enabled obstacle list.
    config    : HardShieldConfig.

    Returns
    -------
    qdot_safe : Safe joint velocity (7,), may be zero.
    diag      : HardShieldDiagnostics.
    """
    q        = np.asarray(q,       dtype=float)
    qdot_cmd = np.asarray(qdot_cmd, dtype=float)
    diag     = HardShieldDiagnostics()

    if not config.enabled:
        return qdot_cmd.copy(), diag

    col_obs = [o for o in obstacles if o.collision_enabled]
    if not col_obs:
        return qdot_cmd.copy(), diag

    d_min = config.d_hard_min

    # Binary backtrack: alpha = 1, 0.5, 0.25, …, 2^(-n_backtrack)
    for k in range(config.n_backtrack + 1):
        alpha   = 0.5 ** k
        q_next  = q + dt * alpha * qdot_cmd
        cl      = _min_clearance_at_q(q_next, col_obs, config.check_all_links)

        if cl >= d_min:
            diag.accepted_scale      = alpha
            diag.predicted_clearance = cl
            diag.n_backtracks        = k
            diag.shield_triggered    = (k > 0)
            return alpha * qdot_cmd, diag

    # No safe non-zero command found — hold position
    diag.forced_stop        = True
    diag.shield_triggered   = True
    diag.accepted_scale     = 0.0
    diag.n_backtracks       = config.n_backtrack
    # Clearance at current q (not predicted)
    diag.predicted_clearance = _min_clearance_at_q(q, col_obs, config.check_all_links)

    return np.zeros_like(qdot_cmd), diag
