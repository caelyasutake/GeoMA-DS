"""
Reusable scenario builders for benchmarks and integration tests.

Each builder returns a dict of named components that callers can use
without knowing the implementation details.

Scenarios
---------
free_space      — no obstacles; tests basic pipeline end-to-end
narrow_passage  — joint-space corridor; tests multi-IK advantage
contact_task    — obstacle in workspace; tests contact force sensing
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical robot state
# ---------------------------------------------------------------------------
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Panda ready pose
N_JOINTS = 7

# ---------------------------------------------------------------------------
# Synthetic IK solution sets  (hand-crafted; no HJCD-IK GPU required)
# ---------------------------------------------------------------------------
# All solutions are within Panda joint limits and represent distinct arm poses.

_IK_GOALS_FREE: List[np.ndarray] = [
    np.array([ 0.50, -0.60,  0.30, -2.00,  0.20,  1.80,  0.60]),
    np.array([-0.30, -0.80, -0.20, -2.20, -0.10,  1.50,  0.90]),
    np.array([ 0.80, -0.40,  0.50, -1.80,  0.40,  2.00,  0.40]),
    np.array([-0.50, -0.70,  0.10, -2.50,  0.10,  1.40,  1.00]),
    np.array([ 0.20, -1.00,  0.40, -1.60,  0.30,  1.90,  0.70]),
]

# Goal that sits in the "safe" half of the corridor (q[0] < 1.0)
_GOAL_EASY = np.array([ 0.40, -0.90, 0.10, -2.10, 0.05, 1.60, 0.80])

# Goal across the blocked corridor (q[0] > 3.0);
# reachable physically but the straight path is blocked
_GOAL_HARD = np.array([ 3.50, -0.80, 0.20, -2.00, 0.10, 1.70, 0.70])

# For contact task: goal near the obstacle
_GOAL_CONTACT = np.array([ 0.60, -0.50,  0.40, -2.10,  0.20,  1.85,  0.65])

# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------
def make_free_space_fn() -> Callable[[np.ndarray], bool]:
    """Collision checker that always returns True (no obstacles)."""
    return lambda q: True


def make_corridor_fn(
    joint_idx: int = 0,
    lo: float = 1.0,
    hi: float = 3.0,
) -> Callable[[np.ndarray], bool]:
    """
    Joint-space corridor collision checker.

    Returns False (collision) when ``q[joint_idx] ∈ (lo, hi)``,
    otherwise True (free). Creates a narrow passage that can only be
    crossed by going to an alternative goal on the same side as the start.
    """
    def fn(q: np.ndarray) -> bool:
        return not (lo < float(q[joint_idx]) < hi)
    return fn


def make_simenv_collision_fn(obstacles: Optional[Dict] = None) -> Callable[[np.ndarray], bool]:
    """
    MuJoCo-backed collision checker from SimEnv.

    Lazily imports SimEnv so the rest of the module is importable
    without a MuJoCo installation.
    """
    from src.simulation.env import SimEnv, SimEnvConfig
    cfg = SimEnvConfig(obstacles=obstacles or {})
    env = SimEnv(cfg)
    return env.make_collision_fn()


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
def free_space_scenario() -> Dict:
    """
    Scenario 1 — Free Space.

    Components
    ----------
    q_start : np.ndarray
    Q_goals : list of np.ndarray   (5 diverse goal configurations)
    collision_fn : always True
    """
    return {
        "name":         "free_space",
        "q_start":      Q_READY.copy(),
        "Q_goals":      [g.copy() for g in _IK_GOALS_FREE],
        "collision_fn": make_free_space_fn(),
        "obstacles":    {},
    }


def narrow_passage_scenario(use_simenv: bool = False) -> Dict:
    """
    Scenario 2 — Narrow Passage.

    The corridor blocks joint 0 ∈ (1.0, 3.0).

    From q_start (joint0 = 0.0):
      * Goal_easy (joint0 = 0.40): reachable without entering the corridor.
      * Goal_hard (joint0 = 3.50): requires crossing the blocked corridor.

    Components
    ----------
    q_start         : np.ndarray
    Q_goals_multi   : [Goal_easy, Goal_hard]  — multi-IK set
    Q_goals_single  : [Goal_hard]             — single-IK set (hard goal only)
    collision_fn    : corridor checker (joint 0 ∈ (1.0, 3.0) → blocked)
    """
    col_fn = make_corridor_fn(joint_idx=0, lo=1.0, hi=3.0)
    return {
        "name":            "narrow_passage",
        "q_start":         Q_READY.copy(),
        "Q_goals_multi":   [_GOAL_EASY.copy(), _GOAL_HARD.copy()],
        "Q_goals_single":  [_GOAL_HARD.copy()],
        "collision_fn":    col_fn,
        "obstacles":       {},
    }


def contact_task_scenario() -> Dict:
    """
    Scenario 3 — Contact Task.

    A box obstacle sits in the robot's workspace.  The controller drives
    the arm toward a goal configuration that causes a link to contact
    the obstacle, allowing get_contact_forces() to return non-empty.

    Components
    ----------
    q_start   : np.ndarray
    Q_goals   : list containing one contact goal
    obstacles : dict (HJCD-IK format)
    """
    # Box placed at (0.25, 0, 0.60) — intersects arm in GOAL_CONTACT pose
    obstacles = {
        "cuboid": {
            "contact_box": {
                "dims": [0.20, 0.40, 0.20],
                "pose": [0.25, 0.0, 0.60, 1, 0, 0, 0],
            }
        }
    }
    return {
        "name":      "contact_task",
        "q_start":   Q_READY.copy(),
        "Q_goals":   [_GOAL_CONTACT.copy()],
        "obstacles": obstacles,
    }
