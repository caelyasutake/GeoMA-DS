"""Short-horizon kinematic rollout scorer for Morse escape candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.solver.escape.escape_primitives import EscapeCandidate
from src.solver.ds.path_ds import PathDS


@dataclass
class ScoredEscapeCandidate:
    candidate: EscapeCandidate
    score: float
    valid: bool                          # False if any step collides
    rollout_qs: List[np.ndarray]         # all rollout_horizon steps [q1 .. qH]
    execute_prefix_qs: List[np.ndarray]  # first execute_prefix_len steps
    diagnostics: dict = field(default_factory=dict)


def _joint_limit_cost(q: np.ndarray) -> float:
    """Simple soft joint-limit cost: penalise configs outside [-2.9, 2.9] rad.

    Approximate symmetric bounds — actual per-joint Panda limits vary per joint.
    Used as a soft bias only, not as a hard constraint.
    """
    lower, upper = -2.9, 2.9
    below = np.maximum(0.0, lower - q)
    above = np.maximum(0.0, q - upper)
    return float(np.sum(below ** 2 + above ** 2))


def _score_rollout(
    q0: np.ndarray,
    rollout_qs: List[np.ndarray],
    clearance_fn: Callable[[np.ndarray], float],
    goal_error_fn: Callable[[np.ndarray], float],
    energy_fn: Callable[[np.ndarray], float],
    config,
    collided: bool,
) -> tuple[float, dict]:
    if not rollout_qs:
        return -config.w_collision * float(collided), {}

    q_H = rollout_qs[-1]
    c0 = clearance_fn(q0)
    c_H = clearance_fn(q_H)
    ge0 = goal_error_fn(q0)
    ge_H = goal_error_fn(q_H)
    e0 = energy_fn(q0)
    e_H = energy_fn(q_H)

    goal_progress = ge0 - ge_H
    clearance_gain = c_H - c0
    path_length = sum(
        float(np.linalg.norm(rollout_qs[i + 1] - rollout_qs[i]))
        for i in range(len(rollout_qs) - 1)
    ) if len(rollout_qs) > 1 else 0.0
    jl_cost = _joint_limit_cost(q_H)
    energy_increase = max(0.0, e_H - e0 - config.max_energy_increase_allowed)

    score = (
        config.w_goal_progress    * goal_progress
        + config.w_clearance_gain   * clearance_gain
        + config.w_final_clearance  * c_H
        - config.w_collision        * float(collided)
        - config.w_joint_path_length * path_length
        - config.w_joint_limit_cost * jl_cost
        - config.w_energy_increase  * energy_increase
    )

    diag = {
        "goal_progress": goal_progress,
        "clearance_gain": clearance_gain,
        "final_clearance": c_H,
        "path_length": path_length,
        "joint_limit_cost": jl_cost,
        "energy_increase": max(0.0, e_H - e0 - config.max_energy_increase_allowed),
        "collided": collided,
    }
    return float(score), diag


def _simulate_rollout(
    q0: np.ndarray,
    candidate: EscapeCandidate,
    ds: PathDS,
    clearance_fn: Callable[[np.ndarray], float],
    config,
) -> tuple[List[np.ndarray], bool]:
    """Kinematic rollout. Returns (list of q states, collided)."""
    rollout_qs = []
    q = q0.copy()
    collided = False

    # Step 0→1: escape direction seeds the first step
    q = q + config.escape_step_size * candidate.direction
    collided = clearance_fn(q) < config.collision_clearance_threshold
    rollout_qs.append(q.copy())
    if collided:
        # Pad remaining steps with the collision position so rollout_qs length == horizon
        while len(rollout_qs) < config.rollout_horizon:
            rollout_qs.append(q.copy())
        return rollout_qs, True

    # Steps 1→H: DS guides remaining steps
    for _ in range(1, config.rollout_horizon):
        qdot = ds.f(q)
        q = q + config.rollout_dt * qdot * config.escape_step_size
        rollout_qs.append(q.copy())
        if clearance_fn(q) < config.collision_clearance_threshold:
            collided = True
            # Pad to horizon length
            while len(rollout_qs) < config.rollout_horizon:
                rollout_qs.append(q.copy())
            break

    return rollout_qs, collided


def score_all(
    q: np.ndarray,
    candidates: List[EscapeCandidate],
    ds: PathDS,
    clearance_fn: Callable[[np.ndarray], float],
    goal_error_fn: Callable[[np.ndarray], float],
    energy_fn: Callable[[np.ndarray], float],
    config,
) -> List[ScoredEscapeCandidate]:
    """Score all candidates via kinematic rollout. Returns sorted list (best first)."""
    results: List[ScoredEscapeCandidate] = []

    for cand in candidates:
        rollout_qs, collided = _simulate_rollout(q, cand, ds, clearance_fn, config)
        score, diag = _score_rollout(
            q, rollout_qs, clearance_fn, goal_error_fn, energy_fn, config, collided
        )
        prefix = rollout_qs[:config.execute_prefix_len]
        valid = not collided and score >= config.min_score_to_accept
        results.append(ScoredEscapeCandidate(
            candidate=cand,
            score=score,
            valid=valid,
            rollout_qs=rollout_qs,
            execute_prefix_qs=prefix,
            diagnostics=diag,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
