"""Candidate escape direction generator for the Morse escape planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.evaluation.stall_detection import StallDiagnostics, TrapDiagnostics
from src.solver.ik.goal_selection import IKGoalInfo


@dataclass
class EscapeCandidate:
    direction: np.ndarray   # unit vector in joint space, shape (n_dof,)
    source_tag: str         # "ik_family"|"nullspace"|"tangent"|"clearance_grad"|"random"|"joint_basis"|"negative_curvature"
    metadata: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    """Return normalized vector, or None if near-zero."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _nullspace_project(J: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """Project d into the nullspace of J. Returns unit vector or None."""
    J_pinv = np.linalg.pinv(J)
    N = np.eye(d.shape[0]) - J_pinv @ J
    return _unit(N @ d)


def _ik_family_directions(
    q: np.ndarray,
    ik_infos: List[IKGoalInfo],
    J: np.ndarray,
) -> List[EscapeCandidate]:
    candidates = []
    seen_families = set()
    for info in ik_infos:
        if info.family_label in seen_families:
            continue
        seen_families.add(info.family_label)
        raw = info.q_goal - q
        d = _unit(raw)
        if d is None:
            continue
        candidates.append(EscapeCandidate(
            direction=d,
            source_tag="ik_family",
            metadata={"family_label": info.family_label, "goal_idx": info.goal_idx},
        ))
        # Also add nullspace-projected version
        d_null = _nullspace_project(J, raw)
        if d_null is not None:
            candidates.append(EscapeCandidate(
                direction=d_null,
                source_tag="nullspace",
                metadata={"family_label": info.family_label, "source": "ik_family"},
            ))
    return candidates


def _joint_basis_directions(n_dof: int, J: np.ndarray) -> List[EscapeCandidate]:
    candidates = []
    for i in range(n_dof):
        for sign in (+1.0, -1.0):
            e = np.zeros(n_dof)
            e[i] = sign
            candidates.append(EscapeCandidate(
                direction=e,
                source_tag="joint_basis",
                metadata={"joint_idx": i, "sign": sign},
            ))
            d_null = _nullspace_project(J, e)
            if d_null is not None:
                candidates.append(EscapeCandidate(
                    direction=d_null,
                    source_tag="nullspace",
                    metadata={"joint_idx": i, "source": "joint_basis"},
                ))
    return candidates


def _random_directions(
    n_dof: int,
    n: int,
    J: np.ndarray,
    rng: np.random.Generator,
) -> List[EscapeCandidate]:
    candidates = []
    for _ in range(n):
        raw = rng.standard_normal(n_dof)
        d_null = _nullspace_project(J, raw)
        if d_null is not None:
            candidates.append(EscapeCandidate(
                direction=d_null,
                source_tag="random",
                metadata={},
            ))
        else:
            d = _unit(raw)
            if d is not None:
                candidates.append(EscapeCandidate(direction=d, source_tag="random", metadata={}))
    return candidates


def _clearance_gradient_direction(
    q: np.ndarray,
    clearance_fn: Callable[[np.ndarray], float],
    eps: float = 1e-4,
) -> Optional[EscapeCandidate]:
    n_dof = q.shape[0]
    grad = np.zeros(n_dof)
    c0 = clearance_fn(q)
    for i in range(n_dof):
        q_p = q.copy()
        q_p[i] += eps
        grad[i] = (clearance_fn(q_p) - c0) / eps
    d = _unit(grad)
    if d is None:
        return None
    return EscapeCandidate(direction=d, source_tag="clearance_grad", metadata={})


def _obstacle_tangent_directions(
    q: np.ndarray,
    J: np.ndarray,
    clearance_fn: Callable[[np.ndarray], float],
    eps: float = 1e-4,
) -> List[EscapeCandidate]:
    """Map task-space tangents to obstacle normal into joint space via J_pinv."""
    n_dof = q.shape[0]
    J_pinv = np.linalg.pinv(J)
    grad_task = np.zeros(6)
    c0 = clearance_fn(q)
    for i in range(6):
        dq = (J_pinv @ np.eye(6)[:, i]) * eps
        q_p = q + dq
        grad_task[i] = (clearance_fn(q_p) - c0) / eps

    normal_task = grad_task[:3]
    norm = np.linalg.norm(normal_task)
    if norm < 1e-9:
        return []

    normal_task /= norm
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal_task, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal_task, arbitrary)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal_task, t1)
    tangents_task = [t1, -t1, t2, -t2]

    candidates = []
    for t in tangents_task:
        t6 = np.zeros(6)
        t6[:3] = t
        dq = J_pinv @ t6
        d = _unit(dq)
        if d is not None:
            candidates.append(EscapeCandidate(
                direction=d,
                source_tag="tangent",
                metadata={"tangent_task": t.tolist()},
            ))
    return candidates


def generate_candidates(
    q: np.ndarray,
    stall_diag: StallDiagnostics,
    trap_diag: TrapDiagnostics,
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    clearance_fn: Callable[[np.ndarray], float],
    ik_infos: List[IKGoalInfo],
    config,
    rng: np.random.Generator,
) -> List[EscapeCandidate]:
    """Generate candidate escape directions from multiple sources.

    Args:
        q:           Current joint config (n_dof,).
        stall_diag:  Output of detect_stall().
        trap_diag:   Output of detect_trap().
        jacobian_fn: Returns (6, n_dof) Jacobian at q.
        clearance_fn: Returns scalar clearance at q.
        ik_infos:    Available IK goal metadata (for family directions).
        config:      MorseEscapeConfig instance (or duck-typed config).
        rng:         Seeded random generator.

    Returns:
        List of EscapeCandidate (all unit vectors), capped at config.max_candidates.
    """
    J = jacobian_fn(q)
    n_dof = q.shape[0]
    candidates: List[EscapeCandidate] = []

    if config.use_ik_family_dirs and ik_infos:
        candidates.extend(_ik_family_directions(q, ik_infos, J))

    if config.use_joint_basis_dirs:
        candidates.extend(_joint_basis_directions(n_dof, J))

    if config.num_random_dirs > 0:
        candidates.extend(_random_directions(n_dof, config.num_random_dirs, J, rng))

    if config.use_clearance_gradient:
        cg = _clearance_gradient_direction(q, clearance_fn)
        if cg is not None:
            candidates.append(cg)

    if config.use_obstacle_tangents:
        candidates.extend(_obstacle_tangent_directions(q, J, clearance_fn))

    if getattr(config, "negative_curvature", None) and config.negative_curvature.enabled:
        from src.solver.escape.negative_curvature import (
            lanczos_min_eigenvector, make_hvp
        )
        hvp_fn = make_hvp(clearance_fn)
        direction, curvature = lanczos_min_eigenvector(
            q, hvp_fn, n_dof,
            m=config.negative_curvature.n_lanczos_iters,
            rng=rng,
        )
        if direction is not None:
            candidates.append(EscapeCandidate(
                direction=direction,
                source_tag="negative_curvature",
                metadata={"curvature": curvature},
            ))

    return candidates[:config.max_candidates]
