"""Approximate negative-curvature directions via Lanczos / power iteration.

Gated by NegativeCurvatureConfig.enabled — never called unless explicitly enabled.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def make_hvp(
    clearance_fn: Optional[Callable[[np.ndarray], float]],
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eps: float = 1e-4,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a Hessian-vector product function via finite differences of grad_fn.

    If grad_fn is not provided, builds a clearance-based energy gradient
    (energy decreases as clearance increases — so gradient points toward obstacles).
    """
    if grad_fn is None:
        def grad_fn(q: np.ndarray) -> np.ndarray:
            n = q.shape[0]
            g = np.zeros(n)
            c0 = clearance_fn(q) if clearance_fn else 0.0
            for i in range(n):
                qp = q.copy()
                qp[i] += eps
                cp = clearance_fn(qp) if clearance_fn else 0.0
                # Negative: energy increases as clearance decreases
                g[i] = -(cp - c0) / eps
            return g

    def hvp(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return np.zeros_like(v)
        v_unit = v / v_norm
        g_plus = grad_fn(q + eps * v_unit)
        g_minus = grad_fn(q - eps * v_unit)
        return (g_plus - g_minus) / (2.0 * eps)

    return hvp


def lanczos_min_eigenvector(
    q: np.ndarray,
    hvp_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dim: int,
    m: int = 12,
    rng: Optional[np.random.Generator] = None,
    curvature_threshold: float = -0.01,
) -> Tuple[Optional[np.ndarray], float]:
    """Lanczos iteration to approximate the minimum eigenvalue direction.

    Returns (direction, curvature). direction is None if curvature >= threshold.
    """
    if rng is None:
        rng = np.random.default_rng()

    qv = rng.standard_normal(dim)
    norm = np.linalg.norm(qv)
    if norm < 1e-12:
        return None, 0.0
    qv = qv / norm

    Q = []
    alphas = []
    betas = []

    q_prev = np.zeros(dim)
    beta = 0.0

    for j in range(min(m, dim)):
        Q.append(qv.copy())
        z = hvp_fn(q, qv)
        alpha = float(np.dot(qv, z))
        alphas.append(alpha)

        z = z - alpha * qv - beta * q_prev
        beta = float(np.linalg.norm(z))
        betas.append(beta)

        if beta < 1e-10:
            break

        q_prev = qv.copy()
        qv = z / beta

    # Build tridiagonal matrix and find eigenvalues.
    # betas[:k-1] are the k-1 off-diagonal entries (beta_j couples Q[:,j] and Q[:,j+1]).
    # The last appended beta is unused because the Lanczos iteration stopped after k steps.
    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:k - 1], 1) + np.diag(betas[:k - 1], -1)
    eigvals, eigvecs = np.linalg.eigh(T)

    min_idx = int(np.argmin(eigvals))
    min_eigval = float(eigvals[min_idx])
    min_eigvec_T = eigvecs[:, min_idx]  # shape (k,)

    # Map back to original space: direction = Q @ min_eigvec_T
    Q_mat = np.column_stack(Q)          # shape (dim, k)
    direction_full = Q_mat @ min_eigvec_T
    norm = np.linalg.norm(direction_full)
    if norm < 1e-9:
        return None, min_eigval

    direction_full /= norm

    if min_eigval >= curvature_threshold:
        return None, min_eigval

    return direction_full, min_eigval


def negative_curvature_power_iteration(
    q: np.ndarray,
    hvp_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dim: int,
    n_iters: int = 20,
    rng: Optional[np.random.Generator] = None,
    curvature_threshold: float = -0.01,
) -> Tuple[Optional[np.ndarray], float]:
    """Power iteration on -H to find most-negative curvature direction."""
    if rng is None:
        rng = np.random.default_rng()

    v = rng.standard_normal(dim)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None, 0.0
    v = v / norm

    # Track whether at least one iterate updated v; if norm collapses on the
    # first step, the Hessian is near-zero and v remains the random seed.
    updated = False
    for _ in range(n_iters):
        Hv = hvp_fn(q, v)
        v_next = -Hv  # power iteration on -H → finds most-negative mode
        norm = np.linalg.norm(v_next)
        if norm < 1e-8:
            break
        v = v_next / norm
        updated = True

    if not updated:
        return None, 0.0

    curvature = float(np.dot(v, hvp_fn(q, v)))
    if curvature >= curvature_threshold:
        return None, curvature
    return v.copy(), curvature
