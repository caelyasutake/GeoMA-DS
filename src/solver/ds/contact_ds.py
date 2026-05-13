"""
Contact DS — Task-Space Circle Reference

Defines the desired end-effector trajectory for a contact task in which the
EE traces a circle on a planar surface (e.g. a table top).

The reference is specified entirely in **Cartesian task space** — not derived
from joint-space interpolation.

Primary entry point
-------------------
circle_on_plane_reference(t, cfg) -> CircleReference

The returned CircleReference provides:
    x_d(t)     — desired position  (3,)
    xdot_d(t)  — desired velocity  (3,)
    tangent(t) — unit tangent direction (derivative direction) (3,)
    normal     — unit surface normal (constant) (3,)
    phase      — current angle in radians

Circle equation
---------------
    x_d(t) = center + r·cos(ωt)·e1 + r·sin(ωt)·e2

where e1, e2 are orthonormal vectors spanning the contact plane
(automatically derived from the surface normal if not supplied).

    xdot_d(t) = r·ω·(−sin(ωt)·e1 + cos(ωt)·e2)

The normal component of xdot_d is zeroed so the reference implies no
motion into or away from the surface — contact regulation is handled
separately by the task_tracking controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CircleContactConfig:
    """
    Parameters that fully define a circular contact trajectory on a plane.

    Attributes:
        center:    (3,) world-frame centre of the circle.
        radius:    Circle radius (metres).
        omega:     Angular velocity (rad/s) — positive = counter-clockwise
                   when viewed from the direction of ``normal``.
        normal:    (3,) unit normal to the contact surface (e.g. [0,0,1]
                   for a horizontal table).  Will be normalised internally.
        z_contact: Contact height — the z-coordinate (world frame) at which
                   the EE maintains contact.  Forces x_d[2] = z_contact.
        e1:        Optional (3,) first in-plane axis (tangent direction at
                   t=0).  If None, derived automatically from ``normal``.
        e2:        Optional (3,) second in-plane axis.  If None, derived
                   from ``normal × e1``.
    """

    center:    np.ndarray            # (3,)
    radius:    float
    omega:     float                 # rad/s
    normal:    np.ndarray            # (3,)
    z_contact: float
    e1:        Optional[np.ndarray] = None
    e2:        Optional[np.ndarray] = None

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        if self.e1 is not None:
            self.e1 = np.asarray(self.e1, dtype=float)
        if self.e2 is not None:
            self.e2 = np.asarray(self.e2, dtype=float)


# ---------------------------------------------------------------------------
# Reference output
# ---------------------------------------------------------------------------
@dataclass
class CircleReference:
    """
    Task-space reference at a single instant in time.

    Attributes:
        x_d:     Desired EE position (3,).
        xdot_d:  Desired EE velocity (3,) — tangential only (no normal component).
        tangent: Unit tangent direction along the circle (3,).
        normal:  Unit surface normal (3,) — constant over trajectory.
        phase:   Current angle ωt (radians).
    """

    x_d:     np.ndarray   # (3,)
    xdot_d:  np.ndarray   # (3,)
    tangent: np.ndarray   # (3,)
    normal:  np.ndarray   # (3,)
    phase:   float        # rad


# ---------------------------------------------------------------------------
# In-plane orthonormal frame construction
# ---------------------------------------------------------------------------
def _build_plane_frame(
    normal: np.ndarray,
    e1: Optional[np.ndarray],
    e2: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (e1, e2) — an orthonormal basis for the plane perpendicular to n.

    If e1/e2 are provided they are normalised and returned as-is.
    Otherwise e1 is chosen orthogonal to n using a numerically stable
    Gram–Schmidt step, and e2 = n × e1.

    The resulting basis satisfies:
        e1 ⊥ n,  e2 ⊥ n,  e1 ⊥ e2,  ||e1|| = ||e2|| = 1

    Args:
        normal: Unit normal to the plane (normalised on entry).
        e1:     Optional first axis hint.
        e2:     Optional second axis hint.

    Returns:
        (e1, e2) orthonormal pair.
    """
    n = normal / np.linalg.norm(normal)

    if e1 is not None and e2 is not None:
        return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)

    # Pick an arbitrary vector not parallel to n
    arb = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(arb, n))) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt: project out n component
    arb_ortho = arb - float(np.dot(arb, n)) * n
    e1_new = arb_ortho / np.linalg.norm(arb_ortho)

    # e2 completes the right-handed frame
    e2_new = np.cross(n, e1_new)
    e2_new = e2_new / np.linalg.norm(e2_new)

    return e1_new, e2_new


# ---------------------------------------------------------------------------
# Primary reference function
# ---------------------------------------------------------------------------
def circle_on_plane_reference(
    t: float,
    cfg: CircleContactConfig,
) -> CircleReference:
    """
    Compute the task-space circle reference at time ``t``.

    Position:
        x_d(t) = center + r·cos(ωt)·e1 + r·sin(ωt)·e2
        x_d[2] overridden to z_contact (contact height)

    Velocity:
        xdot_d(t) = r·ω·(−sin(ωt)·e1 + cos(ωt)·e2)
        Normal component projected out → pure tangential motion

    Tangent:
        tangent = xdot_d / ||xdot_d||

    Args:
        t:   Current time (seconds from start of WALL_SLIDE phase).
        cfg: CircleContactConfig defining the trajectory.

    Returns:
        CircleReference with x_d, xdot_d, tangent, normal, phase.

    Guarantees:
        tangent · normal = 0   (tangent lies in the contact plane)
        xdot_d · normal = 0    (no desired motion into/away from surface)
        |x_d_xy - center_xy| ≈ radius  (on circle when projected)
        x_d[2] = z_contact             (contact height enforced)
    """
    n = cfg.normal / np.linalg.norm(cfg.normal)
    e1, e2 = _build_plane_frame(n, cfg.e1, cfg.e2)

    angle = float(cfg.omega) * float(t)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Desired position on circle
    x_d = cfg.center + cfg.radius * (cos_a * e1 + sin_a * e2)
    # Enforce contact height (replaces whatever z the circle formula gives)
    x_d = x_d.copy()
    x_d[2] = cfg.z_contact

    # Desired velocity (tangential direction of circle)
    xdot_d_raw = cfg.radius * cfg.omega * (-sin_a * e1 + cos_a * e2)
    # Remove any residual normal component (purely tangential)
    xdot_d = xdot_d_raw - float(np.dot(xdot_d_raw, n)) * n

    # Unit tangent direction
    speed = float(np.linalg.norm(xdot_d))
    tangent = xdot_d / (speed + 1e-9)

    return CircleReference(
        x_d=x_d,
        xdot_d=xdot_d,
        tangent=tangent,
        normal=n,
        phase=angle,
    )
