"""
Scenario-driven collision checker for the BiRRT planner.

Converts a ``ScenarioSpec`` (or raw obstacle dict) into a callable
``(q: np.ndarray) -> bool`` that returns True when the configuration is
collision-free.

Approach — geometric sphere-swept link approximation
-----------------------------------------------------
Each Panda link is approximated by a sphere centred at the link's FK
position.  For each query configuration the checker computes FK for every
link and tests each link sphere against every obstacle primitive.

This is fast enough for the RRT inner loop and consistent with the same
obstacle list used in the HJCD-IK JSON and MuJoCo scene.

For obstacle types:
  box      — AABB check: sphere centre is within (half-extent + radius) of box
  sphere   — sphere-sphere distance check
  cylinder — capsule approximation (axis-aligned cylinder along Z)

Fallback
--------
If no obstacles are specified the returned function always returns True
(equivalent to free space).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.scenarios.scenario_schema import Obstacle, ScenarioSpec

# ---------------------------------------------------------------------------
# Panda link sphere approximation
# Link positions are computed from MJCF forward kinematics that match the
# MuJoCo world frame used by the scene builder and visualizer.
# Radii are conservative (outer bound of each link body).
# ---------------------------------------------------------------------------
_LINK_RADII = {
    0: 0.10,   # link1 shoulder
    1: 0.10,   # link2
    2: 0.09,   # link3
    3: 0.09,   # link4
    4: 0.08,   # link5
    5: 0.07,   # link6
    6: 0.06,   # link7
    7: 0.06,   # hand (EE body, 0.107 m past joint7)
}

# Body local transforms from mujoco/franka_emika_panda/panda.xml:
#   each entry is (local_pos, local_quat_wxyz)
# All joints rotate around their body's local z-axis (axis="0 0 1").
_MJCF_BODIES = [
    ([0.0,     0.0,   0.333], [1.0,  0.0, 0.0, 0.0]),  # link1
    ([0.0,     0.0,   0.0  ], [1.0, -1.0, 0.0, 0.0]),  # link2
    ([0.0,    -0.316, 0.0  ], [1.0,  1.0, 0.0, 0.0]),  # link3
    ([0.0825,  0.0,   0.0  ], [1.0,  1.0, 0.0, 0.0]),  # link4
    ([-0.0825, 0.384, 0.0  ], [1.0, -1.0, 0.0, 0.0]),  # link5
    ([0.0,     0.0,   0.0  ], [1.0,  1.0, 0.0, 0.0]),  # link6
    ([0.088,   0.0,   0.0  ], [1.0,  1.0, 0.0, 0.0]),  # link7
]


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix (normalises input)."""
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)     ],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)     ],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y) ],
    ])


def _panda_link_positions(q: np.ndarray) -> List[np.ndarray]:
    """
    Return list of 8 body origins in world frame (links 1-7 plus hand).

    Uses the MJCF body hierarchy from ``mujoco/franka_emika_panda/panda.xml``
    so positions match MuJoCo world frame exactly.  Index 7 is the hand body
    (0.107 m along link7's post-joint z-axis), which MuJoCo reports as the EE.

    Args:
        q: 7-DOF joint angles (rad).

    Returns:
        List of 8 position vectors [x, y, z] in world frame.
        positions[-1] == hand body == true end-effector position.
    """
    T = np.eye(4)
    positions = []
    for i, ((lx, ly, lz), (qw, qx, qy, qz)) in enumerate(_MJCF_BODIES):
        # Body local transform (pos + orientation from MJCF)
        R_body = _quat_to_rot(qw, qx, qy, qz)
        T_body = np.eye(4)
        T_body[:3, :3] = R_body
        T_body[:3, 3] = [lx, ly, lz]

        # Joint rotation around local z-axis
        c, s = np.cos(float(q[i])), np.sin(float(q[i]))
        T_joint = np.eye(4)
        T_joint[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        T = T @ T_body @ T_joint
        positions.append(T[:3, 3].copy())

    # Hand body: pos=[0,0,0.107] in link7's post-joint frame (panda.xml)
    hand_pos = T[:3, :3] @ np.array([0.0, 0.0, 0.107]) + T[:3, 3]
    positions.append(hand_pos)
    return positions


def _panda_hand_transform(q: np.ndarray):
    """
    Return (hand_pos, R_hand) for the hand body at joint config q.

    hand_pos : world position of the hand body origin (same as
               _panda_link_positions(q)[7]).
    R_hand   : 3x3 world rotation matrix of the hand body frame.

    Use this to compute the panda_grasptarget IK site:
        grasptarget = hand_pos + R_hand @ [0, 0, 0.105]
    """
    T = np.eye(4)
    for i, ((lx, ly, lz), (qw, qx, qy, qz)) in enumerate(_MJCF_BODIES):
        R_body = _quat_to_rot(qw, qx, qy, qz)
        T_body = np.eye(4)
        T_body[:3, :3] = R_body
        T_body[:3, 3] = [lx, ly, lz]
        c, s = np.cos(float(q[i])), np.sin(float(q[i]))
        T_joint = np.eye(4)
        T_joint[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = T @ T_body @ T_joint
    R_hand   = T[:3, :3].copy()
    hand_pos = R_hand @ np.array([0.0, 0.0, 0.107]) + T[:3, 3]
    return hand_pos, R_hand


def _panda_fk_batch(qs: np.ndarray) -> np.ndarray:
    """
    Vectorised forward kinematics for N joint configurations.

    Computes the same link positions as N calls to _panda_link_positions but
    in a single NumPy kernel, avoiding per-config Python loop overhead.

    Args:
        qs: (N, 7) joint angles in radians.

    Returns:
        (N, 8, 3) link positions in world frame (links 1-7 plus hand body).
    """
    N = qs.shape[0]
    c = np.cos(qs)   # (N, 7)
    s = np.sin(qs)   # (N, 7)

    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()   # (N, 3, 3)
    t = np.zeros((N, 3))
    positions = np.empty((N, 8, 3))

    for i in range(7):
        t_body = _BODY_T_CACHE[i]   # (3,) constant
        R_body = _BODY_R_CACHE[i]   # (3, 3) constant

        # t_new = R_old @ t_body + t_old  (translation update uses current world R)
        t = np.matmul(R, t_body) + t          # (N, 3)

        # R_new = R_old @ R_body @ Rz(q[:, i])
        R = np.matmul(R, R_body)              # (N, 3, 3)

        # Apply joint rotation Rz around local Z in-place
        R0 = R[:, :, 0].copy()
        R1 = R[:, :, 1].copy()
        ci = c[:, i, np.newaxis]              # (N, 1)
        si = s[:, i, np.newaxis]
        R[:, :, 0] =  R0 * ci + R1 * si
        R[:, :, 1] = -R0 * si + R1 * ci
        # column 2 unchanged (Rz leaves Z-axis fixed)

        positions[:, i] = t

    # Hand body offset [0, 0, 0.107] in post-joint-7 frame
    positions[:, 7] = np.matmul(R, np.array([0.0, 0.0, 0.107])) + t

    return positions


# Precomputed body transforms and link radii array — populated after
# _quat_to_rot is defined above.  Used by _panda_fk_batch at call time.
_BODY_R_CACHE: List[np.ndarray] = [
    _quat_to_rot(*quat) for _, quat in _MJCF_BODIES
]
_BODY_T_CACHE: List[np.ndarray] = [
    np.array(pos, dtype=float) for pos, _ in _MJCF_BODIES
]
_LINK_RADII_ARRAY: np.ndarray = np.array(
    [_LINK_RADII.get(i, 0.08) for i in range(8)], dtype=float
)


# ---------------------------------------------------------------------------
# Vectorised batch clearance (used by GeoMultiAttractorDS fast path)
# ---------------------------------------------------------------------------

def _batch_clearance_from_lp(
    lp: np.ndarray,
    obs_list: list,
    obs_R_inv_list: list,
) -> np.ndarray:
    """
    Vectorised minimum signed clearance for N configurations.

    Avoids per-configuration Python loops by operating on pre-computed link
    positions using NumPy broadcasting.  Intended for use with _panda_fk_batch
    to compute clearance gradients and attractor scores in one batch FK call.

    Args:
        lp:             (N, 8, 3) link sphere centres in world frame, as
                        returned by _panda_fk_batch().
        obs_list:       List of Obstacle objects (collision_enabled already
                        filtered by the factory).
        obs_R_inv_list: Per-obstacle R.T matrices (None for AABB).  Produced
                        by _precompute_obs_rotations().

    Returns:
        (N,) minimum signed clearances across all links and obstacles.
        Positive = free; negative = penetrating.
    """
    N = lp.shape[0]
    min_cl = np.full(N, np.inf)

    for j, obs in enumerate(obs_list):
        R_inv   = obs_R_inv_list[j]
        obs_pos = np.asarray(obs.position, dtype=float)
        t       = obs.type.lower()

        if t == "box":
            half = np.asarray(obs.size, dtype=float)   # half-extents
            if R_inv is not None:
                # OBB: project centres into obstacle-local frame.
                # Equivalent to  R_inv @ (centre - obs_pos)  for each centre.
                local = (lp - obs_pos) @ R_inv.T        # (N, 8, 3)
            else:
                local = lp - obs_pos                    # (N, 8, 3)
            closest = np.clip(local, -half, half)        # (N, 8, 3)
            dist    = np.linalg.norm(local - closest, axis=2)  # (N, 8)
            cl      = dist - _LINK_RADII_ARRAY           # (N, 8) via broadcast
            min_cl  = np.minimum(min_cl, cl.min(axis=1))

        elif t == "sphere":
            obs_r  = float(obs.size[0])
            dist   = np.linalg.norm(lp - obs_pos, axis=2)      # (N, 8)
            cl     = dist - _LINK_RADII_ARRAY - obs_r
            min_cl = np.minimum(min_cl, cl.min(axis=1))

        else:
            # Cylinder and unknown types: scalar fallback (rare in practice).
            for k in range(N):
                for i in range(8):
                    cl_ki = _scalar_signed_dist(
                        lp[k, i], _LINK_RADII_ARRAY[i], obs, R_inv
                    )
                    if cl_ki < min_cl[k]:
                        min_cl[k] = cl_ki

    return min_cl


def _scalar_signed_dist(
    centre: np.ndarray,
    link_radius: float,
    obs,
    R_inv: Optional[np.ndarray],
) -> float:
    """Signed clearance from a single link sphere to one obstacle (scalar)."""
    obs_pos = np.asarray(obs.position, dtype=float)
    t = obs.type.lower()
    if t == "box":
        half = np.asarray(obs.size, dtype=float)
        return _sphere_box_signed_dist(centre, link_radius, obs_pos, half, R_inv)
    elif t == "sphere":
        obs_r = float(obs.size[0])
        return float(np.linalg.norm(centre - obs_pos)) - link_radius - obs_r
    elif t == "cylinder":
        obs_r = float(obs.size[0])
        obs_h = float(obs.size[1])
        dx = float(centre[0]) - obs_pos[0]
        dy = float(centre[1]) - obs_pos[1]
        dz = float(centre[2]) - obs_pos[2]
        r_xy = np.sqrt(dx * dx + dy * dy)
        cz   = float(np.clip(dz, -obs_h, obs_h))
        if r_xy < 1e-9:
            closest = np.array([obs_pos[0], obs_pos[1], obs_pos[2] + cz])
        elif r_xy <= obs_r:
            closest = np.array([obs_pos[0] + dx, obs_pos[1] + dy, obs_pos[2] + cz])
        else:
            s = obs_r / r_xy
            closest = np.array([obs_pos[0] + dx * s, obs_pos[1] + dy * s, obs_pos[2] + cz])
        return float(np.linalg.norm(centre - closest)) - link_radius
    return float("inf")


# ---------------------------------------------------------------------------
# Obstacle geometry tests
# ---------------------------------------------------------------------------
def _sphere_vs_box(
    centre: np.ndarray,
    radius: float,
    box_pos: np.ndarray,
    box_half: np.ndarray,
) -> bool:
    """Return True if sphere overlaps axis-aligned box (world frame)."""
    closest = np.clip(centre, box_pos - box_half, box_pos + box_half)
    dist_sq = float(np.sum((centre - closest) ** 2))
    return dist_sq < radius * radius


def _sphere_box_signed_dist(
    centre: np.ndarray,
    radius: float,
    box_pos: np.ndarray,
    box_half: np.ndarray,
    R_inv: Optional[np.ndarray] = None,
) -> float:
    """
    Signed distance from a sphere to an oriented box.

    Positive = free (no overlap); negative = penetrating.

    Args:
        centre:   Sphere centre in world frame.
        radius:   Sphere radius.
        box_pos:  Box centre in world frame.
        box_half: Box half-extents in box-local frame.
        R_inv:    Transpose of the box's rotation matrix (R_inv = R.T where
                  R maps local→world).  None means axis-aligned (identity).
    """
    if R_inv is not None:
        local = R_inv @ (centre - box_pos)
    else:
        local = centre - box_pos
    closest = np.clip(local, -box_half, box_half)
    return float(np.linalg.norm(local - closest)) - radius


def _precompute_obs_rotations(obs_list) -> List[Optional[np.ndarray]]:
    """
    Precompute R_inv = R.T for each obstacle.

    Returns a list parallel to obs_list.  Each entry is the inverse rotation
    matrix for a box obstacle with non-identity orientation, or None for
    axis-aligned boxes and all non-box obstacles.
    """
    result: List[Optional[np.ndarray]] = []
    for obs in obs_list:
        if obs.type.lower() == "box":
            q = obs.orientation_wxyz
            if not np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6):
                result.append(_quat_to_rot(*q).T)
            else:
                result.append(None)
        else:
            result.append(None)
    return result


def _sphere_vs_sphere(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
) -> bool:
    """Return True if two spheres overlap."""
    return float(np.linalg.norm(c1 - c2)) < (r1 + r2)


def _sphere_vs_cylinder(
    centre: np.ndarray,
    radius: float,
    cyl_pos: np.ndarray,
    cyl_radius: float,
    cyl_half_height: float,
) -> bool:
    """
    Axis-aligned cylinder (axis = Z through cyl_pos).
    Return True if sphere overlaps the cylinder (side or caps).
    """
    dx = centre[0] - cyl_pos[0]
    dy = centre[1] - cyl_pos[1]
    dz = centre[2] - cyl_pos[2]

    # Radial distance in XY plane
    r_xy = np.sqrt(dx * dx + dy * dy)
    # Closest Z on cylinder to sphere centre
    cz = np.clip(dz, -cyl_half_height, cyl_half_height)
    # Closest point on cylinder surface
    if r_xy < 1e-9:
        closest = np.array([cyl_pos[0], cyl_pos[1], cyl_pos[2] + cz])
    else:
        scale = cyl_radius / r_xy
        if r_xy <= cyl_radius:
            # Inside the XY footprint — closest point is on the cap or surface
            closest = np.array([
                cyl_pos[0] + dx,
                cyl_pos[1] + dy,
                cyl_pos[2] + cz,
            ])
        else:
            closest = np.array([
                cyl_pos[0] + dx * scale,
                cyl_pos[1] + dy * scale,
                cyl_pos[2] + cz,
            ])
    dist = float(np.linalg.norm(centre - closest))
    return dist < radius


def _in_collision_with_obstacle(
    link_pos: np.ndarray,
    link_radius: float,
    obs: Obstacle,
    R_inv: Optional[np.ndarray] = None,
) -> bool:
    """
    Test a single link sphere against a single obstacle.

    Args:
        R_inv: Precomputed R.T for box OBB rotation (None = axis-aligned).
               Ignored for sphere / cylinder obstacles.
    """
    obs_pos = np.array(obs.position, dtype=float)
    t = obs.type.lower()

    if t == "box":
        half = np.array(obs.size, dtype=float)   # stored as half-extents
        return _sphere_box_signed_dist(link_pos, link_radius, obs_pos, half, R_inv) < 0.0
    elif t == "sphere":
        return _sphere_vs_sphere(link_pos, link_radius, obs_pos, float(obs.size[0]))
    elif t == "cylinder":
        return _sphere_vs_cylinder(
            link_pos, link_radius,
            obs_pos, float(obs.size[0]), float(obs.size[1]),
        )
    return False   # unknown type — assume free


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_collision_fn(
    spec: Optional[ScenarioSpec] = None,
    obstacles: Optional[List[Obstacle]] = None,
    margin: float = 0.0,
) -> Callable[[np.ndarray], bool]:
    """
    Build a collision-checking callable from a ScenarioSpec or obstacle list.

    Returns a function ``fn(q) -> bool`` where True = collision-free.

    Args:
        spec:      Canonical scenario (uses spec.collision_obstacles()).
        obstacles: Direct list of Obstacle objects (overrides spec if given).
        margin:    Extra clearance added to each link radius (inflates effective
                   obstacle sizes for conservative planning).

    Returns:
        Callable that takes a 7-DOF joint config and returns True if free.
    """
    if obstacles is not None:
        obs_list = [o for o in obstacles if o.collision_enabled]
    elif spec is not None:
        obs_list = spec.collision_obstacles()
    else:
        obs_list = []

    if not obs_list:
        # No obstacles — always free
        return lambda q: True

    # Precompute OBB rotation inverses once — avoids per-call quaternion→matrix.
    obs_R_inv = _precompute_obs_rotations(obs_list)

    def _check(q: np.ndarray) -> bool:
        link_positions = _panda_link_positions(q)
        for i, link_pos in enumerate(link_positions):
            r = _LINK_RADII.get(i, 0.08) + margin
            for j, obs in enumerate(obs_list):
                if _in_collision_with_obstacle(link_pos, r, obs, obs_R_inv[j]):
                    return False   # collision detected
        return True   # all links clear

    return _check


def make_free_space_fn() -> Callable[[np.ndarray], bool]:
    """Convenience: always-free collision function."""
    return lambda q: True
