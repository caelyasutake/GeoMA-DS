"""
Canonical scenario builders.

Each function returns a ``ScenarioSpec`` that is the single source of truth
for obstacles, start config, IK goals, and parameters used by:
  * HJCD-IK JSON generator
  * BiRRT planner collision checker
  * MuJoCo scene constructor
  * Demo visualization

Available scenarios
-------------------
free_space_scenario()       — no obstacles, 5 diverse IK goals
narrow_passage_scenario()   — box wall creates a workspace passage
contact_task_scenario()     — box obstacle for contact sensing
cluttered_tabletop_scenario() — multiple obstacles on a tabletop
wall_contact_scenario()     — horizontal wall for contact-circle experiment
random_obstacle_field_scenario() — random box field, parameterised
u_shape_scenario()          — non-convex U-shaped obstacle (topological trap)
left_open_u_scenario()      — right-opening U cavity benchmark (arm starts inside, exits via arms)
"""

from __future__ import annotations

import numpy as np

from src.scenarios.scenario_schema import Obstacle, ScenarioSpec

# ---------------------------------------------------------------------------
# Canonical Panda ready pose
# ---------------------------------------------------------------------------
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ---------------------------------------------------------------------------
# Shared IK goal sets (hand-crafted; no GPU required for unit tests)
# ---------------------------------------------------------------------------
_IK_GOALS_FREE = [
    np.array([ 0.50, -0.60,  0.30, -2.00,  0.20,  1.80,  0.60]),
    np.array([-0.30, -0.80, -0.20, -2.20, -0.10,  1.50,  0.90]),
    np.array([ 0.80, -0.40,  0.50, -1.80,  0.40,  2.00,  0.40]),
    np.array([-0.50, -0.70,  0.10, -2.50,  0.10,  1.40,  1.00]),
    np.array([ 0.20, -1.00,  0.40, -1.60,  0.30,  1.90,  0.70]),
]

# Narrow-passage IK goals
_GOAL_EASY = np.array([ 0.40, -0.90,  0.10, -2.10,  0.05,  1.60,  0.80])
_GOAL_HARD = np.array([ 3.50, -0.80,  0.20, -2.00,  0.10,  1.70,  0.70])

# Contact task goal
_GOAL_CONTACT = np.array([ 0.60, -0.50,  0.40, -2.10,  0.20,  1.85,  0.65])


# ---------------------------------------------------------------------------
# Scenario 1 — Free space
# ---------------------------------------------------------------------------
def free_space_scenario() -> ScenarioSpec:
    """
    No obstacles.  5 diverse IK goals.  Tests basic pipeline end-to-end.
    """
    return ScenarioSpec(
        name="free_space",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.45, 0.0, 0.50],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[g.copy() for g in _IK_GOALS_FREE],
        obstacles=[],
        goal_radius=0.05,
        planner=dict(max_iterations=8_000, step_size=0.15, goal_bias=0.15),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(cam_azimuth=140.0, cam_elevation=-25.0),
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Narrow passage (wall obstacle)
# ---------------------------------------------------------------------------
def narrow_passage_scenario() -> ScenarioSpec:
    """
    A vertical box wall cuts through the workspace creating a narrow passage.

    The wall is placed so that:
      - _GOAL_EASY (joint0 = 0.40) is reachable without passing the wall.
      - _GOAL_HARD (joint0 = 3.50) is blocked by the wall in joint space.

    The same wall appears in:
      - HJCD-IK JSON (collision-aware IK generation)
      - BiRRT planner (geometric collision checker)
      - MuJoCo scene (visual + physics body)
    """
    wall = Obstacle(
        name="passage_wall",
        type="box",
        position=[0.50, 0.0, 0.40],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        # half-extents: thin (x=0.03), wide (y=0.40), tall (z=0.40)
        size=[0.03, 0.40, 0.40],
        rgba=(0.60, 0.60, 0.80, 0.80),
        collision_enabled=True,
    )
    return ScenarioSpec(
        name="narrow_passage",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.40, -0.15, 0.50],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[_GOAL_EASY.copy(), _GOAL_HARD.copy()],
        obstacles=[wall],
        goal_radius=0.05,
        planner=dict(max_iterations=8_000, step_size=0.15, goal_bias=0.15),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(cam_azimuth=140.0, cam_elevation=-25.0),
    )


# ---------------------------------------------------------------------------
# Scenario 3 — Contact task
# ---------------------------------------------------------------------------
def contact_task_scenario() -> ScenarioSpec:
    """
    Box obstacle to the side and forward of the arm.

    Box is placed at [0.22, 0.38, 0.62] — verified clear of all Panda links
    at Q_READY (hand at [0.307, 0, 0.590] → y=0 < box y_min=0.28, free).
    _GOAL_CONTACT drives the hand to [0.19, 0.391, 0.618] which intersects
    the box, producing a contact interaction.
    """
    box = Obstacle(
        name="contact_box",
        type="box",
        position=[0.22, 0.38, 0.62],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        size=[0.08, 0.10, 0.08],      # half-extents — full dims 0.16×0.20×0.16
        rgba=(0.85, 0.35, 0.20, 0.85),
        collision_enabled=True,
    )
    return ScenarioSpec(
        name="contact_task",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.22, 0.38, 0.62],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[_GOAL_CONTACT.copy()],
        obstacles=[box],
        goal_radius=0.05,
        planner=dict(max_iterations=8_000, step_size=0.15, goal_bias=0.15),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(cam_azimuth=140.0, cam_elevation=-25.0),
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Cluttered tabletop
# ---------------------------------------------------------------------------
def cluttered_tabletop_scenario() -> ScenarioSpec:
    """
    Multiple cylinders and boxes scattered on a virtual tabletop.

    Demonstrates the multi-IK advantage in a cluttered environment:
    many IK solutions are generated and those that are collision-free
    are preferentially selected by the planner.
    """
    obstacles = [
        Obstacle(
            name="pillar_left",
            type="cylinder",
            position=[0.30, -0.18, 0.275],
            size=[0.028, 0.275],          # radius, half-height
            rgba=(0.70, 0.50, 0.30, 0.9),
        ),
        Obstacle(
            name="pillar_right",
            type="cylinder",
            position=[0.30,  0.18, 0.275],
            size=[0.028, 0.275],
            rgba=(0.70, 0.50, 0.30, 0.9),
        ),
        Obstacle(
            name="pillar_center",
            type="cylinder",
            position=[0.40,  0.00, 0.20],
            size=[0.035, 0.20],
            rgba=(0.70, 0.50, 0.30, 0.9),
        ),
        Obstacle(
            name="side_box",
            type="box",
            position=[0.20,  0.28, 0.10],
            size=[0.08, 0.08, 0.10],
            rgba=(0.50, 0.60, 0.80, 0.85),
        ),
        Obstacle(
            name="robot_stand",
            type="box",
            position=[-0.05, 0.00, -0.40],
            size=[0.15, 0.125, 0.40],
            rgba=(0.40, 0.40, 0.40, 1.0),
        ),
    ]
    ik_goals = [
        np.array([ 0.50, -0.60,  0.30, -2.00,  0.20,  1.80,  0.60]),
        np.array([-0.30, -0.80, -0.20, -2.20, -0.10,  1.50,  0.90]),
        np.array([ 0.20, -1.00,  0.40, -1.60,  0.30,  1.90,  0.70]),
    ]
    return ScenarioSpec(
        name="cluttered_tabletop",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.35, 0.15, 0.40],
            "quaternion_wxyz": [0.924, 0.0, 0.383, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(cam_azimuth=130.0, cam_elevation=-20.0),
    )


# ---------------------------------------------------------------------------
# Scenario 5 — Wall contact + perturbation + return
# ---------------------------------------------------------------------------
# Wall is horizontal (parallel to the ground), like a table surface.
# Top surface at z = 0.37.  EE traces a circle of radius 0.08 m on it.
#
# The actual circular joint-space waypoints are generated at runtime in
# wall_contact_demo.py using Jacobian-pseudoinverse IK; the ik_goals below
# are approximate approach goals used only for HJCD-IK JSON generation.
#
# Rough approach goal: EE near circle centre [0.40, 0.0, 0.37].
# (joints found by adjusting shoulder/elbow to lower EE from Q_READY)
_WALL_APPROACH_GOAL = np.array([0.0, 0.10, 0.0, -2.70, 0.0, 2.80, 0.785])


def wall_contact_scenario() -> ScenarioSpec:
    """
    Wall contact + perturbation + return experiment (horizontal wall).

    A flat horizontal platform is placed at z_centre=0.35 (top surface at
    z=0.37).  The arm presses down onto the surface and traces a circle of
    radius 0.08 m.  Then an external upward torque lifts it off; the DS +
    energy tank restore contact.

    Circular trajectory parameters (used by wall_contact_demo.py):
        circle_center: [0.40, 0.0]  (x, y in world frame)
        circle_radius: 0.08 m
        circle_z:      0.37 m       (top surface of wall)
        circle_n_pts:  12           (waypoints per full revolution)
    """
    wall = Obstacle(
        name="contact_wall",
        type="box",
        position=[0.40, 0.0, 0.35],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        size=[0.25, 0.25, 0.02],   # half-extents: wide x/y, thin z
        rgba=(0.30, 0.55, 0.85, 0.85),
        collision_enabled=True,
        friction=(0.05, 0.005, 0.0001),  # low sliding friction — EE must slide on surface
    )
    return ScenarioSpec(
        name="wall_contact",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.40, 0.0, 0.37],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[_WALL_APPROACH_GOAL.copy()],
        obstacles=[wall],
        goal_radius=0.05,
        planner=dict(max_iterations=8_000, step_size=0.15, goal_bias=0.15),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=5.0),
        visualization=dict(
            cam_azimuth=130.0,
            cam_elevation=-35.0,
            circle_center=[0.40, 0.0],
            circle_radius=0.15,
            circle_z=0.37,
            circle_n_pts=12,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 5b — Contact passivity wall (dedicated passivity-benchmark scenario)
# ---------------------------------------------------------------------------

def contact_passivity_wall_scenario() -> ScenarioSpec:
    """
    Horizontal wall for the contact-passivity approach-comparison benchmark.

    Identical geometry to wall_contact_scenario() but with a tighter circle
    (radius 0.08 m) so approach accuracy affects contact quality more visibly.
    Used by benchmarks/eval_contact_passivity.py to compare:
        vanilla DS  vs  BiRRT + DS  vs  DiffIK
    on approach dynamics and contact establishment quality.
    """
    wall = Obstacle(
        name="contact_wall",
        type="box",
        position=[0.40, 0.0, 0.35],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        size=[0.25, 0.25, 0.02],
        rgba=(0.30, 0.55, 0.85, 0.85),
        collision_enabled=True,
        friction=(0.05, 0.005, 0.0001),
    )
    return ScenarioSpec(
        name="contact_passivity_wall",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.40, 0.0, 0.37],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[_WALL_APPROACH_GOAL.copy()],
        obstacles=[wall],
        goal_radius=0.05,
        planner=dict(max_iterations=8_000, step_size=0.15, goal_bias=0.15),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=5.0),
        visualization=dict(
            cam_azimuth=130.0,
            cam_elevation=-35.0,
            circle_center=[0.40, 0.0],
            circle_radius=0.08,
            circle_z=0.37,
            circle_n_pts=12,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 6 — Random obstacle field
# ---------------------------------------------------------------------------

def random_obstacle_field_scenario(
    seed: int = 0,
    n_obstacles: int = 4,
    obstacle_size: float = 0.07,
) -> ScenarioSpec:
    """
    Randomly placed box obstacles in the Panda workspace.

    Obstacles are sampled uniformly from a workspace region in front of the
    robot, with a fixed random seed so scenarios are reproducible.

    Args:
        seed:          RNG seed for obstacle placement.
        n_obstacles:   Number of random box obstacles.
        obstacle_size: Half-extent of each box (metres).
    """
    import random as _random
    rng = _random.Random(seed)

    obstacles = []
    # Workspace box: x ∈ [0.25, 0.55], y ∈ [-0.30, 0.30], z ∈ [0.15, 0.55]
    for i in range(n_obstacles):
        x = rng.uniform(0.25, 0.55)
        y = rng.uniform(-0.30, 0.30)
        z = rng.uniform(0.15, 0.55)
        obstacles.append(Obstacle(
            name=f"rand_box_{i}",
            type="box",
            position=[x, y, z],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[obstacle_size, obstacle_size, obstacle_size],
            rgba=(0.80, 0.45, 0.20, 0.85),
            collision_enabled=True,
        ))

    ik_goals = [
        np.array([ 0.50, -0.60,  0.30, -2.00,  0.20,  1.80,  0.60]),
        np.array([-0.30, -0.80, -0.20, -2.20, -0.10,  1.50,  0.90]),
        np.array([ 0.80, -0.40,  0.50, -1.80,  0.40,  2.00,  0.40]),
    ]

    return ScenarioSpec(
        name="random_obstacle_field",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [0.45, 0.0, 0.45],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=10_000, step_size=0.12, goal_bias=0.12),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(cam_azimuth=140.0, cam_elevation=-25.0),
    )


# ---------------------------------------------------------------------------
# Scenario 7 — U-shaped obstacle (topological trap)
# ---------------------------------------------------------------------------

def u_shape_scenario(
    seed: int = 0,
    opening_width: float = 0.20,
    depth: float = 0.20,
    thickness: float = 0.03,
    height: float = 0.30,
    rotation_rad: float = 0.0,
) -> ScenarioSpec:
    """
    Non-convex U-shaped obstacle that creates a topological trap.

    The U is placed in front of the robot (positive x, world frame) at a
    height where the Panda arm links naturally operate.  Verified geometry:

      * Q_READY (q=[0,-0.785,0,-2.356,0,1.571,0.785]) puts the hand at
        [0.307, 0, 0.590], which is OUTSIDE the U opening (x_open=0.35).
      * The "clean" IK goal (q[0]=0) sends the arm straight through the
        U opening with link5 at y≈0, giving clearance from both walls.
      * "Wide" IK goals (q[0]=±0.45) swing link5 into the U arm walls,
        causing a collision that the planner must avoid.

    The U is built from 3 boxes:
      u_left   — left arm of U (y > 0 side)
      u_right  — right arm of U (y < 0 side)
      u_bottom — connecting wall (far end, seals the U)

    NOTE on rotation_rad:
      Box positions are rotated around the U opening pivot.  Individual box
      orientations (orientation_wxyz) are set to reflect the rotation so
      MuJoCo renders correctly.  The planner collision checker uses AABB
      (axis-aligned), so for rotation_rad != 0 collision checking is
      approximate; use rotation_rad=0 (default) for exact results.

    Args:
        seed:          RNG seed (reserved for future stochastic placement).
        opening_width: Width of the U gap (m).  Default 0.20.
        depth:         Length of U arms in x (m).  Default 0.20.
        thickness:     Wall thickness (m).  Default 0.03.
        height:        Wall height (m).  Default 0.30.
        rotation_rad:  Rotation of the whole U around its opening pivot (rad).
    """
    # ---- Geometry (un-rotated frame) ----------------------------------------
    # Opening at x = x_open, bottom wall inner face at x = x_open + depth.
    x_open = 0.35       # x-coordinate of U opening (robot side)
    z_c    = 0.45       # vertical centre of the walls
    y_c    = 0.0        # lateral centre

    # Centre of each arm (left/right walls)
    arm_cx = x_open + depth / 2
    arm_cy_left  =  opening_width / 2 + thickness / 2
    arm_cy_right = -(opening_width / 2 + thickness / 2)
    # Centre of bottom wall (outer face at x_open + depth + thickness)
    bot_cx = x_open + depth + thickness / 2
    bot_cy = 0.0

    # Half-extents for each wall
    arm_half = [depth / 2,          thickness / 2,                    height / 2]
    bot_half = [thickness / 2,      opening_width / 2 + thickness,    height / 2]

    # ---- Apply rotation around the opening pivot [x_open, y_c] -------------
    def _rot2d(px: float, py: float, angle: float):
        dx, dy = px - x_open, py - y_c
        c, s = float(np.cos(angle)), float(np.sin(angle))
        return x_open + c * dx - s * dy, y_c + s * dx + c * dy

    lx, ly = _rot2d(arm_cx, arm_cy_left,  rotation_rad)
    rx, ry = _rot2d(arm_cx, arm_cy_right, rotation_rad)
    bx, by = _rot2d(bot_cx, bot_cy,       rotation_rad)

    # Quaternion for z-axis rotation: w = cos(a/2), z = sin(a/2)
    qw = float(np.cos(rotation_rad / 2))
    qz = float(np.sin(rotation_rad / 2))
    rot_wxyz = [qw, 0.0, 0.0, qz]

    obstacles = [
        Obstacle(
            name="u_left",
            type="box",
            position=[lx, ly, z_c],
            orientation_wxyz=rot_wxyz,
            size=list(arm_half),
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="u_right",
            type="box",
            position=[rx, ry, z_c],
            orientation_wxyz=rot_wxyz,
            size=list(arm_half),
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="u_bottom",
            type="box",
            position=[bx, by, z_c],
            orientation_wxyz=rot_wxyz,
            size=list(bot_half),
            rgba=(0.55, 0.18, 0.18, 0.85),
            collision_enabled=True,
        ),
    ]

    # ---- Goal: inside the U at 75 % depth -----------------------------------
    gx_raw = x_open + depth * 0.75
    gx, gy = _rot2d(gx_raw, 0.0, rotation_rad)

    # ---- IK goals -----------------------------------------------------------
    # Verified with the MJCF FK (matches MuJoCo world frame).
    # At the default geometry (opening_width=0.20, depth=0.20):
    #
    #   clean_center  q0=0.00: hand=(0.400, 0,     0.497) — through centre of opening ✓
    #   wide_left     q0=0.45: link5 hits u_left wall   (collision) ✓
    #   wide_right    q0=-0.45: link5 hits u_right wall  (collision) ✓
    #   deep_left     q0=0.30: link5 also hits u_left    (collision) ✓
    #
    # This creates a meaningful test: the planner must select "clean_center"
    # (or find an alternative path) rather than heading straight to a
    # wide/deep goal that is blocked by the U walls.
    ik_goals = [
        # Clean — arm straight forward (q0=0), link5 at y≈0, fits through opening
        np.array([ 0.00, -0.40,  0.00, -2.60,  0.00,  2.20,  0.785]),
        # Wide-left — shoulder rotated to +y side, link5 hits u_left
        np.array([ 0.45, -0.40,  0.00, -2.60,  0.00,  2.20,  0.785]),
        # Wide-right — shoulder rotated to -y side, link5 hits u_right
        np.array([-0.45, -0.40,  0.00, -2.60,  0.00,  2.20,  0.785]),
        # Moderate-left (different reach) — also hits u_left
        np.array([ 0.30, -0.30,  0.00, -2.50,  0.00,  2.00,  0.785]),
    ]

    return ScenarioSpec(
        name="u_shape",
        q_start=Q_READY.copy(),
        target_pose={
            "position": [float(gx), float(gy), z_c],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(
            cam_azimuth=140.0,
            cam_elevation=-25.0,
            u_opening_width=opening_width,
            u_depth=depth,
            u_rotation_rad=rotation_rad,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 8 — Left-opening U / C-shape barrier (canonical left-to-right benchmark)
# ---------------------------------------------------------------------------

def left_open_u_scenario(
    seed: int = 0,
    arm_length: float = 0.35,
    gap_width: float = 0.40,
    wall_thickness: float = 0.03,
    wall_height: float = 0.35,
    center_x: float = 0.50,
    center_y: float = 0.00,
    center_z: float = 0.40,
) -> ScenarioSpec:
    """
    U/C-shaped barrier benchmark — opening faces +Y (rotated 90° CW).

    The Panda arm starts inside the U cavity at POSITIVE Y (near the opening).
    The goal is at NEGATIVE Y, past the bottom spine.  The opening faces +Y.

    Geometry (XY plane, viewed from above)::

                        opening (+Y)
             ┌──────────────────────────────┐
             │                              │
        left arm                       right arm
             │                              │
             └──────── bottom_spine ────────┘
                           (-Y)

                        o ← Start (EE at +Y, inside cavity)

                        x ← Goal (-Y, past spine)

    Obstacle layout (3 boxes):
      left_arm     — vertical arm at −X, thin in X, wide in Y
      right_arm    — vertical arm at +X, thin in X, wide in Y
      bottom_spine — horizontal bar at −Y, wide in X, thin in Y

    Verified geometry at default parameters
    (arm_length=0.35, gap_width=0.40, wall_thickness=0.03, wall_height=0.35):

      * opening at y = center_y + half_arm = +0.175  (top, +Y side)
      * spine at  y = center_y − half_arm − half_thick = −0.190  (bottom)
      * left_arm  at x = center_x − half_gap − half_thick = 0.285
      * right_arm at x = center_x + half_gap + half_thick = 0.715
      * q_start=[0.20, 0.30, 0, -1.60, 0, 1.50, 0.785] → EE=(0.630,+0.128,0.530)
        — inside cavity (0.285 < 0.630 < 0.715 in x, 0 < +0.128 < 0.175 in y) ✓

    IK goal classification (verified with collision.py FK):
      * via_wide  q0=-1.00 → EE=(0.348,-0.541,0.530)  FREE  (arm swings far -y)
      * via_med   q0=-0.60 → EE=(0.531,-0.363,0.530)  FREE  (arm clears spine)
      * blk_near  q0=-0.30 → L4 clips bottom_spine (d=0.009)  BLOCKED
      * blk_mid   q0=-0.40 → L4 clips bottom_spine (d=0.014)  BLOCKED

    Multi-IK advantage: two IK goals are FREE, two are BLOCKED.
    Mean pairwise joint-space distance = 0.383 > 0.3 ✓

    Args:
        seed:            RNG seed (reserved).
        arm_length:      Length of each vertical arm in Y (m).  Default 0.35.
        gap_width:       Clear gap between the two arms in X (m).  Default 0.40.
        wall_thickness:  Wall thickness (m).  Default 0.03.
        wall_height:     Full wall height in Z (m).  Default 0.35.
        center_x:        X-coordinate of the opening centre (m).  Default 0.50.
        center_y:        Y-coordinate of the barrier centre (m).  Default 0.00.
        center_z:        Z-coordinate of the barrier centre (m).  Default 0.40.
    """
    # ---- Derived geometry ---------------------------------------------------
    half_thick = wall_thickness / 2.0          # 0.015
    half_arm   = arm_length / 2.0              # 0.175
    half_z     = wall_height / 2.0             # 0.175
    half_gap   = gap_width / 2.0               # 0.200

    # Left arm: vertical box at −X, running in Y direction
    left_cx  = center_x - half_gap - half_thick    # 0.285
    # Right arm: vertical box at +X, running in Y direction
    right_cx = center_x + half_gap + half_thick    # 0.715
    # Bottom spine: horizontal box at −Y, running in X direction
    spine_cy = center_y - half_arm - half_thick    # −0.190
    spine_hx = half_gap + wall_thickness           #  0.230  (spine half-extent in X)
    # Opening: top face of the arms (+Y side)
    opening_y = center_y + half_arm               #  0.175

    obstacles = [
        Obstacle(
            name="left_arm",
            type="box",
            position=[left_cx, center_y, center_z],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[half_thick, half_arm, half_z],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="right_arm",
            type="box",
            position=[right_cx, center_y, center_z],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[half_thick, half_arm, half_z],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="bottom_spine",
            type="box",
            position=[center_x, spine_cy, center_z],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[spine_hx, half_thick, half_z],
            rgba=(0.55, 0.18, 0.18, 0.85),
            collision_enabled=True,
        ),
    ]

    # ---- Starting configuration ---------------------------------------------
    # q_start = [0.20, 0.30, 0, -1.60, 0, 1.50, 0.785]
    # Verified FK: EE=(0.630, +0.128, 0.530) — inside cavity at +Y.
    # All links clear of all three obstacle boxes.
    q_start = np.array([0.20, 0.30, 0.0, -1.60, 0.0, 1.50, 0.785])

    # ---- IK goals -----------------------------------------------------------
    # All four goals share q1..q6; only q0 (shoulder pan) varies.
    # Negative q0 swings the arm toward −Y (goal side); the magnitude determines
    # whether the arm clears the bottom_spine.
    #
    # Verified with collision.py FK at default parameters:
    #
    #   via_wide  q0=-1.00 → EE=(0.348,-0.541,0.530)  L4 clears spine  [FREE]
    #   via_med   q0=-0.60 → EE=(0.531,-0.363,0.530)  L4 clears spine  [FREE]
    #   blk_near  q0=-0.30 → EE=(0.615,-0.190,0.530)  L4 clips spine   [BLOCKED]
    #   blk_mid   q0=-0.40 → EE=(0.593,-0.251,0.530)  L4 clips spine   [BLOCKED]
    #
    # Mean pairwise joint-space distance = 0.383 > 0.3 ✓
    ik_goals = [
        # via_wide — wide −y swing, L4 clears bottom_spine
        np.array([-1.00,  0.30,  0.00, -1.60,  0.00,  1.50,  0.785]),
        # via_med — moderate −y swing, also clears
        np.array([-0.60,  0.30,  0.00, -1.60,  0.00,  1.50,  0.785]),
        # blk_near — small −y swing, L4 clips bottom_spine
        np.array([-0.30,  0.30,  0.00, -1.60,  0.00,  1.50,  0.785]),
        # blk_mid — medium −y swing, L4 clips bottom_spine
        np.array([-0.40,  0.30,  0.00, -1.60,  0.00,  1.50,  0.785]),
    ]

    # Goal workspace position: at −Y, past the bottom spine.
    goal_x = center_x                    # 0.50 — centred laterally
    goal_y = spine_cy - 0.08             # −0.270 — clearly past the spine
    goal_z = center_z                    # 0.40

    return ScenarioSpec(
        name="left_open_u",
        q_start=q_start,
        target_pose={
            "position":         [goal_x, goal_y, goal_z],
            "quaternion_wxyz":  [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12),
        controller=dict(K_c=2.0, K_r=1.0, K_n=0.3, d_gain=3.0),
        visualization=dict(
            cam_azimuth=90.0,
            cam_elevation=-40.0,
            cam_lookat=[center_x, center_y, center_z],
            cam_distance=1.80,
            left_open_u=True,
            spine_y=spine_cy,
            opening_y=opening_y,
        ),
    )


# Alias — expose under additional canonical names
c_barrier_scenario           = left_open_u_scenario
left_to_right_u_trap_scenario = left_open_u_scenario


# ---------------------------------------------------------------------------
# Scenario 9 — Frontal I-Barrier left-to-right benchmark
# ---------------------------------------------------------------------------

_FRONTAL_I_PARAMS = {
    "easy":   dict(bar_half_y=0.28, post_half_z=0.33, thickness_x=0.04),
    "medium": dict(bar_half_y=0.32, post_half_z=0.37, thickness_x=0.06),
    "hard":   dict(bar_half_y=0.38, post_half_z=0.41, thickness_x=0.08),
}


def build_frontal_i_barrier_lr(difficulty: str = "medium") -> ScenarioSpec:
    """
    Frontal I-barrier left-to-right benchmark.

    The barrier forms a gate at x = x_post.  Three red box obstacles:
      - top_bar    : thin in X, long in Y (depth), thin in Z — at high Z
      - bottom_bar : thin in X, long in Y (depth), thin in Z — at low Z
      - center_post: thin in X, thin in Y,           tall in Z — at z_mid

    The gate lies in the XZ plane at x=0.48.  The start EE [0.48, -0.24, 0.555]
    is in front of the barrier (y < 0); the goal EE [0.48, +0.24, 0.555] is
    behind it (y > 0).  Both share x=0.48 and z=0.555 (grasptarget; panda_hand
    lands at z=0.45).  The center_post blocks direct front-to-back (Y) passage;
    the arm must route around it in the X direction while staying between the
    top and bottom bars.

    The start and goal EE positions are collision-free for all difficulty
    variants (verified by bounding-box analysis in the design spec).

    Args:
        difficulty: "easy", "medium", or "hard".

    Returns:
        ScenarioSpec with 3 obstacles and 4 IK goals.
    """
    if difficulty not in _FRONTAL_I_PARAMS:
        raise ValueError(
            f"difficulty must be 'easy', 'medium', or 'hard'; got {difficulty!r}"
        )
    p = _FRONTAL_I_PARAMS[difficulty]

    # ---- Shared geometry ---------------------------------------------------
    x_post      = 0.48
    y_barrier   = 0.00
    z_mid       = 0.45
    thickness_z = 0.04   # bar height (Z)
    post_thin   = 0.04   # post width in both X and Y

    bar_half_y  = p["bar_half_y"]    # half-span of bars in Y (depth)
    post_half_z = p["post_half_z"]   # half-height of center post in Z
    thx         = p["thickness_x"]   # full thickness of bars/post in X
    thx_half    = thx / 2.0

    # Bar Z positions: flush with the top/bottom of the post
    z_top    = z_mid + post_half_z - thickness_z / 2.0
    z_bottom = z_mid - post_half_z + thickness_z / 2.0

    obstacles = [
        Obstacle(
            name="top_bar",
            type="box",
            position=[x_post, y_barrier, z_top],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            # thin in X, long in Y, thin in Z
            size=[thx_half, bar_half_y, thickness_z / 2.0],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="bottom_bar",
            type="box",
            position=[x_post, y_barrier, z_bottom],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[thx_half, bar_half_y, thickness_z / 2.0],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="center_post",
            type="box",
            position=[x_post, y_barrier, z_mid],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            # thin in X and Y, tall in Z
            size=[thx_half, post_thin / 2.0, post_half_z],
            rgba=(0.55, 0.18, 0.18, 0.85),
            collision_enabled=True,
        ),
    ]

    # ---- Visual markers (collision_enabled=False) --------------------------
    # Green sphere at start EE, red sphere at goal EE (grasptarget positions).
    # Layout (Y axis): start (s) at y=-0.24, center_post at y=0, goal (g) at y=+0.24.
    _start_ee = [0.48, -0.24, 0.555]
    _goal_ee  = [0.48, +0.24, 0.555]
    obstacles += [
        Obstacle(
            name="marker_start",
            type="sphere",
            position=_start_ee,
            size=[0.025],
            rgba=(0.20, 0.75, 0.20, 0.90),
            collision_enabled=False,
        ),
        Obstacle(
            name="marker_goal",
            type="sphere",
            position=_goal_ee,
            size=[0.025],
            rgba=(0.85, 0.20, 0.20, 0.90),
            collision_enabled=False,
        ),
    ]

    # ---- Start config (grasptarget EE = [0.48, -0.24, 0.555]) ---------------
    # HJCD-IK uses panda_grasptarget (0.105m past panda_hand) as its EE frame.
    # panda_hand lands at z=0.45 when grasptarget is at z=0.555.
    # FK check: _panda_link_positions(q)[-1] (hand) = [0.48, -0.24, 0.45].
    q_start = np.array([ 1.3517, -1.2759, -1.6638, -2.1596,  1.77,    1.3333,  0.4533])

    # ---- IK goals (8 configs with grasptarget EE = [0.48, +0.24, 0.555]) ---
    # Generated by HJCD-IK; panda_hand lands at [0.48, +0.24, 0.45].
    # Multiple elbow families present (joint1 ≈ ±1.8 and ±1.35).
    ik_goals = [
        np.array([ 1.7855,  1.2316, -1.4597, -2.1615, -1.7994,  1.2952,  1.1105]),
        np.array([ 1.7921,  1.7143, -1.6492, -2.1545, -1.4933,  1.7148,  1.1366]),
        np.array([-1.346,  -1.6618,  1.5131, -2.1539, -1.5264,  1.6689,  1.1389]),
        np.array([ 1.745,   0.9935, -1.354,  -2.1761, -1.971,   1.0942,  1.0534]),
        np.array([ 1.7921,  1.3031, -1.4887, -2.1586, -1.7522,  1.3566,  1.1213]),
        np.array([-1.3705, -1.1242,  1.7274, -2.167,  -1.8736,  1.2037,  1.089 ]),
        np.array([ 1.7983,  1.6,    -1.6044, -2.1536, -1.5649,  1.615,   1.14  ]),
        np.array([-1.3597, -1.2007,  1.6947, -2.1629, -1.8204,  1.2687,  1.105 ]),
    ]

    return ScenarioSpec(
        name=f"frontal_i_barrier_lr_{difficulty}",
        q_start=q_start,
        target_pose={
            "position":        [0.48, 0.24, 0.555],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12, collision_margin=0.03),
        controller=dict(K_c=2.0, K_r=2.0, K_n=2.0, d_gain=8.0, max_speed=2.0, ds_goal_radius=0.5),
        visualization=dict(
            cam_azimuth=90.0,
            cam_elevation=-15.0,
            cam_lookat=[x_post, y_barrier, z_mid],
            cam_distance=1.6,
            frontal_i_barrier=True,
            difficulty=difficulty,
        ),
    )


def frontal_i_barrier_lr_easy()   -> ScenarioSpec:
    return build_frontal_i_barrier_lr("easy")

def frontal_i_barrier_lr_medium() -> ScenarioSpec:
    return build_frontal_i_barrier_lr("medium")

def frontal_i_barrier_lr_hard()   -> ScenarioSpec:
    return build_frontal_i_barrier_lr("hard")


# ---------------------------------------------------------------------------
# Scenario 10 — Frontal Cross-Barrier left-to-right benchmark
# ---------------------------------------------------------------------------

_FRONTAL_CROSS_PARAMS = {
    "easy":   dict(bar_half_y=0.28, post_half_z=0.33, thickness_x=0.04, cross_half_x=0.18),
    "medium": dict(bar_half_y=0.32, post_half_z=0.37, thickness_x=0.06, cross_half_x=0.20),
    "hard":   dict(bar_half_y=0.38, post_half_z=0.41, thickness_x=0.08, cross_half_x=0.22),
}


def build_frontal_cross_barrier(difficulty: str = "medium") -> ScenarioSpec:
    """
    Frontal Cross-barrier left-to-right benchmark.

    Extends the I-barrier by adding a horizontal bar (`horiz_bar`) that spans
    X at the mid-height z_mid.  The gate in the XZ plane now has a "+" shape:
      - center_post blocks the vertical middle (as in I-barrier)
      - horiz_bar   blocks the horizontal middle

    This creates 4 corner windows instead of 2 side passages, requiring the
    arm to commit to a specific quadrant (top-left, top-right, bottom-left,
    bottom-right) — a harder topological decision than the I-barrier.

    Obstacles (all at x = x_post):
      - top_bar    : high Z, long in Y, thin in X/Z
      - bottom_bar : low Z, long in Y, thin in X/Z
      - center_post: mid Z, tall in Z, thin in X/Y
      - horiz_bar  : mid Z, wide in X, thin in Y/Z  ← cross piece

    Same start/goal EE positions and IK goals as the frontal I-barrier.

    Args:
        difficulty: "easy", "medium", or "hard".

    Returns:
        ScenarioSpec with 4 obstacles and 8 IK goals.
    """
    if difficulty not in _FRONTAL_CROSS_PARAMS:
        raise ValueError(
            f"difficulty must be 'easy', 'medium', or 'hard'; got {difficulty!r}"
        )
    p = _FRONTAL_CROSS_PARAMS[difficulty]

    x_post       = 0.48
    y_barrier    = 0.00
    z_mid        = 0.45
    thickness_z  = 0.04
    post_thin    = 0.04

    bar_half_y   = p["bar_half_y"]
    post_half_z  = p["post_half_z"]
    thx          = p["thickness_x"]
    thx_half     = thx / 2.0
    cross_half_x = p["cross_half_x"]

    z_top    = z_mid + post_half_z - thickness_z / 2.0
    z_bottom = z_mid - post_half_z + thickness_z / 2.0

    obstacles = [
        Obstacle(
            name="top_bar",
            type="box",
            position=[x_post, y_barrier, z_top],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[thx_half, bar_half_y, thickness_z / 2.0],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="bottom_bar",
            type="box",
            position=[x_post, y_barrier, z_bottom],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[thx_half, bar_half_y, thickness_z / 2.0],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="center_post",
            type="box",
            position=[x_post, y_barrier, z_mid],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[thx_half, post_thin / 2.0, post_half_z],
            rgba=(0.55, 0.18, 0.18, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="horiz_bar",
            type="box",
            position=[x_post, y_barrier, z_mid],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            # spans X, thin in Y and Z — crosses center_post to form "+"
            size=[cross_half_x, post_thin / 2.0, thickness_z / 2.0],
            rgba=(0.55, 0.18, 0.55, 0.85),
            collision_enabled=True,
        ),
    ]

    # Visual markers
    _start_ee = [0.48, -0.24, 0.555]
    _goal_ee  = [0.48, +0.24, 0.555]
    obstacles += [
        Obstacle(
            name="marker_start",
            type="sphere",
            position=_start_ee,
            size=[0.025],
            rgba=(0.20, 0.75, 0.20, 0.90),
            collision_enabled=False,
        ),
        Obstacle(
            name="marker_goal",
            type="sphere",
            position=_goal_ee,
            size=[0.025],
            rgba=(0.85, 0.20, 0.20, 0.90),
            collision_enabled=False,
        ),
    ]

    q_start = np.array([ 1.3517, -1.2759, -1.6638, -2.1596,  1.77,    1.3333,  0.4533])

    ik_goals = [
        np.array([ 1.7855,  1.2316, -1.4597, -2.1615, -1.7994,  1.2952,  1.1105]),
        np.array([ 1.7921,  1.7143, -1.6492, -2.1545, -1.4933,  1.7148,  1.1366]),
        np.array([-1.346,  -1.6618,  1.5131, -2.1539, -1.5264,  1.6689,  1.1389]),
        np.array([ 1.745,   0.9935, -1.354,  -2.1761, -1.971,   1.0942,  1.0534]),
        np.array([ 1.7921,  1.3031, -1.4887, -2.1586, -1.7522,  1.3566,  1.1213]),
        np.array([-1.3705, -1.1242,  1.7274, -2.167,  -1.8736,  1.2037,  1.089 ]),
        np.array([ 1.7983,  1.6,    -1.6044, -2.1536, -1.5649,  1.615,   1.14  ]),
        np.array([-1.3597, -1.2007,  1.6947, -2.1629, -1.8204,  1.2687,  1.105 ]),
    ]

    return ScenarioSpec(
        name=f"frontal_cross_barrier_{difficulty}",
        q_start=q_start,
        target_pose={
            "position":        [0.48, 0.24, 0.555],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=ik_goals,
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12, collision_margin=0.03),
        controller=dict(K_c=2.0, K_r=2.0, K_n=2.0, d_gain=8.0, max_speed=2.0, ds_goal_radius=0.5),
        visualization=dict(
            cam_azimuth=90.0,
            cam_elevation=-15.0,
            cam_lookat=[x_post, y_barrier, z_mid],
            cam_distance=1.6,
            frontal_cross_barrier=True,
            difficulty=difficulty,
        ),
    )


def frontal_cross_barrier_easy()   -> ScenarioSpec:
    return build_frontal_cross_barrier("easy")

def frontal_cross_barrier_medium() -> ScenarioSpec:
    return build_frontal_cross_barrier("medium")

def frontal_cross_barrier_hard()   -> ScenarioSpec:
    return build_frontal_cross_barrier("hard")


# ---------------------------------------------------------------------------
# Frontal XZ-Cross benchmark
# ---------------------------------------------------------------------------

# Cross center: [0.42, 0.0, 0.47] — places start EE [0.48, -0.24, 0.555] in
# top-right quadrant (x=0.48>0.42, z=0.555>0.47) and goal in bottom-left.
_XZ_CROSS_CENTER = np.array([0.42, 0.0, 0.47])

# IK goals for target grasptarget=[0.28, 0.24, 0.35] (bottom-left quadrant).
# Generated by numerical IK search; all collision-free w.r.t. the cross bars
# (clearance > 0.05 m).  Elbow families: mix of fwd/center/back.
_XZ_CROSS_IK_GOALS = [
    np.array([-1.6441, -0.4356,  2.0766, -1.8416,  0.3100,  0.7626, -0.1673]),
    np.array([ 0.9641,  0.0888,  0.2729, -2.2662, -0.9028,  0.9918,  1.0027]),
    np.array([ 1.0467,  0.5949,  0.0054, -1.9363, -0.8186,  0.3576, -0.3477]),
    np.array([-0.6650,  0.2005,  1.7115, -2.0772, -0.4840,  1.1016, -0.9255]),
    np.array([-0.3426,  0.0210,  1.3320, -2.0635, -0.4206,  1.1240,  1.3560]),
    np.array([ 1.7216,  0.5185, -1.4747, -1.9455,  0.6005,  0.8079, -2.0272]),
    np.array([-1.6299, -1.4697,  1.8611, -1.8096, -0.6386,  0.1871,  0.5825]),
    np.array([ 1.6706, -0.4864, -0.7303, -2.3313, -0.4216,  1.5790,  2.0192]),
]


def build_frontal_xz_cross() -> ScenarioSpec:
    """
    Frontal XZ-cross benchmark.

    Two bars form a "+" in the XZ plane, centered at [0.42, 0.0, 0.47]:
      - vertical_bar  : long in Z (±0.225 m), thin in X (±0.055 m)
      - horizontal_bar: long in X (±0.225 m), thin in Z (±0.055 m)
    Both bars have depth ±0.08 m in Y.

    The existing benchmark q_start places the grasptarget EE at [0.48, -0.24, 0.555],
    which is in the top-right quadrant (x>cross_x, z>cross_z).  The goal is at
    [0.28, +0.24, 0.35], in the bottom-left quadrant — diagonally opposite.

    This forces the arm to detour around the cross intersection; a local reactive
    controller is trapped, while GeoMA-DS has diverse route families to exploit.
    """
    cx, cy, cz = _XZ_CROSS_CENTER
    bar_thickness = 0.055   # half-extent in thin direction (X or Z)
    bar_half_xz   = 0.225   # half-span in long direction
    bar_depth_y   = 0.08    # half-depth in Y (barrier face direction)

    obstacles = [
        Obstacle(
            name="xz_cross_vertical_bar",
            type="box",
            position=[cx, cy, cz],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[bar_thickness, bar_depth_y, bar_half_xz],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="xz_cross_horizontal_bar",
            type="box",
            position=[cx, cy, cz],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[bar_half_xz, bar_depth_y, bar_thickness],
            rgba=(0.55, 0.18, 0.55, 0.85),
            collision_enabled=True,
        ),
    ]

    # Visual markers
    _start_ee = [0.48, -0.24, 0.555]
    _goal_ee  = [0.28, +0.24, 0.350]
    obstacles += [
        Obstacle(
            name="marker_start",
            type="sphere",
            position=_start_ee,
            size=[0.025],
            rgba=(0.20, 0.75, 0.20, 0.90),
            collision_enabled=False,
        ),
        Obstacle(
            name="marker_goal",
            type="sphere",
            position=_goal_ee,
            size=[0.025],
            rgba=(0.85, 0.20, 0.20, 0.90),
            collision_enabled=False,
        ),
    ]

    q_start = np.array([ 1.3517, -1.2759, -1.6638, -2.1596,  1.77,    1.3333,  0.4533])

    assert _goal_ee[0] < cx, "goal must be left of cross center (x)"
    assert _goal_ee[2] < cz, "goal must be below cross center (z)"
    assert _start_ee[0] > cx, "start must be right of cross center (x)"
    assert _start_ee[2] > cz, "start must be above cross center (z)"

    return ScenarioSpec(
        name="frontal_xz_cross",
        q_start=q_start,
        target_pose={
            "position":        [0.28, 0.24, 0.350],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[g.copy() for g in _XZ_CROSS_IK_GOALS],
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12, collision_margin=0.03),
        controller=dict(K_c=2.0, K_r=2.0, K_n=2.0, d_gain=8.0, max_speed=2.0, ds_goal_radius=0.5),
        visualization=dict(
            cam_azimuth=90.0,
            cam_elevation=-15.0,
            cam_lookat=[cx, cy, cz],
            cam_distance=1.6,
            frontal_xz_cross=True,
        ),
    )


# ---------------------------------------------------------------------------
# Frontal YZ-cross benchmark
# ---------------------------------------------------------------------------

# Cross center: [0.60, 0.0, 0.48] — places start EE [0.48, -0.24, 0.555] in
# top-left quadrant (y=-0.24<0.0, z=0.555>0.48) and goal in bottom-right.
# cx=0.60 positions the cross slightly in front of the goal (goal at x=0.65).
_YZ_CROSS_CENTER = np.array([0.60, 0.0, 0.48])

# IK goals for target grasptarget=[0.65, 0.24, 0.35] (bottom-right quadrant).
# Generated by numerical IK search; all collision-free w.r.t. the cross bars
# (clearance >= 0.042 m).  Elbow families: diverse.
_YZ_CROSS_IK_GOALS = [
    np.array([ 0.0964,  1.7628,  1.2009, -0.5782,  1.9435,  0.8008,  2.4446]),
    np.array([-2.6944, -0.9617, -2.8002, -1.3275, -2.0911,  0.8465, -2.4850]),
    np.array([ 1.4962,  1.7628, -1.3847, -1.8828,  1.3446,  3.7395, -2.6515]),
    np.array([-2.8973, -1.3763, -2.8973, -0.9433,  2.8973,  0.5683,  1.9663]),
    np.array([-1.8315, -0.7569,  2.1346, -2.1787, -2.8245,  1.5336,  2.5596]),
    np.array([-1.6374, -1.5190,  1.6422, -1.4938,  2.8489,  3.5475, -0.9303]),
    np.array([ 2.1502, -1.7470, -1.5645, -2.1322,  2.7374,  2.2951,  2.6631]),
    np.array([ 1.5128,  0.4905, -1.2160, -1.9809, -2.4681,  2.7573,  0.6971]),
]


def build_frontal_yz_cross() -> ScenarioSpec:
    """
    Frontal YZ-cross benchmark.

    Two bars form a "+" in the Y-Z plane, centered at [0.60, 0.0, 0.48]:
      - vertical_bar  : long in Z (+-0.30 m), thin in Y (+-0.015 m)
      - horizontal_bar: long in Y (+-0.30 m), thin in Z (+-0.015 m)
    Both bars have depth +-0.025 m in X (barrier face direction).
    Cross is positioned slightly in front of (cx=0.60) the goal (x=0.65).

    The existing benchmark q_start places the grasptarget EE at [0.48, -0.24, 0.555],
    which is in the top-left quadrant (y<cross_y, z>cross_z).  The goal is at
    [0.65, 0.24, 0.35], in the bottom-right quadrant — diagonally opposite.

    This forces the arm to detour around the cross intersection; a local reactive
    controller is trapped, while GeoMA-DS has diverse route families to exploit.
    """
    cx, cy, cz = _YZ_CROSS_CENTER
    bar_depth_x  = 0.025   # half-depth in X (barrier face direction)
    bar_thick    = 0.015   # half-extent in thin direction (Y or Z)
    bar_half_yz  = 0.30    # half-span in long direction

    obstacles = [
        Obstacle(
            name="yz_cross_vertical_bar",
            type="box",
            position=[cx, cy, cz],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[bar_depth_x, bar_thick, bar_half_yz],
            rgba=(0.72, 0.22, 0.22, 0.85),
            collision_enabled=True,
        ),
        Obstacle(
            name="yz_cross_horizontal_bar",
            type="box",
            position=[cx, cy, cz],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            size=[bar_depth_x, bar_half_yz, bar_thick],
            rgba=(0.55, 0.18, 0.55, 0.85),
            collision_enabled=True,
        ),
    ]

    # Visual markers
    _start_ee = [0.48, -0.24, 0.555]
    _goal_ee  = [0.65, 0.24, 0.350]
    obstacles += [
        Obstacle(
            name="marker_start",
            type="sphere",
            position=_start_ee,
            size=[0.025],
            rgba=(0.20, 0.75, 0.20, 0.90),
            collision_enabled=False,
        ),
        Obstacle(
            name="marker_goal",
            type="sphere",
            position=_goal_ee,
            size=[0.025],
            rgba=(0.85, 0.20, 0.20, 0.90),
            collision_enabled=False,
        ),
    ]

    q_start = np.array([ 1.3517, -1.2759, -1.6638, -2.1596,  1.77,    1.3333,  0.4533])

    if not (_start_ee[1] < cy):
        print("[warn] frontal_yz_cross: start EE y not in top-left quadrant (want y < cy)")
    if not (_start_ee[2] > cz):
        print("[warn] frontal_yz_cross: start EE z not in top-left quadrant (want z > cz)")
    if not (_goal_ee[1] > cy):
        print("[warn] frontal_yz_cross: goal EE y not in bottom-right quadrant (want y > cy)")
    if not (_goal_ee[2] < cz):
        print("[warn] frontal_yz_cross: goal EE z not in bottom-right quadrant (want z < cz)")

    return ScenarioSpec(
        name="frontal_yz_cross",
        q_start=q_start,
        target_pose={
            "position":        [0.65, 0.24, 0.350],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        ik_goals=[g.copy() for g in _YZ_CROSS_IK_GOALS],
        obstacles=obstacles,
        goal_radius=0.05,
        planner=dict(max_iterations=12_000, step_size=0.12, goal_bias=0.12, collision_margin=0.03),
        controller=dict(K_c=2.0, K_r=2.0, K_n=2.0, d_gain=8.0, max_speed=2.0, ds_goal_radius=0.5),
        visualization=dict(
            cam_azimuth=90.0,
            cam_elevation=-15.0,
            cam_lookat=[cx, cy, cz],
            cam_distance=1.6,
            frontal_yz_cross=True,
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCENARIO_REGISTRY = {
    "free_space":            free_space_scenario,
    "narrow_passage":        narrow_passage_scenario,
    "contact_task":          contact_task_scenario,
    "cluttered_tabletop":    cluttered_tabletop_scenario,
    "wall_contact":          wall_contact_scenario,
    "random_obstacle_field": random_obstacle_field_scenario,
    "u_shape":               u_shape_scenario,
    # Left-to-right barrier benchmark (canonical name + aliases)
    "left_open_u":           left_open_u_scenario,
    "c_barrier":             left_open_u_scenario,
    "left_to_right_barrier": left_open_u_scenario,
    # Frontal I-barrier benchmark (vertical gate in XZ plane)
    "i_barrier":                   frontal_i_barrier_lr_medium,   # benchmark canonical name
    "frontal_i_barrier":           frontal_i_barrier_lr_medium,   # legacy alias
    "frontal_i_barrier_lr":        frontal_i_barrier_lr_medium,
    "frontal_i_barrier_lr_easy":   frontal_i_barrier_lr_easy,
    "frontal_i_barrier_lr_medium": frontal_i_barrier_lr_medium,
    "frontal_i_barrier_lr_hard":   frontal_i_barrier_lr_hard,
    # Frontal Cross-barrier benchmark (4-window gate in XZ plane)
    "frontal_cross":               frontal_cross_barrier_medium,  # legacy alias
    "frontal_cross_barrier":        frontal_cross_barrier_medium,
    "frontal_cross_barrier_easy":   frontal_cross_barrier_easy,
    "frontal_cross_barrier_medium": frontal_cross_barrier_medium,
    "frontal_cross_barrier_hard":   frontal_cross_barrier_hard,
    # Frontal XZ-cross benchmark (diagonal quadrant crossing)
    "frontal_xz_cross":             build_frontal_xz_cross,
    # Frontal YZ-cross benchmark (diagonal quadrant crossing in Y-Z plane)
    "cross_barrier":                build_frontal_yz_cross,       # benchmark canonical name
    "frontal_yz_cross":             build_frontal_yz_cross,       # legacy alias
}


def get_scenario(name: str) -> ScenarioSpec:
    """Look up and build a scenario by name."""
    if name not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario {name!r}. Available: {list(SCENARIO_REGISTRY)}"
        )
    return SCENARIO_REGISTRY[name]()
