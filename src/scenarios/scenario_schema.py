"""
Canonical scenario and obstacle schema.

A single ``ScenarioSpec`` object is the authoritative source for:
  * robot start configuration
  * end-effector target pose
  * IK goal specification
  * obstacle list  (same set flows to HJCD-IK JSON, planner, and MuJoCo)
  * controller / planner / visualization parameters

Obstacle format mirrors HJCD-IK JSON so conversion is trivial.

Usage::

    from src.scenarios.scenario_schema import ScenarioSpec, Obstacle
    from src.scenarios.scenario_builders import narrow_passage_scenario

    spec = narrow_passage_scenario()
    obs_dict = spec.obstacles_as_hjcd_dict()   # for HJCD-IK / panda_scene
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    """
    Single obstacle definition shared across all pipeline layers.

    Attributes
    ----------
    name:               Unique string identifier (used as MuJoCo body name).
    type:               One of "box", "sphere", "cylinder".
    position:           [x, y, z] in world frame (metres).
    orientation_wxyz:   Quaternion [w, x, y, z] (default: identity).
    size:               Geometry-dependent:
                          box      -> [half_x, half_y, half_z]  (half-extents)
                          sphere   -> [radius]
                          cylinder -> [radius, half_height]
    rgba:               Display colour [r, g, b, a] (0-1).
    collision_enabled:  If False, the obstacle is visual-only and excluded from
                        HJCD-IK JSON and planner collision checking.
    """
    name:             str
    type:             str                          # "box" | "sphere" | "cylinder"
    position:         List[float]                  # [x, y, z]
    orientation_wxyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    size:             List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    rgba:             Tuple[float, float, float, float] = (0.85, 0.35, 0.20, 0.85)
    collision_enabled: bool = True
    # MuJoCo contact friction [sliding, torsional, rolling].
    # None → MuJoCo default (1.0, 0.005, 0.0001).
    # Set very low sliding friction (e.g. 0.01) for surfaces that the EE slides along.
    friction: Optional[Tuple[float, float, float]] = None

    def pose_list(self) -> List[float]:
        """Return [x, y, z, qw, qx, qy, qz] for HJCD-IK / panda_scene."""
        return list(self.position) + list(self.orientation_wxyz)

    def as_hjcd_entry(self) -> Tuple[str, str, dict]:
        """
        Return (hjcd_type_key, name, spec_dict) for HJCD-IK obstacle dict.

        HJCD-IK uses:
          cuboid  -> {"dims": [full_x, full_y, full_z], "pose": [...]}
          cylinder -> {"radius": r, "height": full_h, "pose": [...]}
          sphere  -> {"radius": r, "pose": [...]}
        """
        pose = self.pose_list()
        t = self.type.lower()
        if t == "box":
            # size stores half-extents; HJCD-IK wants full dims
            dims = [self.size[0] * 2, self.size[1] * 2, self.size[2] * 2]
            return "cuboid", self.name, {"dims": dims, "pose": pose}
        elif t == "cylinder":
            r, hh = self.size[0], self.size[1]
            return "cylinder", self.name, {"radius": r, "height": hh * 2, "pose": pose}
        elif t == "sphere":
            return "sphere", self.name, {"radius": self.size[0], "pose": pose}
        else:
            raise ValueError(f"Unknown obstacle type: {self.type!r}")


# ---------------------------------------------------------------------------
# ScenarioSpec
# ---------------------------------------------------------------------------
@dataclass
class ScenarioSpec:
    """
    Canonical scenario definition.

    All pipeline layers (IK, planner, MuJoCo) must read from this object.

    Attributes
    ----------
    name:           Scenario identifier string (used in output paths).
    q_start:        Robot start joint configuration (7-DOF Panda).
    target_pose:    EE target as dict with keys
                      "position": [x, y, z]
                      "quaternion_wxyz": [w, x, y, z]
                    May be None for joint-space-only scenarios.
    ik_goals:       Pre-computed IK goal configurations (list of np.ndarray).
                    If empty, IK must be solved online via HJCD-IK.
    obstacles:      List of Obstacle objects.
    goal_radius:    Convergence threshold in joint space (rad).
    planner:        Dict of BiRRT parameters.
    controller:     Dict of DS/impedance controller parameters.
    visualization:  Dict of rendering parameters.
    """
    name:          str
    q_start:       np.ndarray
    target_pose:   Optional[Dict]                  = None
    ik_goals:      List[np.ndarray]                = field(default_factory=list)
    obstacles:     List[Obstacle]                  = field(default_factory=list)
    goal_radius:   float                           = 0.05
    planner:       Dict                            = field(default_factory=dict)
    controller:    Dict                            = field(default_factory=dict)
    visualization: Dict                            = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Obstacle accessors
    # ------------------------------------------------------------------
    def collision_obstacles(self) -> List[Obstacle]:
        """Return only obstacles with collision_enabled=True."""
        return [o for o in self.obstacles if o.collision_enabled]

    def obstacles_as_hjcd_dict(self) -> Dict:
        """
        Convert collision-enabled obstacles to HJCD-IK obstacle dict.

        Returns a dict of the form:
          {
            "cuboid":   {name: {dims: [...], pose: [...]}},
            "cylinder": {name: {radius: ..., height: ..., pose: [...]}},
            "sphere":   {name: {radius: ..., pose: [...]}},
          }

        Only keys that have at least one entry are included.
        """
        out: Dict[str, Dict] = {}
        for obs in self.collision_obstacles():
            hjcd_type, name, spec = obs.as_hjcd_entry()
            out.setdefault(hjcd_type, {})[name] = spec
        return out

    def obstacles_as_panda_scene_dict(self) -> Dict:
        """
        Convert ALL obstacles (including visual-only) to panda_scene format.

        Same structure as HJCD-IK dict but includes visual-only obstacles
        so the MuJoCo scene is fully populated, and carries per-obstacle
        ``_rgba`` and ``_visual_only`` sentinel fields for panda_scene.py.
        """
        out: Dict[str, Dict] = {}
        for obs in self.obstacles:
            hjcd_type, name, spec = obs.as_hjcd_entry()
            # Carry individual RGBA so panda_scene can colour each obstacle
            spec = dict(spec, _rgba=list(obs.rgba))
            # For visual-only obstacles, set contype=0 flag via a sentinel
            if not obs.collision_enabled:
                spec = dict(spec, _visual_only=True)
            out.setdefault(hjcd_type, {})[name] = spec
        return out

    def obstacle_friction_map(self) -> Dict[str, Optional[Tuple[float, float, float]]]:
        """
        Return {obstacle_name: friction_tuple_or_None} for all obstacles.

        ``None`` means "use MuJoCo default" (sliding=1.0).
        Callers can use this to apply per-obstacle friction after model compilation.
        """
        return {o.name: o.friction for o in self.obstacles}

    def n_collision_obstacles(self) -> int:
        return len(self.collision_obstacles())

    def n_obstacles(self) -> int:
        return len(self.obstacles)
