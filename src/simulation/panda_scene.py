"""
Panda Scene Builder for MuJoCo Visualization.

Loads the Franka Panda from the local ``mujoco/franka_emika_panda/`` folder
(MJCF with mesh assets and proper visual geometry) and optionally injects
obstacle bodies and EE/goal markers into the scene.

Primary entry points
--------------------
load_panda_scene(obstacles, ee_target, goal_markers)
    Build and return a compiled MjModel from the local MJCF assets.

validate_panda_model(model)
    Confirm the model contains Franka Panda bodies/joints/geoms.

configure_marker_colors(model)
    Set rgba values on marker geoms after model compilation.

Accepted body naming conventions
---------------------------------
* ``panda_link0`` … ``panda_link7``, ``panda_hand``   (URDF-based models)
* ``link0``       … ``link7``,       ``hand``          (MJCF menagerie)

Both are recognised by validate_panda_model.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT    = Path(__file__).resolve().parents[2]
_MUJOCO_ROOT  = _REPO_ROOT / "mujoco"
_PANDA_DIR    = _MUJOCO_ROOT / "franka_emika_panda"
_SCENE_XML    = _PANDA_DIR  / "scene.xml"        # has floor + lights
_PANDA_XML    = _PANDA_DIR  / "panda.xml"        # robot only

# Fallback: URDF-based path (used only when MJCF unavailable)
_HJCD_ROOT    = _REPO_ROOT / "external" / "HJCD-IK"
_PANDA_URDF   = _HJCD_ROOT / "include" / "test_urdf" / "panda.urdf"

# ---------------------------------------------------------------------------
# Panda body name sets (both naming conventions)
# ---------------------------------------------------------------------------
_PANDA_LINK_NAMES_LONG  = {f"panda_link{i}" for i in range(8)} | {"panda_hand"}
_PANDA_LINK_NAMES_SHORT = {f"link{i}" for i in range(8)} | {"hand"}

# MJCF Panda joints (7 arm + 2 finger = 9 DOF)
N_ARM_JOINTS = 7

# Neutral arm configuration (Panda ready pose)
NEUTRAL_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ---------------------------------------------------------------------------
# RGBA colour palette for markers
# ---------------------------------------------------------------------------
COLOR_OBSTACLE = (0.85, 0.35, 0.20, 0.85)
COLOR_TARGET   = (1.00, 0.20, 0.20, 0.90)
COLOR_GOAL_ALL = (0.60, 0.60, 0.60, 0.70)
COLOR_SAFE     = (0.20, 0.80, 0.30, 0.80)
COLOR_SELECTED = (1.00, 0.60, 0.00, 1.00)
COLOR_CIRCLE   = (0.00, 0.85, 0.95, 0.90)  # cyan — circle trajectory waypoints


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_panda_model(model: mujoco.MjModel) -> dict:
    """
    Confirm the MjModel contains Franka Panda bodies/joints/geoms.

    Accepts both naming conventions:
      * long  — ``panda_link0`` … ``panda_link7``, ``panda_hand``
      * short — ``link0``       … ``link7``,       ``hand``

    Args:
        model: Compiled MjModel to inspect.

    Returns:
        dict with keys:
          body_names      (list[str])
          joint_names     (list[str])
          n_bodies        (int)
          n_joints        (int)
          n_geoms         (int)
          panda_detected  (bool)
    """
    body_names  = [model.body(i).name  for i in range(model.nbody)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    body_set = set(body_names)
    panda_detected = bool(
        body_set & _PANDA_LINK_NAMES_LONG
        or body_set & _PANDA_LINK_NAMES_SHORT
    )

    return {
        "body_names":     body_names,
        "joint_names":    joint_names,
        "n_bodies":       model.nbody,
        "n_joints":       model.njnt,
        "n_geoms":        model.ngeom,
        "panda_detected": panda_detected,
    }


# ---------------------------------------------------------------------------
# Primary loader — MJCF from local mujoco/ folder
# ---------------------------------------------------------------------------
def load_panda_scene(
    obstacles:    Dict         = None,
    ee_target:    Optional[np.ndarray] = None,
    goal_markers: Optional[List[Tuple[np.ndarray, str]]] = None,
) -> mujoco.MjModel:
    """
    Load the Franka Panda from ``mujoco/franka_emika_panda/scene.xml`` and
    inject optional obstacle/marker bodies.

    Args:
        obstacles:    HJCD-IK obstacle dict (keys: cuboid / cylinder / sphere).
                      Each entry: {name: {pose: [...], dims/radius/...}}.
        ee_target:    EE target position [x, y, z] in world frame (red sphere).
        goal_markers: List of (pos_xyz, kind) where kind ∈
                      {"all", "safe", "selected"}.

    Returns:
        Compiled MjModel.  Call ``validate_panda_model()`` to confirm.

    Raises:
        FileNotFoundError: if the local mujoco/ folder is missing.
        RuntimeError:      if the compiled model has no Panda bodies.
    """
    if not _SCENE_XML.exists():
        raise FileNotFoundError(
            f"Panda MJCF not found at {_SCENE_XML}.\n"
            f"Expected a local mujoco/ folder at {_MUJOCO_ROOT}."
        )

    if obstacles is None:
        obstacles = {}
    if goal_markers is None:
        goal_markers = []

    # If no extras needed, load the scene XML directly (most efficient)
    if not obstacles and ee_target is None and not goal_markers:
        model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        _check_panda(model)
        return model

    # Otherwise inject extras by modifying the scene XML in a temp file
    model = _load_with_extras(obstacles, ee_target, goal_markers)
    _check_panda(model)
    return model


def _check_panda(model: mujoco.MjModel) -> None:
    """Raise RuntimeError if the model contains no Panda bodies."""
    info = validate_panda_model(model)
    if not info["panda_detected"]:
        raise RuntimeError(
            "MuJoCo model does not contain Franka Panda bodies.\n"
            f"Found bodies: {info['body_names']}"
        )


def _quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr, cosr))
    sinp = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = float(np.arcsin(float(sinp)))
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny, cosy))
    return roll, pitch, yaw


def _load_with_extras(
    obstacles:    Dict,
    ee_target:    Optional[np.ndarray],
    goal_markers: List[Tuple[np.ndarray, str]],
) -> mujoco.MjModel:
    """
    Inject extra bodies into the scene XML and compile.

    We extend the worldbody of scene.xml with additional free bodies
    (markers, obstacles). The scene.xml is parsed, new elements added,
    and compiled via a temporary directory so the <include> path resolves.
    """
    tree = ET.parse(str(_SCENE_XML))
    root = tree.getroot()

    # Find or create <worldbody>
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # ---- Obstacles -----------------------------------------------------------
    obs_idx = 0
    for geom_type, entries in obstacles.items():
        for obs_name, spec in entries.items():
            pose = spec.get("pose", [0, 0, 0, 1, 0, 0, 0])
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            qw, qx, qy, qz = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
            rpy = _quat_to_rpy(qw, qx, qy, qz)
            pos_s   = f"{x} {y} {z}"
            euler_s = f"{np.degrees(rpy[0]):.4f} {np.degrees(rpy[1]):.4f} {np.degrees(rpy[2]):.4f}"

            body = ET.SubElement(worldbody, "body")
            body.set("name",  f"obs_{obs_idx}_{obs_name}")
            body.set("pos",   pos_s)
            body.set("euler", euler_s)

            geom = ET.SubElement(body, "geom")
            # Use obstacle name in geom name so tests can find it
            geom.set("name", f"obs_geom_{obs_name}")
            gt = geom_type.lower()
            if gt in ("cuboid", "box"):
                dims = spec.get("dims", spec.get("size", [0.1, 0.1, 0.1]))
                geom.set("type", "box")
                geom.set("size", f"{dims[0]/2} {dims[1]/2} {dims[2]/2}")
            elif gt == "cylinder":
                r = spec.get("radius", 0.05)
                h = spec.get("height", spec.get("length", 0.1))
                geom.set("type", "cylinder")
                geom.set("size", f"{r} {h/2}")
            elif gt == "sphere":
                geom.set("type", "sphere")
                geom.set("size", str(spec.get("radius", 0.05)))
            else:
                obs_idx += 1
                continue

            # Per-obstacle colour: use _rgba if provided, else fall back to default
            rgba = spec.get("_rgba", None)
            if rgba is not None and len(rgba) == 4:
                geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
            else:
                geom.set("rgba", f"{COLOR_OBSTACLE[0]} {COLOR_OBSTACLE[1]} "
                                 f"{COLOR_OBSTACLE[2]} {COLOR_OBSTACLE[3]}")

            # Visual-only obstacles (collision_enabled=False) have contype=0
            if spec.get("_visual_only", False):
                geom.set("contype",     "0")
                geom.set("conaffinity", "0")
            else:
                geom.set("contype",     "1")
                geom.set("conaffinity", "1")
            obs_idx += 1

    # ---- EE target marker ----------------------------------------------------
    if ee_target is not None:
        x, y, z = float(ee_target[0]), float(ee_target[1]), float(ee_target[2])
        body = ET.SubElement(worldbody, "body")
        body.set("name", "ee_target_marker")
        body.set("pos",  f"{x} {y} {z}")
        geom = ET.SubElement(body, "geom")
        geom.set("name",    "ee_target_geom")
        geom.set("type",    "sphere")
        geom.set("size",    "0.04")
        geom.set("rgba",    f"{COLOR_TARGET[0]} {COLOR_TARGET[1]} "
                            f"{COLOR_TARGET[2]} {COLOR_TARGET[3]}")
        geom.set("contype",     "0")
        geom.set("conaffinity", "0")

    # ---- Goal markers --------------------------------------------------------
    _MARKER_RGBA = {
        "selected": COLOR_SELECTED,
        "safe":     COLOR_SAFE,
        "circle":   COLOR_CIRCLE,
    }
    _MARKER_SIZE = {
        "circle": "0.018",   # smaller dot — marks a path, not a goal
    }
    for i, (pos, kind) in enumerate(goal_markers):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        rgba = _MARKER_RGBA.get(kind, COLOR_GOAL_ALL)
        size = _MARKER_SIZE.get(kind, "0.035")

        body = ET.SubElement(worldbody, "body")
        body.set("name", f"goal_marker_{i}_{kind}")
        body.set("pos",  f"{x} {y} {z}")
        geom = ET.SubElement(body, "geom")
        geom.set("name",    f"goal_geom_{i}")
        geom.set("type",    "sphere")
        geom.set("size",    size)
        geom.set("rgba",    f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        geom.set("contype",     "0")
        geom.set("conaffinity", "0")

    # Write the augmented XML as a sibling file of scene.xml so that
    # the <include file="panda.xml"/> and mesh asset paths resolve correctly.
    # We use NamedTemporaryFile in _PANDA_DIR itself (not a subdirectory).
    tmp_path = _PANDA_DIR / "_scene_augmented_tmp.xml"
    try:
        tree.write(str(tmp_path), encoding="unicode", xml_declaration=False)
        model = mujoco.MjModel.from_xml_path(str(tmp_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return model


# ---------------------------------------------------------------------------
# Marker colour configuration (post-compilation)
# ---------------------------------------------------------------------------
def configure_marker_colors(model: mujoco.MjModel) -> None:
    """
    Apply colour overrides to marker geoms injected by ``load_panda_scene()``.

    Safe to call even if no markers were added — silently no-ops on
    non-marker geoms.
    """
    for geom_id in range(model.ngeom):
        body_id   = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name

        if body_name == "ee_target_marker":
            model.geom_rgba[geom_id] = list(COLOR_TARGET)
        elif body_name.startswith("goal_marker_"):
            kind = body_name.split("_")[-1]
            rgba = {
                "selected": COLOR_SELECTED,
                "safe":     COLOR_SAFE,
                "circle":   COLOR_CIRCLE,
            }.get(kind, COLOR_GOAL_ALL)
            model.geom_rgba[geom_id] = list(rgba)
        elif body_name.startswith("obs_"):
            model.geom_rgba[geom_id] = list(COLOR_OBSTACLE)


# ---------------------------------------------------------------------------
# Obstacle friction configuration (post-compilation)
# ---------------------------------------------------------------------------
def apply_obstacle_friction(
    model: mujoco.MjModel,
    friction: Tuple[float, float, float],
) -> None:
    """
    Set contact friction on all obstacle geoms in an MJCF-compiled model.

    Obstacle bodies are identified by the ``obs_`` prefix added by
    ``_load_with_extras()``.  Call this immediately after ``load_panda_scene()``
    before creating ``MjData``.

    Args:
        model:    Compiled MjModel (mutated in-place).
        friction: (sliding, torsional, rolling) friction coefficients.
                  Use low sliding values (e.g. 0.05) for surfaces the EE slides on.

    Example::

        model = load_panda_scene(obstacles=obs_dict)
        apply_obstacle_friction(model, friction=(0.05, 0.005, 0.0001))
    """
    for geom_id in range(model.ngeom):
        body_id   = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name
        if body_name.startswith("obs_"):
            model.geom_friction[geom_id, 0] = friction[0]
            model.geom_friction[geom_id, 1] = friction[1]
            model.geom_friction[geom_id, 2] = friction[2]
