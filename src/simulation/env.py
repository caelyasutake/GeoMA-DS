"""
Phase 6 — MuJoCo Simulation Environment

SimEnv wraps a Franka Panda (or compatible) robot loaded from URDF and
provides:

  * robot model         — FK, Jacobian, gravity torques
  * obstacle loading    — cuboid / cylinder / sphere (HJCD-IK JSON format)
  * collision checking  — sphere-swept robot links vs. obstacles
  * contact-force sensing — MuJoCo contact forces in simulation

Usage::

    env = SimEnv()                         # default Panda, no obstacles
    env = SimEnv(obstacles=obs_dict)       # with HJCD-IK-format obstacles
    ok  = env.is_collision_free(q)         # pure query (non-destructive)
    G   = env.gravity_torques(q)           # gravity compensation torques
    pos, quat = env.ee_pose(q)             # FK
    J   = env.jacobian(q)                  # 6 × n_joints Jacobian
    env.set_state(q, qdot)                 # set simulation state
    env.step(tau)                          # forward-simulate one timestep
    contacts = env.get_contact_forces()    # list of contact dicts

Injectable callables (for Phase 2 / Phase 3 / Phase 5)::

    col_fn  = env.make_collision_fn()      # (q) -> bool
    grav_fn = env.make_gravity_fn()        # (q) -> tau_g
    jac_fn  = env.make_jacobian_fn()       # (q) -> J (6 × n)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
_HJCD_ROOT  = Path(__file__).resolve().parents[2] / "external" / "HJCD-IK"
_PANDA_URDF = str(_HJCD_ROOT / "include" / "test_urdf" / "panda.urdf")

# Sphere radii used for swept-link collision detection.
# Keys must match link names in the Panda URDF.
DEFAULT_LINK_SPHERES: Dict[str, float] = {
    "panda_link1": 0.08,
    "panda_link2": 0.08,
    "panda_link3": 0.08,
    "panda_link4": 0.08,
    "panda_link5": 0.07,
    "panda_link6": 0.06,
    # panda_link7 intentionally omitted: a virtual fingertip contact sphere is
    # added below at the correct geometric contact height so the arm stops at
    # exactly the table surface rather than 160 mm above it.
}

# Default MuJoCo timestep (s) — 500 Hz
_DEFAULT_TIMESTEP = 0.002


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """Convert quaternion [w,x,y,z] → URDF RPY [roll, pitch, yaw] (intrinsic XYZ)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = float(np.arcsin(float(np.clip(sinp, -1.0, 1.0))))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def _strip_meshes(text: str) -> str:
    """Remove <visual> and <collision> blocks (which reference mesh files)."""
    text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.DOTALL)
    text = re.sub(r"<collision>.*?</collision>", "", text, flags=re.DOTALL)
    return text


def _build_urdf(
    urdf_path: str,
    link_spheres: Dict[str, float],
    obstacles: Dict,
    base_link: str = "panda_link0",
) -> str:
    """
    Build a URDF XML string with:
      * sphere collision geoms added to specified links
      * obstacle bodies attached as fixed joints to ``base_link``

    Args:
        urdf_path:    Path to original Panda URDF.
        link_spheres: {link_name: radius} for collision spheres.
        obstacles:    HJCD-IK obstacle dict (keys: ``cuboid``, ``cylinder``,
                      ``sphere``).  Each entry: {name: {pose: [...], ...}}.
        base_link:    Parent link for all obstacle bodies.

    Returns:
        URDF XML string ready for ``mujoco.MjModel.from_xml_string()``.
    """
    with open(urdf_path) as fh:
        text = fh.read()
    text = _strip_meshes(text)
    root = ET.fromstring(text)

    # --- Add sphere geoms to robot links ----------------------------------
    for link_elem in root.findall("link"):
        name = link_elem.get("name", "")
        if name in link_spheres:
            r = link_spheres[name]
            col = ET.SubElement(link_elem, "collision")
            ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(geom, "sphere", radius=str(r))

    # --- Add obstacle bodies ----------------------------------------------
    obs_idx = 0
    for geom_type, entries in obstacles.items():
        for obs_name, spec in entries.items():
            pose = spec.get("pose", [0, 0, 0, 1, 0, 0, 0])
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            qw, qx, qy, qz = (
                float(pose[3]),
                float(pose[4]),
                float(pose[5]),
                float(pose[6]),
            )
            roll, pitch, yaw = _quat_to_rpy(qw, qx, qy, qz)
            xyz_str = f"{x} {y} {z}"
            rpy_str = f"{roll} {pitch} {yaw}"

            link_name = f"_obs_{obs_idx}_{obs_name}"

            # Link element
            link_elem = ET.SubElement(root, "link")
            link_elem.set("name", link_name)
            col = ET.SubElement(link_elem, "collision")
            ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
            geom_el = ET.SubElement(col, "geometry")

            gt_lower = geom_type.lower()
            if gt_lower in ("cuboid", "box"):
                dims = spec.get("dims", spec.get("size", [0.1, 0.1, 0.1]))
                ET.SubElement(
                    geom_el, "box",
                    size=f"{dims[0]} {dims[1]} {dims[2]}",
                )
            elif gt_lower == "cylinder":
                radius = float(spec.get("radius", 0.05))
                height = float(spec.get("height", spec.get("length", 0.1)))
                ET.SubElement(geom_el, "cylinder",
                               radius=str(radius), length=str(height))
            elif gt_lower == "sphere":
                radius = float(spec.get("radius", 0.05))
                ET.SubElement(geom_el, "sphere", radius=str(radius))
            else:
                obs_idx += 1
                continue  # skip unknown geometry types

            # Fixed joint to base_link
            joint_el = ET.SubElement(root, "joint")
            joint_el.set("name", f"_obs_joint_{obs_idx}")
            joint_el.set("type", "fixed")
            ET.SubElement(joint_el, "parent", link=base_link)
            ET.SubElement(joint_el, "child", link=link_name)
            ET.SubElement(joint_el, "origin", xyz=xyz_str, rpy=rpy_str)

            obs_idx += 1

    # --- Add EE flange sphere and fingertip contact sphere to panda_link7 ----
    # MuJoCo's URDF compiler merges fixed-joint child bodies (panda_link8,
    # panda_hand, fingers) into panda_link7, so both spheres go here.
    #
    # Flange sphere: xyz="0 0 0.107" — matches panda_joint8 origin (the
    # physical flange / EE mounting plate).  Radius 0.06 m matches
    # _LINK_RADII[7] in collision.py so the planner and physics agree.
    #
    # Fingertip sphere: xyz="0 0 0.212", radius 0.005 m — retained for
    # contact-task scenarios where the fingertip surface height matters.
    for link_elem in root.findall("link"):
        if link_elem.get("name", "") == "panda_link7":
            # EE flange — primary collision/avoidance sphere
            fl_col  = ET.SubElement(link_elem, "collision")
            ET.SubElement(fl_col, "origin", xyz="0 0 0.107", rpy="0 0 0")
            fl_geom = ET.SubElement(fl_col, "geometry")
            ET.SubElement(fl_geom, "sphere", radius="0.06")
            # Fingertip — fine-contact sensing sphere (kept for contact tasks)
            ft_col  = ET.SubElement(link_elem, "collision")
            ET.SubElement(ft_col, "origin", xyz="0 0 0.212", rpy="0 0 0")
            ft_geom = ET.SubElement(ft_col, "geometry")
            ET.SubElement(ft_geom, "sphere", radius="0.005")
            break

    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# SimEnv
# ---------------------------------------------------------------------------
@dataclass
class SimEnvConfig:
    """Configuration for SimEnv."""

    urdf_path:   str            = _PANDA_URDF
    ee_link:     str            = "panda_link7"
    n_joints:    int            = 7
    base_link:   str            = "panda_link0"
    timestep:    float          = _DEFAULT_TIMESTEP
    link_spheres: Optional[Dict[str, float]] = None   # None → DEFAULT_LINK_SPHERES
    obstacles:    Optional[Dict]             = None   # HJCD-IK format; None → no obstacles
    # MuJoCo friction applied to ALL obstacle geoms after compilation.
    # Tuple: (sliding, torsional, rolling).  None → keep MuJoCo defaults (1.0, 0.005, 0.0001).
    # Use a low sliding value (e.g. 0.05) when the EE must slide on obstacle surfaces.
    obstacle_friction: Optional[Tuple[float, float, float]] = None
    # When True, load physics model from MJCF (scene.xml) instead of URDF.
    # Enables full visual geometry (hand + fingers) for contact with correct height.
    use_mjcf: bool = False
    # EE body name to use when use_mjcf=True.  The MJCF menagerie uses "hand".
    mjcf_ee_link: str = "hand"
    # Fixed offset (m) from the EE body origin in the body's own frame.
    # ee_pose() and jacobian() are evaluated at body_pos + R @ ee_offset_body.
    # Default [0, 0, 0.107] places the EE at the flange (panda_joint8 origin,
    # i.e. panda_link8), matching _panda_link_positions[-1] in collision.py.
    # Contact-task scenarios override this to [0, 0, 0.212] (fingertip height).
    ee_offset_body: Optional[np.ndarray] = None  # set in SimEnv.__init__


class SimEnv:
    """
    MuJoCo-based simulation environment for the Franka Panda robot.

    Provides:
      - robot kinematics (FK, Jacobian)
      - dynamics (gravity torques, forward simulation)
      - collision detection (sphere geoms + obstacle bodies)
      - contact force sensing

    Two MjData objects are maintained:
      ``_data``   — persistent simulation state (mutated by ``step()``/
                    ``set_state()``/``reset()``).
      ``_qdata``  — scratch query data (used by non-destructive methods
                    ``is_collision_free()``, ``gravity_torques()``,
                    ``ee_pose()``, ``jacobian()``).
    """

    def __init__(self, config: Optional[SimEnvConfig] = None) -> None:
        if config is None:
            config = SimEnvConfig()
        self._cfg = config
        self._n = config.n_joints

        if config.use_mjcf:
            # ---- MJCF physics path ----------------------------------------
            from src.simulation.panda_scene import load_panda_scene, apply_obstacle_friction
            self._model = load_panda_scene(obstacles=config.obstacles or {})
            self._model.opt.timestep = config.timestep
            # Disable built-in PD actuators so we can drive via qfrc_applied
            self._disable_mjcf_actuators()
            # Configure collision groups: robot vs obstacle
            self._configure_mjcf_collision_filter()
            ee_link_name = config.mjcf_ee_link  # "hand"
        else:
            # ---- URDF physics path (default) --------------------------------
            link_spheres = (
                config.link_spheres
                if config.link_spheres is not None
                else DEFAULT_LINK_SPHERES
            )
            obstacles = config.obstacles or {}

            urdf_str = _build_urdf(
                config.urdf_path,
                link_spheres,
                obstacles,
                base_link=config.base_link,
            )

            self._model = mujoco.MjModel.from_xml_string(urdf_str)
            self._model.opt.timestep = config.timestep

            # Configure collision filter BEFORE creating data objects so that
            # the initial mj_forward uses the correct contype/conaffinity.
            self._configure_collision_filter(config.base_link)
            ee_link_name = config.ee_link  # "panda_link7"

        self._data  = mujoco.MjData(self._model)   # simulation state
        self._qdata = mujoco.MjData(self._model)   # query scratch

        self._ee_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, ee_link_name
        )
        if self._ee_id < 0:
            raise ValueError(
                f"EE link '{ee_link_name}' not found in model. "
                f"Available bodies: {[self._model.body(i).name for i in range(self._model.nbody)]}"
            )

        # Offset from EE body origin to the true EE contact point.
        # Default: [0, 0, 0.107] — the EE flange (panda_joint8 origin).
        # Contact tasks override via SimEnvConfig.ee_offset_body.
        if config.ee_offset_body is not None:
            self._ee_offset = np.asarray(config.ee_offset_body, dtype=float)
        else:
            self._ee_offset = np.array([0.0, 0.0, 0.107])

        # Run initial forward pass to populate kinematics
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------
    def _configure_collision_filter(self, base_link: str) -> None:
        """
        Disable self-collision between robot link spheres while allowing
        robot–obstacle collision detection.

        Collision rule (MuJoCo):
          Two geoms collide iff
            (g1.contype & g2.conaffinity) | (g2.contype & g1.conaffinity) ≠ 0

        Assignments:
          Robot geoms   → contype = 1, conaffinity = 2
          Obstacle geoms → contype = 2, conaffinity = 1

        Robot–robot:     (1 & 2) | (1 & 2) = 0  → no self-collision ✓
        Robot–obstacle:  (1 & 1) | (2 & 2) ≠ 0  → detected         ✓
        Obstacle–obs:    (2 & 1) | (2 & 1) = 0  → ignored           ✓
        """
        # Identify robot body IDs: bodies whose name starts with 'panda_'.
        # NOTE: panda_link0 becomes the MuJoCo 'world' body (body 0 with
        # name 'world') after URDF loading.  Obstacle bodies attached to it
        # via fixed joints are merged into body 0 as well.  Because
        # panda_link0 carries NO sphere geoms (it is not in link_spheres),
        # every geom in body 0 ('world') is an obstacle geom — so we must
        # NOT include body 0 in the robot set.
        robot_body_ids: set = set()
        for i in range(self._model.nbody):
            name = self._model.body(i).name
            if name.startswith("panda_"):
                robot_body_ids.add(i)

        friction = self._cfg.obstacle_friction
        for geom_id in range(self._model.ngeom):
            body_id = int(self._model.geom_bodyid[geom_id])
            if body_id in robot_body_ids:
                self._model.geom_contype[geom_id]     = 1
                self._model.geom_conaffinity[geom_id] = 2
            else:
                # Obstacle geom
                self._model.geom_contype[geom_id]     = 2
                self._model.geom_conaffinity[geom_id] = 1
                if friction is not None:
                    self._model.geom_friction[geom_id, 0] = friction[0]  # sliding
                    self._model.geom_friction[geom_id, 1] = friction[1]  # torsional
                    self._model.geom_friction[geom_id, 2] = friction[2]  # rolling

    def _disable_mjcf_actuators(self) -> None:
        """
        Zero all actuator gains and biases in the MJCF model.

        The MJCF menagerie Panda scene has 8 position PD actuators
        (kp=4500, kd=450).  With ctrl=0 they apply enormous restoring
        torques toward q=0.  Zeroing gainprm / biasprm turns them off so
        we can drive the robot exclusively via qfrc_applied.
        """
        for i in range(self._model.nu):
            self._model.actuator_gainprm[i, :] = 0.0
            self._model.actuator_biasprm[i, :] = 0.0

    def _configure_mjcf_collision_filter(self) -> None:
        """
        Set contype / conaffinity for the MJCF model so that:
          * robot links can collide with obstacles
          * robot links do NOT self-collide
          * obstacle–obstacle contacts are ignored

        Robot body names in the MJCF menagerie:
            link0 … link7, hand, left_finger, right_finger

        Collision rule:
            Robot geoms    → contype=1, conaffinity=2
            Obstacle geoms → contype=2, conaffinity=1
            Robot–robot:     (1&2)|(1&2) = 0  no self-collision ✓
            Robot–obstacle:  (1&1)|(2&2) ≠ 0  detected         ✓
            Obs–obs:         (2&1)|(2&1) = 0  ignored           ✓

        Fingertip pad boxes (type=box on finger bodies) are disabled to reduce
        the number of simultaneous contacts from ~26 to ~3.  The finger mesh
        geoms are retained for correct contact height.
        """
        _MJCF_BOX = 6   # mjtGeom.mjGEOM_BOX
        robot_body_names: set = (
            {f"link{i}" for i in range(8)}
            | {"hand", "left_finger", "right_finger"}
        )
        finger_body_names: set = {"left_finger", "right_finger"}
        robot_body_ids: set = set()
        for i in range(self._model.nbody):
            if self._model.body(i).name in robot_body_names:
                robot_body_ids.add(i)

        friction = self._cfg.obstacle_friction
        for geom_id in range(self._model.ngeom):
            body_id = int(self._model.geom_bodyid[geom_id])
            body_name = self._model.body(body_id).name
            ct = int(self._model.geom_contype[geom_id])
            ca = int(self._model.geom_conaffinity[geom_id])
            gtype = int(self._model.geom_type[geom_id])

            if body_id in robot_body_ids:
                if ct > 0 or ca > 0:
                    # Disable fingertip pad box geoms — they create too many
                    # simultaneous contacts (6 boxes × 2 fingers = 12 extra) and
                    # cause numerical instability.  Keep the finger mesh geoms.
                    if body_name in finger_body_names and gtype == _MJCF_BOX:
                        self._model.geom_contype[geom_id]     = 0
                        self._model.geom_conaffinity[geom_id] = 0
                    else:
                        self._model.geom_contype[geom_id]     = 1
                        self._model.geom_conaffinity[geom_id] = 2
            elif ct > 0 or ca > 0:
                # Non-robot collision geom → obstacle
                self._model.geom_contype[geom_id]     = 2
                self._model.geom_conaffinity[geom_id] = 1
                if friction is not None and body_name.startswith("obs_"):
                    self._model.geom_friction[geom_id, 0] = friction[0]
                    self._model.geom_friction[geom_id, 1] = friction[1]
                    self._model.geom_friction[geom_id, 2] = friction[2]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def n_geoms(self) -> int:
        return self._model.ngeom

    @property
    def n_contacts(self) -> int:
        """Number of active contacts in the current simulation state."""
        return self._data.ncon

    @property
    def q(self) -> np.ndarray:
        """Current joint positions (simulation state)."""
        return self._data.qpos[: self._n].copy()

    @property
    def qdot(self) -> np.ndarray:
        """Current joint velocities (simulation state)."""
        return self._data.qvel[: self._n].copy()

    @property
    def time(self) -> float:
        """Simulation time (s)."""
        return float(self._data.time)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def set_state(
        self,
        q: np.ndarray,
        qdot: Optional[np.ndarray] = None,
    ) -> None:
        """Set simulation joint positions (and optionally velocities)."""
        q = np.asarray(q, dtype=float)
        self._data.qpos[: self._n] = q
        if qdot is not None:
            self._data.qvel[: self._n] = np.asarray(qdot, dtype=float)
        mujoco.mj_forward(self._model, self._data)

    def reset(self, q: Optional[np.ndarray] = None) -> None:
        """Reset simulation to zero (or specified) joint configuration."""
        mujoco.mj_resetData(self._model, self._data)
        if q is not None:
            self._data.qpos[: self._n] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Non-destructive queries  (use self._qdata as scratch)
    # ------------------------------------------------------------------
    def _set_qdata(self, q: np.ndarray) -> None:
        """Populate scratch query data at configuration q."""
        mujoco.mj_resetData(self._model, self._qdata)
        self._qdata.qpos[: self._n] = np.asarray(q, dtype=float)
        self._qdata.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._qdata)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """
        Return True if the robot at configuration ``q`` has zero contacts
        with obstacles or itself.

        Uses the sphere-swept link model. Does NOT modify simulation state.
        """
        self._set_qdata(q)
        return self._qdata.ncon == 0

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """
        Return the gravity compensation torque vector G(q) (shape ``(n_joints,)``).

        Computed as ``qfrc_bias`` with zero velocity (Coriolis = 0).
        Does NOT modify simulation state.
        """
        self._set_qdata(q)
        return self._qdata.qfrc_bias[: self._n].copy()

    def ee_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics for the end-effector.

        Args:
            q: Joint configuration, shape ``(n_joints,)``.

        Returns:
            Tuple ``(pos, quat)`` where
              * pos  — Cartesian position [x, y, z], shape (3,)
              * quat — orientation quaternion [w, x, y, z], shape (4,)

        Does NOT modify simulation state.
        """
        self._set_qdata(q)
        pos  = self._qdata.xpos[self._ee_id].copy()
        if self._ee_offset is not None:
            R   = self._qdata.xmat[self._ee_id].reshape(3, 3)
            pos = pos + R @ self._ee_offset
        quat = self._qdata.xquat[self._ee_id].copy()
        return pos, quat

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Geometric Jacobian at the EE body, shape ``(6, n_joints)``.

        Rows 0-2: linear velocity Jacobian (jacp).
        Rows 3-5: angular velocity Jacobian (jacr).

        Does NOT modify simulation state.
        """
        self._set_qdata(q)
        nv   = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        if self._ee_offset is not None:
            # Evaluate Jacobian at the physical contact point (offset from body
            # origin).  mj_jac() accepts a world-frame point and produces the
            # correct linear Jacobian including the lever-arm contribution.
            R        = self._qdata.xmat[self._ee_id].reshape(3, 3)
            ee_point = self._qdata.xpos[self._ee_id] + R @ self._ee_offset
            mujoco.mj_jac(
                self._model, self._qdata, jacp, jacr, ee_point, self._ee_id
            )
        else:
            mujoco.mj_jacBody(self._model, self._qdata, jacp, jacr, self._ee_id)
        J = np.vstack([jacp, jacr])
        return J[:, : self._n]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def step(self, tau: np.ndarray, dt: Optional[float] = None) -> None:
        """
        Apply joint torques and advance the simulation by one timestep.

        Args:
            tau: Joint torques, shape ``(n_joints,)``.
            dt:  Override timestep (s). If None, uses ``config.timestep``.
        """
        tau = np.asarray(tau, dtype=float)
        if dt is not None:
            self._model.opt.timestep = float(dt)
        self._data.qfrc_applied[: self._n] = tau
        mujoco.mj_step(self._model, self._data)

    # ------------------------------------------------------------------
    # Contact force sensing (from current simulation state)
    # ------------------------------------------------------------------
    def get_contact_forces(self) -> List[Dict]:
        """
        Read all active contacts from the current simulation state.

        Returns:
            List of dicts, one per contact::

                {
                    "pos":    np.ndarray (3,),   # contact point in world frame
                    "force":  np.ndarray (3,),   # contact force in world frame
                    "torque": np.ndarray (3,),   # contact torque in world frame
                    "geom1":  int,               # first geom index
                    "geom2":  int,               # second geom index
                    "dist":   float,             # penetration depth
                }
        """
        contacts = []
        force_buf = np.zeros(6)
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            mujoco.mj_contactForce(self._model, self._data, i, force_buf)
            # mj_contactForce returns [f_normal, f_tan1, f_tan2, t_normal, t_tan1, t_tan2]
            # in the contact local frame (first axis = contact normal in world).
            # Rotate to world frame: F_world = contact.frame.T @ force_contact
            c_frame = np.array(contact.frame).reshape(3, 3)   # rows = frame axes in world
            force_world  = c_frame.T @ force_buf[:3]
            torque_world = c_frame.T @ force_buf[3:]
            contacts.append(
                {
                    "pos":    np.array(contact.pos).copy(),
                    "force":  force_world.copy(),
                    "torque": torque_world.copy(),
                    "geom1":  int(contact.geom1),
                    "geom2":  int(contact.geom2),
                    "dist":   float(contact.dist),
                }
            )
        return contacts

    # ------------------------------------------------------------------
    # Injectable callables (for Phase 2 / 3 / 5 dependency injection)
    # ------------------------------------------------------------------
    def make_collision_fn(self) -> Callable[[np.ndarray], bool]:
        """Return a callable ``(q) -> bool`` for collision checking."""
        return self.is_collision_free

    def make_gravity_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a callable ``(q) -> G(q)`` for gravity compensation."""
        return self.gravity_torques

    def make_jacobian_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a callable ``(q) -> J`` (6 × n_joints Jacobian)."""
        return self.jacobian

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"SimEnv(n_joints={self._n}, nbody={self._model.nbody}, "
            f"ngeom={self._model.ngeom}, ncon={self._data.ncon})"
        )
