"""
MuJoCo Rendering Environment

Wraps a compiled MjModel (loaded from the local Panda MJCF assets) with:

  * off-screen rendering via ``mujoco.Renderer``
  * joint-config setter (7-DOF arm, ignoring finger joints)
  * FK helper for EE Cartesian position
  * explicit camera configuration

Usage::

    from src.simulation.panda_scene import load_panda_scene
    from src.simulation.mujoco_env import MuJoCoRenderEnv, RenderConfig

    model = load_panda_scene(obstacles=..., ee_target=..., goal_markers=...)
    env   = MuJoCoRenderEnv(model, RenderConfig())

    env.set_joint_config(q)          # set 7-DOF arm pose
    frame = env.render()             # np.ndarray (H, W, 3) uint8
    frame = env.render_at(q)         # shorthand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mujoco
import numpy as np

# Number of arm joints (excluding finger joints)
N_ARM = 7


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RenderConfig:
    """Off-screen rendering parameters."""

    height: int = 480
    width:  int = 640

    # Camera parameters (free camera — explicitly set, no MuJoCo defaults)
    cam_lookat:   Tuple[float, float, float] = (0.3, 0.0, 0.4)
    cam_distance: float = 2.2
    cam_azimuth:  float = 140.0   # degrees — shows arm from front-left
    cam_elevation: float = -25.0  # degrees — slightly above horizon


# ---------------------------------------------------------------------------
# MuJoCoRenderEnv
# ---------------------------------------------------------------------------
class MuJoCoRenderEnv:
    """
    Off-screen rendering environment for the Franka Panda.

    Parameters
    ----------
    model:  Compiled MjModel (from ``load_panda_scene()``).
    config: Rendering configuration.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: Optional[RenderConfig] = None,
    ) -> None:
        if config is None:
            config = RenderConfig()
        self._cfg = config

        self._model = model
        self._data  = mujoco.MjData(model)

        # On Windows, mujoco.Renderer requires an explicit GL context to be
        # made current before it can render.  Without this, render() returns
        # all-black frames even though no error is raised.
        self._gl_ctx: Optional[object] = None
        try:
            if hasattr(mujoco, "GLContext"):
                self._gl_ctx = mujoco.GLContext(config.width, config.height)
                self._gl_ctx.make_current()
        except Exception:
            pass  # GL context optional — viewer path works without it

        self._renderer = mujoco.Renderer(model, config.height, config.width)

        # Build camera — explicitly configured, not relying on MuJoCo defaults
        self._camera = mujoco.MjvCamera()
        self._camera.type        = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:]   = list(config.cam_lookat)
        self._camera.distance    = config.cam_distance
        self._camera.azimuth     = config.cam_azimuth
        self._camera.elevation   = config.cam_elevation

        # Locate EE body (accept both naming conventions)
        self._ee_id = self._find_ee_body()

        # Run initial forward pass
        mujoco.mj_forward(model, self._data)

    def _find_ee_body(self) -> int:
        """
        Find the end-effector body id.

        Tries: "hand" (MJCF menagerie), "panda_link7", "link7" in that order.
        """
        for candidate in ("hand", "panda_link7", "link7", "panda_hand"):
            bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, candidate
            )
            if bid >= 0:
                return bid
        raise RuntimeError(
            "Cannot find Panda end-effector body. "
            f"Available: {[self._model.body(i).name for i in range(self._model.nbody)]}"
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def set_joint_config(self, q: np.ndarray) -> None:
        """
        Set the 7-DOF arm joint positions and run forward kinematics.

        Only the first ``N_ARM`` entries of qpos are set; finger joints
        (indices 7, 8 in the MJCF 9-DOF model) are left at their current
        values.
        """
        q = np.asarray(q, dtype=float)
        n = min(len(q), N_ARM, self._model.nq)
        self._data.qpos[:n] = q[:n]
        self._data.qvel[:]  = 0.0
        mujoco.mj_forward(self._model, self._data)

    def ee_pos(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return EE Cartesian position [x, y, z].

        If ``q`` is None uses the current model state.
        """
        if q is not None:
            self.set_joint_config(q)
        return self._data.xpos[self._ee_id].copy()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(
        self,
        camera: Optional[mujoco.MjvCamera] = None,
    ) -> np.ndarray:
        """
        Render current scene to an RGB array.

        Args:
            camera: Optional MjvCamera override.  Defaults to the free
                    camera defined in ``RenderConfig`` (explicitly set).

        Returns:
            RGB image, shape (H, W, 3), dtype uint8.
        """
        cam = camera if camera is not None else self._camera
        self._renderer.update_scene(self._data, camera=cam)
        return self._renderer.render()

    def render_at(self, q: np.ndarray) -> np.ndarray:
        """Set joint config then render — convenience method."""
        self.set_joint_config(q)
        return self.render()

    def render_with_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        """
        Set joint positions AND velocities then render.

        Unlike ``render_at`` (which zeros velocity), this method preserves the
        exact simulation state — including velocity — so that the rendered
        frame faithfully represents the actual dynamics state at that instant.

        Both ``qpos`` and ``qvel`` are taken from the first ``N_ARM`` entries;
        finger joints retain their current values.

        Args:
            qpos: Joint positions (7,).
            qvel: Joint velocities (7,).

        Returns:
            RGB image, shape (H, W, 3), dtype uint8.
        """
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        nq = min(len(qpos), N_ARM, self._model.nq)
        nv = min(len(qvel), N_ARM, self._model.nv)
        self._data.qpos[:nq] = qpos[:nq]
        self._data.qvel[:nv] = qvel[:nv]
        mujoco.mj_forward(self._model, self._data)
        return self.render()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def height(self) -> int:
        return self._cfg.height

    @property
    def width(self) -> int:
        return self._cfg.width

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def __repr__(self) -> str:
        return (
            f"MuJoCoRenderEnv(nq={self._model.nq}, "
            f"{self.width}×{self.height}px, ee_body={self._model.body(self._ee_id).name})"
        )
