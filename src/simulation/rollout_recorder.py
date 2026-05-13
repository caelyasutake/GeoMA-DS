"""
Rollout Recorder

Records the exact simulation state (qpos, qvel, contact forces, EE pose) at
every step of a SimEnv rollout.  Renders GIF frames **inline** — i.e., each
frame is captured during simulation by mirroring the recorded state into the
render model and calling mj_forward, NOT from a post-hoc reconstruction.

This is the single source of truth for:
    * The saved GIF (wall_contact_rollout.gif)
    * Contact diagnostics
    * Circle-tracking metrics

Design guarantees
-----------------
* Frames are captured from the exact (qpos, qvel) pair produced by mj_step.
* No Euler reconstruction, no separate marker scene, no guessed states.
* The viewer and the GIF share the same recorded snapshots.

Usage::

    recorder = RolloutRecorder(render_env=render_env, capture_every=5)

    for i in range(n_steps):
        env.step(tau, dt=DT)
        recorder.record_step(
            step=i, t=i*DT, phase=phase,
            env=env, tau=tau,
            tank_energy=res.tank_energy,
            pf_clipped=res.pf_clipped,
        )

    recorder.render_gif(out_dir / "rollout.gif", fps=20)
    metrics = recorder.compute_metrics(
        circle_center=center, circle_radius=radius,
        contact_z=z_surface,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-step snapshot
# ---------------------------------------------------------------------------
@dataclass
class StepSnapshot:
    """
    Exact simulation state captured at one timestep.

    All array fields are copies of the SimEnv state — not references.
    """

    step:         int
    t:            float
    phase:        str
    qpos:         np.ndarray    # (7,) joint positions from SimEnv
    qvel:         np.ndarray    # (7,) joint velocities from SimEnv
    contacts:     List[Dict]    # raw list from env.get_contact_forces()
    contact_mag:  float         # ||Σ F_contact||
    ee_pos:       np.ndarray    # (3,) EE Cartesian position
    tank_energy:  float
    pf_clipped:   bool
    tau:          np.ndarray    # (7,) applied joint torque
    frame:        Optional[np.ndarray] = None  # (H,W,3) uint8 — None if not captured


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class RolloutRecorder:
    """
    Records simulation states and renders GIF frames inline.

    Args:
        render_env:     MuJoCoRenderEnv used for inline frame capture.
                        If None, no frames are captured (metrics only).
        capture_every:  Capture a rendered frame every this many steps.
                        Lower → smoother GIF, more memory.
    """

    def __init__(
        self,
        render_env=None,        # Optional[MuJoCoRenderEnv]
        capture_every: int = 5,
    ) -> None:
        self._render_env   = render_env
        self._capture_every = max(1, int(capture_every))
        self._snapshots: List[StepSnapshot] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_step(
        self,
        step:         int,
        t:            float,
        phase:        str,
        env,                    # SimEnv — duck-typed to avoid circular import
        tau:          np.ndarray,
        tank_energy:  float,
        pf_clipped:   bool,
    ) -> StepSnapshot:
        """
        Record the exact state of ``env`` AFTER ``env.step()`` has been called.

        Reads ``env.q``, ``env.qdot``, ``env.get_contact_forces()``, and
        ``env.ee_pose(q)`` directly from the simulator — no reconstruction.

        A rendered frame is captured (via ``render_env.render_with_state()``)
        every ``capture_every`` steps.

        Args:
            step:        Integer step index (0-based).
            t:           Simulation time (s) at this step.
            phase:       Phase label string (e.g. "WALL_SLIDE").
            env:         SimEnv instance (already stepped).
            tau:         Applied joint torques this step (7,).
            tank_energy: Tank energy s after this step.
            pf_clipped:  Whether passivity filter clipped this step.

        Returns:
            The StepSnapshot that was appended.
        """
        qpos = env.q.copy()
        qvel = env.qdot.copy()

        contacts = env.get_contact_forces()
        total_force = np.zeros(3)
        for c in contacts:
            total_force += np.asarray(c["force"], dtype=float)
        contact_mag = float(np.linalg.norm(total_force))

        ee_pos, _ = env.ee_pose(qpos)

        frame: Optional[np.ndarray] = None
        if self._render_env is not None and (step % self._capture_every == 0):
            frame = self._render_frame(qpos, qvel)

        snap = StepSnapshot(
            step=step,
            t=t,
            phase=phase,
            qpos=qpos,
            qvel=qvel,
            contacts=contacts,
            contact_mag=contact_mag,
            ee_pos=ee_pos,
            tank_energy=tank_energy,
            pf_clipped=pf_clipped,
            tau=np.asarray(tau, dtype=float).copy(),
            frame=frame,
        )
        self._snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------
    def _render_frame(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        """
        Mirror state into render model → mj_forward → render.

        Uses ``render_with_state(qpos, qvel)`` which preserves velocity so
        the rendered state matches the dynamics state exactly.
        """
        return self._render_env.render_with_state(qpos, qvel)

    # ------------------------------------------------------------------
    # GIF export
    # ------------------------------------------------------------------
    def render_gif(
        self,
        out_path: Path,
        fps: int = 20,
    ) -> None:
        """
        Save a GIF from the frames captured inline during simulation.

        Frames are the exact simulation states mirrored into the render
        model — NOT a separate replay or reconstruction.

        Args:
            out_path: Destination file path (.gif).
            fps:      Frames per second in the output GIF.

        Raises:
            ImportError: if Pillow is not installed.
            ValueError:  if no frames were captured.
        """
        frames = [s.frame for s in self._snapshots if s.frame is not None]
        if not frames:
            raise ValueError(
                "RolloutRecorder has no captured frames.  "
                "Pass render_env= at construction time and ensure "
                "capture_every is not larger than n_steps."
            )

        from PIL import Image

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        duration_ms = max(1, int(round(1000.0 / fps)))
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            str(out_path),
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        circle_center: Tuple[float, float],
        circle_radius: float,
        contact_z:     float,
        slide_phase:   str = "WALL_SLIDE",
    ) -> dict:
        """
        Compute circle-tracking and contact quality metrics.

        Metrics are computed only from snapshots in ``slide_phase`` (i.e.
        the WALL_SLIDE phase).  Contact fraction and force metrics cover all
        phases combined.

        Args:
            circle_center: (cx, cy) world-frame circle centre.
            circle_radius: Desired radius (m).
            contact_z:     Table surface z-coordinate.
            slide_phase:   Phase label for the circle-tracking segment.

        Returns:
            dict with keys:
              circle_tracking_rmse      — 3-D RMSE from desired circle (m)
              circle_radius_rmse        — RMSE of projected radius error (m)
              arc_completion_ratio      — fraction of full 2π arc traversed
              contact_established       — True if any contact detected in slide
              contact_maintained_fraction — fraction of all steps with contact
              mean_contact_force        — mean |F_contact| when in contact (N)
              std_contact_force         — std  |F_contact| when in contact (N)
              mean_table_height_error   — mean |ee_z − contact_z| during slide (m)
              final_phase_progress      — arc angle reached at end of slide (rad)
        """
        cx, cy = float(circle_center[0]), float(circle_center[1])

        slide_snaps = [s for s in self._snapshots if s.phase == slide_phase]
        all_snaps   = self._snapshots

        # ---- Contact metrics (all phases) --------------------------------
        contact_steps = [s for s in all_snaps if s.contact_mag > 0.1]
        contact_maintained_fraction = (
            len(contact_steps) / max(1, len(all_snaps))
        )
        contact_established = len(
            [s for s in slide_snaps if s.contact_mag > 0.1]
        ) > 0

        if contact_steps:
            forces = [s.contact_mag for s in contact_steps]
            mean_cf = float(np.mean(forces))
            std_cf  = float(np.std(forces))
        else:
            mean_cf = 0.0
            std_cf  = 0.0

        # ---- Circle-tracking metrics (slide phase only) ------------------
        if not slide_snaps:
            return {
                "circle_tracking_rmse":      float("nan"),
                "circle_radius_rmse":        float("nan"),
                "arc_completion_ratio":      0.0,
                "contact_established":       contact_established,
                "contact_maintained_fraction": contact_maintained_fraction,
                "mean_contact_force":        mean_cf,
                "std_contact_force":         std_cf,
                "mean_table_height_error":   float("nan"),
                "final_phase_progress":      0.0,
            }

        ee_positions = np.array([s.ee_pos for s in slide_snaps])   # (N,3)

        # Desired positions: x_d[i] = center + radius*[cos(ωt_i), sin(ωt_i), 0]
        # We compute the "ideal" circle position closest to each EE position
        # using the angle from the EE projected onto the table plane.
        ee_x = ee_positions[:, 0]
        ee_y = ee_positions[:, 1]
        ee_z = ee_positions[:, 2]

        # Angle of each EE point projected to table
        angles = np.arctan2(ee_y - cy, ee_x - cx)

        # Desired position for each step (on circle, at contact_z)
        x_d_arr = np.column_stack([
            cx + circle_radius * np.cos(angles),
            cy + circle_radius * np.sin(angles),
            np.full(len(angles), contact_z),
        ])

        # 3-D RMSE
        diffs = ee_positions - x_d_arr
        circle_tracking_rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))

        # Radius error (projected)
        radii = np.sqrt((ee_x - cx)**2 + (ee_y - cy)**2)
        circle_radius_rmse = float(np.sqrt(np.mean((radii - circle_radius)**2)))

        # Table height error
        mean_table_height_error = float(np.mean(np.abs(ee_z - contact_z)))

        # Arc completion: total angular change traversed
        angle_diffs = np.diff(np.unwrap(angles))
        total_angle = float(np.abs(np.sum(angle_diffs)))
        arc_completion_ratio = min(1.0, total_angle / (2.0 * np.pi))
        final_phase_progress = float(total_angle)

        return {
            "circle_tracking_rmse":        circle_tracking_rmse,
            "circle_radius_rmse":          circle_radius_rmse,
            "arc_completion_ratio":        arc_completion_ratio,
            "contact_established":         contact_established,
            "contact_maintained_fraction": contact_maintained_fraction,
            "mean_contact_force":          mean_cf,
            "std_contact_force":           std_cf,
            "mean_table_height_error":     mean_table_height_error,
            "final_phase_progress":        final_phase_progress,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def snapshots(self) -> List[StepSnapshot]:
        """All recorded snapshots in chronological order."""
        return self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def __repr__(self) -> str:
        n_frames = sum(1 for s in self._snapshots if s.frame is not None)
        return (
            f"RolloutRecorder(steps={len(self._snapshots)}, "
            f"frames={n_frames}, capture_every={self._capture_every})"
        )
