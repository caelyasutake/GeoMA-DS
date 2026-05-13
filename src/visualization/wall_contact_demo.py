"""
Wall Contact + Perturbation + Return Experiment (horizontal wall, circular trajectory).

The wall is a flat horizontal platform (parallel to the ground).  The EE presses
down onto the top surface and traces a **Cartesian circle** on it.  An external
torque then lifts the arm off; the DS + energy tank restore contact.

Key improvements over the previous version
------------------------------------------
* WALL_SLIDE uses a **task-space tracking controller** (task_tracking.py) that
  explicitly follows a Cartesian circle defined by ``circle_on_plane_reference``.
  The circle is specified in task space — not derived from joint-space interpolation.

* Tangential motion (following the circle) and normal contact regulation
  (pressing into the table) are **decoupled** via P_t / P_n projectors.

* The rollout GIF is generated from frames captured **inline** during simulation
  using ``RolloutRecorder`` — not from a post-hoc state reconstruction.  Each
  frame is rendered with exact (qpos, qvel) from the physics simulation step.

* The interactive viewer and saved GIF share the same recorded ``StepSnapshot``
  sequence — only the rendering mode differs.

Phases
------
APPROACH     PathDS drives arm from Q_READY down to the circle start.
WALL_SLIDE   Task-space controller follows the Cartesian circle on the wall.
PERTURBATION External upward torque lifts arm off the surface.
RETURN       PathDS drives arm back to the nearest circle point.

Usage::

    python -m src.visualization.wall_contact_demo
    python -m src.visualization.wall_contact_demo --viewer
    python -m src.visualization.wall_contact_demo --headless --no-animation
    python -m src.visualization.wall_contact_demo --perturb-magnitude 12.0 --perturb-joint 1

Outputs (outputs/demo/wall_contact/)::

    summary.json               — experiment metrics and phase outcomes
    metrics.png                — 6-panel figure (contact, tank, passivity, etc.)
    scenario.json              — canonical ScenarioSpec for reproducibility
    wall_contact_rollout.gif   — animated rollout from RolloutRecorder inline frames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.scenarios.scenario_builders import wall_contact_scenario
from src.scenarios.scenario_schema import ScenarioSpec
from src.simulation.env import SimEnv, SimEnvConfig
from src.simulation.panda_scene import (
    load_panda_scene, validate_panda_model, NEUTRAL_QPOS,
    apply_obstacle_friction,
)
from src.simulation.mujoco_env import MuJoCoRenderEnv, RenderConfig
from src.simulation.rollout_recorder import RolloutRecorder, StepSnapshot
from src.solver.ds.path_ds import PathDS, DSConfig
from src.solver.ds.contact_ds import (
    CircleContactConfig, circle_on_plane_reference
)
from src.solver.controller.impedance import ControllerConfig, step as ctrl_step
from src.solver.controller.task_tracking import (
    TaskTrackingConfig, task_space_step
)
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.ik.filter import PANDA_Q_MIN, PANDA_Q_MAX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 0.002     # simulation timestep (s)

PHASE_COLORS = {
    "APPROACH":     "#E3F2FD",
    "WALL_SLIDE":   "#E8F5E9",
    "PERTURBATION": "#FFEBEE",
    "RETURN":       "#FFF8E1",
}


# ---------------------------------------------------------------------------
# Per-step data record (for metrics figure — separate from StepSnapshot)
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    step: int
    t: float
    phase: str
    q: np.ndarray
    qdot: np.ndarray
    contact_force: np.ndarray   # (3,) resultant world-frame force
    contact_mag: float
    tank_energy: float
    beta_R: float
    z: float
    pf_clipped: bool
    pf_power_nom: float
    tau: np.ndarray             # actual applied torque (incl. perturbation)
    ee_pos: np.ndarray          # (3,) EE Cartesian position
    wall_dist: float            # ee_z - wall_z_surface  (positive = above, ≤0 = contact)


# ---------------------------------------------------------------------------
# Jacobian IK (position-only, pseudoinverse) — used for APPROACH seed only
# ---------------------------------------------------------------------------
def _jacobian_ik(
    env: SimEnv,
    q_init: np.ndarray,
    target_pos: np.ndarray,
    n_iter: int = 400,
    lr: float = 0.4,
    tol: float = 0.004,
    max_step: float = 0.08,
) -> Tuple[np.ndarray, float]:
    """Iterative Jacobian-pseudoinverse IK for EE position."""
    q = q_init.copy()
    final_err = np.inf
    for _ in range(n_iter):
        pos, _ = env.ee_pose(q)
        err = target_pos - pos
        final_err = float(np.linalg.norm(err))
        if final_err < tol:
            break
        J = env.jacobian(q)[:3, :]
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ (err * lr)
        dq = np.clip(dq, -max_step, max_step)
        q = np.clip(q + dq, PANDA_Q_MIN, PANDA_Q_MAX)
    return q, final_err


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
def _get_phase(step, approach_steps, slide_steps, perturb_steps, return_steps):
    if step < approach_steps:
        return "APPROACH"
    if step < approach_steps + slide_steps:
        return "WALL_SLIDE"
    if step < approach_steps + slide_steps + perturb_steps:
        return "PERTURBATION"
    return "RETURN"


def _phase_boundaries(approach_steps, slide_steps, perturb_steps, return_steps):
    a, s, p, r = approach_steps, slide_steps, perturb_steps, return_steps
    return {
        "APPROACH":     [0,           a - 1],
        "WALL_SLIDE":   [a,           a + s - 1],
        "PERTURBATION": [a + s,       a + s + p - 1],
        "RETURN":       [a + s + p,   a + s + p + r - 1],
    }


# ---------------------------------------------------------------------------
# Circle markers for MuJoCo scene
# ---------------------------------------------------------------------------
def _build_circle_markers(
    center_xy: Tuple[float, float],
    radius: float,
    z: float,
    n_pts: int,
) -> List[Tuple[np.ndarray, str]]:
    """Return (position, "circle") tuples for load_panda_scene goal_markers."""
    cx, cy = center_xy
    angles  = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    return [
        (np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), z]), "circle")
        for a in angles
    ]


# ---------------------------------------------------------------------------
# Contact force aggregation
# ---------------------------------------------------------------------------
def _aggregate_contact_force(contacts: list) -> np.ndarray:
    total = np.zeros(3)
    for c in contacts:
        total += np.asarray(c["force"], dtype=float)
    return total


# ---------------------------------------------------------------------------
# Metrics figure
# ---------------------------------------------------------------------------
def _annotate_phases(ax, phase_boundaries: dict, dt: float) -> None:
    for phase, (start, end) in phase_boundaries.items():
        t0 = start * dt
        t1 = (end + 1) * dt
        ax.axvspan(t0, t1, alpha=0.20, color=PHASE_COLORS.get(phase, "#FFF"), zorder=0)
    for phase, (start, _) in phase_boundaries.items():
        if start > 0:
            ax.axvline(start * dt, color="gray", linestyle="--",
                       linewidth=0.8, zorder=1)


def build_metrics_figure(
    log: List[StepRecord],
    phase_boundaries: dict,
    dt: float,
    scenario_name: str,
    seed: int,
    wall_z: float,
    circle_center: Tuple[float, float],
    circle_radius: float,
) -> plt.Figure:
    t_arr    = np.array([r.t           for r in log])
    contacts = np.array([r.contact_mag for r in log])
    tank_arr = np.array([r.tank_energy for r in log])
    z_arr    = np.array([r.z           for r in log])
    wdist    = np.array([r.wall_dist   for r in log])
    clipped  = np.array([r.pf_clipped  for r in log], dtype=float)
    ee_xy    = np.array([[r.ee_pos[0], r.ee_pos[1]] for r in log])
    q_hist   = np.array([r.q           for r in log])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Wall Contact Experiment (Cartesian circle, task-space ctrl) — "
        f"{scenario_name}  seed={seed}",
        fontsize=11,
    )

    # [0,0] Contact force magnitude
    ax = axes[0, 0]
    ax.plot(t_arr, contacts, color="#E53935", linewidth=0.9)
    _annotate_phases(ax, phase_boundaries, dt)
    ax.set_xlabel("t (s)"); ax.set_ylabel("|F_contact| (N)")
    ax.set_title("Contact Force Magnitude"); ax.grid(True, alpha=0.4)

    # [0,1] Tank energy
    ax = axes[0, 1]
    ax.plot(t_arr, tank_arr, color="#00897B", linewidth=0.9)
    _annotate_phases(ax, phase_boundaries, dt)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, label="s=0")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Tank energy s")
    ax.set_title("Energy Tank"); ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    # [0,2] Passivity metric z
    ax = axes[0, 2]
    ax.plot(t_arr, z_arr, color="#7B1FA2", linewidth=0.9)
    _annotate_phases(ax, phase_boundaries, dt)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("t (s)"); ax.set_ylabel("z = qdot·qdot_res")
    ax.set_title("Passivity Metric (z)"); ax.grid(True, alpha=0.4)

    # [1,0] EE Z distance to wall surface
    ax = axes[1, 0]
    ax.plot(t_arr, wdist * 1000, color="#1565C0", linewidth=0.9)
    _annotate_phases(ax, phase_boundaries, dt)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8,
               label="wall surface")
    ax.set_xlabel("t (s)"); ax.set_ylabel("EE z − wall_z  (mm)")
    ax.set_title("EE Height above Wall"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)

    # [1,1] EE trajectory in XY (top-down view) with target circle
    ax = axes[1, 1]
    phase_list = [r.phase for r in log]
    for phase, color in [
        ("APPROACH",     "#2196F3"),
        ("WALL_SLIDE",   "#4CAF50"),
        ("PERTURBATION", "#F44336"),
        ("RETURN",       "#FF9800"),
    ]:
        mask = np.array([p == phase for p in phase_list])
        if mask.any():
            ax.plot(ee_xy[mask, 0], ee_xy[mask, 1], ".", markersize=1.5,
                    color=color, label=phase)
    theta_c = np.linspace(0, 2 * np.pi, 200)
    cx, cy  = circle_center
    ax.plot(cx + circle_radius * np.cos(theta_c),
            cy + circle_radius * np.sin(theta_c),
            "k--", linewidth=1.0, label="target circle")
    ax.set_aspect("equal"); ax.set_xlabel("EE x (m)"); ax.set_ylabel("EE y (m)")
    ax.set_title("EE XY Trajectory (top-down)"); ax.legend(fontsize=6, markerscale=3)
    ax.grid(True, alpha=0.4)

    # [1,2] PF clipping ratio per phase
    ax = axes[1, 2]
    phase_names = list(phase_boundaries.keys())
    ratios = []
    for ph in phase_names:
        s_idx, e_idx = phase_boundaries[ph]
        ph_clip = clipped[s_idx:e_idx + 1]
        ratios.append(float(np.mean(ph_clip)) if len(ph_clip) > 0 else 0.0)
    colors_bar = [PHASE_COLORS.get(p, "#CCC") for p in phase_names]
    bars = ax.bar(phase_names, ratios, color=colors_bar, edgecolor="gray")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Clip ratio")
    ax.set_title("PF Clipping per Phase"); ax.tick_params(axis="x", labelsize=8)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.4, axis="y")

    patches = [mpatches.Patch(color=PHASE_COLORS[p], label=p, alpha=0.6)
               for p in PHASE_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _save_scenario_json(spec: ScenarioSpec, path: Path) -> None:
    import dataclasses

    def _ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _ser(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_ser(spec), indent=2, default=str))


# ---------------------------------------------------------------------------
# Unified rollout — single execution path for viewer and headless
# ---------------------------------------------------------------------------
def run_rollout(
    env: SimEnv,
    spec: ScenarioSpec,
    circle_cfg: CircleContactConfig,
    q_approach_target: np.ndarray,    # joint config for circle start
    circle_qs: List[np.ndarray],      # joint-space waypoints (for RETURN phase)
    tank: EnergyTank,
    ctrl_cfg: ControllerConfig,
    task_cfg: TaskTrackingConfig,
    pf_cfg: PassivityFilterConfig,
    approach_steps: int,
    slide_steps: int,
    perturb_steps: int,
    return_steps: int,
    perturb_magnitude: float,
    perturb_joint: int,
    recorder: Optional[RolloutRecorder] = None,
) -> Tuple[List[StepRecord], List[np.ndarray]]:
    """
    Single execution path for simulation.  Both viewer and headless modes
    call this function — only the recorder/rendering differs.

    APPROACH:     PathDS from q_start to circle start.
    WALL_SLIDE:   Task-space tracking controller following Cartesian circle.
    PERTURBATION: Same controller + external torque on perturb_joint.
    RETURN:       PathDS from current config back to nearest circle waypoint.

    Args:
        env:               SimEnv (initialised at q_start, qdot=0).
        spec:              ScenarioSpec (provides q_start, controller params).
        circle_cfg:        CircleContactConfig for task-space reference.
        q_approach_target: Joint config to approach as circle start (from IK).
        circle_qs:         Joint-space waypoints on circle (for RETURN DS).
        tank:              EnergyTank (mutated in place).
        ctrl_cfg:          ControllerConfig for APPROACH / RETURN phases.
        task_cfg:          TaskTrackingConfig for WALL_SLIDE phase.
        pf_cfg:            PassivityFilterConfig.
        approach_steps:    Number of steps in APPROACH phase.
        slide_steps:       Number of steps in WALL_SLIDE phase.
        perturb_steps:     Number of steps in PERTURBATION phase.
        return_steps:      Number of steps in RETURN phase.
        perturb_magnitude: External torque (Nm) on perturb_joint.
        perturb_joint:     Joint index (0-based) for perturbation.
        recorder:          Optional RolloutRecorder (captures inline frames).

    Returns:
        (log, q_history) where:
            log       — List[StepRecord] with per-step metrics.
            q_history — List[np.ndarray] of recorded joint configs (for viewer).
    """
    ctrl_params = spec.controller

    # ---- APPROACH DS: q_start → circle start ----------------------------
    approach_ds = PathDS(
        [spec.q_start, q_approach_target],
        config=DSConfig(
            K_c=ctrl_params.get("K_c", 2.0),
            K_r=ctrl_params.get("K_r", 1.0),
            K_n=ctrl_params.get("K_n", 0.3),
            goal_radius=spec.goal_radius,
        ),
    )

    total_steps  = approach_steps + slide_steps + perturb_steps + return_steps
    phase_bounds = _phase_boundaries(approach_steps, slide_steps,
                                     perturb_steps, return_steps)

    log: List[StepRecord]       = []
    q_history: List[np.ndarray] = [spec.q_start.copy()]

    q    = env.q.copy()
    qdot = env.qdot.copy()

    active_ds  = approach_ds
    prev_phase = "APPROACH"
    return_ds  = None

    # Slide time counter (resets at WALL_SLIDE start)
    t_slide = 0.0

    for i in range(total_steps):
        phase = _get_phase(i, approach_steps, slide_steps,
                           perturb_steps, return_steps)

        # ---- Phase transitions ------------------------------------------
        if phase == "WALL_SLIDE" and prev_phase == "APPROACH":
            print(f"[wall_contact] Step {i}: → WALL_SLIDE (task-space controller)")
            t_slide = 0.0

        elif phase == "RETURN" and prev_phase == "PERTURBATION":
            dists = [float(np.linalg.norm(q - cq)) for cq in circle_qs]
            nearest_idx = int(np.argmin(dists))
            return_ds = PathDS(
                [env.q.copy(), circle_qs[nearest_idx]],
                config=DSConfig(
                    K_c=ctrl_params.get("K_c", 2.0) * 1.5,
                    K_r=ctrl_params.get("K_r", 1.0) * 0.5,
                    K_n=ctrl_params.get("K_n", 0.3),
                    goal_radius=spec.goal_radius,
                ),
            )
            active_ds = return_ds
            print(f"[wall_contact] Step {i}: → RETURN DS (target WP {nearest_idx})")

        # ---- Controller -------------------------------------------------
        if phase == "WALL_SLIDE" or phase == "PERTURBATION":
            # Task-space circle tracking
            ee_pos, _ = env.ee_pose(q)
            contacts   = env.get_contact_forces()
            F_contact  = _aggregate_contact_force(contacts)
            J          = env.jacobian(q)

            ref = circle_on_plane_reference(t_slide, circle_cfg)
            res = task_space_step(
                q, qdot, ee_pos, ref, F_contact, J, tank, DT,
                config=task_cfg,
                passivity_filter_cfg=pf_cfg,
            )
            t_slide += DT

            beta_R       = res.beta_R
            z_metric     = res.z
            pf_clipped   = res.pf_clipped
            pf_power_nom = res.pf_power_nom
            tau          = res.tau.copy()
        else:
            # Joint-space PathDS (APPROACH or RETURN)
            res_jt = ctrl_step(q, qdot, active_ds, tank, DT, config=ctrl_cfg)
            beta_R       = res_jt.beta_R
            z_metric     = res_jt.z
            pf_clipped   = res_jt.pf_clipped
            pf_power_nom = res_jt.pf_power_nom
            tau          = res_jt.tau.copy()

            contacts  = env.get_contact_forces()
            F_contact = _aggregate_contact_force(contacts)
            ee_pos, _ = env.ee_pose(q)

        # ---- Perturbation -----------------------------------------------
        if phase == "PERTURBATION":
            tau[perturb_joint] += perturb_magnitude

        # ---- Physics step -----------------------------------------------
        env.step(tau, dt=DT)

        # ---- Read post-step state ---------------------------------------
        q    = env.q.copy()
        qdot = env.qdot.copy()
        q_history.append(q.copy())

        # Re-read contacts after physics step (now includes contact resolution)
        contacts     = env.get_contact_forces()
        F_contact_ps = _aggregate_contact_force(contacts)
        contact_mag  = float(np.linalg.norm(F_contact_ps))
        ee_pos_ps, _ = env.ee_pose(q)
        wall_dist    = float(ee_pos_ps[2] - circle_cfg.z_contact)

        log.append(StepRecord(
            step=i, t=i * DT, phase=phase,
            q=q.copy(), qdot=qdot.copy(),
            contact_force=F_contact_ps,
            contact_mag=contact_mag,
            tank_energy=tank.energy,
            beta_R=beta_R, z=z_metric,
            pf_clipped=pf_clipped, pf_power_nom=pf_power_nom,
            tau=tau, ee_pos=ee_pos_ps.copy(),
            wall_dist=wall_dist,
        ))

        # ---- Record snapshot for GIF ------------------------------------
        if recorder is not None:
            recorder.record_step(
                step=i, t=i * DT, phase=phase,
                env=env, tau=tau,
                tank_energy=tank.energy,
                pf_clipped=pf_clipped,
            )

        prev_phase = phase

    return log, q_history


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------
def _launch_viewer(
    spec: ScenarioSpec,
    snapshots: List[StepSnapshot],
    q_history: np.ndarray,
    speed: float = 1.0,
    obstacle_friction: Optional[Tuple[float, float, float]] = None,
    circle_markers: Optional[List[Tuple[np.ndarray, str]]] = None,
) -> None:
    try:
        import mujoco
        import mujoco.viewer as mjviewer
    except ImportError:
        print("[viewer] mujoco.viewer not available — skipping.")
        return

    viz = spec.visualization
    render_cfg = RenderConfig(
        cam_azimuth=viz.get("cam_azimuth", 130.0),
        cam_elevation=viz.get("cam_elevation", -35.0),
    )

    model  = load_panda_scene(obstacles=spec.obstacles_as_hjcd_dict(),
                              goal_markers=circle_markers or [])
    if obstacle_friction is not None:
        apply_obstacle_friction(model, obstacle_friction)
    mjdata = mujoco.MjData(model)

    # Use snapshots (exact qpos from simulation) rather than q_history
    # to ensure viewer shows the true simulated trajectory.
    use_snaps = len(snapshots) > 0

    sim_dt       = float(model.opt.timestep)
    steps_per_wp = max(1, int(round(DT / sim_dt)))
    wall_dt      = DT / max(speed, 1e-3)

    print("[viewer] Opening interactive MuJoCo viewer …")
    print("[viewer]   Drag to rotate | Scroll to zoom | Close window to continue")

    with mjviewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.lookat[:] = render_cfg.cam_lookat
        viewer.cam.distance   = render_cfg.cam_distance
        viewer.cam.azimuth    = render_cfg.cam_azimuth
        viewer.cam.elevation  = render_cfg.cam_elevation

        n_ctrl = min(7, model.nu)

        if use_snaps:
            # Replay from exact recorded qpos (preferred — matches simulation)
            qpos0 = snapshots[0].qpos
            mjdata.qpos[:min(len(qpos0), model.nq)] = qpos0
            mjdata.ctrl[:n_ctrl] = qpos0[:n_ctrl]
            mujoco.mj_forward(model, mjdata)
            viewer.sync()
            time.sleep(0.3)

            for snap in snapshots:
                if not viewer.is_running():
                    break
                qp = snap.qpos
                qv = snap.qvel
                mjdata.qpos[:min(len(qp), model.nq)] = qp
                mjdata.qvel[:min(len(qv), model.nv)] = qv
                mujoco.mj_forward(model, mjdata)
                viewer.sync()
                time.sleep(wall_dt)
        else:
            # Fallback: q_history (legacy)
            if len(q_history) > 0:
                mjdata.qpos[:n_ctrl] = q_history[0][:n_ctrl]
                mjdata.ctrl[:n_ctrl] = q_history[0][:n_ctrl]
                mujoco.mj_forward(model, mjdata)
                viewer.sync()
                time.sleep(0.3)

            for q_target in q_history:
                if not viewer.is_running():
                    break
                mjdata.ctrl[:n_ctrl] = q_target[:n_ctrl]
                for _ in range(steps_per_wp):
                    mujoco.mj_step(model, mjdata)
                viewer.sync()
                time.sleep(wall_dt)

        print("[viewer] Playback complete — close the viewer window to continue.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print("[viewer] Viewer closed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> dict:
    """
    Run the wall contact experiment and return summary dict.

    All outputs are written to ``--output-dir / wall_contact /``.
    """
    parser = argparse.ArgumentParser(
        description="Wall contact + perturbation + return (horizontal wall, Cartesian circle)"
    )
    parser.add_argument("--output-dir",        default="outputs/demo", type=Path)
    parser.add_argument("--headless",          action="store_true")
    parser.add_argument("--no-animation",      action="store_true")
    parser.add_argument("--no-frames",         action="store_true")
    parser.add_argument("--fps",               type=int,   default=20)
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--viewer",            action="store_true")
    parser.add_argument("--viewer-speed",      type=float, default=1.0)
    parser.add_argument("--approach-steps",    type=int,   default=600,
                        help="Steps for APPROACH phase")
    parser.add_argument("--slide-steps",       type=int,   default=800,
                        help="Steps for WALL_SLIDE phase (one full circle)")
    parser.add_argument("--perturb-steps",     type=int,   default=80,
                        help="Steps for PERTURBATION phase")
    parser.add_argument("--return-steps",      type=int,   default=600,
                        help="Steps for RETURN phase")
    parser.add_argument("--perturb-magnitude", type=float, default=12.0,
                        help="External torque (Nm) injected during PERTURBATION")
    parser.add_argument("--perturb-joint",     type=int,   default=1,
                        help="Joint index (0-based) to perturb; joint 1 lifts EE upward")
    parser.add_argument("--circle-radius",     type=float, default=None,
                        help="Override circle radius from scenario (metres)")
    parser.add_argument("--circle-n-pts",      type=int,   default=None,
                        help="Override number of circle waypoints (for RETURN DS seed)")
    parser.add_argument("--wall-friction",     type=float, default=None,
                        help="Sliding friction for obstacle geoms (overrides scenario). "
                             "Default: use scenario value (0.05).")
    parser.add_argument("--omega",             type=float, default=1.5,
                        help="Circle angular velocity (rad/s). "
                             "Default: 1.5 rad/s (v_tan = r·ω ≈ 0.225 m/s < v_tan_max)."
                             "Set to None-like values via code; passing 0 uses 2π/slide_duration.")
    parser.add_argument("--K-p",              type=float, default=5.0,
                        help="Task-space position gain K_p")
    parser.add_argument("--K-v",              type=float, default=1.0,
                        help="Task-space velocity gain K_v")
    parser.add_argument("--K-f",              type=float, default=0.0,
                        help="Contact force feedback gain K_f. Default 0 (disabled). "
                             "CAUTION: K_f>0 causes bouncing on rigid contacts because "
                             "MuJoCo reports F_n≈arm-weight (~50N); use --F-contact-bias instead.")
    parser.add_argument("--F-desired",        type=float, default=3.0,
                        help="Desired normal contact force (N)")
    parser.add_argument("--F-contact-bias",   type=float, default=2.0,
                        help="Constant downward bias force (N) applied via Jacobian-transpose. "
                             "Stable alternative to K_f feedback for rigid contacts. Default: 2.0 N.")
    parser.add_argument("--capture-every",    type=int,   default=5,
                        help="Capture GIF frame every N steps")
    args = parser.parse_args(argv)

    matplotlib.use("Agg")

    # ---- Scenario -----------------------------------------------------------
    spec    = wall_contact_scenario()
    out_dir = Path(args.output_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    viz            = spec.visualization
    circle_center  = tuple(viz["circle_center"])           # (cx, cy)
    circle_radius  = args.circle_radius or viz["circle_radius"]
    circle_z       = float(viz["circle_z"])                # top surface z
    circle_n_pts   = args.circle_n_pts  or viz["circle_n_pts"]

    # Angular velocity: default 1.5 rad/s → v_tan = 0.15×1.5 = 0.225 m/s < v_tan_max
    # If user passes 0.0, fall back to one full revolution over the slide phase.
    slide_duration = args.slide_steps * DT
    if args.omega == 0.0:
        omega = 2.0 * np.pi / slide_duration
    else:
        omega = args.omega

    print(f"[wall_contact] Scenario : {spec.name}")
    print(f"[wall_contact] Output   : {out_dir}")
    print(f"[wall_contact] Wall top surface: z = {circle_z:.3f} m")
    print(f"[wall_contact] Circle   : centre {circle_center}, "
          f"r={circle_radius:.3f} m, ω={omega:.3f} rad/s")
    print(f"[wall_contact] Phases   : APPROACH={args.approach_steps}  "
          f"SLIDE={args.slide_steps}  PERTURB={args.perturb_steps}  "
          f"RETURN={args.return_steps}")
    print(f"[wall_contact] Task controller: K_p={args.K_p} K_v={args.K_v} "
          f"K_f={args.K_f} F_des={args.F_desired} F_bias={args.F_contact_bias}")

    # ---- Friction -----------------------------------------------------------
    _fmap = spec.obstacle_friction_map()
    _obs_friction = next(
        (f for f in _fmap.values() if f is not None),
        None,
    )
    if args.wall_friction is not None:
        _obs_friction = (args.wall_friction, 0.005, 0.0001)
    if _obs_friction is not None:
        print(f"[wall_contact] Obstacle friction: sliding={_obs_friction[0]:.4f}")

    # ---- SimEnv -------------------------------------------------------------
    # Virtual fingertip sphere: added to panda_link7 at offset [0,0,0.212] in
    # link7 body frame.  In the contact configuration link7's +z body axis
    # points downward, so the sphere bottom lands exactly at table surface
    # (z=0.370) when link7_z≈0.587.  ee_offset_body shifts EE tracking to the
    # fingertip sphere centre (0.212 m from link7 origin in body frame).
    env = SimEnv(SimEnvConfig(
        obstacles=spec.obstacles_as_hjcd_dict(),
        timestep=DT,
        obstacle_friction=_obs_friction,
        ee_offset_body=np.array([0.0, 0.0, 0.212]),
    ))
    env.set_state(spec.q_start, np.zeros(7))
    grav_fn = env.make_gravity_fn()

    # ---- Generate approach target via Jacobian IK ---------------------------
    # Target slightly below table surface so the approach DS always drives
    # the fingertip sphere into contact (physics prevents actual penetration).
    # This guarantees contact is established before WALL_SLIDE starts.
    print("[wall_contact] Computing IK for circle start position…")
    cx, cy = circle_center
    _approach_z = circle_z - 0.010   # 10 mm below table surface (physically blocked)
    circle_start_pos = np.array([cx + circle_radius, cy, _approach_z])

    q_approach_target, err_c = _jacobian_ik(
        env, spec.q_start, circle_start_pos,
    )
    if err_c > 0.03:
        warnings.warn(
            f"Circle start IK error: {err_c:.4f} m — "
            "approach target may be inaccurate"
        )

    p_start, _ = env.ee_pose(q_approach_target)
    print(f"[wall_contact] Circle start EE: "
          f"x={p_start[0]:.3f} y={p_start[1]:.3f} z={p_start[2]:.3f} "
          f"(IK err={err_c:.4f} m)")

    # ---- Generate joint-space waypoints for RETURN DS seed ------------------
    # (used only during RETURN to find nearest waypoint — not for control)
    print(f"[wall_contact] Pre-computing {circle_n_pts} circle waypoints for RETURN…")
    angles_wp = np.linspace(0.0, 2.0 * np.pi, circle_n_pts, endpoint=False)
    circle_qs: List[np.ndarray] = []
    q_seed = q_approach_target.copy()
    for a in angles_wp:
        tgt = np.array([cx + circle_radius * np.cos(a),
                        cy + circle_radius * np.sin(a),
                        circle_z])
        q_sol, e = _jacobian_ik(env, q_seed, tgt)
        circle_qs.append(q_sol)
        q_seed = q_sol
    print(f"[wall_contact] {len(circle_qs)} circle waypoints ready.")

    # ---- Circle markers for MuJoCo scene -----------------------------------
    circle_markers = _build_circle_markers(
        circle_center, circle_radius, circle_z, circle_n_pts
    )

    # ---- Circle contact config (task-space reference) ----------------------
    circle_cfg = CircleContactConfig(
        center=np.array([cx, cy, circle_z]),
        radius=circle_radius,
        omega=omega,
        normal=np.array([0.0, 0.0, 1.0]),   # horizontal table → +z normal
        z_contact=circle_z,
    )

    # ---- Controller configs -------------------------------------------------
    ctrl_params = spec.controller
    task_cfg = TaskTrackingConfig(
        K_p=args.K_p,
        K_v=args.K_v,
        K_f=args.K_f,
        F_desired=args.F_desired,
        F_contact_bias=args.F_contact_bias,
        alpha_contact=1.0,           # full normal force regulation
        damping=ctrl_params.get("d_gain", 5.0),
        lambda_reg=0.01,
        gravity_fn=grav_fn,
        orthogonalize_residual=True,
        alpha=0.5,
        xdot_max=0.50,               # final combined cap
        v_tan_max=0.25,              # cap circle-tracking speed independently
        v_norm_max=0.20,             # cap normal approach speed independently
    )
    ctrl_cfg = ControllerConfig(
        d_gain=ctrl_params.get("d_gain", 5.0),
        f_n_gain=0.0,
        gravity_fn=grav_fn,
        orthogonalize_residual=True,
        alpha=0.5,
        passivity_filter=PassivityFilterConfig(),
    )
    pf_cfg = PassivityFilterConfig()

    tank = EnergyTank(TankConfig(
        s_init=1.0, s_min=0.01, s_max=2.0,
        epsilon_min=-0.05, epsilon_max=0.1,
    ))

    # ---- Render env for inline frame capture --------------------------------
    recorder: Optional[RolloutRecorder] = None

    if not args.no_animation:
        try:
            render_model = load_panda_scene(
                obstacles=spec.obstacles_as_hjcd_dict(),
                goal_markers=circle_markers,
            )
            if _obs_friction is not None:
                apply_obstacle_friction(render_model, _obs_friction)
            render_cfg_obj = RenderConfig(
                cam_azimuth=viz.get("cam_azimuth", 130.0),
                cam_elevation=viz.get("cam_elevation", -35.0),
            )
            render_env = MuJoCoRenderEnv(render_model, render_cfg_obj)
            recorder = RolloutRecorder(
                render_env=render_env,
                capture_every=args.capture_every,
            )
            print(f"[wall_contact] RolloutRecorder ready "
                  f"(capture every {args.capture_every} steps)")
        except Exception as exc:
            print(f"[wall_contact] WARNING: could not init render env: {exc}")
            recorder = None

    # ---- Run unified rollout ------------------------------------------------
    total_steps = (args.approach_steps + args.slide_steps
                   + args.perturb_steps + args.return_steps)
    print(f"[wall_contact] Simulating {total_steps} steps "
          f"({total_steps * DT:.1f} s)…")

    env.set_state(spec.q_start, np.zeros(7))  # reset before rollout

    log, q_history = run_rollout(
        env=env,
        spec=spec,
        circle_cfg=circle_cfg,
        q_approach_target=q_approach_target,
        circle_qs=circle_qs,
        tank=tank,
        ctrl_cfg=ctrl_cfg,
        task_cfg=task_cfg,
        pf_cfg=pf_cfg,
        approach_steps=args.approach_steps,
        slide_steps=args.slide_steps,
        perturb_steps=args.perturb_steps,
        return_steps=args.return_steps,
        perturb_magnitude=args.perturb_magnitude,
        perturb_joint=args.perturb_joint,
        recorder=recorder,
    )
    q_history_arr = np.array(q_history)
    print("[wall_contact] Simulation complete.")

    # ---- Summary metrics ----------------------------------------------------
    phase_bounds = _phase_boundaries(
        args.approach_steps, args.slide_steps,
        args.perturb_steps, args.return_steps,
    )

    def _ph(ph):
        return [r for r in log if r.phase == ph]

    slide_log   = _ph("WALL_SLIDE")
    perturb_log = _ph("PERTURBATION")
    return_log  = _ph("RETURN")

    contact_detected_slide  = any(r.contact_mag > 0.1 for r in slide_log)
    contact_restored_return = any(r.contact_mag > 0.1 for r in return_log)
    max_perturb_tau = float(max(
        (abs(r.tau[args.perturb_joint]) for r in perturb_log), default=0.0
    ))
    min_tank   = float(min(r.tank_energy for r in log))
    max_tank   = float(max(r.tank_energy for r in log))
    clip_ratio = float(np.mean([r.pf_clipped for r in log]))

    # Circle tracking metrics from RolloutRecorder (if available)
    if recorder is not None and len(recorder) > 0:
        circ_metrics = recorder.compute_metrics(
            circle_center=circle_center,
            circle_radius=circle_radius,
            contact_z=circle_z,
            slide_phase="WALL_SLIDE",
        )
    else:
        # Fallback: compute from log
        circ_metrics = _compute_circle_metrics_from_log(
            log, circle_center, circle_radius, circle_z
        )

    # Legacy metrics from log (for backwards compatibility)
    if slide_log:
        slide_radii = [
            float(np.sqrt((r.ee_pos[0] - cx)**2 + (r.ee_pos[1] - cy)**2))
            for r in slide_log
        ]
        mean_radius_err = float(np.mean(np.abs(np.array(slide_radii) - circle_radius)))
        max_height_off  = float(max(r.wall_dist for r in slide_log))
    else:
        mean_radius_err = float("nan")
        max_height_off  = float("nan")

    summary = {
        "scenario":                 spec.name,
        "n_steps":                  total_steps,
        "dt":                       DT,
        "seed":                     args.seed,
        "wall_z_surface":           circle_z,
        "circle_center":            list(circle_center),
        "circle_radius":            circle_radius,
        "circle_n_pts":             circle_n_pts,
        "omega":                    omega,
        "phase_boundaries":         phase_bounds,
        "contact_detected_slide":   contact_detected_slide,
        "contact_restored_return":  contact_restored_return,
        "perturb_magnitude":        args.perturb_magnitude,
        "perturb_joint":            args.perturb_joint,
        "max_observed_perturb_tau": max_perturb_tau,
        "min_tank_energy":          min_tank,
        "max_tank_energy":          max_tank,
        "clipped_ratio":            clip_ratio,
        "mean_circle_radius_err_m": mean_radius_err,
        "max_height_above_wall_m":  max_height_off,
        "n_scenario_obstacles":     spec.n_obstacles(),
        "n_collision_obstacles":    spec.n_collision_obstacles(),
        # --- New circle-tracking and contact metrics ---
        "circle_tracking_rmse":        circ_metrics["circle_tracking_rmse"],
        "circle_radius_rmse":          circ_metrics["circle_radius_rmse"],
        "arc_completion_ratio":        circ_metrics["arc_completion_ratio"],
        "contact_established":         circ_metrics["contact_established"],
        "contact_maintained_fraction": circ_metrics["contact_maintained_fraction"],
        "mean_contact_force":          circ_metrics["mean_contact_force"],
        "std_contact_force":           circ_metrics["std_contact_force"],
        "mean_table_height_error":     circ_metrics["mean_table_height_error"],
        "final_phase_progress":        circ_metrics["final_phase_progress"],
    }

    # ---- Save outputs -------------------------------------------------------
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[wall_contact] Saved: {out_dir / 'summary.json'}")

    _save_scenario_json(spec, out_dir / "scenario.json")
    print(f"[wall_contact] Saved: {out_dir / 'scenario.json'}")

    fig = build_metrics_figure(
        log, phase_bounds, DT, spec.name, args.seed,
        wall_z=circle_z, circle_center=circle_center, circle_radius=circle_radius,
    )
    metrics_path = out_dir / "metrics.png"
    fig.savefig(str(metrics_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[wall_contact] Saved: {metrics_path}")

    if not args.headless:
        _save_static_robot_render(spec, out_dir,
                                  obstacle_friction=_obs_friction,
                                  circle_markers=circle_markers)

    if not args.no_animation and recorder is not None:
        gif_path = out_dir / "wall_contact_rollout.gif"
        try:
            recorder.render_gif(gif_path, fps=args.fps)
            print(f"[wall_contact] Saved: {gif_path}")
            if not args.no_frames:
                _save_individual_frames(recorder, out_dir)
        except Exception as exc:
            print(f"[wall_contact] WARNING: GIF save failed: {exc}")
    elif not args.no_animation:
        # Fallback: render from q_history (legacy path)
        _save_rollout_animation_legacy(
            spec, q_history_arr, out_dir,
            fps=args.fps, no_frames=args.no_frames,
            obstacle_friction=_obs_friction,
            circle_markers=circle_markers,
        )

    if args.viewer and not args.headless:
        _launch_viewer(
            spec,
            snapshots=recorder.snapshots if recorder else [],
            q_history=q_history_arr,
            speed=args.viewer_speed,
            obstacle_friction=_obs_friction,
            circle_markers=circle_markers,
        )

    print("\n[wall_contact] Results:")
    print(f"  contact_detected_slide:        {contact_detected_slide}")
    print(f"  contact_restored_return:       {contact_restored_return}")
    print(f"  circle_tracking_rmse:          "
          f"{circ_metrics['circle_tracking_rmse']:.4f} m")
    print(f"  arc_completion_ratio:          "
          f"{circ_metrics['arc_completion_ratio']:.3f}")
    print(f"  contact_maintained_fraction:   "
          f"{circ_metrics['contact_maintained_fraction']:.3f}")
    print(f"  mean_radius_err:               {mean_radius_err * 1000:.1f} mm")
    print(f"  max_height_above_wall:         {max_height_off * 1000:.1f} mm")
    print(f"  min_tank_energy:               {min_tank:.4f}")
    print(f"  clipped_ratio:                 {clip_ratio:.3f}")

    return summary


# ---------------------------------------------------------------------------
# Fallback circle metrics from StepRecord log
# ---------------------------------------------------------------------------
def _compute_circle_metrics_from_log(
    log: List[StepRecord],
    circle_center: Tuple[float, float],
    circle_radius: float,
    contact_z: float,
    slide_phase: str = "WALL_SLIDE",
) -> dict:
    """Compute circle-tracking metrics directly from StepRecord log."""
    cx, cy = float(circle_center[0]), float(circle_center[1])
    slide_log = [r for r in log if r.phase == slide_phase]

    contact_steps = [r for r in log if r.contact_mag > 0.1]
    contact_maintained_fraction = len(contact_steps) / max(1, len(log))
    contact_established = any(r.contact_mag > 0.1 for r in slide_log)

    if contact_steps:
        forces = [r.contact_mag for r in contact_steps]
        mean_cf = float(np.mean(forces))
        std_cf  = float(np.std(forces))
    else:
        mean_cf = 0.0
        std_cf  = 0.0

    if not slide_log:
        return {
            "circle_tracking_rmse": float("nan"),
            "circle_radius_rmse": float("nan"),
            "arc_completion_ratio": 0.0,
            "contact_established": contact_established,
            "contact_maintained_fraction": contact_maintained_fraction,
            "mean_contact_force": mean_cf,
            "std_contact_force": std_cf,
            "mean_table_height_error": float("nan"),
            "final_phase_progress": 0.0,
        }

    ee_pos = np.array([r.ee_pos for r in slide_log])
    ee_x, ee_y, ee_z = ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2]

    angles = np.arctan2(ee_y - cy, ee_x - cx)
    x_d = np.column_stack([
        cx + circle_radius * np.cos(angles),
        cy + circle_radius * np.sin(angles),
        np.full(len(angles), contact_z),
    ])
    diffs = ee_pos - x_d
    rmse = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))

    radii = np.sqrt((ee_x - cx)**2 + (ee_y - cy)**2)
    radius_rmse = float(np.sqrt(np.mean((radii - circle_radius)**2)))

    height_err = float(np.mean(np.abs(ee_z - contact_z)))

    angle_diffs = np.diff(np.unwrap(angles))
    total_angle = float(np.abs(np.sum(angle_diffs)))
    arc_ratio = min(1.0, total_angle / (2.0 * np.pi))

    return {
        "circle_tracking_rmse": rmse,
        "circle_radius_rmse": radius_rmse,
        "arc_completion_ratio": arc_ratio,
        "contact_established": contact_established,
        "contact_maintained_fraction": contact_maintained_fraction,
        "mean_contact_force": mean_cf,
        "std_contact_force": std_cf,
        "mean_table_height_error": height_err,
        "final_phase_progress": total_angle,
    }


# ---------------------------------------------------------------------------
# Individual frame export from recorder
# ---------------------------------------------------------------------------
def _save_individual_frames(
    recorder: RolloutRecorder,
    out_dir: Path,
) -> None:
    try:
        from PIL import Image
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        j = 0
        for snap in recorder.snapshots:
            if snap.frame is not None:
                Image.fromarray(snap.frame).save(
                    str(frames_dir / f"frame_{j:05d}.png")
                )
                j += 1
    except ImportError:
        pass
    except Exception as exc:
        print(f"[wall_contact] WARNING: frame export failed: {exc}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _save_static_robot_render(
    spec: ScenarioSpec,
    out_dir: Path,
    obstacle_friction: Optional[Tuple[float, float, float]] = None,
    circle_markers: Optional[List[Tuple[np.ndarray, str]]] = None,
) -> None:
    try:
        import mujoco
        viz = spec.visualization
        render_cfg = RenderConfig(
            cam_azimuth=viz.get("cam_azimuth", 130.0),
            cam_elevation=viz.get("cam_elevation", -35.0),
        )
        model = load_panda_scene(obstacles=spec.obstacles_as_hjcd_dict(),
                                 goal_markers=circle_markers or [])
        if obstacle_friction is not None:
            apply_obstacle_friction(model, obstacle_friction)
        info  = validate_panda_model(model)
        if not info["panda_detected"]:
            raise RuntimeError("Panda bodies not detected.")
        render_env = MuJoCoRenderEnv(model, render_cfg)
        frame = render_env.render_at(NEUTRAL_QPOS)
        from PIL import Image
        path = out_dir / "robot_static_check.png"
        Image.fromarray(frame).save(str(path))
        print(f"[wall_contact] Saved: {path}")
    except ImportError:
        print("[wall_contact] PIL not available — skipping static render")
    except Exception as exc:
        print(f"[wall_contact] WARNING: static render failed: {exc}")


def _save_rollout_animation_legacy(
    spec: ScenarioSpec,
    q_history: np.ndarray,
    out_dir: Path,
    fps: int = 20,
    no_frames: bool = False,
    every_n: int = 5,
    obstacle_friction: Optional[Tuple[float, float, float]] = None,
    circle_markers: Optional[List[Tuple[np.ndarray, str]]] = None,
) -> None:
    """Legacy animation path (renders from q only — no velocity state)."""
    try:
        from PIL import Image
        import mujoco
        viz = spec.visualization
        render_cfg = RenderConfig(
            cam_azimuth=viz.get("cam_azimuth", 130.0),
            cam_elevation=viz.get("cam_elevation", -35.0),
        )
        model = load_panda_scene(obstacles=spec.obstacles_as_hjcd_dict(),
                                 goal_markers=circle_markers or [])
        if obstacle_friction is not None:
            apply_obstacle_friction(model, obstacle_friction)
        render_env = MuJoCoRenderEnv(model, render_cfg)

        indices = list(range(0, len(q_history), every_n))
        if (len(q_history) - 1) not in indices:
            indices.append(len(q_history) - 1)

        frames = [render_env.render_at(q_history[i]) for i in indices]

        gif_path = out_dir / "wall_contact_rollout.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(gif_path), save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        print(f"[wall_contact] Saved (legacy): {gif_path}")

        if not no_frames:
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(frames):
                Image.fromarray(frame).save(
                    str(frames_dir / f"frame_{j:05d}.png"))

    except ImportError:
        print("[wall_contact] PIL not available — skipping rollout animation")
    except Exception as exc:
        print(f"[wall_contact] WARNING: rollout animation failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
