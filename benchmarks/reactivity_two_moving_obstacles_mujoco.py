"""
GeoMA-DS Reactivity Demo — Single Moving Obstacle, MuJoCo Rendered.

Proves obstacle reactivity with IK attractors frozen at t=0.
One sphere oscillates in Z through the arm's workspace height while the arm
performs a Y-sweep reach (I-barrier base, no static obstacles).

Run:
    conda run -n mj python benchmarks/reactivity_two_moving_obstacles_mujoco.py

Output:
    outputs/final/mujoco/reactivity_two_moving_obstacles/
        reactivity_two_moving_obstacles.gif   (primary)
        reactivity_two_moving_obstacles.mp4   (if libx264 available)
        metrics.json
        summary.md
        start.png / mid_1.png / mid_2.png / final.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco

from src.scenarios.scenario_builders import SCENARIO_REGISTRY, ScenarioSpec
from src.solver.ds.factory import ik_goals_to_attractors, _make_clearance_fn
from src.solver.ds.geo_multi_attractor_ds import (
    GeoMultiAttractorDS,
    GeoMultiAttractorDSConfig,
)
from src.solver.ik.coverage_expansion import _grasptarget_pos
from src.solver.planner.collision import _panda_link_positions, _LINK_RADII
from src.simulation.panda_scene import load_panda_scene

OUT_DIR = Path("outputs/final/mujoco/reactivity_two_moving_obstacles")

# ---------------------------------------------------------------------------
# Sphere parameters.
#
# The arm sweeps in Y (≈−0.24 → +0.24 m) with wrist near X≈0.49, Z≈0.40.
# The sphere is placed at X=0.48 (same X as the goal target), Y=0.0 (centre
# of the arm's Y sweep), and oscillates in Z from Z_CENTER−AMP to Z_CENTER+AMP,
# blocking the arm's straight-line path each half-period.
# ---------------------------------------------------------------------------
_R        = 0.065          # sphere radius, m
_SPH_X    = 0.48           # fixed X — same as goal target X
_SPH_Y    = 0.0            # fixed Y — centre of arm's Y sweep
_Z_CENTER = 0.40           # Z midpoint = arm trajectory height
_AMP      = 0.22           # Z oscillation amplitude (sweeps Z_CENTER ± AMP)
_PERIOD   = 250            # steps per full oscillation (5 s at dt=0.02)

# Motion law: z(step) = Z_CENTER + AMP * sin(2π * step / T + φ)
# φ = +π/2 → z(0) = Z_CENTER + AMP = 0.62  (starts at top)
_PHI = np.pi / 2

# Simulation
_DT       = 0.02
_MAX_SPD  = 2.0
_LINK_R   = np.array([_LINK_RADII.get(i, 0.08) for i in range(8)], dtype=float)
_TASK_THRESH_M = 0.05

# Camera — same as single-sphere mujoco demo
_CAM_LOOKAT    = (0.48, 0.0, 0.42)
_CAM_DISTANCE  = 1.8
_CAM_AZIMUTH   = 180.0
_CAM_ELEVATION = -15.0
_H, _W = 480, 640

_SPH_RGBA = [0.20, 0.80, 0.90, 0.95]   # teal


# ---------------------------------------------------------------------------
# Sphere position
# ---------------------------------------------------------------------------

def _sph_center(step: int) -> np.ndarray:
    z = _Z_CENTER + _AMP * np.sin(2.0 * np.pi * step / _PERIOD + _PHI)
    return np.array([_SPH_X, _SPH_Y, float(z)])


# ---------------------------------------------------------------------------
# Dynamic clearance functions — close over sphere_state["c"]
# ---------------------------------------------------------------------------

def _make_dynamic_fns(sphere_state: dict):
    """
    Return (clearance_fn, batch_from_lp_fn) that read sphere_state["c"] on every call.
    """
    def _link_sphere_cl(lp, center: np.ndarray) -> float:
        return float(min(
            np.linalg.norm(np.array(p) - center) - r - _R
            for p, r in zip(lp, _LINK_R)
        ))

    def clearance_fn(q: np.ndarray) -> float:
        lp = _panda_link_positions(np.asarray(q, dtype=float))
        return _link_sphere_cl(lp, sphere_state["c"])

    def batch_from_lp_fn(lp: np.ndarray) -> np.ndarray:
        # lp: (N, 8, 3)
        c = sphere_state["c"]
        d  = np.linalg.norm(lp - c[None, None, :], axis=-1)   # (N, 8)
        return (d - _LINK_R[None, :] - _R).min(axis=-1)        # (N,)

    return clearance_fn, batch_from_lp_fn


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

def _build_spec() -> ScenarioSpec:
    base = SCENARIO_REGISTRY["frontal_i_barrier_lr_easy"]()
    return ScenarioSpec(
        name="reach_one_moving_obs",
        q_start=base.q_start,
        target_pose=base.target_pose,
        ik_goals=base.ik_goals,
        obstacles=[],
        goal_radius=base.goal_radius,
    )


def _sph_clearance_from_lp(lp, center: np.ndarray) -> float:
    return float(min(
        np.linalg.norm(np.array(p) - center) - r - _R
        for p, r in zip(lp, _LINK_R)
    ))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(spec: ScenarioSpec, n_steps: int):
    """
    Run GeoMA-DS with one sinusoidal moving sphere.

    Returns:
        q_hist   (T, 7)  joint trajectory
        sph_hist (T, 3)  sphere centre per step
        cls_all  list[float]  clearance per step
        metrics  dict
    """
    sphere_state = {"c": _sph_center(0)}
    clearance_fn, batch_fn = _make_dynamic_fns(sphere_state)

    static_cl_fn = _make_clearance_fn([])
    ik_goals     = [np.asarray(q, dtype=float) for q in spec.ik_goals]
    attractors   = ik_goals_to_attractors(ik_goals, static_cl_fn)

    cfg = GeoMultiAttractorDSConfig(enable_timing=True)
    ds  = GeoMultiAttractorDS(
        attractors,
        clearance_fn=clearance_fn,
        config=cfg,
        batch_from_lp_fn=batch_fn,
    )

    q         = spec.q_start.copy()
    q_hist    : List[np.ndarray] = [q.copy()]
    sph_hist  : List[np.ndarray] = [_sph_center(0)]
    cls_all   : List[float]      = []
    alphas    : List[float]      = []
    step_ms   : List[float]      = []
    col_cnt   = 0
    conv_step = -1
    n_sw      = 0
    prev_att  : Optional[int] = None
    react_step = -1

    t_wall = time.perf_counter()

    for step in range(n_steps):
        sphere_state["c"] = _sph_center(step)

        qdot, res = ds.compute(q, dt=_DT)

        att = int(res.active_attractor_idx)
        if prev_att is not None and att != prev_att:
            n_sw += 1
        prev_att = att

        cl = float(res.clearance)
        cls_all.append(cl)
        if cl < 0.0:
            col_cnt += 1

        alpha = float(res.obs_blend_alpha)
        alphas.append(alpha)
        if react_step < 0 and alpha > 0.0:
            react_step = step

        if res.timing_ms:
            step_ms.append(res.timing_ms.get("total_ms", 0.0))

        if res.goal_error < spec.goal_radius:
            conv_step = step
            q_hist.append(q.copy())
            sph_hist.append(_sph_center(step))
            break

        spd = float(np.linalg.norm(qdot))
        if spd > _MAX_SPD:
            qdot *= _MAX_SPD / spd
        q = q + _DT * qdot
        q_hist.append(q.copy())
        sph_hist.append(_sph_center(step + 1))

    wall_s  = time.perf_counter() - t_wall
    mean_ms = float(np.mean(step_ms)) if step_ms else 0.0
    p95_ms  = float(np.percentile(step_ms, 95)) if step_ms else 0.0

    active_att = ds.attractors[ds._active_idx]
    gt_final   = _grasptarget_pos(q)
    gt_att     = _grasptarget_pos(active_att.q_goal)
    red_sphere = np.array(spec.target_pose["position"], dtype=float)
    gt_to_red  = float(np.linalg.norm(gt_final - red_sphere))
    task_ok    = gt_to_red < _TASK_THRESH_M
    min_cl     = float(np.min(cls_all)) if cls_all else float("nan")
    success    = conv_step >= 0 and task_ok and col_cnt == 0 and min_cl > 0.0

    metrics = {
        "success":                      success,
        "convergence_step":             conv_step,
        "simulated_duration_s":         len(q_hist) * _DT,
        "wall_clock_s":                 wall_s,
        "min_clearance_m":              min_cl,
        "collision_count":              col_cnt,
        "n_attractors":                 len(ds.attractors),
        "n_switches":                   n_sw,
        "active_attractor_idx":         int(ds._active_idx),
        "active_family":                active_att.family,
        "grasptarget_to_red_sphere_m":  gt_to_red,
        "final_grasptarget_error_m":    gt_to_red,
        "att_grasptarget_to_red_m":     float(np.linalg.norm(gt_att - red_sphere)),
        "gate_converged":               conv_step >= 0,
        "gate_grasptarget_to_sphere":   task_ok,
        "gate_no_collision":            col_cnt == 0,
        "gate_positive_clearance":      min_cl > 0.0,
        "reaction_latency_steps":       react_step,
        "max_blend_alpha":              float(np.max(alphas)) if alphas else 0.0,
        "alpha_gt_0_9_fraction":        float(np.mean([a > 0.9 for a in alphas])) if alphas else 0.0,
        "mean_step_ms":                 mean_ms,
        "p95_step_ms":                  p95_ms,
        "estimated_hz":                 round(1000.0 / mean_ms) if mean_ms > 0 else 0,
        "ik_recomputed_after_t0":       False,
        "coverage_recomputed_after_t0": False,
        "clearance_fn_sees_obs":        True,
        "batch_fn_sees_obs":            True,
        "sphere_params": {
            "radius_m":      _R,
            "z_amplitude_m": _AMP,
            "z_center_m":    _Z_CENTER,
            "period_steps":  _PERIOD,
            "sph_x":         _SPH_X,
            "sph_y":         _SPH_Y,
            "phi_rad":       float(_PHI),
        },
    }
    return np.array(q_hist), np.array(sph_hist), cls_all, metrics


# ---------------------------------------------------------------------------
# MuJoCo scene — one visual-only sphere body
# ---------------------------------------------------------------------------

def _build_mujoco_env(spec: ScenarioSpec):
    """
    Load Panda scene injecting one visual sphere body and a red EE marker.

    Returns (ctx, renderer, data, model, sph_id, cam, n_arm).
    Raises RuntimeError if the body is not found by name.
    """
    ee_target = np.array(spec.target_pose["position"])
    obs_dict = {
        "sphere": {
            "sphere_1": {
                "radius":       _R,
                "pose":         [1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0],
                "_rgba":        _SPH_RGBA,
                "_visual_only": True,
            },
        }
    }
    model = load_panda_scene(obstacles=obs_dict, ee_target=ee_target)

    ctx = mujoco.GLContext(_W, _H)
    ctx.make_current()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, _H, _W)

    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = list(_CAM_LOOKAT)
    cam.distance  = _CAM_DISTANCE
    cam.azimuth   = _CAM_AZIMUTH
    cam.elevation = _CAM_ELEVATION

    all_body_names = [model.body(i).name for i in range(model.nbody)]
    sphere_names   = [n for n in all_body_names if "sphere" in n.lower()]
    print(f"  Sphere-related bodies in model: {sphere_names}")

    sph_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obs_0_sphere_1")
    if sph_id < 0:
        raise RuntimeError(
            f"Sphere body lookup failed.\n"
            f"  Expected 'obs_0_sphere_1' (got id={sph_id}).\n"
            f"  All sphere-related body names: {sphere_names}\n"
            f"  Investigate panda_scene.py obs_idx counter."
        )

    n_arm = min(7, model.nq)
    return ctx, renderer, data, model, sph_id, cam, n_arm


def _set_arm(model, data, q: np.ndarray, n_arm: int) -> None:
    data.qpos[:n_arm] = np.asarray(q, dtype=float)[:n_arm]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _render_frame(
    renderer: mujoco.Renderer,
    model,
    data,
    cam,
    q: np.ndarray,
    c: np.ndarray,
    sph_id: int,
    n_arm: int,
) -> np.ndarray:
    model.body_pos[sph_id] = c.copy()
    _set_arm(model, data, q, n_arm)
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def _build_model_for_viewer(spec: ScenarioSpec):
    """Build MuJoCo model+data with one visual sphere — no offscreen renderer."""
    ee_target = np.array(spec.target_pose["position"])
    obs_dict = {
        "sphere": {
            "sphere_1": {
                "radius":       _R,
                "pose":         [1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0],
                "_rgba":        _SPH_RGBA,
                "_visual_only": True,
            },
        }
    }
    model  = load_panda_scene(obstacles=obs_dict, ee_target=ee_target)
    data   = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sph_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obs_0_sphere_1")
    if sph_id < 0:
        raise RuntimeError("Sphere body 'obs_0_sphere_1' not found in model.")
    return model, data, sph_id, min(7, model.nq)


def _build_ds(spec: ScenarioSpec, sphere_state: dict):
    """Construct GeoMultiAttractorDS with frozen attractors and live closure fns."""
    clearance_fn, batch_fn = _make_dynamic_fns(sphere_state)
    static_cl_fn = _make_clearance_fn([])
    ik_goals   = [np.asarray(q, dtype=float) for q in spec.ik_goals]
    attractors = ik_goals_to_attractors(ik_goals, static_cl_fn)
    cfg = GeoMultiAttractorDSConfig(enable_timing=True)
    return GeoMultiAttractorDS(
        attractors,
        clearance_fn=clearance_fn,
        config=cfg,
        batch_from_lp_fn=batch_fn,
    )


def run_interactive(spec: ScenarioSpec, n_steps: int, speed: float = 1.0) -> None:
    """
    Run the demo live in a MuJoCo interactive viewer.

    Camera orbit/pan/zoom via mouse. Press Ctrl-C in the terminal to quit.
    The simulation loops automatically after convergence.

    Args:
        speed: playback multiplier (1.0 = real-time, 2.0 = 2× faster).
    """
    try:
        import mujoco.viewer as _mjviewer
    except ImportError:
        print("  [error] mujoco.viewer not available (needs mujoco>=3.0 with display)")
        return

    model, data, sph_id, n_arm = _build_model_for_viewer(spec)
    target_dt = _DT / max(speed, 0.01)

    print(f"  Opening interactive viewer  (speed={speed}x  Ctrl-C to quit)")
    print(f"  Mouse: left-drag=orbit  right-drag=pan  scroll=zoom")

    sphere_state: dict = {"c": _sph_center(0)}

    with _mjviewer.launch_passive(model, data) as handle:
        run_idx = 0
        while handle.is_running():
            run_idx += 1
            sphere_state["c"] = _sph_center(0)
            ds = _build_ds(spec, sphere_state)
            q  = spec.q_start.copy()
            print(f"  [run {run_idx}] start", flush=True)

            for step in range(n_steps):
                if not handle.is_running():
                    break
                t0 = time.perf_counter()

                sphere_state["c"] = _sph_center(step)
                model.body_pos[sph_id] = sphere_state["c"]

                qdot, res = ds.compute(q, dt=_DT)
                spd = float(np.linalg.norm(qdot))
                if spd > _MAX_SPD:
                    qdot *= _MAX_SPD / spd

                data.qpos[:n_arm] = q[:n_arm]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                handle.sync()

                if res.goal_error < spec.goal_radius:
                    print(
                        f"  [run {run_idx}] converged  step={step}"
                        f"  min_cl={res.clearance:.4f}"
                        f"  alpha={res.obs_blend_alpha:.3f}",
                        flush=True,
                    )
                    hold_end = time.perf_counter() + 1.5
                    while handle.is_running() and time.perf_counter() < hold_end:
                        handle.sync()
                        time.sleep(0.016)
                    break

                q = q + _DT * qdot

                elapsed = time.perf_counter() - t0
                slack   = target_dt - elapsed
                if slack > 0:
                    time.sleep(slack)


# ---------------------------------------------------------------------------
# Video / snapshot / metrics helpers
# ---------------------------------------------------------------------------

def _save_gif(frames: List[np.ndarray], path: Path, fps: int) -> Optional[str]:
    try:
        from PIL import Image
        gif = path.with_suffix(".gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(gif), save_all=True, append_images=imgs[1:],
                     duration=max(1, int(1000 / fps)), loop=0)
        print(f"  -> {gif}")
        return str(gif)
    except Exception as e:
        print(f"  [warn] GIF save failed: {e}")
        return None


def _save_mp4(frames: List[np.ndarray], path: Path, fps: int) -> Optional[str]:
    try:
        import imageio
        mp4 = path.with_suffix(".mp4")
        writer = imageio.get_writer(str(mp4), fps=fps, codec="libx264",
                                    pixelformat="yuv420p", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  -> {mp4}")
        return str(mp4)
    except Exception as e:
        print(f"  [warn] MP4 save failed: {e}")
        return None


def _save_snapshot(frame: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(frame).save(str(path))
        print(f"  -> {path}")
    except Exception as e:
        print(f"  [warn] snapshot save failed: {e}")


def _save_metrics(metrics: dict, path: Path) -> None:
    def _fix(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and (v != v or abs(v) == float("inf")):
            return str(v)
        if isinstance(v, dict):
            return {k: _fix(w) for k, w in v.items()}
        return v
    clean = {k: _fix(v) for k, v in metrics.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"  -> {path}")


def _save_summary(
    metrics: dict,
    gif_path: Optional[str],
    mp4_path: Optional[str],
    out_dir: Path,
) -> None:
    m  = metrics
    sp = m.get("sphere_params", {})
    conv   = m.get("convergence_step", -1)
    conv_s = str(conv) if conv >= 0 else "DNF"
    lines = [
        "# GeoMA-DS Reactivity Demo — Single Moving Obstacle",
        "",
        "**Proves obstacle reactivity** with frozen IK attractors.",
        "One sphere oscillates in Z through the arm's workspace height.",
        "IK attractors built once at t=0.",
        "",
        "## Setup",
        "",
        "| Property | Value |",
        "|:---------|------:|",
        f"| Scenario | reach_one_moving_obs (I-barrier base, no static obstacles) |",
        f"| Sphere radius | {sp.get('radius_m', _R)} m |",
        f"| Sphere position | X={sp.get('sph_x', _SPH_X)}, Y={sp.get('sph_y', _SPH_Y)} m (fixed) |",
        f"| Z oscillation | center={sp.get('z_center_m', _Z_CENTER)} m, ±{sp.get('z_amplitude_m', _AMP)} m |",
        f"| Period | {sp.get('period_steps', _PERIOD)} steps "
        f"({sp.get('period_steps', _PERIOD) * _DT:.1f} s) |",
        f"| Phase | +π/2 (starts Z=top) |",
        f"| IK recomputed at t>0 | NO |",
        f"| Coverage recomputed at t>0 | NO |",
        f"| clearance_fn sees obs | YES |",
        f"| batch_fn sees obs | YES |",
        f"| Renderer | MuJoCo 3-D ({mujoco.__version__}) |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|:-------|------:|",
        f"| Success | {'yes' if m.get('success') else 'no'} |",
        f"| Convergence step | {conv_s} |",
        f"| Simulated duration | {m.get('simulated_duration_s', 0):.2f} s |",
        f"| Min clearance | {m.get('min_clearance_m', float('nan')):.4f} m |",
        f"| Collision count | {m.get('collision_count', 0)} |",
        f"| Reaction latency | {m.get('reaction_latency_steps', -1)} steps |",
        f"| Max blend alpha | {m.get('max_blend_alpha', 0):.3f} |",
        f"| Fraction alpha > 0.9 | {m.get('alpha_gt_0_9_fraction', 0):.3f} |",
        f"| n_attractors | {m.get('n_attractors', 0)} |",
        f"| n_switches | {m.get('n_switches', 0)} |",
        f"| Mean step time | {m.get('mean_step_ms', 0):.2f} ms ({m.get('estimated_hz', 0)} Hz) |",
        f"| P95 step time | {m.get('p95_step_ms', 0):.2f} ms |",
        f"| grasptarget→goal | {m.get('final_grasptarget_error_m', float('nan')):.4f} m |",
        "",
    ]
    if gif_path:
        lines.append(f"**GIF:** `{gif_path}`")
    if mp4_path:
        lines.append(f"**MP4:** `{mp4_path}`")
    lines += ["", f"_Generated: {time.strftime('%Y-%m-%d %H:%M')}_"]
    p = out_dir / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",       type=int,   default=800)
    parser.add_argument("--fps",         type=int,   default=25)
    parser.add_argument("--every-n",     type=int,   default=2,
                        help="Render every Nth simulation step")
    parser.add_argument("--interactive", action="store_true",
                        help="Open live MuJoCo viewer instead of saving GIF/MP4")
    parser.add_argument("--speed",       type=float, default=1.0,
                        help="Playback speed multiplier for --interactive (default 1.0)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== GeoMA-DS Reactivity Demo — Single Moving Obstacle ===")
    print(f"  mujoco {mujoco.__version__}  PLUGIN_HANDLES={len(mujoco.PLUGIN_HANDLES)}")
    print(f"  Sphere (teal):  X={_SPH_X}  Y={_SPH_Y}  Z_ctr={_Z_CENTER}  "
          f"A=±{_AMP}  φ=+π/2  → starts Z={_Z_CENTER+_AMP:.2f}")
    print(f"  Period: T={_PERIOD} steps ({_PERIOD * _DT:.1f} s),  "
          f"dt={_DT} s,  max_steps={args.steps}")
    print()

    spec = _build_spec()

    if args.interactive:
        run_interactive(spec, args.steps, speed=args.speed)
        return

    # --- Simulation ---
    print("  Running simulation ...")
    q_hist, sph_hist, cls_all, metrics = run_sim(spec, args.steps)
    conv   = metrics["convergence_step"]
    conv_s = str(conv) if conv >= 0 else "DNF"
    print(f"  conv={conv_s}  col={metrics['collision_count']}  "
          f"min_cl={metrics['min_clearance_m']:.4f}  "
          f"alpha_max={metrics['max_blend_alpha']:.3f}  "
          f"hz={metrics['estimated_hz']}")

    print()
    print("  Acceptance check:")
    print(f"    collision_count == 0       : {metrics['collision_count'] == 0}")
    print(f"    min_clearance > 0          : {metrics['min_clearance_m'] > 0}")
    print(f"    target reached             : {metrics['gate_converged']}")
    print(f"    grasptarget_error < 0.05 m : {metrics['final_grasptarget_error_m'] < 0.05}")
    print(f"    IK frozen at t=0           : True (frozen attractors)")
    print(f"    coverage frozen at t=0     : True")
    print(f"    success                    : {metrics['success']}")

    # --- MuJoCo env ---
    print()
    print("  Building MuJoCo render environment ...")
    ctx, renderer, data, model, sph_id, cam, n_arm = _build_mujoco_env(spec)
    print(f"  sph_body_id={sph_id}  nq={model.nq}  n_arm={n_arm}")

    # --- Render frames ---
    print("  Rendering frames ...")
    indices = list(range(0, len(q_hist), args.every_n))
    if len(q_hist) - 1 not in indices:
        indices.append(len(q_hist) - 1)

    frames: List[np.ndarray] = []
    t_render = time.perf_counter()
    for i, idx in enumerate(indices):
        c  = sph_hist[min(idx, len(sph_hist) - 1)]
        fr = _render_frame(renderer, model, data, cam, q_hist[idx], c, sph_id, n_arm)
        frames.append(fr)
        if (i + 1) % 20 == 0 or i + 1 == len(indices):
            print(f"    {i+1}/{len(indices)} frames", end="\r")
    print()
    t_frames = time.perf_counter() - t_render
    print(f"  Rendered {len(frames)} frames in {t_frames:.1f}s "
          f"({t_frames / len(frames) * 1000:.0f} ms/frame)")

    if max(f.mean() for f in frames) < 2.0:
        print("  [ERROR] All frames are black — OpenGL context not rendering.")
        renderer.close()
        return

    # --- Snapshots ---
    print()
    print("  Saving snapshots ...")
    snap_steps = {
        "start": 0,
        "mid_1": _PERIOD // 4,
        "mid_2": _PERIOD // 2,
        "final": len(q_hist) - 1,
    }
    for label, step_idx in snap_steps.items():
        idx = min(step_idx, len(q_hist) - 1)
        c   = sph_hist[min(idx, len(sph_hist) - 1)]
        fr  = _render_frame(renderer, model, data, cam, q_hist[idx], c, sph_id, n_arm)
        _save_snapshot(fr, OUT_DIR / f"{label}.png")

    renderer.close()

    # --- Video ---
    print()
    print("  Saving video ...")
    vid_path = OUT_DIR / "reactivity_two_moving_obstacles"
    gif_path = _save_gif(frames, vid_path, args.fps)
    mp4_path = _save_mp4(frames, vid_path, args.fps)

    # --- Metrics / summary ---
    print()
    print("  Saving metrics/summary ...")
    _save_metrics(metrics, OUT_DIR / "metrics.json")
    _save_summary(metrics, gif_path, mp4_path, OUT_DIR)

    print()
    print(f"  Done. Outputs in {OUT_DIR}/")
    if gif_path:
        print(f"  GIF : {gif_path}")
    if mp4_path:
        print(f"  MP4 : {mp4_path}")
    print(f"  success={metrics['success']}")


if __name__ == "__main__":
    main()
