"""
GeoMA-DS MuJoCo Visual Demos.

Runs four GeoMA-DS scenarios in MuJoCo simulation, records q_history, renders
frames off-screen, and exports PNG snapshots, GIF/MP4, metrics.json, and
summary.md for each demo.

Demos:
  A — I-barrier success           (frontal_i_barrier_lr_easy)
  B — Cross baseline failure      (frontal_cross_barrier_easy, no expansion)
  C — Cross + expansion success   (frontal_cross_barrier_easy, with expansion)
  D — Reactivity: moving sphere descending through wrist path (I-barrier)

Output root: outputs/final/mujoco/

Usage:
    python -m benchmarks.final_geo_ma_mujoco_demos
    python -m benchmarks.final_geo_ma_mujoco_demos --steps 800 --fps 30
    python -m benchmarks.final_geo_ma_mujoco_demos --every-n 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.scenarios.scenario_builders import SCENARIO_REGISTRY
from src.scenarios.scenario_schema import ScenarioSpec
from src.simulation.mujoco_env import MuJoCoRenderEnv, RenderConfig
from src.simulation.panda_scene import load_panda_scene
from src.solver.ds.factory import _make_clearance_fn, build_geo_multi_attractor_ds
from src.solver.ds.geo_multi_attractor_ds import GeoMultiAttractorDSConfig
from src.solver.ik.coverage_expansion import CoverageConfig
from src.solver.planner.collision import (
    _panda_link_positions, _panda_hand_transform, _LINK_RADII,
)
import benchmarks.reactivity_moving_obstacle_mujoco as _rmo

OUT_ROOT = Path("outputs/final/mujoco")

_DT                  = 0.02
_MAX_SPD             = 2.0
_LINK_RADII_L        = [float(_LINK_RADII.get(i, 0.08)) for i in range(8)]
_GRASPTARGET_OFFSET  = np.array([0.0, 0.0, 0.105])  # panda_hand -> panda_grasptarget in hand Z
_TASK_THRESH_M       = 0.05  # grasptarget-to-sphere success threshold

# Reactivity sphere
_SPHERE_POS    = np.array([0.0, 0.52, 0.50])
_SPHERE_RADIUS = 0.08
_T_OBS_START   = 130

# Camera: elevated 3/4 view, consistent across all demos
_CAM = RenderConfig(
    height=480, width=640,
    cam_lookat=(0.3, 0.0, 0.4),
    cam_distance=2.2,
    cam_azimuth=140.0,
    cam_elevation=-25.0,
)

# Sphere obstacle colour (bright teal — easy to see against orange obstacles)
_SPHERE_RGBA = [0.20, 0.80, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Scene / render environment construction
# ---------------------------------------------------------------------------

def _grasptarget_pos(q: np.ndarray) -> np.ndarray:
    """panda_grasptarget = panda_hand + R_hand @ [0, 0, 0.105]."""
    hp, R = _panda_hand_transform(q)
    return np.array(hp, dtype=float) + R @ _GRASPTARGET_OFFSET


def _ee_target_pos(spec: ScenarioSpec) -> Optional[np.ndarray]:
    """
    Extract grasptarget world position from ScenarioSpec.target_pose.

    HJCD-IK solved for panda_grasptarget; target_pose["position"] is the
    grasptarget position (the red sphere).  panda_hand is 0.105m below it
    along the hand's local Z axis.
    """
    if spec.target_pose is None:
        return None
    pos = spec.target_pose.get("position")
    return np.array(pos, dtype=float) if pos is not None else None


def _hand_target_from_ik_goals(spec: ScenarioSpec) -> Optional[np.ndarray]:
    """
    Mean panda_hand position of spec.ik_goals (all solve for same target, elbow_fwd family).
    Used to place the orange hand-frame marker in the scene.
    Returns None if no ik_goals.
    """
    if not spec.ik_goals:
        return None
    from src.solver.planner.collision import _panda_link_positions
    hand_positions = [np.array(_panda_link_positions(q)[-1], dtype=float)
                      for q in spec.ik_goals[:4]]  # use first 4 to be fast
    return np.mean(hand_positions, axis=0)


def _build_render_env(
    spec: ScenarioSpec,
    extra_obstacles: Optional[Dict] = None,
) -> MuJoCoRenderEnv:
    """
    Build a MuJoCoRenderEnv from a ScenarioSpec.

    Markers injected:
      - Red sphere (from marker_goal in spec.obstacles): grasptarget target position.
        HJCD-IK solved for panda_grasptarget; this is the correct success target.
      - Orange sphere (hand_frame_marker): mean panda_hand position across IK goals
        (= grasptarget_pos - 0.105m in hand Z, approximately [0.48, 0.24, 0.45]).

    Args:
        spec:             ScenarioSpec with obstacles.
        extra_obstacles:  Additional obstacles to inject (e.g., the reactivity
                          sphere). Dict in panda_scene obstacle format.

    Returns:
        MuJoCoRenderEnv ready to render frames.
    """
    obs_dict = spec.obstacles_as_panda_scene_dict()
    if extra_obstacles:
        for geom_type, entries in extra_obstacles.items():
            obs_dict.setdefault(geom_type, {}).update(entries)

    # Add orange hand-frame marker so both EE frames are visible in the scene.
    hand_tgt = _hand_target_from_ik_goals(spec)
    if hand_tgt is not None:
        x, y, z = float(hand_tgt[0]), float(hand_tgt[1]), float(hand_tgt[2])
        obs_dict.setdefault("sphere", {})["hand_frame_marker"] = {
            "radius": 0.018,
            "pose": [x, y, z, 1.0, 0.0, 0.0, 0.0],
            "_rgba": [1.00, 0.55, 0.05, 0.90],   # orange
            "_visual_only": True,
        }

    # ee_target = grasptarget position (the red sphere shown by load_panda_scene)
    ee_target = _ee_target_pos(spec)
    model = load_panda_scene(obstacles=obs_dict, ee_target=ee_target)
    return MuJoCoRenderEnv(model, _CAM)


def _build_reactivity_sphere_dict() -> Dict:
    """Return the panda_scene obstacle dict for the surprise sphere."""
    x, y, z = _SPHERE_POS
    return {
        "sphere": {
            "surprise_sphere": {
                "radius": _SPHERE_RADIUS,
                "pose": [x, y, z, 1.0, 0.0, 0.0, 0.0],
                "_rgba": _SPHERE_RGBA,
            }
        }
    }


# ---------------------------------------------------------------------------
# GeoMA-DS simulation (pure numpy — no MuJoCo physics)
# ---------------------------------------------------------------------------

def _run_geo_ma(
    spec: ScenarioSpec,
    use_expansion: bool,
    n_steps: int,
) -> Tuple[np.ndarray, List[float], List[int], dict]:
    """
    Run GeoMA-DS on spec for up to n_steps.

    Returns:
        q_history:   (T, 7) joint trajectory
        clearances:  list of clearance at each step
        att_idxs:    active attractor index at each step
        metrics:     raw numbers dict (conv_step, min_clearance, etc.)
    """
    cfg     = GeoMultiAttractorDSConfig(enable_timing=True)
    cc      = CoverageConfig(verbose=False) if use_expansion else None
    t_plan0 = time.perf_counter()
    ds      = build_geo_multi_attractor_ds(spec, config=cfg, coverage_config=cc)
    planner_s = time.perf_counter() - t_plan0

    q       = spec.q_start.copy()
    q_hist  = [q.copy()]
    cls     = []
    att_i   = []
    step_ms = []
    col_cnt = 0
    n_sw    = 0
    prev_att: Optional[int] = None
    conv_step = -1
    switch_steps: List[int] = []
    last_res = None

    t_wall = time.perf_counter()
    for step in range(n_steps):
        qdot, res = ds.compute(q, dt=_DT)
        last_res = res

        att = int(res.active_attractor_idx)
        if prev_att is not None and att != prev_att:
            n_sw += 1
            switch_steps.append(step)
        prev_att = att
        att_i.append(att)

        cl = float(res.clearance)
        cls.append(cl)
        if cl < 0.0:
            col_cnt += 1

        if res.timing_ms:
            step_ms.append(res.timing_ms.get("total_ms", 0.0))

        if res.goal_error < spec.goal_radius:
            conv_step = step
            q_hist.append(q.copy())
            break

        spd = float(np.linalg.norm(qdot))
        if spd > _MAX_SPD:
            qdot *= _MAX_SPD / spd
        q = q + _DT * qdot
        q_hist.append(q.copy())

    run_s  = time.perf_counter() - t_wall
    wall_s = run_s  # kept for backward compat field name

    mean_ms = float(np.mean(step_ms)) if step_ms else 0.0
    p95_ms  = float(np.percentile(step_ms, 95)) if step_ms else 0.0

    n_steps_run   = len(step_ms)
    control_hz    = round(1000.0 / mean_ms) if mean_ms > 0 else 0
    total_s       = planner_s + run_s
    end_to_end_hz = round(float(n_steps_run) / total_s) if total_s > 0 else 0
    planner_ms    = planner_s * 1000.0

    # ---- Frame-corrected final metrics ----
    # HJCD-IK solved for panda_grasptarget. target_pose["position"] = grasptarget target = red sphere.
    # panda_hand is 0.105m below the red sphere along hand-frame Z.
    active_att = ds.attractors[ds._active_idx]
    q_goal_act = active_att.q_goal
    err_active = float(last_res.goal_error) if last_res is not None else float("nan")
    err_any    = min(float(np.linalg.norm(q - att.q_goal)) for att in ds.attractors)

    # Grasptarget positions
    gt_final  = _grasptarget_pos(q)
    gt_att    = _grasptarget_pos(q_goal_act)
    red_sphere = np.array(spec.target_pose["position"]) if spec.target_pose else gt_att

    grasptarget_to_sphere     = float(np.linalg.norm(gt_final - red_sphere))
    att_grasptarget_to_sphere = float(np.linalg.norm(gt_att   - red_sphere))

    # panda_hand positions (reference, not the success criterion)
    hand_q = np.array(_panda_link_positions(q)[-1])
    hand_g = np.array(_panda_link_positions(q_goal_act)[-1])
    hand_to_hand = float(np.linalg.norm(hand_q - hand_g))

    min_cl = float(np.min(cls)) if cls else float("nan")
    task_ok = grasptarget_to_sphere < _TASK_THRESH_M
    success_verified = (conv_step >= 0 and task_ok and col_cnt == 0 and min_cl > 0.0)

    metrics = {
        # ---- validity ----
        "success":                                  success_verified,
        "convergence_step":                         conv_step,
        "converged_by_metric":                      "joint_error_to_active_att" if conv_step >= 0 else "none",
        # ---- timing ----
        "simulated_duration_s":                     len(q_hist) * _DT,
        "wall_clock_render_s":                      wall_s,
        "planner_ms":                               planner_ms,
        "planner_s":                                planner_s,
        "run_s":                                    run_s,
        "total_s":                                  total_s,
        "control_hz":                               control_hz,
        "end_to_end_hz":                            end_to_end_hz,
        # ---- clearance / collision ----
        "min_clearance_m":                          min_cl,
        "collision_count":                          col_cnt,
        # ---- attractors ----
        "n_attractors":                             len(ds.attractors),
        "n_switches":                               n_sw,
        "active_attractor_idx":                     int(ds._active_idx),
        "active_family":                            active_att.family,
        # ---- joint-space error ----
        "joint_goal_error_to_active_attractor":     err_active,
        "min_joint_goal_error_to_any_attractor":    err_any,
        # ---- task-space error (frame-corrected) ----
        # HJCD-IK solved for panda_grasptarget; red sphere = grasptarget target.
        "grasptarget_to_red_sphere_m":              grasptarget_to_sphere,
        "att_grasptarget_to_red_sphere_m":          att_grasptarget_to_sphere,
        "hand_to_hand_m":                           hand_to_hand,
        # ---- success gates ----
        "gate_converged":                           conv_step >= 0,
        "gate_grasptarget_to_sphere":               task_ok,
        "gate_no_collision":                        col_cnt == 0,
        "gate_positive_clearance":                  min_cl > 0.0,
        # ---- step timing (legacy + new) ----
        "mean_step_ms":                             mean_ms,
        "p95_step_ms":                              p95_ms,
        "estimated_hz":                             control_hz,   # legacy alias
        # ---- internal (not written to metrics.json) ----
        "_switch_steps":                            switch_steps,
    }
    return np.array(q_hist), cls, att_i, metrics


def _run_reactivity(n_steps: int) -> Tuple[np.ndarray, List[float], List[float], dict]:
    """
    Reactivity demo: build attractors without sphere, reveal it at step 130.

    Returns:
        q_history:  (T, 7)
        cls_all:    combined clearance per step
        cls_sph:    sphere-only clearance per step
        metrics:    raw numbers + reactivity fields
    """
    spec         = SCENARIO_REGISTRY["frontal_i_barrier_lr_easy"]()
    obstacles    = spec.collision_obstacles()
    base_cl_fn   = _make_clearance_fn(obstacles)
    sphere_state = {"active": False, "center": np.array([1000., 1000., 1000.])}

    def combined_cl(q: np.ndarray) -> float:
        sc = sphere_state["center"]
        lp = _panda_link_positions(q)
        sph = float(min(
            np.linalg.norm(np.array(pos) - sc) - r - _SPHERE_RADIUS
            for pos, r in zip(lp, _LINK_RADII_L)
        ))
        return min(base_cl_fn(q), sph)

    cfg       = GeoMultiAttractorDSConfig(enable_timing=True)
    t_plan0   = time.perf_counter()
    ds        = build_geo_multi_attractor_ds(spec, config=cfg)
    planner_s = time.perf_counter() - t_plan0
    ds.clearance_fn = combined_cl

    q       = spec.q_start.copy()
    q_hist  = [q.copy()]
    cls_all, cls_sph = [], []
    step_ms = []
    col_cnt = 0
    n_sw    = 0
    prev_att: Optional[int] = None
    conv_step = -1
    reaction_step = -1
    last_res = None
    all_goal_errs: List[float] = []

    t_wall = time.perf_counter()
    for step in range(n_steps):
        if step == _T_OBS_START:
            sphere_state["center"] = _SPHERE_POS.copy()
            sphere_state["active"] = True

        qdot, res = ds.compute(q, dt=_DT)
        last_res = res

        att = int(res.active_attractor_idx)
        if prev_att is not None and att != prev_att:
            n_sw += 1
        prev_att = att

        all_goal_errs.append(float(res.goal_error))

        cl_all = float(res.clearance)
        cls_all.append(cl_all)
        if cl_all < 0.0:
            col_cnt += 1

        sc = sphere_state["center"]
        lp = _panda_link_positions(q)
        sph_cl = float(min(
            np.linalg.norm(np.array(pos) - sc) - r - _SPHERE_RADIUS
            for pos, r in zip(lp, _LINK_RADII_L)
        ))
        cls_sph.append(sph_cl)

        if sphere_state["active"] and reaction_step < 0 and res.obs_blend_alpha > 0.0:
            reaction_step = step

        if res.timing_ms:
            step_ms.append(res.timing_ms.get("total_ms", 0.0))

        if res.goal_error < spec.goal_radius:
            conv_step = step
            q_hist.append(q.copy())
            break

        spd = float(np.linalg.norm(qdot))
        if spd > _MAX_SPD:
            qdot *= _MAX_SPD / spd
        q = q + _DT * qdot
        q_hist.append(q.copy())

    run_s   = time.perf_counter() - t_wall
    wall_s  = run_s  # kept for backward compat field name
    mean_ms = float(np.mean(step_ms)) if step_ms else 0.0
    p95_ms  = float(np.percentile(step_ms, 95)) if step_ms else 0.0

    n_steps_run   = len(step_ms)
    control_hz    = round(1000.0 / mean_ms) if mean_ms > 0 else 0
    total_s       = planner_s + run_s
    end_to_end_hz = round(float(n_steps_run) / total_s) if total_s > 0 else 0
    planner_ms    = planner_s * 1000.0

    # ---- Frame-corrected final metrics ----
    active_att = ds.attractors[ds._active_idx]
    q_goal_act = active_att.q_goal
    err_active = float(last_res.goal_error) if last_res is not None else float("nan")
    err_any    = min(float(np.linalg.norm(q - att.q_goal)) for att in ds.attractors)

    gt_final   = _grasptarget_pos(q)
    gt_att     = _grasptarget_pos(q_goal_act)
    red_sphere = np.array(spec.target_pose["position"]) if spec.target_pose else gt_att
    grasptarget_to_sphere     = float(np.linalg.norm(gt_final - red_sphere))
    att_grasptarget_to_sphere = float(np.linalg.norm(gt_att   - red_sphere))

    hand_q     = np.array(_panda_link_positions(q)[-1])
    hand_g     = np.array(_panda_link_positions(q_goal_act)[-1])
    hand_to_hand = float(np.linalg.norm(hand_q - hand_g))

    min_cl = float(np.min(cls_all)) if cls_all else float("nan")
    task_ok = grasptarget_to_sphere < _TASK_THRESH_M
    success_verified = (conv_step >= 0 and task_ok and col_cnt == 0 and min_cl > 0.0)

    metrics = {
        # ---- validity ----
        "success":                                  success_verified,
        "convergence_step":                         conv_step,
        "converged_by_metric":                      "joint_error_to_active_att" if conv_step >= 0 else "none",
        # ---- timing ----
        "simulated_duration_s":                     len(q_hist) * _DT,
        "wall_clock_render_s":                      wall_s,
        "planner_ms":                               planner_ms,
        "planner_s":                                planner_s,
        "run_s":                                    run_s,
        "total_s":                                  total_s,
        "control_hz":                               control_hz,
        "end_to_end_hz":                            end_to_end_hz,
        # ---- clearance / collision ----
        "min_clearance_m":                          min_cl,
        "collision_count":                          col_cnt,
        # ---- attractors ----
        "n_attractors":                             len(ds.attractors),
        "n_switches":                               n_sw,
        "active_attractor_idx":                     int(ds._active_idx),
        "active_family":                            active_att.family,
        # ---- joint-space error ----
        "joint_goal_error_to_active_attractor":     err_active,
        "min_joint_goal_error_to_any_attractor":    err_any,
        "min_goal_error_during_run":                float(min(all_goal_errs)) if all_goal_errs else float("nan"),
        # ---- task-space error (frame-corrected) ----
        "grasptarget_to_red_sphere_m":              grasptarget_to_sphere,
        "att_grasptarget_to_red_sphere_m":          att_grasptarget_to_sphere,
        "hand_to_hand_m":                           hand_to_hand,
        # ---- success gates ----
        "gate_converged":                           conv_step >= 0,
        "gate_grasptarget_to_sphere":               task_ok,
        "gate_no_collision":                        col_cnt == 0,
        "gate_positive_clearance":                  min_cl > 0.0,
        # ---- step timing (legacy + new) ----
        "mean_step_ms":                             mean_ms,
        "p95_step_ms":                              p95_ms,
        "estimated_hz":                             control_hz,   # legacy alias
        # ---- reactivity-specific ----
        "obstacle_appearance_step":                 _T_OBS_START,
        "reaction_latency_steps":                   max(0, reaction_step - _T_OBS_START)
                                                    if reaction_step >= 0 else -1,
        "min_sphere_clearance_m":                   float(np.min(cls_sph[_T_OBS_START:]))
                                                    if len(cls_sph) > _T_OBS_START else float("nan"),
        "validity_note":                            "DNF: sphere patches scalar clearance_fn only; "
                                                    "gradient+scoring use base obstacles. "
                                                    "Arm trapped at ~1.42 rad error.",
    }
    return np.array(q_hist), cls_all, cls_sph, metrics, spec


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_frames(
    env: MuJoCoRenderEnv,
    q_history: np.ndarray,
    every_n: int = 2,
    env_post: Optional[MuJoCoRenderEnv] = None,
    switch_at: int = -1,
) -> List[np.ndarray]:
    """
    Render q_history frames.

    Args:
        env:       Primary render environment.
        q_history: (T, 7) joint trajectory.
        every_n:   Subsample factor.
        env_post:  If provided, use this env for frames at index >= switch_at
                   (reactivity: sphere appears at switch_at).
        switch_at: Step index at which to switch to env_post.

    Returns:
        List of RGB arrays (H, W, 3) uint8. Empty if GL unavailable.
    """
    frames: List[np.ndarray] = []
    indices = list(range(0, len(q_history), every_n))
    if len(q_history) - 1 not in indices:
        indices.append(len(q_history) - 1)

    for idx in indices:
        if env_post is not None and switch_at >= 0 and idx >= switch_at:
            frame = env_post.render_at(q_history[idx])
        else:
            frame = env.render_at(q_history[idx])
        frames.append(frame)

    # Black-frame guard
    if frames and float(frames[0].mean()) < 1.0:
        print("  [warn] off-screen renderer returned black frames — "
              "GL context unavailable. Saving PNG frames skipped.")
        return []
    return frames


def _snapshot_frames(
    env: MuJoCoRenderEnv,
    q_history: np.ndarray,
    key_step: int,
    env_post: Optional[MuJoCoRenderEnv] = None,
    post_start: int = -1,
) -> Dict[str, np.ndarray]:
    """Return {label: frame} for start/mid/key/final snapshots."""
    n   = len(q_history)
    mid = n // 2
    snaps: Dict[str, int] = {
        "start":  0,
        "mid":    mid,
        "key":    max(0, min(key_step, n - 1)),
        "final":  n - 1,
    }
    result = {}
    for label, idx in snaps.items():
        use_post = (env_post is not None and post_start >= 0 and idx >= post_start)
        e = env_post if use_post else env
        result[label] = e.render_at(q_history[idx])
    return result


def _save_video(frames: List[np.ndarray], out_path: Path, fps: int = 30) -> Optional[str]:
    """
    Save frames as MP4 (imageio+ffmpeg) or fall back to GIF (PIL).

    Returns the path written, or None if nothing could be saved.
    """
    if not frames:
        return None
    mp4_path = out_path.with_suffix(".mp4")
    gif_path = out_path.with_suffix(".gif")
    # Try MP4 first
    try:
        import imageio
        writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264",
                                    pixelformat="yuv420p", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  → {mp4_path}")
        return str(mp4_path)
    except Exception:
        pass

    # Fall back to GIF
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(
            str(gif_path),
            save_all=True,
            append_images=imgs[1:],
            duration=max(1, int(1000 / fps)),
            loop=0,
        )
        print(f"  → {gif_path}")
        return str(gif_path)
    except Exception as e:
        print(f"  [warn] could not save video: {e}")
        return None


def _save_frames(
    frames: List[np.ndarray],
    snaps: Dict[str, np.ndarray],
    out_dir: Path,
) -> List[str]:
    """Save full frame sequence and snapshot PNGs. Returns list of saved paths."""
    paths: List[str] = []
    try:
        from PIL import Image
    except ImportError:
        print("  [warn] PIL not available — skipping PNG output")
        return []

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames):
        p = frames_dir / f"frame_{i:05d}.png"
        Image.fromarray(f).save(str(p))
        paths.append(str(p))

    for label, frame in snaps.items():
        if frame is not None and float(frame.mean()) > 1.0:
            p = out_dir / f"snap_{label}.png"
            Image.fromarray(frame).save(str(p))
            paths.append(str(p))

    return paths


def _save_metrics(metrics: dict, path: Path) -> None:
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    # Make JSON-serialisable
    def _fix(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
            return str(v)
        return v
    clean = {k: _fix(v) for k, v in clean.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"  → {path}")


def _save_summary(
    label: str,
    scenario: str,
    expansion: bool,
    metrics: dict,
    video_path: Optional[str],
    out_dir: Path,
) -> None:
    m       = metrics
    success = m.get("success", False)
    conv    = m.get("convergence_step", -1)
    conv_s  = str(conv) if conv >= 0 else "DNF"
    lines = [
        f"# GeoMA-DS MuJoCo Demo — {label}",
        "",
        f"**Scenario:** `{scenario}`",
        f"**Coverage expansion:** {'yes' if expansion else 'no'}",
        "",
        "## Result",
        "",
        f"| Metric | Value |",
        f"|:-------|------:|",
        f"| Success | {'yes' if success else 'no'} |",
        f"| Convergence step | {conv_s} |",
        f"| Simulated duration | {m.get('simulated_duration_s', 0):.2f} s |",
        f"| Min clearance | {m.get('min_clearance_m', float('nan')):.4f} m |",
        f"| Collisions | {m.get('collision_count', 0)} |",
        f"| Attractors | {m.get('n_attractors', 0)} |",
        f"| Switches | {m.get('n_switches', 0)} |",
        f"| Mean step time | {m.get('mean_step_ms', 0):.2f} ms |",
        "",
        "## Timing",
        "",
        f"| Metric | Value |",
        f"|:-------|------:|",
        f"| Planner (build) | {m.get('planner_ms', 0):.0f} ms |",
        f"| Control Hz | {m.get('control_hz', 0)} Hz |",
        f"| End-to-end Hz | {m.get('end_to_end_hz', 0)} Hz |",
        f"| Total wall time | {m.get('total_s', m.get('wall_clock_render_s', 0)):.2f} s |",
    ]
    if "reaction_latency_steps" in m:
        lines += [
            f"| Obstacle appearance step | {m.get('obstacle_appearance_step', _T_OBS_START)} |",
            f"| Reaction latency | {m.get('reaction_latency_steps', -1)} steps |",
            f"| Min sphere clearance | {m.get('min_sphere_clearance_m', float('nan')):.4f} m |",
        ]
    lines += [""]
    if video_path:
        lines.append(f"**Video:** `{video_path}`")
    lines += [
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M')}_",
    ]
    out = out_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Per-demo runners
# ---------------------------------------------------------------------------

def demo_a(n_steps: int, every_n: int, fps: int) -> dict:
    """I-barrier success."""
    print("\n=== Demo A: I-barrier ===")
    out_dir = OUT_ROOT / "i_barrier"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SCENARIO_REGISTRY["frontal_i_barrier_lr_easy"]()
    q_history, cls, att_i, metrics = _run_geo_ma(spec, use_expansion=False, n_steps=n_steps)
    metrics["scenario"] = "frontal_i_barrier_lr_easy"
    metrics["method"]   = "GeoMA-DS"
    metrics["coverage_expansion"] = False

    conv  = metrics["convergence_step"]
    sw_steps = metrics.get("_switch_steps", [])
    key_step = sw_steps[0] if sw_steps else (conv // 2 if conv > 0 else len(q_history) // 2)
    conv_s = str(conv) if conv >= 0 else "DNF"
    print(f"  conv={conv_s}  min_cl={metrics['min_clearance_m']:.4f}  "
          f"planner={metrics['planner_ms']:.0f}ms  ctrl={metrics['control_hz']}Hz  e2e={metrics['end_to_end_hz']}Hz")

    env    = _build_render_env(spec)
    snaps  = _snapshot_frames(env, q_history, key_step=key_step)
    frames = _render_frames(env, q_history, every_n=every_n)
    _save_frames(frames, snaps, out_dir)
    vid = _save_video(frames, out_dir / "i_barrier", fps=fps)
    _save_metrics(metrics, out_dir / "metrics.json")
    _save_summary("A — I-barrier", "frontal_i_barrier_lr_easy", False, metrics, vid, out_dir)
    return {"label": "A", "out_dir": out_dir, "video": vid, "metrics": metrics}


def demo_b(n_steps: int, every_n: int, fps: int) -> dict:
    """Cross baseline failure (no expansion)."""
    print("\n=== Demo B: Cross baseline (no expansion) ===")
    out_dir = OUT_ROOT / "cross_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SCENARIO_REGISTRY["frontal_cross_barrier_easy"]()
    q_history, cls, att_i, metrics = _run_geo_ma(spec, use_expansion=False, n_steps=n_steps)
    metrics["scenario"] = "frontal_cross_barrier_easy"
    metrics["method"]   = "GeoMA-DS"
    metrics["coverage_expansion"] = False

    conv = metrics["convergence_step"]
    mid  = len(q_history) // 2
    conv_s = str(conv) if conv >= 0 else "DNF"
    print(f"  conv={conv_s}  min_cl={metrics['min_clearance_m']:.4f}  "
          f"planner={metrics['planner_ms']:.0f}ms  ctrl={metrics['control_hz']}Hz  e2e={metrics['end_to_end_hz']}Hz")

    env    = _build_render_env(spec)
    snaps  = _snapshot_frames(env, q_history, key_step=mid)
    frames = _render_frames(env, q_history, every_n=every_n)
    _save_frames(frames, snaps, out_dir)
    vid = _save_video(frames, out_dir / "cross_baseline", fps=fps)
    _save_metrics(metrics, out_dir / "metrics.json")
    _save_summary("B — Cross baseline", "frontal_cross_barrier_easy", False, metrics, vid, out_dir)
    return {"label": "B", "out_dir": out_dir, "video": vid, "metrics": metrics}


def demo_c(n_steps: int, every_n: int, fps: int) -> dict:
    """Cross + coverage expansion success."""
    print("\n=== Demo C: Cross + expansion ===")
    out_dir = OUT_ROOT / "cross_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SCENARIO_REGISTRY["frontal_cross_barrier_easy"]()
    q_history, cls, att_i, metrics = _run_geo_ma(spec, use_expansion=True, n_steps=n_steps)
    metrics["scenario"] = "frontal_cross_barrier_easy"
    metrics["method"]   = "GeoMA-DS + coverage expansion"
    metrics["coverage_expansion"] = True

    conv     = metrics["convergence_step"]
    sw_steps = metrics.get("_switch_steps", [])
    key_step = sw_steps[0] if sw_steps else (conv // 2 if conv > 0 else len(q_history) // 2)
    conv_s   = str(conv) if conv >= 0 else "DNF"
    print(f"  conv={conv_s}  min_cl={metrics['min_clearance_m']:.4f}  sw={metrics['n_switches']}  "
          f"planner={metrics['planner_ms']:.0f}ms  ctrl={metrics['control_hz']}Hz  e2e={metrics['end_to_end_hz']}Hz")

    env    = _build_render_env(spec)
    snaps  = _snapshot_frames(env, q_history, key_step=key_step)
    frames = _render_frames(env, q_history, every_n=every_n)
    _save_frames(frames, snaps, out_dir)
    vid = _save_video(frames, out_dir / "cross_expanded", fps=fps)
    _save_metrics(metrics, out_dir / "metrics.json")
    _save_summary("C — Cross expanded", "frontal_cross_barrier_easy", True, metrics, vid, out_dir)
    return {"label": "C", "out_dir": out_dir, "video": vid, "metrics": metrics}


def demo_d(n_steps: int, every_n: int, fps: int) -> dict:
    """Reactivity: moving sphere descends through wrist path (delegates to reactivity_moving_obstacle_mujoco)."""
    print("\n=== Demo D: Reactivity (moving obstacle) ===")
    out_dir = OUT_ROOT / "reactivity_moving_obstacle"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _rmo._build_spec()
    q_hist, sph_hist, cls_all, cls_sph, metrics = _rmo.run_sim(spec, n_steps)
    metrics["scenario"] = "reach_no_barrier (I-barrier q_start)"
    metrics["method"]   = "GeoMA-DS (reactivity)"
    metrics["coverage_expansion"] = False

    conv   = metrics["convergence_step"]
    conv_s = str(conv) if conv >= 0 else "DNF"
    print(f"  conv={conv_s}  min_cl={metrics['min_clearance_m']:.4f}  "
          f"sph_min_cl={metrics['sphere_min_clearance_m']:.4f}  "
          f"latency={metrics['reaction_latency_steps']}")

    # Build MuJoCo render env (body-based moving sphere)
    import mujoco
    ctx, renderer, data, model, sphere_body_id, cam, n_arm = _rmo._build_mujoco_env(spec)

    # Render frames
    indices = list(range(0, len(q_hist), every_n))
    if len(q_hist) - 1 not in indices:
        indices.append(len(q_hist) - 1)
    frames: List[np.ndarray] = []
    for idx in indices:
        sc = sph_hist[min(idx, len(sph_hist) - 1)]
        fr = _rmo._render_frame(renderer, model, data, cam, q_hist[idx], sc, sphere_body_id, n_arm)
        frames.append(fr)

    # Snapshots
    snap_steps = {
        "start":    0,
        "crossing": (_rmo._T_APPEAR + _rmo._T_GONE) // 2,
        "clear":    min(_rmo._T_GONE + 10, len(q_hist) - 1),
        "final":    len(q_hist) - 1,
    }
    for label, sidx in snap_steps.items():
        sidx = min(sidx, len(q_hist) - 1)
        sc   = sph_hist[min(sidx, len(sph_hist) - 1)]
        fr   = _rmo._render_frame(renderer, model, data, cam, q_hist[sidx], sc, sphere_body_id, n_arm)
        _rmo._save_snapshot(fr, out_dir / f"snap_{label}.png")

    renderer.close()

    # Save video and outputs
    vid_stem = out_dir / "reactivity_moving_obstacle"
    gif = _rmo._save_gif(frames, vid_stem, fps)
    mp4 = _rmo._save_mp4(frames, vid_stem, fps)
    vid = mp4 or gif
    _rmo._save_metrics(metrics, out_dir / "metrics.json")
    _rmo._save_summary(metrics, gif, mp4, out_dir)

    return {"label": "D", "out_dir": out_dir, "video": vid, "metrics": metrics}


# ---------------------------------------------------------------------------
# demo_index.md update
# ---------------------------------------------------------------------------

def _update_demo_index(results: List[dict]) -> None:
    idx_path = Path("outputs/final/demo_index.md")
    existing = ""
    if idx_path.exists():
        existing = idx_path.read_text(encoding="utf-8")

    section_lines = [
        "",
        "## MuJoCo Visual Demos",
        "",
        "Full 3D simulation of the Panda arm executing each scenario.",
        "",
        "| Demo | Description | Proves |",
        "|:-----|:------------|:-------|",
    ]
    desc_map = {
        "A": ("I-barrier success",
              "DS converges collision-free through vertical gate"),
        "B": ("Cross baseline failure",
              "DS halts when all IK solutions share one homotopy class"),
        "C": ("Cross + expansion success",
              "Coverage expansion unlocks missing window; DS converges"),
        "D": ("Reactivity — moving sphere descends through wrist path",
              "DS deflects arm in real-time, no replan, no collision"),
    }
    for r in results:
        label = r["label"]
        d, p  = desc_map.get(label, (label, ""))
        section_lines.append(f"| {label} | {d} | {p} |")

    section_lines += [
        "",
        "### Output Paths",
        "",
    ]
    for r in results:
        label = r["label"]
        out_d = r["out_dir"]
        vid   = r.get("video")
        lines = [f"**Demo {label}** — `{out_d}/`"]
        if vid:
            lines.append(f"- Video: `{vid}`")
        lines.append(f"- Metrics: `{out_d / 'metrics.json'}`")
        lines.append(f"- Summary: `{out_d / 'summary.md'}`")
        lines.append(f"- Snapshots: `{out_d}/snap_*.png`")
        section_lines.extend(lines + [""])

    # Remove previous MuJoCo section if present
    marker = "## MuJoCo Visual Demos"
    if marker in existing:
        existing = existing[:existing.index(marker)].rstrip()

    updated = existing + "\n" + "\n".join(section_lines) + "\n"
    idx_path.write_text(updated, encoding="utf-8")
    print(f"\n  → {idx_path} (updated)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=800)
    parser.add_argument("--every-n", type=int, default=2,
                        help="Render every Nth step (higher = fewer frames, smaller files)")
    parser.add_argument("--fps",     type=int, default=30)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(demo_a(args.steps, args.every_n, args.fps))
    results.append(demo_b(args.steps, args.every_n, args.fps))
    results.append(demo_c(args.steps, args.every_n, args.fps))
    results.append(demo_d(args.steps, args.every_n, args.fps))

    _update_demo_index(results)
    print("\nDone. All MuJoCo demos written to", OUT_ROOT)


if __name__ == "__main__":
    main()
