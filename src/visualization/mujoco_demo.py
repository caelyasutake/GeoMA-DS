"""
Phase 10 — MuJoCo Visualization Demo

Runs one benchmark scenario end-to-end with the Franka Panda robot in a
MuJoCo scene and saves:

  outputs/demo/<scenario>/
    mujoco_rollout.gif    — animated robot rollout
    frame_XXXXX.png       — individual rendered frames (optional)
    metrics.png           — 6-panel summary figure
    summary.json          — pipeline metrics with correct naming

Usage::

    python -m src.visualization.mujoco_demo
    python -m src.visualization.mujoco_demo --scenario narrow_passage --seed 0
    python -m src.visualization.mujoco_demo --scenario free_space --headless
    python -m src.visualization.mujoco_demo --no-animation --no-frames

Scenarios
---------
narrow_passage      (default)  joint-space corridor; demonstrates multi-IK
free_space                     no obstacles; validates basic pipeline
contact_task                   box obstacle; contact force sensing
u_shape                        non-convex U obstacle; topological trap benchmark
left_open_u                    left-to-right C-barrier benchmark (canonical)
c_barrier                      alias for left_open_u
left_to_right_barrier          alias for left_open_u
cluttered_tabletop             multiple obstacles on a virtual tabletop
random_obstacle_field          randomly placed box field
frontal_i_barrier_lr           frontal I-barrier benchmark (medium difficulty)
frontal_i_barrier_lr_easy      frontal I-barrier, easy variant
frontal_i_barrier_lr_medium    frontal I-barrier, medium variant (same as above)
frontal_i_barrier_lr_hard      frontal I-barrier, hard variant
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Canonical scenario builders
# ---------------------------------------------------------------------------
from src.scenarios.scenario_builders import SCENARIO_REGISTRY, get_scenario
from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.ik.problem_json import build_problem_json, save_problem_json
from src.solver.planner.collision import make_collision_fn

# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------
from src.solver.planner.birrt import plan, PlannerConfig
from src.solver.ds.path_ds import PathDS, DSConfig
from src.solver.controller.impedance import (
    ControllerConfig, ControlResult, simulate,
    step as ctrl_step,
)
from src.solver.controller.cbf_filter import CBFConfig
from src.solver.ds.modulation import ModulationConfig
from src.solver.controller.hard_shield import HardShieldConfig
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.evaluation.success_criteria import SuccessConfig, evaluate_success, classify_path_tracking
from src.solver.ds.path_tracking import distance_to_path_q

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
from src.simulation.panda_scene import (
    load_panda_scene,
    validate_panda_model,
    NEUTRAL_QPOS,
)
from src.simulation.mujoco_env import MuJoCoRenderEnv, RenderConfig
from src.visualization.plotting import (
    plot_ik_goals,
    plot_rrt_path,
    plot_executed_trajectory,
    plot_tank_energy,
    plot_passivity_metrics,
    plot_distance_to_goal,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS    = 7
DT          = 0.05       # legacy — kept for build_metrics_figure fallback
SIM_DT      = 0.002      # physics timestep (s) — stable for Panda impedance control
N_PHYS_MAX  = 5000       # max physics steps (= 10 s)
VIS_SUBSAMPLE = 10       # record q every N physics steps ->  ≤300 vis frames
N_STEPS     = 300        # target visualization frame count
GOAL_RADIUS = 0.05


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    spec: ScenarioSpec,
    seed: int,
    n_phys_max: int = N_PHYS_MAX,
    vanilla_ds: bool = False,
    single_ik: bool = False,
    use_modulation: bool = False,
    diffik_ds: bool = False,
) -> dict:
    """
    Execute planning + control pipeline from a canonical ScenarioSpec.

    Args:
        vanilla_ds: Skip BiRRT entirely; use a straight-line DS path from
                    q_start to the nearest IK goal in joint space.  This is
                    the no-planner baseline — the DS gradient pulls the arm
                    directly toward the goal with no obstacle-aware routing.

    Returns a dict with all data needed for visualization and summary.
    """
    q_start = spec.q_start
    Q_goals = spec.ik_goals
    _planner_params    = spec.planner or {}
    _collision_margin  = float(_planner_params.get("collision_margin", 0.0))
    col_fn  = make_collision_fn(spec, margin=_collision_margin)

    # Filter IK goals to collision-free ones so the planner only considers
    # goals that the arm can actually reach without arm links inside obstacles.
    safe_mask = [col_fn(q) for q in Q_goals]
    Q_goals_safe = [q for q, s in zip(Q_goals, safe_mask) if s]
    safe_orig_indices = [i for i, s in enumerate(safe_mask) if s]

    if not Q_goals_safe:
        Q_goals_safe = Q_goals
        safe_orig_indices = list(range(len(Q_goals)))

    # ---- Diff-IK DS: completely separate execution path -------------------
    if diffik_ds:
        return _run_diffik_demo(spec, seed, n_phys_max)

    # ---- Plan (or bypass for vanilla DS) -----------------------------------
    if vanilla_ds:
        # Pick the IK goal nearest to q_start in joint space.
        q_start_arr = np.asarray(q_start, dtype=float)
        dists = [float(np.linalg.norm(np.asarray(g) - q_start_arr)) for g in Q_goals]
        best_idx = int(np.argmin(dists))
        q_goal = np.asarray(Q_goals[best_idx], dtype=float)
        selected_idx = best_idx
        print(f"[pipeline] Vanilla DS: skipping BiRRT, straight-line path to goal {best_idx} "
              f"(dist={dists[best_idx]:.3f} rad)")
        from src.solver.planner.birrt import PlanResult as _PlanResult
        plan_result = _PlanResult(
            success=True,
            path=[q_start_arr, q_goal],
            goal_idx=best_idx,
            nodes_explored=0,
            time_to_solution=0.0,
            collision_checks=0,
            iterations=0,
        )
    else:
        cfg_plan = PlannerConfig(max_iterations=12_000, step_size=0.12,
                                  goal_bias=0.12, seed=seed)
        goals_for_planner = Q_goals_safe[:1] if single_ik else Q_goals_safe
        if single_ik:
            print(f"[pipeline] Single-IK: restricting planner to best goal only "
                  f"(orig idx={safe_orig_indices[0]})")
        plan_result = plan(q_start, goals_for_planner, env=col_fn, config=cfg_plan)

        if not plan_result.success:
            return {
                "success": False, "plan_result": plan_result,
                "Q_goals": Q_goals, "safe_mask": safe_mask,
                "q_start": q_start, "results": [],
                "q_history": np.array([q_start]),
                "q_goal": Q_goals_safe[0] if Q_goals_safe else q_start,
                "selected_idx": safe_orig_indices[0], "path": [],
            }

        # Map goal_idx (into Q_goals_safe) back to original Q_goals index
        selected_idx = safe_orig_indices[plan_result.goal_idx]
        q_goal = Q_goals[selected_idx]

    # Verify IK goal EE position via FK (confirms hand/flange target, not link7)
    from src.solver.planner.collision import _panda_link_positions
    _fk_goal = _panda_link_positions(q_goal)
    _fk_start = _panda_link_positions(spec.q_start)
    print(f"[pipeline] Start EE (hand): {np.round(_fk_start[-1], 4)}  "
          f"link7: {np.round(_fk_start[-2], 4)}")
    print(f"[pipeline] Goal  EE (hand): {np.round(_fk_goal[-1], 4)}  "
          f"link7: {np.round(_fk_goal[-2], 4)}"
          f"  (goal idx={selected_idx})")

    # ---- DS ----------------------------------------------------------------
    ctrl_params = spec.controller or {}
    ds = PathDS(plan_result.path,
                config=DSConfig(
                    K_c=ctrl_params.get("K_c", 2.0),
                    K_r=ctrl_params.get("K_r", 1.0),
                    K_n=ctrl_params.get("K_n", 0.3),
                    goal_radius=ctrl_params.get("ds_goal_radius", GOAL_RADIUS),
                    max_speed=ctrl_params.get("max_speed", float("inf")),
                ))

    # ---- Planned path geometry (for summary metrics) ----------------------
    path_arr = np.array(plan_result.path)
    if len(path_arr) > 1:
        diffs = np.diff(path_arr, axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        path_length = 0.0

    # ---- Obstacle avoidance: CBF (default) or modulation ------------------
    cbf_params = ctrl_params.get("cbf", {})
    if use_modulation:
        # Modulation replaces CBF; disable CBF when modulation is requested.
        cbf_cfg = CBFConfig(enabled=False)
        mod_cfg = ModulationConfig(
            enabled=True,
            safety_margin=ctrl_params.get("mod_safety_margin", 0.03),
            rho=ctrl_params.get("mod_rho", 1.0),
            tangent_gain=ctrl_params.get("mod_tangent_gain", 0.5),
        )
        shield_cfg = HardShieldConfig(enabled=True, d_hard_min=0.01)
        print("[pipeline] Obstacle avoidance: task-space modulation + hard shield (CBF disabled)")
    else:
        cbf_cfg = CBFConfig(
            enabled=cbf_params.get("enabled", True),   # on by default in demo
            d_safe=cbf_params.get("d_safe", 0.03),
            d_buffer=cbf_params.get("d_buffer", 0.05),
            alpha=cbf_params.get("alpha", 8.0),
            contact_mode=cbf_params.get("contact_mode", "avoid_all"),
            allowed_contact_obstacle_names=cbf_params.get("allowed_contact_obstacle_names", []),
            allowed_contact_phases=cbf_params.get("allowed_contact_phases", []),
        )
        mod_cfg = None
        shield_cfg = None
    col_obs = spec.collision_obstacles()

    # ---- Physics rollout via SimEnv (MuJoCo) ------------------------------
    # Uses real robot dynamics so the controller torques produce physically
    # correct motion — unlike the old unit-mass Euler integrator which
    # diverged immediately.  CBF is applied each step, pushing the arm away
    # from obstacle surfaces before the impedance torque is computed.
    q_history_list: list = [np.asarray(q_start, dtype=float)]
    results: list = []
    final_error: float = float("inf")
    terminal_success: bool = False
    ever_in_goal: bool = False
    used_sim: bool = False
    _demo_goal_errors: list = []
    _demo_clearances: list = []
    _demo_deviations_q: list = []
    _step_times: list = []

    try:
        from src.simulation.env import SimEnv, SimEnvConfig

        sim_env = SimEnv(SimEnvConfig(
            obstacles=spec.obstacles_as_hjcd_dict(),
            timestep=SIM_DT,
        ))
        sim_env.set_state(np.asarray(q_start, dtype=float), np.zeros(N_JOINTS))
        grav_fn = sim_env.make_gravity_fn()

        pf_cfg   = PassivityFilterConfig()
        tank     = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0,
                                          epsilon_min=-0.05, epsilon_max=0.1))
        ctrl_cfg = ControllerConfig(
            d_gain=ctrl_params.get("d_gain", 3.0),
            f_n_gain=0.0,
            orthogonalize_residual=True,
            alpha=0.5,
            passivity_filter=pf_cfg,
            gravity_fn=grav_fn,
            cbf=cbf_cfg,
            modulation=mod_cfg,
            hard_shield=shield_cfg,
        )

        q    = sim_env.q.copy()
        qdot = sim_env.qdot.copy()

        for i in range(n_phys_max):
            _t0 = time.perf_counter()
            res = ctrl_step(q, qdot, ds, tank, SIM_DT,
                            config=ctrl_cfg, obstacles=col_obs)
            sim_env.step(res.tau, dt=SIM_DT)
            _step_times.append(time.perf_counter() - _t0)
            q    = sim_env.q.copy()
            qdot = sim_env.qdot.copy()
            results.append(res)

            # Subsample configurations for visualization frames
            if (i + 1) % VIS_SUBSAMPLE == 0:
                q_history_list.append(q.copy())

            goal_err = float(np.linalg.norm(q - q_goal))
            _demo_goal_errors.append(goal_err)
            _demo_deviations_q.append(distance_to_path_q(q, plan_result.path))
            if res.cbf is not None and res.cbf.min_clearance is not None:
                _demo_clearances.append(res.cbf.min_clearance)

            if goal_err < GOAL_RADIUS:
                ever_in_goal = True

        final_error = float(np.linalg.norm(q - q_goal))
        terminal_success = final_error < GOAL_RADIUS
        used_sim = True
        # Report control frequency achieved
        if _step_times:
            _mean_ms  = 1e3 * float(np.mean(_step_times))
            _p95_ms   = 1e3 * float(np.percentile(_step_times, 95))
            _max_ms   = 1e3 * float(np.max(_step_times))
            _mean_hz  = 1.0 / float(np.mean(_step_times))
            print(f"[pipeline] Control loop: mean {_mean_ms:.2f} ms  "
                  f"p95 {_p95_ms:.2f} ms  max {_max_ms:.2f} ms  "
                  f"→ {_mean_hz:.0f} Hz  (sim dt={SIM_DT*1e3:.1f} ms)")
        # Report final EE position to confirm hand/flange tracking
        _fk_final = _panda_link_positions(q)
        print(f"[pipeline] Final EE (hand): {np.round(_fk_final[-1], 4)}  "
              f"link7: {np.round(_fk_final[-2], 4)}"
              f"  goal_err={final_error:.4f} rad")

    except Exception as exc:
        # Fallback: linear interpolation of the RRT path.
        # This does NOT apply CBF but at least produces a viewable trajectory.
        print(f"[mujoco_demo] WARNING: physics rollout failed ({exc!r}), "
              f"falling back to RRT path interpolation — CBF will not be applied.")
        n_wp = len(path_arr)
        if n_wp >= 2:
            t_path = np.linspace(0.0, 1.0, n_wp)
            t_vis  = np.linspace(0.0, 1.0, N_STEPS + 1)
            q_history_list = list(np.stack([
                np.interp(t_vis, t_path, path_arr[:, j]) for j in range(N_JOINTS)
            ], axis=1))
        elif n_wp == 1:
            q_history_list = list(np.tile(path_arr[0], (N_STEPS + 1, 1)))
        else:
            q_history_list = list(np.tile(q_start, (N_STEPS + 1, 1)))

        final_error = float(np.linalg.norm(path_arr[-1] - q_goal))
        terminal_success = final_error < GOAL_RADIUS
        dists = [float(np.linalg.norm(qh - q_goal)) for qh in path_arr]
        ever_in_goal = any(d < GOAL_RADIUS for d in dists)

        pf_cfg   = PassivityFilterConfig()
        tank     = EnergyTank(TankConfig())
        ctrl_cfg = ControllerConfig(d_gain=3.0, f_n_gain=0.0,
                                     orthogonalize_residual=True, alpha=0.5,
                                     passivity_filter=pf_cfg, cbf=cbf_cfg)
        results = simulate(np.asarray(q_start, dtype=float), np.zeros(N_JOINTS),
                           ds, tank, dt=DT, n_steps=N_STEPS, config=ctrl_cfg)

    # ---- Assemble q_history (pad / trim to exactly N_STEPS+1 frames) -------
    q_history = np.array(q_history_list)
    if len(q_history) < N_STEPS + 1:
        pad = np.tile(q_history[-1], (N_STEPS + 1 - len(q_history), 1))
        q_history = np.vstack([q_history, pad])
    q_history = q_history[: N_STEPS + 1]

    # The effective dt per visualization frame (for distance-to-goal time axis)
    vis_dt = SIM_DT * VIS_SUBSAMPLE if used_sim else DT

    # ---- CBF aggregate metrics (from controller simulation results) ---------
    cbf_summary: Optional[dict] = None
    cbf_diags = [r.cbf for r in results if r.cbf is not None]
    if cbf_diags:
        clearances   = [d.min_clearance for d in cbf_diags if d.min_clearance is not None]
        corrections  = [d.correction_norm for d in cbf_diags]
        n_total      = len(cbf_diags)
        d_safe       = ctrl_cfg.cbf.d_safe if ctrl_cfg.cbf else 0.03
        cbf_summary = {
            "cbf_enabled":             True,
            "cbf_min_clearance":       float(min(clearances)) if clearances else None,
            "cbf_active_fraction":     sum(1 for d in cbf_diags if d.n_active > 0) / max(1, n_total),
            "cbf_mean_correction":     float(np.mean(corrections)) if corrections else 0.0,
            "cbf_max_correction":      float(max(corrections)) if corrections else 0.0,
            "cbf_n_slack_activations": sum(1 for d in cbf_diags if d.cbf_slack_used),
            "cbf_n_near_grazing":      sum(1 for c in clearances if c < d_safe + 0.005),
            "cbf_n_unintended_col":    sum(1 for c in clearances if c < 0),
        }

    # ---- Success diagnostics (shared evaluator, same criteria as benchmark) -
    _succ_cfg = SuccessConfig(goal_radius=GOAL_RADIUS)
    _succ_diag = evaluate_success(
        goal_errors=_demo_goal_errors,
        clearances=_demo_clearances if _demo_clearances else None,
        config=_succ_cfg,
    )
    # Path tracking verdict
    _n_steps = max(1, len(_demo_goal_errors))
    _mean_dev = float(np.mean(_demo_deviations_q)) if _demo_deviations_q else 0.0
    _max_dev  = float(np.max(_demo_deviations_q))  if _demo_deviations_q else 0.0
    _pt_verdict = classify_path_tracking(_mean_dev, _max_dev, 0, _n_steps)

    return {
        "success":          True,
        "plan_result":      plan_result,
        "Q_goals":          Q_goals,
        "safe_mask":        safe_mask,
        "selected_idx":     selected_idx,
        "q_start":          q_start,
        "q_goal":           q_goal,
        "path":             plan_result.path,
        "results":          results,
        "q_history":        q_history,
        "final_goal_error": final_error,
        "terminal_success": _succ_diag.terminal_success,
        "ever_in_goal":     ever_in_goal,
        "path_length":      path_length,
        "cbf_summary":      cbf_summary,
        "vis_dt":           vis_dt,
        # Structured success diagnostics (parity with benchmark)
        "success_diagnostics": {
            "terminal_success":       _succ_diag.terminal_success,
            "failure_reason":         _succ_diag.failure_reason,
            "reached_goal_once":      _succ_diag.reached_goal_once,
            "stayed_in_goal_steps":   _succ_diag.stayed_in_goal_steps,
            "min_goal_error_ever":    _succ_diag.min_goal_error_ever,
            "final_goal_error":       final_error,
            "had_collision":          _succ_diag.had_collision,
            "min_clearance_rollout":  _succ_diag.min_clearance_rollout,
            "mean_path_deviation_q":  _mean_dev,
            "max_path_deviation_q":   _max_dev,
            "path_tracking_verdict":  _pt_verdict,
            "n_steps":                _n_steps,
            "goal_radius_used":       GOAL_RADIUS,
        },
        "ctrl_freq": {
            "mean_hz":   round(1.0 / float(np.mean(_step_times)), 1) if _step_times else None,
            "mean_ms":   round(1e3 * float(np.mean(_step_times)), 3) if _step_times else None,
            "p95_ms":    round(1e3 * float(np.percentile(_step_times, 95)), 3) if _step_times else None,
            "max_ms":    round(1e3 * float(np.max(_step_times)), 3) if _step_times else None,
            "sim_dt_ms": SIM_DT * 1e3,
        },
    }


# ---------------------------------------------------------------------------
# Vanilla DS + Diff-IK demo helper
# ---------------------------------------------------------------------------
def _run_diffik_demo(spec: ScenarioSpec, seed: int, n_phys_max: int) -> dict:
    """
    Run a task-space DS + differential IK + modulation demo.
    No BiRRT, no PathDS.  Returns the same dict shape as run_pipeline.
    """
    from src.solver.controller.vanilla_ds_diffik import (
        VanillaDSDiffIKController, VanillaDSConfig, DiffIKConfig,
    )
    from src.solver.ds.modulation import ModulationConfig
    from src.solver.controller.hard_shield import HardShieldConfig, enforce_hard_clearance
    from src.simulation.env import SimEnv, SimEnvConfig

    q_start   = np.asarray(spec.q_start, dtype=float)
    Q_goals   = spec.ik_goals
    col_obs   = spec.collision_obstacles()
    ctrl_params = spec.controller or {}

    # Pick nearest IK goal as the joint-space goal (for metrics)
    dists = [float(np.linalg.norm(np.asarray(g) - q_start)) for g in Q_goals]
    best_idx  = int(np.argmin(dists))
    q_goal    = np.asarray(Q_goals[best_idx], dtype=float)

    # Build sim env
    env = SimEnv(SimEnvConfig(
        obstacles=spec.obstacles_as_hjcd_dict(),
        timestep=SIM_DT,
    ))
    env.set_state(q_start, np.zeros(N_JOINTS))

    # Get task-space goal via FK
    x_goal, _ = env.ee_pose(q_goal)
    print(f"[diffik-demo] Task-space goal: {np.round(x_goal, 4)}  "
          f"joint-goal idx={best_idx}")

    ds_cfg  = VanillaDSConfig(
        pos_gain=ctrl_params.get("diffik_pos_gain", 2.0),
        max_task_speed=ctrl_params.get("diffik_max_task_speed", 0.25),
        goal_tolerance=ctrl_params.get("diffik_goal_tol_m", 0.01),
    )
    ik_cfg  = DiffIKConfig(
        damping=ctrl_params.get("diffik_damping", 1e-2),
        max_joint_speed=ctrl_params.get("diffik_max_joint_speed", 1.0),
    )
    mod_cfg = ModulationConfig(
        enabled=bool(col_obs),
        safety_margin=ctrl_params.get("mod_safety_margin", 0.03),
        rho=ctrl_params.get("mod_rho", 1.0),
        tangent_gain=ctrl_params.get("mod_tangent_gain", 0.5),
    )
    shield_cfg = HardShieldConfig(enabled=bool(col_obs), d_hard_min=0.01)

    ctrl = VanillaDSDiffIKController(
        ds_config=ds_cfg,
        diffik_config=ik_cfg,
        modulation_config=mod_cfg,
        q_nominal=q_start.copy(),
    )

    q    = env.q.copy()
    qdot = env.qdot.copy()
    grav_fn = env.make_gravity_fn()
    D_imp   = 5.0 * np.eye(N_JOINTS)

    q_history_list: List[np.ndarray] = []
    goal_errors: List[float] = []
    step_times:  List[float] = []
    ever_in_goal = False
    final_error  = float("inf")

    for i in range(n_phys_max):
        _t0 = time.perf_counter()
        x_ee, _ = env.ee_pose(q)
        J_full  = env.jacobian(q)
        J_pos   = J_full[:3, :]

        qdot_des, diag = ctrl.step(q, x_goal, J_pos, x_ee, obstacles=col_obs)

        if shield_cfg.enabled and col_obs:
            qdot_des, _ = enforce_hard_clearance(q, qdot_des, SIM_DT, col_obs, shield_cfg)

        G   = grav_fn(q)
        tau = G - D_imp @ (qdot - qdot_des)

        env.step(tau, dt=SIM_DT)
        step_times.append(time.perf_counter() - _t0)
        q    = env.q.copy()
        qdot = env.qdot.copy()

        if (i + 1) % VIS_SUBSAMPLE == 0:
            q_history_list.append(q.copy())

        goal_err = float(np.linalg.norm(q - q_goal))
        goal_errors.append(goal_err)
        if goal_err < GOAL_RADIUS:
            ever_in_goal = True

        if diag.goal_reached:
            print(f"[diffik-demo] Task-space goal reached at step {i} "
                  f"(EE err={diag.goal_error_m*100:.1f} cm)")
            break

    final_error = float(np.linalg.norm(q - q_goal))
    terminal_success = final_error < GOAL_RADIUS

    if step_times:
        _mean_ms = 1e3 * float(np.mean(step_times))
        _p95_ms  = 1e3 * float(np.percentile(step_times, 95))
        _hz      = 1.0 / float(np.mean(step_times))
        print(f"[diffik-demo] Control loop: mean {_mean_ms:.2f} ms  "
              f"p95 {_p95_ms:.2f} ms  → {_hz:.0f} Hz")

    from src.solver.planner.birrt import PlanResult as _PR
    dummy_plan = _PR(success=True, path=[q_start, q_goal],
                     goal_idx=best_idx, nodes_explored=0,
                     time_to_solution=0.0, collision_checks=0, iterations=0)

    q_history = np.array(q_history_list) if q_history_list else np.array([q_start])
    return {
        "success":          True,
        "plan_result":      dummy_plan,
        "Q_goals":          Q_goals,
        "safe_mask":        [True] * len(Q_goals),
        "selected_idx":     best_idx,
        "q_start":          q_start,
        "q_goal":           q_goal,
        "path":             [q_start, q_goal],
        "results":          [],
        "q_history":        q_history,
        "final_goal_error": final_error,
        "terminal_success": terminal_success,
        "ever_in_goal":     ever_in_goal,
        "path_length":      0.0,
        "cbf_summary":      {},
        "vis_dt":           SIM_DT * VIS_SUBSAMPLE,
        "solver":           "vanilla_ds_diffik_modulation",
        "success_diagnostics": {
            "terminal_success":      terminal_success,
            "failure_reason":        "" if terminal_success else "stalled_before_goal",
            "reached_goal_once":     ever_in_goal,
            "stayed_in_goal_steps":  0,
            "min_goal_error_ever":   float(min(goal_errors)) if goal_errors else float("inf"),
            "final_goal_error":      final_error,
            "had_collision":         False,
            "min_clearance_rollout": float("inf"),
            "n_steps":               len(goal_errors),
            "goal_radius_used":      GOAL_RADIUS,
        },
        "ctrl_freq": {
            "mean_hz":   round(1.0 / float(np.mean(step_times)), 1) if step_times else None,
            "mean_ms":   round(1e3 * float(np.mean(step_times)), 3) if step_times else None,
            "p95_ms":    round(1e3 * float(np.percentile(step_times, 95)), 3) if step_times else None,
            "max_ms":    round(1e3 * float(np.max(step_times)), 3) if step_times else None,
            "sim_dt_ms": SIM_DT * 1e3,
        } if step_times else None,
    }


# ---------------------------------------------------------------------------
# MuJoCo rendering
# ---------------------------------------------------------------------------
def _compute_ee_positions(Q_goals: list, q_start: np.ndarray) -> dict:
    """
    Use a temporary SimEnv to compute EE Cartesian positions for goal configs.

    Returns dict with:
        "goals": list of (pos, kind) for all goals
        "start": start EE position
    """
    from src.simulation.env import SimEnv, SimEnvConfig
    try:
        env_tmp = SimEnv(SimEnvConfig())
        goal_markers = []
        for i, q in enumerate(Q_goals):
            pos, _ = env_tmp.ee_pose(q)
            goal_markers.append((pos, "safe"))  # will override selected later
        start_pos, _ = env_tmp.ee_pose(q_start)
        return {"goals": goal_markers, "start": start_pos}
    except Exception:
        # FK not available — use zero positions
        return {"goals": [(np.zeros(3), "safe")] * len(Q_goals),
                "start": np.zeros(3)}


def _render_cfg_for_spec(spec: ScenarioSpec, base_cfg: RenderConfig) -> RenderConfig:
    """
    Override RenderConfig camera with per-scenario visualization settings.

    Reads optional keys from spec.visualization:
        cam_azimuth, cam_elevation, cam_lookat, cam_distance
    """
    vis = spec.visualization if spec else {}
    return RenderConfig(
        height=base_cfg.height,
        width=base_cfg.width,
        cam_lookat=tuple(vis.get("cam_lookat", list(base_cfg.cam_lookat))),
        cam_distance=float(vis.get("cam_distance", base_cfg.cam_distance)),
        cam_azimuth=float(vis.get("cam_azimuth", base_cfg.cam_azimuth)),
        cam_elevation=float(vis.get("cam_elevation", base_cfg.cam_elevation)),
    )


def build_render_env(
    data: dict,
    spec: ScenarioSpec,
    render_cfg: RenderConfig,
) -> MuJoCoRenderEnv:
    """
    Build a MuJoCoRenderEnv from the canonical ScenarioSpec.

    Loads the Franka Panda via load_panda_scene() (which uses the real
    mesh-based MJCF from mujoco/franka_emika_panda/), validates that Panda
    bodies are present, then wraps in MuJoCoRenderEnv.

    Uses obstacles_as_panda_scene_dict() so per-obstacle rgba colours and
    visual-only flags are passed through to panda_scene.py.
    """
    Q_goals      = data["Q_goals"]
    q_start      = data["q_start"]
    selected_idx = data.get("selected_idx", None)
    safe_mask    = data.get("safe_mask", [True] * len(Q_goals))

    # Use panda_scene_dict so per-obstacle rgba is preserved
    obstacles = spec.obstacles_as_panda_scene_dict()

    # Apply per-scenario camera overrides
    scene_cfg = _render_cfg_for_spec(spec, render_cfg)

    # Compute EE Cartesian positions using the render env's own FK
    # (we build a temporary env to get EE positions before adding markers)
    tmp_model = load_panda_scene()
    tmp_env   = MuJoCoRenderEnv(tmp_model, scene_cfg)

    goal_markers: List[Tuple[np.ndarray, str]] = []
    for i, q in enumerate(Q_goals):
        pos  = tmp_env.ee_pos(q)
        if selected_idx is not None and i == selected_idx:
            kind = "selected"
        elif not safe_mask[i]:
            kind = "all"   # gray — blocked goal (arm links in collision at this config)
        else:
            kind = "safe"  # green — collision-free but not selected
        goal_markers.append((pos, kind))

    ee_target = None
    if selected_idx is not None and selected_idx < len(goal_markers):
        ee_target = goal_markers[selected_idx][0]

    # Load the full scene with markers injected
    model = load_panda_scene(
        obstacles=obstacles,
        ee_target=ee_target,
        goal_markers=goal_markers,
    )

    # Validate Panda is actually present
    info = validate_panda_model(model)
    if not info["panda_detected"]:
        raise RuntimeError(
            "Panda bodies not found in rendered model — "
            "marker-only scenes are not acceptable."
        )

    return MuJoCoRenderEnv(model, scene_cfg)


def render_rollout(
    render_env: MuJoCoRenderEnv,
    q_history: np.ndarray,
    every_n: int = 3,
) -> List[np.ndarray]:
    """
    Render the robot at each step in q_history, returning a list of RGB frames.

    Args:
        render_env: Initialized MuJoCoRenderEnv.
        q_history:  Shape (T, n_joints).
        every_n:    Only render every Nth step (to reduce frame count).

    Returns:
        List of RGB arrays, each shape (H, W, 3).
        Empty list if off-screen GL context is unavailable (black-frame guard).
    """
    frames = []
    indices = list(range(0, len(q_history), every_n))
    if (len(q_history) - 1) not in indices:
        indices.append(len(q_history) - 1)

    for idx in indices:
        frame = render_env.render_at(q_history[idx])
        frames.append(frame)

    # Black-frame guard: if the first frame is all zeros the GL context is
    # missing (common on Windows without an explicit GLContext).  Warn and
    # return empty so callers skip the GIF and fall back to matplotlib plots.
    if frames and float(frames[0].mean()) < 1.0:
        print("[mujoco_demo] WARNING: off-screen renderer returned black frames "
              "(no GL context on Windows). Skipping GIF/PNG output. "
              "Use --viewer for 3D visualization.")
        return []

    return frames


def save_frames_to_dir(
    frames: List[np.ndarray],
    out_dir: Path,
    prefix: str = "frame_",
) -> List[str]:
    """Save a list of RGB frames as PNG files."""
    try:
        from PIL import Image
    except ImportError:
        print("[demo] WARNING: PIL not available — skipping PNG frame export")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = out_dir / f"{prefix}{i:05d}.png"
        Image.fromarray(frame).save(str(path))
        paths.append(str(path))
    return paths


def save_animation(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
) -> bool:
    """
    Save frames as an animated GIF.

    Returns True on success, False if the Pillow dependency is missing.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[demo] WARNING: Pillow not installed — skipping animation. "
              "Install with: pip install Pillow")
        return False

    if not frames:
        return False

    imgs = [Image.fromarray(f) for f in frames]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return True


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------
def _launch_viewer(
    render_env: MuJoCoRenderEnv,
    q_history: np.ndarray,
    speed: float = 1.0,
    start_delay: float = 0.3,
) -> None:
    """
    Open an interactive MuJoCo viewer and play back the trajectory.

    Uses ``mujoco.viewer.launch_passive`` with ``mj_step`` (not mj_forward)
    so MuJoCo's contact solver is active and the arm cannot pass through
    obstacles.  The built-in MJCF position actuators (gainprm/biasprm) are
    used: setting ``mjdata.ctrl[:7] = q_target`` drives the arm toward the
    desired configuration while contacts are enforced by the physics engine.

    After the final waypoint the viewer stays open for inspection.

    Args:
        render_env: Initialized MuJoCoRenderEnv (contains model + data).
        q_history:  Joint trajectory, shape (T, 7).
        speed:      Playback speed multiplier (1.0 = real-time at DT=0.05 s).
    """
    try:
        import mujoco.viewer as mjviewer
    except ImportError:
        print("[viewer] mujoco.viewer not available — skipping interactive view.")
        return

    import time

    model  = render_env.model
    mjdata = mujoco.MjData(model)

    # MuJoCo simulation timestep (from model) and steps per trajectory waypoint
    sim_dt      = float(model.opt.timestep)          # typically 0.002 s
    steps_per_wp = max(1, int(round(DT / sim_dt)))   # e.g. 25 steps per waypoint

    print("[viewer] Opening interactive MuJoCo viewer …")
    print("[viewer]   Collision enforcement: ON (mj_step + MJCF actuators)")
    print("[viewer]   Drag to rotate  |  Scroll to zoom  |  Close window to continue")

    # Wall-clock pause between waypoints so playback matches real time
    wall_dt = DT / max(speed, 1e-3)

    with mjviewer.launch_passive(model, mjdata) as viewer:
        # Match camera to render config
        viewer.cam.lookat[:] = render_env._cfg.cam_lookat
        viewer.cam.distance   = render_env._cfg.cam_distance
        viewer.cam.azimuth    = render_env._cfg.cam_azimuth
        viewer.cam.elevation  = render_env._cfg.cam_elevation

        n_ctrl = min(7, model.nu)   # number of controllable joints

        # Initialise arm at the first waypoint (skip zero-time teleport)
        if len(q_history) > 0:
            mjdata.qpos[:n_ctrl] = q_history[0][:n_ctrl]
            mjdata.ctrl[:n_ctrl] = q_history[0][:n_ctrl]
            mujoco.mj_forward(model, mjdata)
            viewer.sync()
            time.sleep(start_delay)

        for q_target in q_history:
            if not viewer.is_running():
                break

            # Drive arm toward q_target via built-in PD actuators + physics step
            mjdata.ctrl[:n_ctrl] = q_target[:n_ctrl]
            for _ in range(steps_per_wp):
                mujoco.mj_step(model, mjdata)
            viewer.sync()
            time.sleep(wall_dt)

        # Keep window open after playback ends
        print("[viewer] Playback complete — close the viewer window to continue.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print("[viewer] Viewer closed.")


# ---------------------------------------------------------------------------
# Scenario / HJCD-IK JSON persistence helpers (CLAUDE.md required)
# ---------------------------------------------------------------------------
def _save_scenario_json(spec: ScenarioSpec, path) -> None:
    """Serialise the canonical ScenarioSpec to JSON for reproducibility."""
    import dataclasses

    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _serialise(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_serialise(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(_serialise(spec), indent=2, default=str)
    )


def _save_hjcd_problem_json(spec: ScenarioSpec, path) -> None:
    """Generate and save the HJCD-IK problem JSON from the scenario spec."""
    if spec.target_pose is None:
        # No target pose — write a stub with empty obstacles
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"note": "no target_pose in scenario"}, indent=2))
        return
    save_problem_json(spec, path)


# ---------------------------------------------------------------------------
# Metrics figure
# ---------------------------------------------------------------------------
def build_metrics_figure(data: dict, scenario_name: str, dt: float = DT):
    """Build the 3×3 summary metrics figure using plotting.py utilities."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    sd = data.get("success_diagnostics", {})
    verdict = sd.get("path_tracking_verdict", "")
    status = "SUCCESS" if sd.get("terminal_success") else f"FAIL: {sd.get('failure_reason','')}"
    fig.suptitle(
        f"Multi-IK DS — {scenario_name}  seed={data.get('seed', 0)}\n"
        f"{status}  |  path: {verdict}  |  min_err={sd.get('min_goal_error_ever', float('inf')):.3f} rad",
        fontsize=11,
    )

    Q_goals      = data["Q_goals"]
    q_start      = data["q_start"]
    safe_mask    = data["safe_mask"]
    results      = data["results"]
    q_history    = data["q_history"]
    selected_idx = data.get("selected_idx")
    q_goal       = data.get("q_goal")
    path         = data.get("path", [])

    # ---- Row 0: Planning ---------------------------------------------------
    plot_ik_goals(axes[0, 0], Q_goals, q_start,
                  safe_mask=safe_mask, selected_idx=selected_idx,
                  title="IK Goal Set (joint space)")

    if data["success"] and path:
        plot_rrt_path(axes[0, 1], path, q_start=q_start,
                      title="Planned RRT Path")
    else:
        axes[0, 1].text(0.5, 0.5, "Planning FAILED",
                        ha="center", va="center",
                        transform=axes[0, 1].transAxes,
                        fontsize=14, color="red")
        axes[0, 1].set_title("RRT Path")

    if data["success"] and len(q_history) > 1:
        plot_executed_trajectory(axes[0, 2], q_history,
                                  path=path,
                                  title="Executed vs Planned")
    else:
        axes[0, 2].set_title("Executed Trajectory")

    # ---- Row 1: Goal error + path deviation + clearance --------------------
    ts = np.arange(len(q_history)) * dt if len(q_history) > 1 else np.array([0.0])

    # Goal error over time
    ax_ge = axes[1, 0]
    if q_goal is not None and len(q_history) > 1:
        goal_errs = [float(np.linalg.norm(q_history[i] - q_goal))
                     for i in range(len(q_history))]
        ax_ge.plot(ts, goal_errs, color="#2176AE", linewidth=1.2)
        ax_ge.axhline(GOAL_RADIUS, color="green", linestyle="--",
                      linewidth=0.8, label=f"goal_radius={GOAL_RADIUS}")
        ax_ge.set_xlabel("Time (s)")
        ax_ge.set_ylabel("||q - q_goal|| (rad)")
        ax_ge.set_title("Goal Error over Time")
        ax_ge.legend(fontsize=7)
        min_err = min(goal_errs)
        ax_ge.set_title(f"Goal Error  (min={min_err:.3f} rad)")
    else:
        ax_ge.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax_ge.transAxes, color="grey")
        ax_ge.set_title("Goal Error")

    # Path deviation over time (joint space)
    ax_pd = axes[1, 1]
    if path and len(q_history) > 1:
        try:
            from src.solver.ds.path_tracking import distance_to_path_q
            devs = [distance_to_path_q(q_history[i], path)
                    for i in range(len(q_history))]
            ax_pd.plot(ts, devs, color="#E07B39", linewidth=1.2)
            ax_pd.set_xlabel("Time (s)")
            ax_pd.set_ylabel("Distance to path (rad)")
            ax_pd.set_title(f"Path Deviation  (mean={np.mean(devs):.3f}, max={np.max(devs):.3f})")
        except Exception:
            ax_pd.text(0.5, 0.5, "unavailable", ha="center", va="center",
                       transform=ax_pd.transAxes, color="grey")
            ax_pd.set_title("Path Deviation")
    else:
        ax_pd.set_title("Path Deviation")

    # CBF clearance over time
    ax_cl = axes[1, 2]
    cbf_clearances = []
    if results:
        cbf_clearances = [r.cbf.min_clearance for r in results
                          if r.cbf is not None and r.cbf.min_clearance is not None]
    if cbf_clearances:
        ts_cbf = np.arange(len(cbf_clearances)) * dt
        ax_cl.plot(ts_cbf, cbf_clearances, color="#6A0DAD", linewidth=1.0)
        ax_cl.axhline(0.0, color="red", linestyle="--", linewidth=0.8, label="collision")
        ax_cl.axhline(0.03, color="orange", linestyle=":", linewidth=0.8, label="d_safe")
        ax_cl.set_xlabel("Time (s)")
        ax_cl.set_ylabel("Min clearance (m)")
        ax_cl.set_title(f"Obstacle Clearance  (min={min(cbf_clearances):.3f} m)")
        ax_cl.legend(fontsize=7)
    else:
        ax_cl.text(0.5, 0.5, "No CBF data", ha="center", va="center",
                   transform=ax_cl.transAxes, color="grey")
        ax_cl.set_title("Obstacle Clearance")

    # ---- Row 2: Controller metrics ----------------------------------------
    if results:
        plot_tank_energy(axes[2, 0], results, dt=dt)
        plot_passivity_metrics(axes[2, 1], results, dt=dt)
        if q_goal is not None and len(q_history) > 1:
            plot_distance_to_goal(axes[2, 2], q_history, q_goal,
                                  dt=dt, goal_radius=GOAL_RADIUS,
                                  title="Distance to Goal")
        else:
            axes[2, 2].set_title("(no control data)")
    else:
        for ax in axes[2]:
            ax.text(0.5, 0.5, "No control data",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Obstacle layout validator (for left_open_u / CLAUDE.md required fields)
# ---------------------------------------------------------------------------
def _validate_left_open_u_layout(spec: ScenarioSpec) -> bool:
    """
    Validate the 90°-CW-rotated left-opening U/C barrier geometry.

    Geometry (XY plane, viewed from above)::

                    opening (+Y)
        left_arm ──────────────── right_arm
           |                          |
           └────────bottom_spine──────┘
                         (-Y)

    Checks:
      - exactly 3 collision-enabled obstacles
      - 2 vertical arm walls: thin in X, wide in Y (size[0] < size[1])
      - 1 horizontal spine:   wide in X, thin in Y (size[0] > size[1])
      - spine Y is between start EE Y (positive) and goal Y (negative)
      - arms are on opposite X sides of the workspace centre
    """
    obs = [o for o in spec.obstacles if o.collision_enabled]
    if len(obs) != 3:
        return False

    try:
        # Rotated geometry: arms thin in X wide in Y, spine wide in X thin in Y
        v_arms = [o for o in obs if o.size[0] < o.size[1]]   # thin in X — the arm walls
        h_bar  = [o for o in obs if o.size[0] > o.size[1]]   # wide in X — the spine
        if len(v_arms) != 2 or len(h_bar) != 1:
            return False

        spine = h_bar[0]
        spine_y = float(spine.position[1])

        # Arms must be on opposite X sides of centre
        arm_xs = sorted(float(a.position[0]) for a in v_arms)
        if arm_xs[1] <= arm_xs[0]:
            return False

        # Get start EE Y via FK
        start_ee_y = None
        try:
            from src.solver.planner.collision import _panda_link_positions
            lp = _panda_link_positions(spec.q_start)
            start_ee_y = float(lp[-1][1])
        except Exception:
            pass

        # Get goal Y
        goal_y = (float(spec.target_pose["position"][1])
                  if spec.target_pose else None)

        # Spine must lie between start (positive Y) and goal (negative Y)
        if start_ee_y is not None and goal_y is not None:
            if not (goal_y < spine_y < start_ee_y):
                return False

        return True
    except Exception:
        return False


def _validate_frontal_i_barrier_layout(spec: "ScenarioSpec") -> dict:
    """
    Validate the frontal I-barrier (vertical gate in XZ plane) geometry.

    Geometry (XZ plane, viewed from the front)::

        top_bar  ─────────────────
                        |
                   center_post
                        |
        bottom_bar ─────────────────

    All three members share approximately the same Y position.
    The robot transfers from start EE (y < post_center_y) to
    goal EE (y > post_center_y), routing around the center_post.

    Returns a dict with:
      valid            bool
      transfer_axis    "Y"
      post_name        str or None
      start_side_y     float or None  (signed: start_y - post_center_y)
      goal_side_y      float or None  (signed: goal_y - post_center_y)
      reason           str
    """
    result = {
        "valid": False,
        "transfer_axis": "Y",
        "post_name": None,
        "start_side_y": None,
        "goal_side_y": None,
        "reason": "",
    }

    try:
        obs = [o for o in spec.obstacles if o.collision_enabled]
        if len(obs) != 3:
            result["reason"] = f"expected 3 collision-enabled obstacles, got {len(obs)}"
            return result

        # Identify center_post: tall in Z, thin in X and Y
        posts = [
            o for o in obs
            if (o.size[2] > o.size[0]
                and o.size[2] > o.size[1]
                and o.size[2] > 2 * max(o.size[0], o.size[1]))
        ]
        if len(posts) != 1:
            result["reason"] = f"expected exactly 1 center_post candidate, found {len(posts)}"
            return result
        post = posts[0]
        result["post_name"] = post.name

        # Identify bars: the remaining 2 obstacles, long in Y, thin in Z
        bars = [o for o in obs if o is not post]
        if len(bars) != 2:
            result["reason"] = "could not identify 2 bar obstacles"
            return result

        post_center_z = float(post.position[2])
        post_center_y = float(post.position[1])

        bar_zs = [float(b.position[2]) for b in bars]
        top_bars    = [b for b, z in zip(bars, bar_zs) if z > post_center_z]
        bottom_bars = [b for b, z in zip(bars, bar_zs) if z < post_center_z]
        if len(top_bars) != 1 or len(bottom_bars) != 1:
            result["reason"] = (
                "bars must have one above and one below post center Z; "
                f"found {len(top_bars)} above, {len(bottom_bars)} below"
            )
            return result

        # All 3 share approximately the same Y position (within 0.01)
        all_ys = [float(o.position[1]) for o in obs]
        if max(all_ys) - min(all_ys) > 0.01:
            result["reason"] = (
                f"obstacles Y positions not coplanar: range={max(all_ys)-min(all_ys):.4f}"
            )
            return result

        # Compute start EE Y via FK
        start_y = None
        try:
            from src.solver.planner.collision import _panda_link_positions
            lp = _panda_link_positions(spec.q_start)
            start_y = float(lp[-1][1])
        except Exception:
            pass

        # Compute goal Y
        goal_y = (float(spec.target_pose["position"][1])
                  if spec.target_pose else None)

        if start_y is not None:
            result["start_side_y"] = start_y - post_center_y
        if goal_y is not None:
            result["goal_side_y"] = goal_y - post_center_y

        # Start and goal must be on opposite sides of post_center_y
        if start_y is not None and goal_y is not None:
            if (start_y - post_center_y) * (goal_y - post_center_y) >= 0:
                result["reason"] = (
                    f"start_y={start_y:.4f} and goal_y={goal_y:.4f} are on "
                    f"the same side of post_center_y={post_center_y:.4f}"
                )
                return result

        result["valid"] = True
        result["reason"] = "ok"
        return result

    except Exception as exc:
        result["reason"] = f"exception: {exc}"
        return result


# ---------------------------------------------------------------------------
# Summary JSON builder — CORRECTED metric names per CLAUDE.md
# ---------------------------------------------------------------------------
def build_summary(data: dict, scenario_name: str, spec: Optional[ScenarioSpec] = None) -> dict:
    """
    Build the summary.json payload with correctly named metrics.

    Metric naming (CLAUDE.md):
      planner_path_length    = geometric path length in joint space (sum of ||Δq||)
      planner_waypoint_count = number of waypoints in the path
      final_goal_error       = ||q_final - q_goal|| at termination
      terminal_success       = final_goal_error < threshold (NOT transient)
      ever_entered_goal_region = True if any step was within goal_radius
    """
    results   = data["results"]
    Q_goals   = data["Q_goals"]
    safe_mask = data["safe_mask"]

    clipped    = sum(1 for r in results if r.pf_clipped) if results else 0
    clip_ratio = clipped / max(len(results), 1) if results else None
    min_tank   = min(r.tank_energy for r in results) if results else None
    unhandled  = (
        sum(1 for r in results if r.pf_power_nom > 1e-10 and not r.pf_clipped)
        if results else None
    )

    final_goal_error = data.get("final_goal_error", float("nan"))
    terminal_success = data.get("terminal_success", False)

    # CLAUDE.md new required fields
    scene_name   = spec.name if spec else scenario_name
    goal_pos     = (spec.target_pose["position"]
                    if spec and spec.target_pose else None)
    if spec is None:
        layout_valid = None
    elif any(spec.name.startswith(p) for p in ("frontal_i_barrier", "image_gate")):
        layout_valid = _validate_frontal_i_barrier_layout(spec)
    else:
        layout_valid = _validate_left_open_u_layout(spec)

    # start_position_workspace: EE position at q_start (computed from FK)
    start_ee_pos = None
    if spec is not None:
        try:
            from src.solver.planner.collision import _panda_link_positions
            lp = _panda_link_positions(spec.q_start)
            start_ee_pos = [round(float(v), 4) for v in lp[-1]]
        except Exception:
            pass

    return {
        "scenario":                 scenario_name,
        "scene_name":               scene_name,
        "seed":                     data.get("seed", 0),
        "goal_radius":              GOAL_RADIUS,
        "planning_success":         data["success"],
        "n_ik_goals":               len(Q_goals),
        "n_safe_goals":             sum(safe_mask),
        "selected_goal_idx":        data.get("selected_idx"),
        # Planner metrics — distinct names per CLAUDE.md
        "planner_waypoint_count":   len(data.get("path", [])),
        "planner_path_length":      round(data.get("path_length", 0.0), 4),
        "planner_iterations":       (data["plan_result"].iterations
                                     if data["success"] else None),
        # Goal completion — strict per CLAUDE.md
        "final_goal_error":         round(float(final_goal_error), 6),
        "terminal_success":         terminal_success,
        "ever_entered_goal_region": data.get("ever_in_goal", False),
        # Controller metrics
        "clipped_ratio":            round(clip_ratio, 4) if clip_ratio is not None else None,
        "min_tank_energy":          round(float(min_tank), 6) if min_tank is not None else None,
        "unhandled_violations":     unhandled,
        "n_steps":                  len(results),
        # Obstacle consistency metrics (CLAUDE.md)
        "n_scenario_obstacles":     spec.n_obstacles() if spec else None,
        "n_collision_obstacles":    spec.n_collision_obstacles() if spec else None,
        # CLAUDE.md Phase 7 required fields
        "start_position_workspace": start_ee_pos,
        "goal_position_workspace":  goal_pos,
        "obstacle_layout_valid":    layout_valid,
        # CBF safety filter aggregate metrics (None when CBF disabled)
        **(data.get("cbf_summary") or {}),
        # Structured success diagnostics (parity with benchmark trial_runner)
        **(data.get("success_diagnostics") or {}),
        # Control frequency stats
        "ctrl_freq":                data.get("ctrl_freq"),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MuJoCo Franka Panda visualization demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario",
                        choices=list(SCENARIO_REGISTRY.keys()),
                        default="narrow_passage")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--output-dir",  default="outputs/demo")
    parser.add_argument("--headless",    action="store_true",
                        help="Do not open interactive windows")
    parser.add_argument("--no-animation", action="store_true",
                        help="Skip GIF generation")
    parser.add_argument("--no-frames",   action="store_true",
                        help="Skip per-step PNG frames")
    parser.add_argument("--render-every", type=int, default=3,
                        help="Render every N-th simulation step")
    parser.add_argument("--fps",         type=int, default=10,
                        help="Frames per second for the animation")
    parser.add_argument("--viewer",      action="store_true",
                        help="Open interactive MuJoCo viewer and play back trajectory")
    parser.add_argument("--viewer-speed", type=float, default=1.0,
                        help="Playback speed multiplier for the viewer (default 1.0)")
    parser.add_argument("--viewer-delay", type=float, default=0.3,
                        help="Seconds to pause after the viewer opens before playback starts (default 0.3)")
    parser.add_argument("--parity", action="store_true",
                        help="Match benchmark settings: 1500 exec steps, same success criteria. "
                             "Use with --scenario frontal_i_barrier_lr_medium --seed N to "
                             "compare directly against a benchmark trial.")
    parser.add_argument("--vanilla-ds", action="store_true",
                        help="Skip BiRRT; use a straight-line DS path to the nearest IK goal. "
                             "Baseline showing what happens with no obstacle-aware planning.")
    parser.add_argument("--single-ik", action="store_true",
                        help="Restrict BiRRT to the single best IK goal only (no multi-goal search).")
    parser.add_argument("--modulation", action="store_true",
                        help="Use task-space modulation obstacle avoidance instead of CBF.")
    parser.add_argument("--diffik-ds", action="store_true",
                        help="Vanilla task-space DS + differential IK + modulation. "
                             "No BiRRT, no PathDS. Fast reactive baseline.")
    args = parser.parse_args(argv)

    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_name = args.scenario
    print(f"[mujoco_demo] Scenario: {scenario_name}  seed={args.seed}")

    # ---- Build canonical scenario spec ------------------------------------
    spec = get_scenario(scenario_name)
    print(f"[mujoco_demo] Obstacles: {spec.n_obstacles()} total, "
          f"{spec.n_collision_obstacles()} collision-enabled")

    # ---- Run pipeline -----------------------------------------------------
    # --parity: use the same step budget as the benchmark (1500 steps @ SIM_DT)
    _n_phys = 1500 if args.parity else N_PHYS_MAX
    if args.parity:
        print(f"[mujoco_demo] --parity mode: n_steps={_n_phys} (matches benchmark)")
    print("[mujoco_demo] Running pipeline …")
    data = run_pipeline(spec, seed=args.seed, n_phys_max=_n_phys,
                        vanilla_ds=args.vanilla_ds,
                        single_ik=args.single_ik,
                        use_modulation=args.modulation,
                        diffik_ds=args.diffik_ds)
    data["seed"] = args.seed

    if data["success"]:
        print(f"[mujoco_demo] Planning OK — waypoints={len(data['path'])}, "
              f"path_length={data['path_length']:.3f} rad")
        print(f"[mujoco_demo] Controller: {len(data['results'])} steps, "
              f"terminal_success={data['terminal_success']}, "
              f"final_error={data['final_goal_error']:.4f} rad")
    else:
        print("[mujoco_demo] WARNING — Planning FAILED. Producing partial output.")

    # ---- Output directory ------------------------------------------------
    out_dir = Path(args.output_dir) / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mujoco_demo] Saving outputs to {out_dir.resolve()}")

    # ---- Summary JSON (correct metric names per CLAUDE.md) ---------------
    summary = build_summary(data, scenario_name, spec=spec)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[mujoco_demo] summary.json saved")
    print(f"  planning_success         = {summary['planning_success']}")
    print(f"  terminal_success         = {summary['terminal_success']}")
    print(f"  failure_reason           = {summary.get('failure_reason', 'n/a')}")
    print(f"  final_goal_error         = {summary['final_goal_error']}")
    print(f"  min_goal_error_ever      = {summary.get('min_goal_error_ever', 'n/a')}")
    print(f"  ever_entered_goal_region = {summary['ever_entered_goal_region']}")
    print(f"  stayed_in_goal_steps     = {summary.get('stayed_in_goal_steps', 'n/a')}")
    print(f"  path_tracking_verdict    = {summary.get('path_tracking_verdict', 'n/a')}")
    print(f"  mean_path_deviation_q    = {summary.get('mean_path_deviation_q', 'n/a')}")
    print(f"  planner_waypoint_count   = {summary['planner_waypoint_count']}")
    print(f"  planner_path_length      = {summary['planner_path_length']}")
    print(f"  clipped_ratio            = {summary['clipped_ratio']}")
    print(f"  min_tank_energy          = {summary['min_tank_energy']}")
    print(f"  unhandled_violations     = {summary['unhandled_violations']}")

    # ---- Save scenario.json and hjcd_problem.json (CLAUDE.md required) --
    scenario_json_path = out_dir / "scenario.json"
    _save_scenario_json(spec, scenario_json_path)
    print(f"[mujoco_demo] scenario.json saved ->  {scenario_json_path}")

    hjcd_path = out_dir / "hjcd_problem.json"
    _save_hjcd_problem_json(spec, hjcd_path)
    print(f"[mujoco_demo] hjcd_problem.json saved ->  {hjcd_path}")

    # ---- MuJoCo rendering ------------------------------------------------
    print("[mujoco_demo] Building MuJoCo render environment …")
    render_cfg = RenderConfig(height=480, width=640)
    try:
        render_env = build_render_env(data, spec, render_cfg)
        print(f"[mujoco_demo] Render env: {render_env}")

        # Validate Panda model
        info = validate_panda_model(render_env.model)
        print(f"[mujoco_demo] Panda detected={info['panda_detected']}, "
              f"nbody={info['n_bodies']}, ngeom={info['n_geoms']}")
        if not info["panda_detected"]:
            raise RuntimeError("Panda not detected — aborting render.")

    except Exception as exc:
        print(f"[mujoco_demo] WARNING: render env failed — {exc}")
        render_env = None

    # ---- Static robot check (neutral pose) --------------------------------
    if render_env is not None:
        print("[mujoco_demo] Saving static robot check …")
        static_frame = render_env.render_at(NEUTRAL_QPOS)
        static_path  = out_dir / "robot_static_check.png"
        try:
            from PIL import Image
            Image.fromarray(static_frame).save(str(static_path))
            print(f"[mujoco_demo] robot_static_check.png saved ->  {static_path}")
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(str(static_path), static_frame)
            print(f"[mujoco_demo] robot_static_check.png saved ->  {static_path}")

    # ---- Scene preview (q_start pose) — CLAUDE.md Phase 7 required ------
    # Saved for all scenarios; especially important for left_open_u to
    # visually verify left-to-right layout matches the intended sketch.
    if render_env is not None:
        print("[mujoco_demo] Saving scene_preview.png (q_start pose) …")
        preview_frame = render_env.render_at(spec.q_start)
        preview_path  = out_dir / "scene_preview.png"
        try:
            from PIL import Image
            Image.fromarray(preview_frame).save(str(preview_path))
            print(f"[mujoco_demo] scene_preview.png saved ->  {preview_path}")
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(str(preview_path), preview_frame)
            print(f"[mujoco_demo] scene_preview.png saved ->  {preview_path}")

    if render_env is not None and len(data["q_history"]) > 1:
        print(f"[mujoco_demo] Rendering {len(data['q_history'])} steps "
              f"(every {args.render_every}) …")
        frames = render_rollout(render_env, data["q_history"],
                                 every_n=args.render_every)
        print(f"[mujoco_demo] {len(frames)} frames rendered")

        # Save frames
        if not args.no_frames:
            saved = save_frames_to_dir(frames, out_dir / "frames")
            if saved:
                print(f"[mujoco_demo] {len(saved)} frames ->  {out_dir / 'frames'}")

        # Save animation
        if not args.no_animation:
            gif_path = str(out_dir / "mujoco_rollout.gif")
            ok = save_animation(frames, gif_path, fps=args.fps)
            if ok:
                print(f"[mujoco_demo] mujoco_rollout.gif saved ->  {gif_path}")

        # Save a representative single frame (middle of rollout)
        if frames:
            mid = len(frames) // 2
            try:
                from PIL import Image
                snap_path = out_dir / "snapshot.png"
                Image.fromarray(frames[mid]).save(str(snap_path))
                print(f"[mujoco_demo] snapshot.png saved ->  {snap_path}")
            except ImportError:
                pass

    # ---- Interactive MuJoCo viewer ---------------------------------------
    if args.viewer and render_env is not None:
        _launch_viewer(render_env, data["q_history"], args.viewer_speed, args.viewer_delay)

    # ---- Metric plots ----------------------------------------------------
    print("[mujoco_demo] Building metrics figure …")
    fig = build_metrics_figure(data, scenario_name, dt=data.get("vis_dt", SIM_DT))
    metrics_path = out_dir / "metrics.png"
    fig.savefig(metrics_path, dpi=120, bbox_inches="tight")
    print(f"[mujoco_demo] metrics.png saved ->  {metrics_path}")
    if not args.headless:
        plt.show()
    plt.close(fig)

    print("[mujoco_demo] Done.")
    return summary


if __name__ == "__main__":
    main()
