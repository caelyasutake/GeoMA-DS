"""
Single-trial execution engine.

Each trial runs the full pipeline for one (scenario, condition, seed) tuple
and returns a TrialMetrics object.  Two entry points:

    run_planning_trial   — IK filter → BiRRT → DS execution (MuJoCo physics)
    run_contact_trial    — APPROACH → WALL_SLIDE → PERTURBATION → RETURN

Design principles
-----------------
* All randomness is seeded via the ``seed`` argument so trials are reproducible.
* MuJoCo physics is used for execution to capture gravity and contact dynamics.
* Controller configs are built from the ControlCondition enum.
* For planning trials, spec.ik_goals provides the IK solution pool
  (bypassing the GPU-dependent HJCD-IK solver for deterministic evaluation).
* Contact trials delegate to a lightweight re-implementation of the
  wall_contact_demo loop (same logic, no CLI overhead).
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import (
    BarrierMetrics, CBFGoalConflictDiagnostics, CBFMetrics, ContactMetrics,
    CtrlFreqMetrics, EscapeMetrics,
    ExecutionMetrics, FastMorseMetrics, HardShieldMetrics, IKMetrics, ModulationMetrics,
    MorseEscapeMetrics, PassivityMetrics, PlanMetrics, RobustnessMetrics, TrialMetrics,
)
from src.evaluation.success_criteria import (
    SuccessConfig, evaluate_success,
    classify_path_tracking, classify_multiik_effect,
)
from src.solver.planner.collision import (
    _panda_link_positions, _LINK_RADII, _sphere_box_signed_dist,
    _precompute_obs_rotations, make_collision_fn,
)
from src.evaluation.baselines import (
    ControlCondition, IKCondition, TrialCondition,
    build_ctrl_config, build_task_ctrl_config,
    ik_diversity_score, ik_goal_spread, select_ik_goals,
)
from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.ik.filter import (
    FilterConfig, FilterResult, RobotState, filter_safe_set,
)
from src.solver.planner.birrt import PlannerConfig, plan, PlanResult
from src.solver.ds.path_ds import PathDS, DSConfig
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.controller.impedance import ControllerConfig, step as ctrl_step
from src.simulation.env import SimEnv, SimEnvConfig

# Family-aware selection and diagnostics (new)
from src.solver.ik.goal_selection import (
    IKGoalInfo, classify_ik_goals, select_family_representatives,
    goals_from_infos, best_alternative, select_escape_family,
)
from src.evaluation.path_quality import path_risk_score, path_clearance_stats
from src.solver.planner.parallel_birrt import (
    GoalPlanCandidate, ParallelPlanResult, ParallelPlanningConfig,
    plan_candidates_parallel,
)
from src.evaluation.stall_detection import (
    StallDetectionConfig, StallHistory, StallDiagnostics, detect_stall,
    TrapDetectionConfig, TrapDiagnostics, detect_trap,
)
from src.solver.escape.morse_escape import (
    MorseEscapeConfig, MorseEscapePlanner, NegativeCurvatureConfig,
)
from src.solver.escape.fast_morse_escape import (
    FastMorseEscapeConfig, FastMorseEscapeController,
)
from src.solver.escape.morse_supervisor import (
    MorseSupervisorConfig, MorseEscapeSupervisor,
)
from src.solver.ds.escape_policy import EscapePolicy, EscapePolicyConfig, EscapeMode
from src.solver.ds.path_tracking import PathTubeConfig, PathTubeTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT_PLANNING = 0.002      # simulation timestep for planning trials (s)
N_EXEC_STEPS = 4000      # max execution steps for planning trials
DT_CONTACT   = 0.002     # simulation timestep for contact trials
GOAL_RADIUS  = 0.05      # rad — distance threshold for "in goal"
D_GRAZE_THRESH = 0.015   # m — clearance below this = near-graze event
STALL_THRESH   = 5e-4    # m — EE step displacement below this = stall step
MAX_REPLANS    = 2        # maximum family switches per trial


# ---------------------------------------------------------------------------
# Barrier clearance helper
# ---------------------------------------------------------------------------

def _make_clearance_fn(spec: "ScenarioSpec"):
    """
    Return a callable ``(q: np.ndarray) -> float`` giving the minimum signed
    obstacle clearance across all collision-enabled obstacles and robot links.

    Positive return value = free space; negative = penetrating.
    Uses the same sphere-swept link geometry as the BiRRT planner.
    Box obstacles use OBB (oriented bounding box) so rotated obstacles
    (e.g. tilted bookshelf panels) are checked correctly.
    Returns ``float("inf")`` if there are no collision-enabled obstacles.
    """
    obs_list = spec.collision_obstacles()
    if not obs_list:
        return lambda q: float("inf")

    # Precompute OBB rotation inverses once per obstacle.
    obs_R_inv = _precompute_obs_rotations(obs_list)

    # Cache per-obstacle geometry for the inner loop.
    _obs_pos  = [np.array(o.position, dtype=float) for o in obs_list]
    _obs_type = [o.type.lower() for o in obs_list]
    _obs_size = [np.array(o.size,     dtype=float) for o in obs_list]

    def _clearance(q: np.ndarray) -> float:
        link_positions = _panda_link_positions(q)
        min_cl = float("inf")
        for i, link_pos in enumerate(link_positions):
            r   = float(_LINK_RADII.get(i, 0.08))
            lp  = np.asarray(link_pos, dtype=float)
            for j, t in enumerate(_obs_type):
                obs_pos = _obs_pos[j]
                if t == "box":
                    cl = _sphere_box_signed_dist(
                        lp, r, obs_pos, _obs_size[j], obs_R_inv[j]
                    )
                elif t == "sphere":
                    cl = float(np.linalg.norm(lp - obs_pos)) - r - float(_obs_size[j][0])
                elif t == "cylinder":
                    cyl_r  = float(_obs_size[j][0])
                    cyl_hh = float(_obs_size[j][1])
                    dx, dy = lp[0] - obs_pos[0], lp[1] - obs_pos[1]
                    dz     = lp[2] - obs_pos[2]
                    r_xy   = float(np.sqrt(dx * dx + dy * dy))
                    cz     = float(np.clip(dz, -cyl_hh, cyl_hh))
                    if r_xy < 1e-9:
                        closest = np.array([obs_pos[0], obs_pos[1], obs_pos[2] + cz])
                    elif r_xy <= cyl_r:
                        closest = np.array([obs_pos[0] + dx, obs_pos[1] + dy, obs_pos[2] + cz])
                    else:
                        scale   = cyl_r / r_xy
                        closest = np.array([obs_pos[0] + dx * scale,
                                            obs_pos[1] + dy * scale, obs_pos[2] + cz])
                    cl = float(np.linalg.norm(lp - closest)) - r
                else:
                    continue
                if cl < min_cl:
                    min_cl = cl
        return min_cl

    return _clearance


def _make_per_obstacle_clearance_fn(spec: "ScenarioSpec"):
    """
    Return a callable ``(q: np.ndarray) -> dict`` mapping obstacle name to
    its minimum signed clearance across all robot links at that configuration.

    Uses the same sphere-swept link geometry as the BiRRT planner.
    Positive clearance = free; negative = penetrating.
    Returns an empty dict if there are no collision-enabled obstacles.
    """
    obs_list = spec.collision_obstacles()
    if not obs_list:
        return lambda q: {}

    # Precompute OBB rotation inverses and geometry once.
    obs_R_inv = _precompute_obs_rotations(obs_list)
    _obs_pos  = [np.array(o.position, dtype=float) for o in obs_list]
    _obs_type = [o.type.lower() for o in obs_list]
    _obs_size = [np.array(o.size,     dtype=float) for o in obs_list]

    def _per_obs_clearance(q: np.ndarray) -> dict:
        link_positions = _panda_link_positions(q)
        result = {}
        for j, obs in enumerate(obs_list):
            obs_pos = _obs_pos[j]
            t       = _obs_type[j]
            min_cl  = float("inf")
            for i, link_pos in enumerate(link_positions):
                r  = float(_LINK_RADII.get(i, 0.08))
                lp = np.asarray(link_pos, dtype=float)
                if t == "box":
                    cl = _sphere_box_signed_dist(
                        lp, r, obs_pos, _obs_size[j], obs_R_inv[j]
                    )
                elif t == "sphere":
                    cl = float(np.linalg.norm(lp - obs_pos)) - r - float(_obs_size[j][0])
                elif t == "cylinder":
                    cyl_r  = float(_obs_size[j][0])
                    cyl_hh = float(_obs_size[j][1])
                    dx, dy = lp[0] - obs_pos[0], lp[1] - obs_pos[1]
                    dz     = lp[2] - obs_pos[2]
                    r_xy   = float(np.sqrt(dx * dx + dy * dy))
                    cz     = float(np.clip(dz, -cyl_hh, cyl_hh))
                    if r_xy < 1e-9:
                        closest = np.array([obs_pos[0], obs_pos[1], obs_pos[2] + cz])
                    elif r_xy <= cyl_r:
                        closest = np.array([obs_pos[0] + dx, obs_pos[1] + dy, obs_pos[2] + cz])
                    else:
                        scale   = cyl_r / r_xy
                        closest = np.array([obs_pos[0] + dx * scale,
                                            obs_pos[1] + dy * scale, obs_pos[2] + cz])
                    cl = float(np.linalg.norm(lp - closest)) - r
                else:
                    continue
                if cl < min_cl:
                    min_cl = cl
            result[obs.name] = min_cl
        return result

    return _per_obs_clearance


# ---------------------------------------------------------------------------
# Per-goal planning helper
# ---------------------------------------------------------------------------

def _plan_to_single_goal(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    col_fn,
    plan_cfg: PlannerConfig,
    clearance_fn=None,
    scaffold_waypoints=None,
) -> PlanResult:
    """Plan from q_start to one goal; returns PlanResult."""
    return plan(q_start, [q_goal], env=col_fn, config=plan_cfg,
                clearance_fn=clearance_fn, scaffold_waypoints=scaffold_waypoints)


def _select_best_plan(
    q_start: np.ndarray,
    candidates: List[IKGoalInfo],
    col_fn,
    plan_cfg: PlannerConfig,
    obs_list,
    near_threshold: float = 0.025,
    clearance_fn=None,
    parallel_cfg: Optional["ParallelPlanningConfig"] = None,
) -> Tuple[
    Optional[IKGoalInfo],
    Optional[PlanResult],
    List[IKGoalInfo],
    List[Optional[PlanResult]],
    Dict,
]:
    """
    Plan to each candidate goal, score by path executability, return the best.

    When parallel_cfg.enabled is True and backend != "sequential", candidate
    planning jobs are dispatched concurrently via plan_candidates_parallel().
    Otherwise falls back to the original sequential loop.

    Returns:
        (best_info, best_result, ordered_candidates, ordered_results, timing)
        where timing is the dict from plan_candidates_parallel.
    """
    par_cfg = parallel_cfg or ParallelPlanningConfig(enabled=False)

    # Build GoalPlanCandidate list for the parallel helper
    par_candidates = [
        GoalPlanCandidate(
            goal_idx=info.goal_idx,
            family_label=info.family_label,
            q_goal=info.q_goal,
        )
        for info in candidates
    ]

    # Dispatch (parallel or sequential via backend="sequential")
    par_results, timing = plan_candidates_parallel(
        q_start=q_start,
        candidates=par_candidates,
        obstacles=obs_list,
        plan_cfg=plan_cfg,
        margin=0.0,
        config=par_cfg,
        near_threshold=near_threshold,
    )

    # Map goal_idx → IKGoalInfo for lookup
    info_by_idx = {info.goal_idx: info for info in candidates}

    scored: List[Tuple[float, IKGoalInfo, PlanResult]] = []
    failed: List[Tuple[IKGoalInfo, PlanResult]] = []

    for pr in par_results:
        info = info_by_idx.get(pr.goal_idx)
        if info is None:
            continue
        result = pr.plan_result

        if result.success and result.path:
            # Use precomputed risk if available; otherwise compute now
            if pr.path_risk_score is not None:
                risk_val = pr.path_risk_score
            else:
                risk_obj = path_risk_score(
                    result.path, q_start, info.q_goal, obs_list,
                    plan_success=True, near_threshold=near_threshold,
                )
                risk_val = risk_obj.total
            scored.append((risk_val, info, result))
        else:
            failed.append((info, result))

    # Sort by risk (ascending = better)
    scored.sort(key=lambda x: x[0])

    all_infos   = [s[1] for s in scored] + [f[0] for f in failed]
    all_results = [s[2] for s in scored] + [f[1] for f in failed]

    if scored:
        _, best_info, best_result = scored[0]
        return best_info, best_result, all_infos, all_results, timing
    return None, None, all_infos, all_results, timing


# ---------------------------------------------------------------------------
# Planning trial
# ---------------------------------------------------------------------------

def run_planning_trial(
    spec: ScenarioSpec,
    condition: TrialCondition,
    seed: int,
    trial_id: int = 0,
    dt: float = DT_PLANNING,
    n_exec_steps: int = N_EXEC_STEPS,
    noise_std: float = 0.0,
    perturb_magnitude: float = 0.0,
    perturb_joint: int = 1,
    perturb_start: int = 200,
    perturb_duration: int = 50,
    planner_override: Optional[Dict] = None,
    morse_override: Optional[Dict] = None,
    supervisor_override: Optional[Dict] = None,
    cbf_override: Optional["CBFConfig"] = None,
    disable_birrt: bool = False,
    q_log: Optional[List] = None,
) -> TrialMetrics:
    """
    Run one planning trial (IK → BiRRT → DS execution in MuJoCo).

    Args:
        spec:               ScenarioSpec (provides ik_goals, obstacles, params).
        condition:          IK and control conditions.
        seed:               RNG seed for planner and stochastic IK selection.
        trial_id:           Unique trial index (for logging).
        dt:                 Simulation timestep.
        n_exec_steps:       Maximum controller steps.
        noise_std:          Std of Gaussian state noise added each step.
        perturb_magnitude:  External torque magnitude (0 = off).
        perturb_joint:      Joint index for perturbation.
        perturb_start:      Step at which perturbation begins.
        perturb_duration:   Number of steps for perturbation.

    Returns:
        TrialMetrics with all available metrics.
    """
    t_start = time.perf_counter()

    tm = TrialMetrics(
        trial_id=trial_id,
        seed=seed,
        scenario=spec.name,
        condition=condition.name,
    )

    try:
        rng = np.random.default_rng(seed)

        # ---- IK goal selection -------------------------------------------
        all_goals = [np.asarray(g, dtype=float) for g in spec.ik_goals]
        if not all_goals:
            tm.error = "spec.ik_goals is empty"
            return tm

        q_start = np.asarray(spec.q_start, dtype=float)
        tank_init = TankConfig().s_init

        # Apply energy-aware filter if requested (uses default FilterConfig)
        if condition.ik == IKCondition.MULTI_IK_ENERGY_AWARE:
            filter_cfg = FilterConfig(w_energy=1.0, w_contact=0.0)
            fr: FilterResult = filter_safe_set(
                all_goals, RobotState(q=q_start), tank_init, filter_cfg
            )
            all_goals = fr.solutions if fr.solutions else all_goals

        # Apply IK condition to select which goals reach the planner
        selected_goals = select_ik_goals(all_goals, condition.ik, seed=seed)
        if not selected_goals:
            tm.error = "No IK goals survived selection"
            return tm

        diversity = ik_diversity_score(all_goals)
        spread    = ik_goal_spread(all_goals)

        # ---- SimEnv for collision checking and execution -----------------
        env = SimEnv(SimEnvConfig(
            obstacles=spec.obstacles_as_hjcd_dict(),
            timestep=dt,
        ))
        env.set_state(q_start, np.zeros(len(q_start)))
        grav_fn = env.make_gravity_fn()
        col_fn  = env.make_collision_fn()

        # ---- Planner config ----------------------------------------------
        planner_params = {**(spec.planner or {}), **(planner_override or {})}
        collision_margin = planner_params.get("collision_margin", 0.0)
        plan_col_fn = make_collision_fn(spec=spec, margin=collision_margin) \
                      if collision_margin > 0.0 else col_fn
        plan_cfg = PlannerConfig(
            max_iterations=planner_params.get("max_iterations", 10_000),
            step_size=planner_params.get("step_size", 0.10),
            goal_bias=planner_params.get("goal_bias", 0.10),
            seed=seed,
            min_step_size=planner_params.get("min_step_size", 0.02),
            clearance_step_scale=planner_params.get("clearance_step_scale", 1.0),
            clearance_step_threshold=planner_params.get("clearance_step_threshold", 0.10),
            gaussian_bias=planner_params.get("gaussian_bias", 0.20),
            gaussian_std=planner_params.get("gaussian_std", 0.15),
            shortcut_iterations=planner_params.get("shortcut_iterations", 300),
            shortcut_n_check=planner_params.get("shortcut_n_check", 12),
        )
        obs_list = spec.collision_obstacles()

        # ---- Parallel planning config (read from planner_override) --------
        pp = planner_params.get("parallel_planning", {})
        parallel_cfg = ParallelPlanningConfig(
            enabled=pp.get("enabled", True),
            max_workers=pp.get("max_workers", 4),
            backend=pp.get("backend", "process"),
            compute_path_risk_in_worker=pp.get("compute_path_risk_in_worker", True),
            preserve_order=pp.get("preserve_order", True),
        )

        # ---- Family-aware selection for MULTI_IK_FULL --------------------
        # Classify all goals by posture family, select one rep per family,
        # then plan to each representative and score by path executability.
        # For all other conditions, fall back to the original single-query BiRRT.
        use_family_planning = (condition.ik == IKCondition.MULTI_IK_FULL)
        use_vanilla_ds = (condition.ik == IKCondition.VANILLA_DS)
        use_diffik_ds = (condition.ctrl == ControlCondition.VANILLA_DS_DIFFIK_MODULATION)

        all_goal_infos: List[IKGoalInfo] = []
        ordered_candidates: List[IKGoalInfo] = []
        ordered_results: List[Optional[PlanResult]] = []
        _parallel_timing: Dict = {}

        t_plan_0 = time.perf_counter()
        _t_classify_s: float = 0.0   # wall-clock time for classify_ik_goals only

        if use_vanilla_ds or use_diffik_ds:
            # No BiRRT: pick nearest IK goal in joint space.
            # For diffik, we only need the goal to derive task-space target via FK.
            dists = [float(np.linalg.norm(g - q_start)) for g in all_goals]
            best_goal_idx = int(np.argmin(dists))
            q_goal_vanilla = all_goals[best_goal_idx]
            plan_result = PlanResult(
                success=True,
                path=[q_start.copy(), q_goal_vanilla],
                goal_idx=best_goal_idx,
                nodes_explored=0,
                time_to_solution=0.0,
                collision_checks=0,
                iterations=0,
            )
            best_info = None
        elif disable_birrt and use_family_planning:
            # BiRRT disabled: classify goals for family metadata, pick best by
            # IK quality score alone, and use a trivial 2-point path.
            # Morse / FastMorse handles any resulting local minima at runtime.
            _t_cls0 = time.perf_counter()
            all_goal_infos = classify_ik_goals(
                q_start, all_goals, obs_list, col_fn=plan_col_fn, margin=0.0
            )
            _t_classify_s = time.perf_counter() - _t_cls0
            _cf = [g for g in all_goal_infos if g.is_collision_free]
            best_info = min(_cf or all_goal_infos, key=lambda x: x.score_prior) \
                        if all_goal_infos else None
            _q_best = best_info.q_goal if best_info else (all_goals[0] if all_goals else q_start)
            plan_result = PlanResult(
                success=True,
                path=[q_start.copy(), _q_best],
                goal_idx=best_info.goal_idx if best_info else 0,
                nodes_explored=0,
                time_to_solution=0.0,
                collision_checks=0,
                iterations=0,
            )
        elif use_family_planning:
            # Classify goals and pick per-family representatives.
            # Use plan_col_fn (sphere-swept, includes EE flange) so static
            # collision checks on IK goals are consistent with the planner.
            _t_cls0 = time.perf_counter()
            all_goal_infos = classify_ik_goals(
                q_start, all_goals, obs_list, col_fn=plan_col_fn, margin=0.0
            )
            _t_classify_s = time.perf_counter() - _t_cls0
            # top_k_per_family=2 so we have fallbacks within each family
            representatives = select_family_representatives(
                all_goal_infos, top_k_per_family=2, require_collision_free=True
            )
            if not representatives:
                representatives = select_family_representatives(
                    all_goal_infos, top_k_per_family=1, require_collision_free=False
                )
            # Plan to each representative (parallel or sequential per config)
            best_info, plan_result, ordered_candidates, ordered_results, _parallel_timing = \
                _select_best_plan(
                    q_start, representatives, plan_col_fn, plan_cfg, obs_list,
                    clearance_fn=_make_clearance_fn(spec) if spec.obstacles else None,
                    parallel_cfg=parallel_cfg,
                )
            if best_info is None:
                plan_result = None
        else:
            # Original: single multi-goal BiRRT query
            plan_result = plan(q_start, selected_goals, env=plan_col_fn, config=plan_cfg)
            best_info = None

        t_plan = time.perf_counter() - t_plan_0

        # ---- Compute path quality metrics --------------------------------
        path_len = 0.0
        planned_min_cl = float("inf")
        planned_mean_cl = float("inf")
        planned_near_frac = 0.0
        interp_min_cl = float("inf")
        risk_score = 0.0

        if plan_result is not None and plan_result.path and len(plan_result.path) > 1:
            segs = [float(np.linalg.norm(plan_result.path[i+1] - plan_result.path[i]))
                    for i in range(len(plan_result.path) - 1)]
            path_len = float(sum(segs))
            if obs_list:
                pqs = path_clearance_stats(plan_result.path, obs_list, n_samples=40)
                planned_min_cl  = pqs.min_clearance
                planned_mean_cl = pqs.mean_clearance
                planned_near_frac = pqs.near_obstacle_fraction

        # ---- Resolve selected goal index ---------------------------------
        # When using family planning, map back to original all_goals index.
        if use_family_planning and best_info is not None:
            selected_idx = best_info.goal_idx
            goal_rank    = selected_idx
            active_goal_info = best_info
            active_q_goal = best_info.q_goal
        elif plan_result is not None:
            selected_idx = plan_result.goal_idx if plan_result.goal_idx is not None else -1
            if 0 <= selected_idx < len(selected_goals):
                sel_q = selected_goals[selected_idx]
                goal_rank = next(
                    (i for i, g in enumerate(all_goals) if np.allclose(g, sel_q, atol=1e-6)),
                    selected_idx,
                )
                active_q_goal = sel_q
            else:
                goal_rank = -1
                active_q_goal = all_goals[0] if all_goals else q_start
            active_goal_info = None
        else:
            selected_idx = -1
            goal_rank = -1
            active_q_goal = all_goals[0] if all_goals else q_start
            active_goal_info = None

        # Collect family metadata for IKMetrics
        if all_goal_infos:
            family_labels = list({g.family_label for g in all_goal_infos if g.is_collision_free})
            fam_str = ",".join(sorted(family_labels))
            sel_fam = active_goal_info.family_label if active_goal_info else ""
        else:
            fam_str = ""
            sel_fam = ""

        tm.ik = IKMetrics(
            n_raw=len(spec.ik_goals),
            n_safe=len(all_goals),
            selected_goal_idx=selected_idx,
            selected_goal_rank=goal_rank,
            used_multi_ik=(len(selected_goals) > 1),
            ik_set_diversity=diversity,
            avg_pairwise_dist=diversity,
            goal_spread=spread,
            selected_family_label=sel_fam,
            n_families_available=len(set(g.family_label for g in all_goal_infos)) if all_goal_infos else 0,
            available_family_labels=fam_str,
            final_selected_goal_idx=selected_idx,
        )

        plan_success = (plan_result is not None and plan_result.success
                        and plan_result.path is not None)
        tm.plan = PlanMetrics(
            success=plan_success,
            time_s=t_plan,
            ik_classify_time_s=_t_classify_s,
            iterations=plan_result.iterations if plan_result else 0,
            nodes_explored=plan_result.nodes_explored if plan_result else 0,
            collision_checks=plan_result.collision_checks if plan_result else 0,
            path_length=path_len,
            n_waypoints=len(plan_result.path) if (plan_result and plan_result.path) else 0,
            goal_idx=selected_idx,
            planned_min_clearance=planned_min_cl,
            planned_mean_clearance=planned_mean_cl,
            planned_near_obs_fraction=planned_near_frac,
            interp_min_clearance=interp_min_cl,
            path_risk_score=risk_score,
            # Parallel planning instrumentation
            parallel_enabled=_parallel_timing.get("backend", "") not in ("", "sequential")
                              and _parallel_timing.get("n_candidates", 0) > 0,
            parallel_backend=_parallel_timing.get("backend", ""),
            parallel_max_workers=parallel_cfg.max_workers if use_family_planning else 0,
            parallel_candidate_count=_parallel_timing.get("n_candidates", 0),
            parallel_success_count=_parallel_timing.get("n_success", 0),
            parallel_wall_time_s=_parallel_timing.get("parallel_wall_time_s", 0.0),
            parallel_sum_candidate_s=_parallel_timing.get("sum_candidate_time_s", 0.0),
            parallel_speedup_estimate=_parallel_timing.get("speedup_estimate", 0.0),
        )

        if not plan_success:
            # Planning failed — execution metrics default to failure
            tm.execution = ExecutionMetrics(terminal_success=False,
                                             final_goal_err=float("inf"))
            tm.wall_time_s = time.perf_counter() - t_start
            return tm

        # ---- Vanilla DS + Differential IK execution path ----------------
        if use_diffik_ds:
            tm = _run_diffik_trial(
                tm, env, spec, q_start, active_q_goal, dt, n_exec_steps, t_start,
            )
            return tm

        # ---- Build DS and controller ------------------------------------
        ctrl_params = spec.controller or {}
        ds_cfg = DSConfig(
            K_c=ctrl_params.get("K_c", 2.0),
            K_r=ctrl_params.get("K_r", 1.0),
            K_n=ctrl_params.get("K_n", 0.3),
            goal_radius=ctrl_params.get("ds_goal_radius", spec.goal_radius),
            max_speed=ctrl_params.get("max_speed", float("inf")),
        )
        ds = PathDS(plan_result.path, config=ds_cfg)

        # ---- Path-tube tracker (primary controller near obstacles) ------
        tube_cfg = PathTubeConfig(
            enabled=True,
            switch_clearance=ctrl_params.get("tube_switch_clearance", 0.08),
            hard_switch_clearance=ctrl_params.get("tube_hard_clearance", 0.05),
            exit_clearance=ctrl_params.get("tube_exit_clearance", 0.10),
            tangent_gain=ctrl_params.get("tube_tangent_gain", 1.0),
            path_attract_gain=ctrl_params.get("tube_attract_gain", 2.0),
        )
        tube_tracker = PathTubeTracker(plan_result.path, config=tube_cfg)
        _in_tube_mode: bool = False   # hysteretic mode flag

        ctrl_cfg, tank_cfg, pf_cfg = build_ctrl_config(
            condition.ctrl,
            d_gain=ctrl_params.get("d_gain", 5.0),
            gravity_fn=grav_fn,
        )
        if cbf_override is not None:
            ctrl_cfg = dataclasses.replace(ctrl_cfg, cbf=cbf_override)
        tank = EnergyTank(tank_cfg)

        # ---- Stall / trap detection setup -------------------------------
        stall_cfg = StallDetectionConfig()
        trap_cfg  = TrapDetectionConfig()
        stall_history = StallHistory()
        blacklisted_goal_indices: List[int] = []
        blacklisted_families: List[str] = []
        family_switch_count = 0
        replan_count = 0
        first_stall_diag: Optional[StallDiagnostics] = None
        stall_step_global: Optional[int] = None
        near_graze_threshold = 0.015

        # ---- Escape mode state -------------------------------------------
        escape_policy = EscapePolicy()
        escape_metrics = EscapeMetrics()

        # ---- Morse escape planner setup ---------------------------------
        _morse_kw = dict(morse_override or {})
        if "negative_curvature" in _morse_kw and isinstance(
            _morse_kw["negative_curvature"], dict
        ):
            _morse_kw["negative_curvature"] = NegativeCurvatureConfig(
                **_morse_kw["negative_curvature"]
            )
        _morse_cfg = MorseEscapeConfig(**_morse_kw)
        _morse_planner = MorseEscapePlanner(_morse_cfg) if _morse_cfg.enabled else None
        _morse_active: bool = False
        _morse_prefix_queue: list = []

        # Fast Morse controller + supervisor (reactive escape, no rollout)
        _fast_cfg         = FastMorseEscapeConfig()
        _supervisor_cfg   = MorseSupervisorConfig(**(supervisor_override or {}))
        _fast_morse_ctrl  = FastMorseEscapeController(_fast_cfg) if _fast_cfg.enabled else None
        _fast_supervisor  = MorseEscapeSupervisor(_supervisor_cfg) if _supervisor_cfg.enabled else None
        _fast_compute_times_ms: list = []

        current_ds_mode  = "normal"
        current_escape_target: Optional[np.ndarray] = None
        _escape_clearance_at_entry: float = float("inf")
        _escape_cbf_active: List[bool] = []
        _escape_cbf_norms: List[float] = []
        _path_progress_at_escape: float = 0.0

        # ---- Path deviation accumulators ---------------------------------
        _path_deviations_q: List[float] = []
        _tube_mode_steps:   int = 0
        _goal_errors:       List[float] = []
        _stall_mask:        List[bool]  = []

        # ---- Execution loop (MuJoCo physics) ----------------------------
        q_goal = active_q_goal.copy()
        q    = env.q.copy()
        qdot = env.qdot.copy()

        powers_before: List[float] = []
        powers_after:  List[float] = []
        tank_levels:   List[float] = []
        beta_Rs:       List[float] = []
        clipped_steps: int = 0
        ever_in_goal:  bool = False
        conv_step:     Optional[int] = None
        exec_path_len: float = 0.0
        q_prev = q.copy()

        # CBF aggregate tracking (populated when ctrl_cfg.cbf is enabled)
        cbf_clearances:    List[float] = []
        cbf_corrections:   List[float] = []
        cbf_active_steps:  int = 0
        cbf_slack_steps:   int = 0
        cbf_override_steps: int = 0
        cbf_d_safe = (ctrl_cfg.cbf.d_safe if ctrl_cfg.cbf is not None else 0.03)
        cbf_d_buffer = (ctrl_cfg.cbf.d_buffer if ctrl_cfg.cbf is not None else 0.05)
        _cbf_on = ctrl_cfg.cbf is not None and ctrl_cfg.cbf.enabled
        _mod_on = ctrl_cfg.modulation is not None and ctrl_cfg.modulation.enabled
        _col_obs = spec.collision_obstacles() if (_cbf_on or _mod_on) else None

        # Modulation aggregate tracking
        mod_gamma_mins:    List[float] = []
        mod_corrections:   List[float] = []
        mod_active_steps:  int = 0
        mod_near_surface_steps: int = 0

        # Hard shield aggregate tracking
        _shield_on = ctrl_cfg.hard_shield is not None and ctrl_cfg.hard_shield.enabled
        shield_triggered_steps: int = 0
        shield_forced_stops:    int = 0
        shield_pred_clearances: List[float] = []
        shield_rejected_steps:  int = 0
        # DS-CBF conflict angle diagnostics (Section 7)
        cbf_high_angle_steps:        int = 0   # steps with correction_angle > 45°
        cbf_center_post_activations: int = 0   # steps where center_post was most critical

        # Goal-CBF conflict angle tracking (for CBFGoalConflictDiagnostics)
        cbf_goal_conflict_angles: List[float] = []   # correction_angle when CBF actively fired
        cbf_correction_angles:    List[float] = []   # correction_angle at every step
        _goal_errors_by_step:     List[float] = []   # ||q - q_goal|| at each step

        # Preflight values (set before execution loop; used in B5 population block)
        _goal_cl:           float = float("inf")
        _activation_cl:     float = float("inf")
        _straight_line_min_cl: float = float("inf")

        # ---- Per-step timing arrays ----------------------------------------
        # All arrays are parallel: index i = step i.
        # Timing boundary identical for all conditions:
        #   _t_ctrl set just before ctrl_step(); timer stopped after env.step().
        _ctrl_step_times:   List[float] = []   # total: ctrl_step + env.step
        _ctrl_compute_times: List[float] = []  # compute only: ctrl_step

        # Per-regime timing buckets (total step time split by obstacle regime)
        _times_near_obs:  List[float] = []   # clearance < NEAR_OBS_THRESH
        _times_far_obs:   List[float] = []   # clearance >= NEAR_OBS_THRESH
        _times_cbf_on:    List[float] = []   # CBF active
        _times_cbf_off:   List[float] = []   # CBF inactive

        # Per-step regime flags (used to count steps in each regime)
        _step_near_obs:   List[bool] = []
        _step_cbf_active: List[bool] = []
        _step_mod_active: List[bool] = []
        _step_tube_mode:  List[bool] = []
        _step_stall_mode: List[bool] = []

        NEAR_OBS_THRESH = 0.05   # metres — same as CBF d_buffer

        # ---- Barrier metric accumulators ------------------------------------
        _bar_clearances:   List[float] = []
        _bar_ee_positions: List[np.ndarray] = []
        _bar_jpl:          float = 0.0   # joint path length
        _bar_stalls:       int   = 0
        _bar_grazes:       int   = 0
        _bar_collisions:   int   = 0
        _bar_progress:     List[float] = []
        _bar_clearance_fn  = _make_clearance_fn(spec) if spec.obstacles else None
        _bar_per_obs_fn    = _make_per_obstacle_clearance_fn(spec) if spec.obstacles else None
        _bar_clearances_per_obs: dict = {}   # obs_name -> List[float]
        _bar_q_prev        = q.copy()
        _bar_ee_prev: Optional[np.ndarray] = None

        if condition.ctrl == ControlCondition.WAYPOINT_PD:
            # Waypoint-following PD baseline
            q, qdot, exec_metrics, pass_metrics = _run_waypoint_pd(
                env, plan_result.path, q_goal, grav_fn, dt, n_exec_steps,
                d_gain=ctrl_params.get("d_gain", 5.0),
                noise_std=noise_std, rng=rng,
                perturb_magnitude=perturb_magnitude,
                perturb_joint=perturb_joint,
                perturb_start=perturb_start,
                perturb_duration=perturb_duration,
            )
            tm.execution = exec_metrics
            tm.passivity = pass_metrics
        else:
            # ------------------------------------------------------------------
            # Goal clearance for final approach gate — computed whenever a
            # clearance function is available, regardless of whether CBF is on.
            # Keeps float("inf") sentinel out of impedance.step() so that the
            # final approach mode only activates for a verified collision-free
            # goal.
            # ------------------------------------------------------------------
            _goal_cl_for_ctrl: Optional[float] = None
            if _bar_clearance_fn is not None and q_goal is not None:
                _goal_cl_for_ctrl = float(_bar_clearance_fn(q_goal))

            # ------------------------------------------------------------------
            # Preflight: warn if goal lies inside CBF activation buffer
            # ------------------------------------------------------------------
            if _cbf_on and _bar_clearance_fn is not None and q_goal is not None:
                _goal_cl = float(_bar_clearance_fn(q_goal))
                _activation_cl = cbf_d_safe + cbf_d_buffer
                _straight_line_min_cl = float("inf")
                # Sample 30 points along q_start → q_goal straight line
                for _alpha in np.linspace(0.0, 1.0, 30):
                    _q_interp = (1.0 - _alpha) * q + _alpha * q_goal
                    _cl_interp = float(_bar_clearance_fn(_q_interp))
                    if _cl_interp < _straight_line_min_cl:
                        _straight_line_min_cl = _cl_interp
                if _goal_cl < _activation_cl:
                    print(
                        f"[CBF-PREFLIGHT] WARNING: selected goal lies inside CBF activation buffer. "
                        f"goal_clearance={_goal_cl:.4f}m  activation_clearance={_activation_cl:.4f}m  "
                        f"(d_safe={cbf_d_safe:.3f}  d_buffer={cbf_d_buffer:.3f})  "
                        f"straight_line_min_cl={_straight_line_min_cl:.4f}m. "
                        f"PathDS may not converge unless margins are relaxed near goal."
                    )

            global_step = 0
            for i in range(n_exec_steps):
                # Optional Gaussian state noise
                if noise_std > 0:
                    q    = q    + rng.standard_normal(len(q))    * noise_std
                    qdot = qdot + rng.standard_normal(len(qdot)) * noise_std

                # Pre-step clearance for DS proximity-aware f_c scaling
                _pre_clearance: Optional[float] = None
                if _bar_clearance_fn is not None:
                    _pre_clearance = _bar_clearance_fn(q)

                # ---- Path-tube tracking (primary near-obstacle controller) --
                _tube_override: Optional[np.ndarray] = None
                if tube_cfg.enabled and _pre_clearance is not None and use_family_planning:
                    _in_tube_mode = tube_tracker.in_tube_mode(_pre_clearance, _in_tube_mode)
                    if _in_tube_mode:
                        _tube_mode_steps += 1
                        w_path, w_ds = tube_tracker.blend_weights(_pre_clearance)
                        qdot_path = tube_tracker.qdot_nom(q, clearance=_pre_clearance)
                        if w_ds < 1.0:
                            qdot_ds_nom = ds.f(q, beta_R=1.0, clearance=_pre_clearance)
                            _tube_override = w_path * qdot_path + w_ds * qdot_ds_nom

                # Always track path deviation for both conditions
                from src.solver.ds.path_tracking import distance_to_path_q
                _dev_q = distance_to_path_q(q, ds.path)
                _path_deviations_q.append(_dev_q)

                # ---- Morse prefix execution (pre-validated kinematic prefix) ----
                if _morse_prefix_queue:
                    q_cmd = _morse_prefix_queue[0]
                    # Safety check: abort if current state or next target is unsafe
                    _curr_cl = _bar_clearance_fn(q) if _bar_clearance_fn else 1.0
                    _cmd_cl  = _bar_clearance_fn(q_cmd) if _bar_clearance_fn else 1.0
                    if (_curr_cl < _morse_cfg.collision_clearance_threshold or
                            _cmd_cl < _morse_cfg.collision_clearance_threshold):
                        _morse_prefix_queue.clear()
                        _morse_active = False
                        if escape_metrics.morse is None:
                            escape_metrics.morse = MorseEscapeMetrics()
                        escape_metrics.morse.prefix_aborted = True
                        escape_metrics.morse.fallback_used = "existing_escape"
                    else:
                        _morse_prefix_queue.pop(0)
                        current_ds_mode = "bridge_target"
                        current_escape_target = q_cmd
                        if not _morse_prefix_queue:
                            _morse_active = False

                # ---- Escape mode: compute escape velocity from policy -------
                _esc_vel: Optional[np.ndarray] = None
                if current_ds_mode != "normal" and _tube_override is None:
                    _esc_vel = escape_policy.escape_velocity(
                        q,
                        clearance_fn=_bar_clearance_fn,
                        n_dof=len(q),
                    )

                # Fast Morse reactive escape: inject as nominal velocity override
                _fast_morse_override: Optional[np.ndarray] = None
                if (_fast_supervisor is not None and _fast_supervisor.state.active
                        and _bar_clearance_fn is not None):
                    _fast_result = _fast_morse_ctrl.compute_qdot(
                        q=q,
                        ds_qdot=ds.f(q, beta_R=1.0),
                        jacobian_fn=lambda _q: env.jacobian(_q),
                        clearance_fn=_bar_clearance_fn,
                        goal_error_fn=lambda _q, _qg=q_goal.copy(): float(
                            np.linalg.norm(_q - _qg)
                        ),
                        ik_infos=all_goal_infos or [],
                        cached_tangent_dirs=_fast_supervisor.state.cached_tangent_dirs,
                        backtrack_dir=_fast_supervisor.state.cached_backtrack_dir,
                    )
                    _fast_compute_times_ms.append(_fast_result.compute_time_ms)
                    if _fast_result.active:
                        _fast_morse_override = _fast_result.qdot_escape
                        if escape_metrics.fast_morse is None:
                            escape_metrics.fast_morse = FastMorseMetrics(
                                activated=True, activation_step=i,
                            )
                        escape_metrics.fast_morse.active_steps += 1

                # ---- Regime flags (set BEFORE timing window so they are
                #      independent of what the controller does this step) ------
                _is_near_obs  = (_pre_clearance is not None
                                 and _pre_clearance < NEAR_OBS_THRESH)
                _is_stall_mode = (current_ds_mode != "normal")
                _is_tube_mode  = bool(_in_tube_mode)

                _nom_override = _tube_override if _tube_override is not None else _fast_morse_override
                _t_ctrl = time.perf_counter()
                res = ctrl_step(q, qdot, ds, tank, dt, config=ctrl_cfg,
                               obstacles=_col_obs, clearance=_pre_clearance,
                               ds_mode=current_ds_mode,
                               ds_escape_target=current_escape_target,
                               escape_velocity=_esc_vel,
                               nominal_velocity_override=_nom_override,
                               q_goal=q_goal,
                               goal_clearance=_goal_cl_for_ctrl)
                _t_after_ctrl = time.perf_counter()
                _ctrl_compute_times.append(_t_after_ctrl - _t_ctrl)
                tau = res.tau.copy()

                # Collect CBF per-step diagnostics
                cbf_active_this_step = False
                cbf_correction_this = 0.0
                if res.cbf is not None:
                    if res.cbf.min_clearance is not None:
                        cbf_clearances.append(res.cbf.min_clearance)
                    if res.cbf.n_active > 0:
                        cbf_active_steps += 1
                        cbf_active_this_step = True
                    if res.cbf.cbf_slack_used:
                        cbf_slack_steps += 1
                    if res.cbf.allowed_contact_override_used:
                        cbf_override_steps += 1
                    cbf_corrections.append(res.cbf.correction_norm)
                    cbf_correction_this = res.cbf.correction_norm
                    # DS-CBF conflict angle diagnostics
                    if res.cbf.correction_angle_deg > 45.0:
                        cbf_high_angle_steps += 1
                    if res.cbf.most_critical_obstacle == "center_post":
                        cbf_center_post_activations += 1

                    # CBFGoalConflictDiagnostics per-step tracking
                    # correction_angle_deg is the angle between qdot_nom (DS, goal-directed)
                    # and qdot_safe (CBF output) — this IS the goal-CBF conflict angle.
                    cbf_correction_angles.append(res.cbf.correction_angle_deg)
                    if res.cbf.n_active > 0 and res.cbf.correction_norm > 1e-6:
                        cbf_goal_conflict_angles.append(res.cbf.correction_angle_deg)

                # Track goal error at each step (needed for final-20% progress)
                _goal_errors_by_step.append(float(np.linalg.norm(q - q_goal)))

                # Hard shield per-step diagnostics
                if res.hard_shield is not None:
                    shield_pred_clearances.append(res.hard_shield.predicted_clearance)
                    if res.hard_shield.shield_triggered:
                        shield_triggered_steps += 1
                    if res.hard_shield.forced_stop:
                        shield_forced_stops += 1
                    if res.hard_shield.accepted_scale < 1.0:
                        shield_rejected_steps += 1

                # Modulation per-step diagnostics
                _is_mod_active = False
                if res.modulation is not None:
                    mod_gamma_mins.append(res.modulation.gamma_min)
                    mod_corrections.append(res.modulation.correction_norm)
                    if res.modulation.gamma_min < 2.0:
                        mod_active_steps += 1
                        _is_mod_active = True
                    if res.modulation.gamma_min < 1.1:
                        mod_near_surface_steps += 1

                # Optional perturbation
                if perturb_magnitude > 0 and perturb_start <= i < perturb_start + perturb_duration:
                    tau[perturb_joint] += perturb_magnitude

                env.step(tau, dt=dt)
                _step_total = time.perf_counter() - _t_ctrl
                _ctrl_step_times.append(_step_total)

                # Regime-split timing buckets
                if _is_near_obs:
                    _times_near_obs.append(_step_total)
                else:
                    _times_far_obs.append(_step_total)
                if cbf_active_this_step:
                    _times_cbf_on.append(_step_total)
                else:
                    _times_cbf_off.append(_step_total)

                # Regime flag arrays (for step-count derivation)
                _step_near_obs.append(_is_near_obs)
                _step_cbf_active.append(cbf_active_this_step)
                _step_mod_active.append(_is_mod_active)
                _step_tube_mode.append(_is_tube_mode)
                _step_stall_mode.append(_is_stall_mode)
                q    = env.q.copy()
                qdot = env.qdot.copy()
                if q_log is not None:
                    q_log.append(q.copy())

                # Barrier metric collection
                _cl_per_obs: Dict[str, float] = {}
                if _bar_clearance_fn is not None:
                    _cl = _bar_clearance_fn(q)
                    _bar_clearances.append(_cl)
                    if _cl < D_GRAZE_THRESH:
                        _bar_grazes += 1
                    if _cl < 0.0:
                        _bar_collisions += 1
                    _bar_jpl += float(np.linalg.norm(q - _bar_q_prev))
                    _bar_q_prev = q.copy()
                    try:
                        _ee_now = np.array(env.ee_pose(q)[0], dtype=float)
                    except Exception:
                        _ee_now = None
                    if _ee_now is not None:
                        _bar_ee_positions.append(_ee_now)
                        if _bar_ee_prev is not None:
                            _step_disp = float(np.linalg.norm(_ee_now - _bar_ee_prev))
                            _bar_progress.append(_step_disp)
                            if _step_disp < STALL_THRESH:
                                _bar_stalls += 1
                        _bar_ee_prev = _ee_now
                    # Per-obstacle clearance tracking
                    if _bar_per_obs_fn is not None:
                        _cl_per_obs = _bar_per_obs_fn(q)
                        for _obs_name, _obs_cl in _cl_per_obs.items():
                            if _obs_name not in _bar_clearances_per_obs:
                                _bar_clearances_per_obs[_obs_name] = []
                            _bar_clearances_per_obs[_obs_name].append(_obs_cl)

                exec_path_len += float(np.linalg.norm(q - q_prev))
                q_prev = q.copy()

                powers_before.append(res.pf_power_nom)
                powers_after.append(res.pf_power_filtered)
                tank_levels.append(res.tank_energy)
                beta_Rs.append(res.beta_R)
                if res.pf_clipped:
                    clipped_steps += 1

                goal_err = float(np.linalg.norm(q - q_goal))
                _goal_errors.append(goal_err)
                _stall_mask.append(
                    len(_bar_progress) > 0 and _bar_progress[-1] < STALL_THRESH
                )
                if goal_err < spec.goal_radius:
                    ever_in_goal = True
                    if conv_step is None:
                        conv_step = i

                # Early stop if converged
                if conv_step is not None and i > conv_step + 50:
                    break

                # ---- Collect escape-mode CBF diagnostics -------------------
                if current_ds_mode != "normal":
                    _escape_cbf_active.append(cbf_active_this_step)
                    _escape_cbf_norms.append(cbf_correction_this)

                # ---- Check escape exit condition ---------------------------
                if current_ds_mode != "normal":
                    _cl_now = _pre_clearance if _pre_clearance is not None else float("inf")
                    if escape_policy.is_escaped(_cl_now) or escape_policy.budget_exhausted():
                        # Return to normal mode
                        current_ds_mode = "normal"
                        current_escape_target = None
                        escape_metrics.escape_mode_total_steps += escape_policy.escape_steps
                        escape_metrics.resumed_normal_after_escape = True
                        escape_metrics.clearance_gain_during_escape = max(
                            0.0, _cl_now - _escape_clearance_at_entry
                        )
                        escape_metrics.path_progress_after_escape = ds.progress(q)
                        escape_policy.reset()
                        stall_history = StallHistory()   # fresh window after escape

                # ---- Stall / trap detection + recovery (Multi-IK only) ------
                if use_family_planning and conv_step is None and current_ds_mode == "normal":
                    near_graze_now = (
                        _bar_clearances[-1] < near_graze_threshold
                        if _bar_clearances else False
                    )
                    obs_near: Dict[str, bool] = {
                        name: (v < near_graze_threshold)
                        for name, v in _cl_per_obs.items()
                    }
                    _angle_this = (res.cbf.correction_angle_deg
                                   if res.cbf is not None else 0.0)
                    _cp_this    = (res.cbf is not None and
                                   res.cbf.most_critical_obstacle == "center_post")
                    stall_history.append(
                        goal_error=goal_err,
                        cbf_was_active=cbf_active_this_step,
                        correction_norm=cbf_correction_this,
                        near_graze=near_graze_now,
                        obstacle_near=obs_near,
                        correction_angle=_angle_this,
                        center_post_critical=_cp_this,
                    )
                    escape_policy.push_trajectory(q)

                    # --- Generic stall ---
                    diag = detect_stall(stall_history, stall_cfg)
                    if diag.stalled:
                        if first_stall_diag is None:
                            first_stall_diag = diag
                            first_stall_diag.stalled_goal_idx = (
                                active_goal_info.goal_idx if active_goal_info else selected_idx
                            )
                            first_stall_diag.stalled_family_label = (
                                active_goal_info.family_label if active_goal_info else ""
                            )
                            stall_step_global = i

                    # --- Trap detection (stricter) ---
                    trap_diag = detect_trap(stall_history, trap_cfg)
                    _force_morse_now = (
                        _morse_planner is not None
                        and _morse_cfg.force_at_step is not None
                        and i >= _morse_cfg.force_at_step
                        and not _morse_active
                    )
                    # Supervisor update (runs each step inside stall-detection block; multi-IK only)
                    if _fast_supervisor is not None:
                        _sup_was_active = _fast_supervisor.state.active
                        _fast_supervisor.update(
                            step=i, q=q, trap_diag=trap_diag,
                            clearance_fn=_bar_clearance_fn,
                            jacobian_fn=lambda _q: env.jacobian(_q),
                            ik_infos=all_goal_infos or [],
                            goal_error=goal_err,
                        )
                        # Stamp metrics on the step the supervisor first activates
                        if not _sup_was_active and _fast_supervisor.state.active:
                            if escape_metrics.fast_morse is None:
                                escape_metrics.fast_morse = FastMorseMetrics(
                                    activated=True, activation_step=i,
                                )
                        # Natural deactivation (trap cleared or goal reached)
                        if _sup_was_active and not _fast_supervisor.state.active:
                            if escape_metrics.fast_morse is not None:
                                escape_metrics.fast_morse.escaped_trap = True
                            _fast_morse_ctrl.reset()
                        # Escalation: reactive escape made no progress
                        if _fast_supervisor.should_escalate():
                            _fast_supervisor.deactivate("no_progress")
                            _fast_morse_ctrl.reset()
                            if escape_metrics.fast_morse is not None:
                                escape_metrics.fast_morse.escaped_trap = False
                                escape_metrics.fast_morse.fallback_used = "existing_escape"
                            # Activate EscapePolicy fallback only if permitted
                            if _supervisor_cfg.enable_full_rollout_fallback:
                                if escape_policy.config.allow_backtracking:
                                    current_ds_mode = "backtrack"
                                    escape_policy.start_escape(EscapeMode.BACKTRACK)
                                    escape_metrics.backtrack_steps = len(escape_policy._backtrack_buf)
                                else:
                                    current_ds_mode = "escape_clearance"
                                    escape_policy.start_escape(EscapeMode.ESCAPE_CLEARANCE)
                    if (trap_diag.trapped or _force_morse_now) and not escape_metrics.trap_detected:
                        escape_metrics.trap_detected = True
                        escape_metrics.trap_step = i
                        escape_metrics.trap_reason = trap_diag.trap_reason
                        _escape_clearance_at_entry = (
                            _pre_clearance if _pre_clearance is not None else 0.0
                        )
                        escape_metrics.path_progress_before_escape = ds.progress(q)

                        # Choose escape strategy
                        escape_metrics.escape_mode_activations += 1

                        # Supervisor handles reactive escape; skip full rollout
                        _supervisor_handling = (
                            _fast_supervisor is not None and _fast_supervisor.state.active
                        )
                        if not _supervisor_handling:
                            # ---- Morse escape (second-line, before existing escape modes) ----
                            if _morse_planner is not None and not _morse_active:
                                _qg_snap = q_goal.copy()
                                _goal_error_fn = lambda _q, _qg=_qg_snap: float(np.linalg.norm(_q - _qg))
                                _energy_fn = lambda _q: ds.V(_q)
                                _jacobian_fn = lambda _q: env.jacobian(_q)
                                _morse_result = _morse_planner.plan_escape(
                                    q=q,
                                    stall_diag=diag,
                                    trap_diag=trap_diag,
                                    ds=ds,
                                    clearance_fn=_bar_clearance_fn,
                                    goal_error_fn=_goal_error_fn,
                                    energy_fn=_energy_fn,
                                    jacobian_fn=_jacobian_fn,
                                    ik_infos=all_goal_infos if all_goal_infos else [],
                                    escape_policy=escape_policy,
                                    seed=seed + i,
                                )
                                escape_metrics.morse = _morse_result.metrics

                                if _morse_result.success and _morse_result.action is not None:
                                    _action = _morse_result.action
                                    if _action.mode == "prefix" and _morse_result.execute_prefix_qs:
                                        _morse_prefix_queue = list(_morse_result.execute_prefix_qs)
                                        _morse_active = True
                                        escape_metrics.morse.escape_succeeded = True
                                        continue  # skip existing escape activation
                                    elif _action.escape_policy_mode is not None:
                                        _mode_map = {
                                            "ESCAPE_CLEARANCE": EscapeMode.ESCAPE_CLEARANCE,
                                            "BACKTRACK": EscapeMode.BACKTRACK,
                                            "BRIDGE_TARGET": EscapeMode.BRIDGE_TARGET,
                                            "REDUCED_FC": EscapeMode.REDUCED_FC,
                                        }
                                        _ep_mode = _mode_map.get(
                                            _action.escape_policy_mode, EscapeMode.ESCAPE_CLEARANCE
                                        )
                                        _params = _action.escape_policy_params or {}
                                        escape_policy.start_escape(
                                            _ep_mode,
                                            bridge_target=_params.get("bridge_target"),
                                            preferred_escape_dir=_params.get("preferred_escape_dir"),
                                        )
                                        _morse_active = True
                                        escape_metrics.morse.escape_succeeded = True
                                        continue  # skip existing escape activation

                            # Fallback: activate existing escape policy unless Morse
                            # explicitly suppresses it. fallback_to_birrt relies on
                            # the family-switch replan below when stall persists.
                            _use_existing_escape = (
                                _morse_planner is None
                                or _morse_cfg.fallback_to_existing_escape
                            )
                            if _use_existing_escape:
                                if escape_policy.config.allow_backtracking:
                                    current_ds_mode = "backtrack"
                                    escape_policy.start_escape(EscapeMode.BACKTRACK)
                                    escape_metrics.backtrack_steps = len(escape_policy._backtrack_buf)
                                else:
                                    current_ds_mode = "escape_clearance"
                                    escape_policy.start_escape(EscapeMode.ESCAPE_CLEARANCE)

                    # --- Family switch after stall (if not in escape mode) ---
                    if diag.stalled and replan_count < MAX_REPLANS and not disable_birrt:
                        if active_goal_info is not None:
                            blacklisted_goal_indices.append(active_goal_info.goal_idx)
                            blacklisted_families.append(active_goal_info.family_label)

                        next_info = select_escape_family(
                            all_goal_infos if all_goal_infos else [],
                            blacklisted_families,
                            blacklisted_goal_indices,
                        ) or best_alternative(
                            all_goal_infos if all_goal_infos else [],
                            blacklisted_goal_indices,
                            blacklisted_families,
                        )
                        if next_info is not None:
                            # Scaffold: surviving waypoints from current path
                            # (latter half, still collision-free) to warm-start
                            # the replan tree.
                            _scaffold: List[np.ndarray] = []
                            if plan_result is not None and plan_result.path:
                                orig = plan_result.path
                                step = max(1, len(orig) // 8)
                                for _wi in range(len(orig) // 2, len(orig), step):
                                    _wp = orig[_wi]
                                    if plan_col_fn(_wp):
                                        _scaffold.append(_wp)

                            replan_result = _plan_to_single_goal(
                                q, next_info.q_goal, plan_col_fn,
                                PlannerConfig(
                                    max_iterations=plan_cfg.max_iterations,
                                    step_size=plan_cfg.step_size,
                                    min_step_size=plan_cfg.min_step_size,
                                    clearance_step_scale=plan_cfg.clearance_step_scale,
                                    clearance_step_threshold=plan_cfg.clearance_step_threshold,
                                    goal_bias=plan_cfg.goal_bias,
                                    gaussian_bias=plan_cfg.gaussian_bias,
                                    gaussian_std=plan_cfg.gaussian_std,
                                    shortcut_iterations=plan_cfg.shortcut_iterations,
                                    shortcut_n_check=plan_cfg.shortcut_n_check,
                                    shortcut_clearance_margin=plan_cfg.shortcut_clearance_margin,
                                    seed=seed + replan_count + 1,
                                ),
                                clearance_fn=_bar_clearance_fn,
                                scaffold_waypoints=_scaffold or None,
                            )
                            if replan_result.success and replan_result.path:
                                ds = PathDS(replan_result.path, config=ds_cfg)
                                tube_tracker = PathTubeTracker(replan_result.path, config=tube_cfg)
                                _in_tube_mode = False
                                q_goal = next_info.q_goal.copy()
                                active_goal_info = next_info
                                replan_count += 1
                                family_switch_count += 1
                                escape_metrics.family_switch_after_escape = True
                                escape_metrics.replan_after_escape = True
                                stall_history = StallHistory()

                global_step += 1

            final_err = float(np.linalg.norm(q - q_goal))
            _n_steps_done = max(1, global_step)
            _tube_frac = _tube_mode_steps / _n_steps_done

            # ---- Success diagnostics via shared evaluator -------------------
            _succ_cfg = SuccessConfig(
                goal_radius=spec.goal_radius,
                sustained_steps=0,
                require_no_collision=False,
                min_clearance_threshold=0.0,
            )
            _succ_diag = evaluate_success(
                goal_errors=_goal_errors,
                clearances=_bar_clearances if _bar_clearances else None,
                config=_succ_cfg,
                stall_mask=_stall_mask if _stall_mask else None,
            )

            # ---- Path deviation and homotopy verdict -----------------------
            _max_dev  = float(np.max(_path_deviations_q)) if _path_deviations_q else 0.0
            _mean_dev = float(np.mean(_path_deviations_q)) if _path_deviations_q else 0.0
            _pt_verdict = classify_path_tracking(
                _mean_dev, _max_dev, tube_tracker.tube_exit_count, _n_steps_done
            )

            # ---- Path progress fraction ------------------------------------
            _path_prog = float(ds.progress(q)) if hasattr(ds, "progress") else 0.0

            # ---- Multi-IK effect verdict ------------------------------------
            _n_fam = (tm.ik.n_families_available if tm.ik else 0)
            _mik_effect = classify_multiik_effect(
                n_families_available=_n_fam,
                family_switch_count=family_switch_count,
                terminal_success=_succ_diag.terminal_success,
            ) if use_family_planning else ""

            tm.execution = ExecutionMetrics(
                terminal_success=_succ_diag.terminal_success,
                final_goal_err=final_err,
                ever_in_goal=ever_in_goal,
                convergence_step=conv_step,
                convergence_time_s=(conv_step * dt if conv_step is not None else None),
                exec_path_length=exec_path_len,
                path_deviation=0.0,
                stall_step=stall_step_global,
                stall_reason=first_stall_diag.stall_reason if first_stall_diag else "",
                family_switch_count=family_switch_count,
                n_replans=replan_count,
                # Path-tube metrics
                max_path_deviation_q=_max_dev,
                mean_path_deviation_q=_mean_dev,
                time_in_path_tube_fraction=_tube_frac,
                path_mode_active_fraction=_tube_frac,
                anchor_freeze_count=tube_tracker.anchor_freeze_count,
                anchor_advance_count=tube_tracker.anchor_advance_count,
                tube_exit_count=tube_tracker.tube_exit_count,
                # Success diagnostics
                failure_reason=_succ_diag.failure_reason,
                min_goal_error_ever=_succ_diag.min_goal_error_ever,
                sustained_goal_steps=_succ_diag.stayed_in_goal_steps,
                path_progress_fraction=_path_prog,
                path_tracking_verdict=_pt_verdict,
                multiik_effect=_mik_effect,
            )

            # Back-fill IK family switch counts
            tm.ik.family_switch_count = family_switch_count
            tm.ik.replanned_count     = replan_count
            tm.ik.stalled_family_label = (
                first_stall_diag.stalled_family_label if first_stall_diag else ""
            )
            if active_goal_info is not None:
                tm.ik.final_selected_goal_idx = active_goal_info.goal_idx
                tm.ik.selected_family_label   = active_goal_info.family_label

            n = len(tank_levels)
            tm.passivity = PassivityMetrics(
                n_violations=sum(1 for p in powers_before if p > 1e-6),
                clipped_ratio=(clipped_steps / max(1, n)),
                mean_power_before=(float(np.mean(powers_before)) if powers_before else 0.0),
                mean_power_after=(float(np.mean(powers_after)) if powers_after else 0.0),
                min_tank_energy=(float(np.min(tank_levels)) if tank_levels else 0.0),
                final_tank_energy=(tank_levels[-1] if tank_levels else 0.0),
                beta_R_zero_fraction=(sum(1 for b in beta_Rs if b < 0.01) / max(1, n)),
            )

            # CBF aggregate metrics (only populated when CBF was enabled)
            if cbf_clearances or cbf_corrections:
                n_steps_done = max(1, n)
                near_grazing_eps = 0.005  # 5 mm above d_safe counts as near-grazing
                tm.cbf = CBFMetrics(
                    min_clearance_rollout=float(min(cbf_clearances)) if cbf_clearances else float("inf"),
                    cbf_active_fraction=cbf_active_steps / n_steps_done,
                    mean_correction_norm=float(np.mean(cbf_corrections)) if cbf_corrections else 0.0,
                    max_correction_norm=float(max(cbf_corrections)) if cbf_corrections else 0.0,
                    n_slack_activations=cbf_slack_steps,
                    n_near_grazing_events=sum(
                        1 for d in cbf_clearances if d < cbf_d_safe + near_grazing_eps
                    ),
                    n_contact_override_steps=cbf_override_steps,
                    n_unintended_collisions=sum(1 for d in cbf_clearances if d < 0),
                    high_angle_fraction=cbf_high_angle_steps / n_steps_done,
                    center_post_activation_count=cbf_center_post_activations,
                )

            # CBF/Goal conflict diagnostics (populated whenever CBF is enabled)
            if _cbf_on and _bar_clearance_fn is not None:
                _n_total = max(1, n)
                # Final 20% window
                _n20 = max(1, _n_total // 5)
                _active_final20 = sum(1 for b in _step_cbf_active[-_n20:] if b)
                _cbf_frac_final20 = _active_final20 / _n20

                # Progress in final 20%: decrease in goal error
                _goal_err_start20 = (
                    _goal_errors_by_step[-_n20]
                    if len(_goal_errors_by_step) > _n20 else float("inf")
                )
                _goal_err_end20 = _goal_errors_by_step[-1] if _goal_errors_by_step else float("inf")
                _progress20 = max(0.0, _goal_err_start20 - _goal_err_end20)

                # Failure classification
                _diag_failure_reason = ""
                if not (tm.execution and tm.execution.terminal_success):
                    _goal_inside_buf = (_goal_cl < _activation_cl)
                    if _goal_inside_buf and _cbf_frac_final20 > 0.5 and _progress20 < 0.01:
                        _diag_failure_reason = "cbf_goal_conflict"
                    elif tm.barrier and tm.barrier.collision_count > 0:
                        _diag_failure_reason = "collision"
                    elif _goal_err_end20 > 0.1:
                        _diag_failure_reason = "no_progress"

                tm.cbf_goal_conflict = CBFGoalConflictDiagnostics(
                    goal_clearance=_goal_cl,
                    activation_clearance=_activation_cl,
                    goal_inside_cbf_buffer=(_goal_cl < _activation_cl),
                    goal_inside_d_safe=(_goal_cl < cbf_d_safe),
                    straight_line_min_clearance=_straight_line_min_cl,
                    d_safe=cbf_d_safe,
                    d_buffer=cbf_d_buffer,
                    final_clearance=(_bar_clearances[-1] if _bar_clearances else float("inf")),
                    final_goal_error=(
                        float(np.linalg.norm(q - q_goal)) if q_goal is not None else float("inf")
                    ),
                    cbf_active_fraction_total=cbf_active_steps / _n_total,
                    cbf_active_fraction_final_20pct=_cbf_frac_final20,
                    mean_cbf_correction_norm=(
                        float(np.mean(cbf_corrections)) if cbf_corrections else 0.0
                    ),
                    max_cbf_correction_norm=(
                        float(max(cbf_corrections)) if cbf_corrections else 0.0
                    ),
                    mean_cbf_angle_deg=(
                        float(np.mean(cbf_correction_angles)) if cbf_correction_angles else 0.0
                    ),
                    max_cbf_angle_deg=(
                        float(max(cbf_correction_angles)) if cbf_correction_angles else 0.0
                    ),
                    mean_goal_cbf_conflict_angle_deg=(
                        float(np.mean(cbf_goal_conflict_angles)) if cbf_goal_conflict_angles else 0.0
                    ),
                    max_goal_cbf_conflict_angle_deg=(
                        float(max(cbf_goal_conflict_angles)) if cbf_goal_conflict_angles else 0.0
                    ),
                    progress_final_20pct=_progress20,
                    failure_reason=_diag_failure_reason,
                )

        # ---- HardShieldMetrics ----------------------------------------
        if _shield_on:
            tm.hard_shield = HardShieldMetrics(
                hard_shield_active_fraction=shield_triggered_steps / max(1, n_steps_done),
                hard_shield_forced_stop_count=shield_forced_stops,
                min_predicted_clearance=float(min(shield_pred_clearances))
                    if shield_pred_clearances else float("inf"),
                n_rejected_steps=shield_rejected_steps,
            )

        # ---- ModulationMetrics ----------------------------------------
        if _mod_on and (mod_gamma_mins or mod_corrections):
            tm.modulation = ModulationMetrics(
                min_gamma_rollout=float(min(mod_gamma_mins)) if mod_gamma_mins else float("inf"),
                modulation_active_fraction=mod_active_steps / max(1, n_steps_done),
                mean_correction_norm=float(np.mean(mod_corrections)) if mod_corrections else 0.0,
                max_correction_norm=float(max(mod_corrections)) if mod_corrections else 0.0,
                n_near_surface_steps=mod_near_surface_steps,
            )

        # ---- CtrlFreqMetrics ------------------------------------------
        if _ctrl_step_times:
            _st  = np.array(_ctrl_step_times)
            _ct  = np.array(_ctrl_compute_times) if _ctrl_compute_times else _st
            _near = np.array(_times_near_obs) if _times_near_obs else None
            _far  = np.array(_times_far_obs)  if _times_far_obs  else None
            _con  = np.array(_times_cbf_on)   if _times_cbf_on   else None
            _coff = np.array(_times_cbf_off)  if _times_cbf_off  else None
            _mean_compute = float(1e3 * np.mean(_ct))
            tm.ctrl_freq = CtrlFreqMetrics(
                # Step counts
                n_steps_executed=len(_st),
                n_steps_near_obstacle=sum(_step_near_obs),
                n_steps_cbf_active=sum(_step_cbf_active),
                n_steps_modulation_active=sum(_step_mod_active),
                n_steps_path_tracking_mode=sum(_step_tube_mode),
                n_steps_stall_logic=sum(_step_stall_mode),
                # Totals
                total_ctrl_time_s=float(np.sum(_st)),
                total_ctrl_compute_time_s=float(np.sum(_ct)),
                # All-step means
                mean_hz=float(1.0 / np.mean(_st)),
                mean_ms=float(1e3 * np.mean(_st)),
                p95_ms=float(1e3 * np.percentile(_st, 95)),
                max_ms=float(1e3 * np.max(_st)),
                mean_ctrl_compute_ms=_mean_compute,
                # Regime-split means
                mean_ctrl_ms_near_obstacle=    float(1e3 * np.mean(_near)) if _near is not None else 0.0,
                mean_ctrl_ms_far_from_obstacle=float(1e3 * np.mean(_far))  if _far  is not None else 0.0,
                # Backward-compat aliases
                n_steps=len(_st),
                ctrl_compute_ms=_mean_compute,
                ctrl_ms_cbf_active=  float(1e3 * np.mean(_con))  if _con  is not None else 0.0,
                ctrl_ms_cbf_inactive=float(1e3 * np.mean(_coff)) if _coff is not None else 0.0,
            )

        # Fast Morse timing summary
        if escape_metrics.fast_morse is not None and _fast_compute_times_ms:
            escape_metrics.fast_morse.mean_compute_time_ms = float(
                np.mean(_fast_compute_times_ms)
            )
            escape_metrics.fast_morse.max_compute_time_ms = float(
                np.max(_fast_compute_times_ms)
            )

        # ---- EscapeMetrics --------------------------------------------
        if use_family_planning and escape_metrics.trap_detected:
            if _escape_cbf_active:
                escape_metrics.escape_cbf_active_fraction = (
                    sum(_escape_cbf_active) / len(_escape_cbf_active)
                )
            if _escape_cbf_norms:
                escape_metrics.escape_mean_correction_norm = float(
                    np.mean(_escape_cbf_norms)
                )
            escape_metrics.escape_success = (
                escape_metrics.resumed_normal_after_escape
                and tm.execution is not None
                and tm.execution.terminal_success
            )
            tm.escape = escape_metrics

        # ---- BarrierMetrics -------------------------------------------
        if _bar_clearance_fn is not None and _bar_clearances:
            ee_arr      = np.array(_bar_ee_positions) if _bar_ee_positions else None
            task_path   = float(np.sum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))) \
                          if ee_arr is not None and len(ee_arr) > 1 else 0.0
            straight    = float(np.linalg.norm(ee_arr[-1] - ee_arr[0])) \
                          if ee_arr is not None and len(ee_arr) > 1 else 0.0
            efficiency  = straight / task_path if task_path > 1e-6 else 0.0
            pr          = np.array(_bar_progress)
            if len(pr) > 2:
                osc = float(np.mean(np.abs(np.diff(np.sign(pr - float(np.mean(pr)))))))
            else:
                osc = 0.0
            n_blocked = max(0, len(spec.ik_goals) - len(selected_goals))
            # Check if selected goal is collision-free via col_fn
            sel_cleared = False
            if 0 <= selected_idx < len(selected_goals):
                sel_cleared = bool(col_fn(selected_goals[selected_idx]))

            # Per-obstacle clearance summaries
            def _obs_min_cl(name: str) -> float:
                vals = _bar_clearances_per_obs.get(name)
                return float(np.min(vals)) if vals else float("inf")

            def _obs_active_frac(name: str) -> float:
                vals = _bar_clearances_per_obs.get(name)
                if not vals:
                    return 0.0
                return float(sum(1 for v in vals if v < D_GRAZE_THRESH) / len(vals))

            tm.barrier = BarrierMetrics(
                min_clearance=float(np.min(_bar_clearances)),
                mean_clearance=float(np.mean(_bar_clearances)),
                near_graze_count=_bar_grazes,
                collision_count=_bar_collisions,
                joint_path_length=_bar_jpl,
                task_path_length=task_path,
                path_efficiency=efficiency,
                stall_steps=_bar_stalls,
                oscillation_index=osc,
                n_ik_goals_blocked=n_blocked,
                selected_goal_cleared_barrier=sel_cleared,
                min_clearance_center_post=_obs_min_cl("center_post"),
                min_clearance_top_bar=_obs_min_cl("top_bar"),
                min_clearance_bottom_bar=_obs_min_cl("bottom_bar"),
                active_fraction_center_post=_obs_active_frac("center_post"),
                active_fraction_top_bar=_obs_active_frac("top_bar"),
                active_fraction_bottom_bar=_obs_active_frac("bottom_bar"),
            )

        # Robustness flag
        tm.robustness = RobustnessMetrics(
            success_under_noise=(tm.execution.terminal_success if noise_std > 0 else False),
            success_under_perturbation=(tm.execution.terminal_success if perturb_magnitude > 0 else False),
            final_error_variance=0.0,
            worst_case_final_err=tm.execution.final_goal_err,
        )

    except Exception as exc:
        tm.error = traceback.format_exc()

    tm.wall_time_s = time.perf_counter() - t_start
    return tm


# ---------------------------------------------------------------------------
# Vanilla DS + Differential IK execution helper
# ---------------------------------------------------------------------------

def _run_diffik_trial(
    tm:            TrialMetrics,
    env:           SimEnv,
    spec:          ScenarioSpec,
    q_start:       np.ndarray,
    q_goal:        np.ndarray,
    dt:            float,
    n_exec_steps:  int,
    t_start:       float,
) -> TrialMetrics:
    """
    Run a vanilla task-space DS + differential IK + modulation trial.

    No BiRRT, no PathDS, no family switching.  May get stuck in local minima.
    The robot moves directly toward the task-space goal using a Cartesian DS
    attractor, modulation obstacle avoidance, and damped differential IK.
    """
    from src.solver.controller.vanilla_ds_diffik import (
        VanillaDSDiffIKController, VanillaDSConfig, DiffIKConfig,
    )
    from src.solver.ds.modulation import ModulationConfig
    from src.solver.controller.hard_shield import HardShieldConfig, enforce_hard_clearance

    ctrl_params  = spec.controller or {}
    col_obs      = spec.collision_obstacles()

    # ---- Get task-space goal via FK of the IK goal ----------------------
    x_goal, _ = env.ee_pose(q_goal)
    goal_tol_m = ctrl_params.get("diffik_goal_tol_m", 0.01)  # 1 cm

    # ---- Controller configs --------------------------------------------
    ds_cfg = VanillaDSConfig(
        pos_gain=ctrl_params.get("diffik_pos_gain", 2.0),
        max_task_speed=ctrl_params.get("diffik_max_task_speed", 0.25),
        goal_tolerance=goal_tol_m,
    )
    ik_cfg = DiffIKConfig(
        damping=ctrl_params.get("diffik_damping", 1e-2),
        max_joint_speed=ctrl_params.get("diffik_max_joint_speed", 1.0),
        use_nullspace_posture=True,
        nullspace_gain=ctrl_params.get("diffik_nullspace_gain", 0.05),
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

    # ---- Execution loop -------------------------------------------------
    q    = env.q.copy()
    qdot = env.qdot.copy()

    ever_in_goal    = False
    conv_step: Optional[int] = None
    exec_path_len   = 0.0
    q_prev          = q.copy()
    goal_err_joint  = float("inf")

    clearances:     List[float] = []
    mod_gammas:     List[float] = []
    mod_active:     int = 0
    n_grazes:       int = 0
    n_collisions:   int = 0
    step_times:     List[float] = []

    _clearance_fn = _make_clearance_fn(spec) if col_obs else None
    JOINT_GOAL_RADIUS = spec.goal_radius  # for "in goal" check
    grav_fn = env.make_gravity_fn()
    D_imp   = 5.0 * np.eye(len(q))

    for i in range(n_exec_steps):
        _t0 = time.perf_counter()

        # FK: current EE position and position Jacobian (3×n)
        x_ee, _ = env.ee_pose(q)
        J_full  = env.jacobian(q)   # (6, n)
        J_pos   = J_full[:3, :]     # (3, n) — position rows only

        # Compute desired joint velocity
        qdot_des, diag = ctrl.step(q, x_goal, J_pos, x_ee, obstacles=col_obs)

        # Hard shield: reject step if predicted clearance < d_hard_min
        if shield_cfg.enabled and col_obs:
            qdot_des, sh_diag = enforce_hard_clearance(
                q, qdot_des, dt, col_obs, shield_cfg
            )

        # Convert to torque via damped impedance: τ = -D * (qdot - qdot_des)
        G   = grav_fn(q)
        tau = G - D_imp @ (qdot - qdot_des)

        env.step(tau, dt=dt)
        step_times.append(time.perf_counter() - _t0)

        q    = env.q.copy()
        qdot = env.qdot.copy()

        # Path length
        exec_path_len += float(np.linalg.norm(q - q_prev))
        q_prev = q.copy()

        # Goal error (joint space for parity with other metrics)
        goal_err_joint = float(np.linalg.norm(q - q_goal))
        if goal_err_joint < JOINT_GOAL_RADIUS and not ever_in_goal:
            ever_in_goal = True
            conv_step    = i

        # Clearance
        if _clearance_fn is not None:
            cl = _clearance_fn(q)
            clearances.append(cl)
            if cl < D_GRAZE_THRESH:
                n_grazes += 1
            if cl < 0:
                n_collisions += 1

        # Modulation diagnostics
        if diag.modulation.n_active > 0:
            mod_active += 1
        if diag.modulation.gamma_min < float("inf"):
            mod_gammas.append(diag.modulation.gamma_min)

    # ---- Pack metrics ---------------------------------------------------
    final_err  = float(np.linalg.norm(q - q_goal))
    term_succ  = final_err < JOINT_GOAL_RADIUS

    # Determine failure reason
    if term_succ:
        failure_reason = ""
    elif n_collisions > 0:
        failure_reason = "collision"
    else:
        failure_reason = "stalled_before_goal"

    tm.execution = ExecutionMetrics(
        terminal_success=term_succ,
        final_goal_err=final_err,
        ever_in_goal=ever_in_goal,
        convergence_step=conv_step,
        convergence_time_s=(conv_step * dt if conv_step is not None else None),
        exec_path_length=exec_path_len,
        failure_reason=failure_reason,
        min_goal_error_ever=final_err,
    )

    tm.barrier = BarrierMetrics(
        min_clearance=float(min(clearances)) if clearances else float("inf"),
        mean_clearance=float(np.mean(clearances)) if clearances else float("inf"),
        near_graze_count=n_grazes,
        collision_count=n_collisions,
        joint_path_length=exec_path_len,
    )

    tm.modulation = ModulationMetrics(
        min_gamma_rollout=float(min(mod_gammas)) if mod_gammas else float("inf"),
        modulation_active_fraction=mod_active / max(1, len(step_times)),
    )

    if step_times:
        _st = np.array(step_times)
        tm.ctrl_freq = CtrlFreqMetrics(
            mean_hz=float(1.0 / np.mean(_st)),
            mean_ms=float(1e3 * np.mean(_st)),
            p95_ms=float(1e3 * np.percentile(_st, 95)),
            max_ms=float(1e3 * np.max(_st)),
            n_steps=len(_st),
        )

    tm.wall_time_s = time.perf_counter() - t_start
    return tm


# ---------------------------------------------------------------------------
# Waypoint PD baseline helper
# ---------------------------------------------------------------------------

def _run_waypoint_pd(
    env: SimEnv,
    path: List[np.ndarray],
    q_goal: np.ndarray,
    gravity_fn,
    dt: float,
    n_steps: int,
    d_gain: float = 5.0,
    p_gain: float = 20.0,
    noise_std: float = 0.0,
    rng=None,
    perturb_magnitude: float = 0.0,
    perturb_joint: int = 1,
    perturb_start: int = 200,
    perturb_duration: int = 50,
) -> Tuple:
    """Simple waypoint-following PD controller. No DS, no passivity."""
    if rng is None:
        rng = np.random.default_rng(0)

    n_joints = len(env.q)
    q = env.q.copy()
    qdot = env.qdot.copy()

    # Assign equal time budget per waypoint
    steps_per_wp = max(1, n_steps // max(1, len(path)))
    wp_idx = 0

    ever_in_goal = False
    conv_step = None
    exec_path_len = 0.0
    q_prev = q.copy()

    D = d_gain * np.eye(n_joints)

    for i in range(n_steps):
        if noise_std > 0 and rng is not None:
            q    = q    + rng.standard_normal(n_joints) * noise_std
            qdot = qdot + rng.standard_normal(n_joints) * noise_std

        # Advance waypoint when close enough
        if wp_idx < len(path) - 1:
            if float(np.linalg.norm(q - path[wp_idx])) < 0.08 or i >= (wp_idx + 1) * steps_per_wp:
                wp_idx = min(wp_idx + 1, len(path) - 1)

        q_target = np.asarray(path[wp_idx], dtype=float)
        G = gravity_fn(q) if gravity_fn else np.zeros(n_joints)
        tau = G + p_gain * (q_target - q) - D @ qdot

        if perturb_magnitude > 0 and perturb_start <= i < perturb_start + perturb_duration:
            tau[perturb_joint] += perturb_magnitude

        env.step(tau, dt=dt)
        q    = env.q.copy()
        qdot = env.qdot.copy()

        exec_path_len += float(np.linalg.norm(q - q_prev))
        q_prev = q.copy()

        goal_err = float(np.linalg.norm(q - q_goal))
        if goal_err < 0.05:
            ever_in_goal = True
            if conv_step is None:
                conv_step = i
            if i > conv_step + 50:
                break

    final_err = float(np.linalg.norm(q - q_goal))
    exec_metrics = ExecutionMetrics(
        terminal_success=(final_err < 0.05),
        final_goal_err=final_err,
        ever_in_goal=ever_in_goal,
        convergence_step=conv_step,
        convergence_time_s=(conv_step * dt if conv_step is not None else None),
        exec_path_length=exec_path_len,
    )
    pass_metrics = PassivityMetrics()   # PD baseline has no passivity
    return q, qdot, exec_metrics, pass_metrics


# ---------------------------------------------------------------------------
# Contact trial
# ---------------------------------------------------------------------------

def run_contact_trial(
    spec: ScenarioSpec,
    condition: TrialCondition,
    seed: int,
    trial_id: int = 0,
    dt: float = DT_CONTACT,
    approach_steps: int = 600,
    slide_steps: int = 1600,
    perturb_steps: int = 80,
    return_steps: int = 600,
    omega: float = 1.5,
    perturb_magnitude: float = 12.0,
    perturb_joint: int = 1,
    K_p: float = 5.0,
    K_v: float = 1.0,
    K_f: float = 2.0,
    F_desired: float = 3.0,
    circle_radius: Optional[float] = None,
) -> TrialMetrics:
    """
    Run one contact-circle trial.

    Replicates the wall_contact_demo logic without CLI overhead.

    Args:
        spec:             ScenarioSpec (must be a contact scenario).
        condition:        TrialCondition (ctrl condition selects contact controller).
        seed:             RNG seed.
        trial_id:         Unique trial ID.
        dt:               Simulation timestep.
        approach_steps:   Steps in APPROACH phase.
        slide_steps:      Steps in WALL_SLIDE phase.
        perturb_steps:    Steps in PERTURBATION phase.
        return_steps:     Steps in RETURN phase.
        omega:            Circle angular velocity (rad/s).
        perturb_magnitude: External torque on perturb_joint.
        perturb_joint:    Joint index (0-based) for perturbation.
        K_p, K_v, K_f, F_desired: Task controller gains.
        circle_radius:    Override circle radius from scenario viz dict.

    Returns:
        TrialMetrics with ContactMetrics populated.
    """
    t_start = time.perf_counter()

    tm = TrialMetrics(
        trial_id=trial_id,
        seed=seed,
        scenario=spec.name,
        condition=condition.name,
    )

    try:
        viz = spec.visualization
        cx, cy    = float(viz["circle_center"][0]), float(viz["circle_center"][1])
        circle_z  = float(viz.get("circle_z", 0.37))
        r         = circle_radius or float(viz.get("circle_radius", 0.15))

        from src.simulation.env import SimEnv, SimEnvConfig
        from src.simulation.panda_scene import load_panda_scene, apply_obstacle_friction
        from src.solver.ds.path_ds import PathDS, DSConfig
        from src.solver.ds.contact_ds import CircleContactConfig, circle_on_plane_reference
        from src.solver.controller.impedance import ControllerConfig, step as ctrl_step
        from src.solver.controller.task_tracking import task_space_step
        from src.solver.tank.tank import EnergyTank, TankConfig
        from src.solver.controller.passivity_filter import PassivityFilterConfig

        _obs_friction = None
        for k, v in spec.obstacle_friction_map().items():
            if v is not None:
                _obs_friction = v
                break

        env = SimEnv(SimEnvConfig(
            obstacles=spec.obstacles_as_hjcd_dict(),
            timestep=dt,
            obstacle_friction=_obs_friction,
            ee_offset_body=np.array([0.0, 0.0, 0.212]),
        ))
        q_start = np.asarray(spec.q_start, dtype=float)
        env.set_state(q_start, np.zeros(len(q_start)))
        grav_fn = env.make_gravity_fn()

        # ---- IK for approach target -------------------------------------
        from src.visualization.wall_contact_demo import _jacobian_ik
        _approach_z = circle_z - 0.010
        circle_start = np.array([cx + r, cy, _approach_z])
        q_approach, _ = _jacobian_ik(env, q_start, circle_start)

        # ---- Circle config ----------------------------------------------
        circle_cfg = CircleContactConfig(
            center=np.array([cx, cy, circle_z]),
            radius=r,
            omega=omega,
            normal=np.array([0.0, 0.0, 1.0]),
            z_contact=circle_z,
        )

        # ---- Approach DS -----------------------------------------------
        ctrl_params = spec.controller or {}
        approach_ds = PathDS(
            [q_start, q_approach],
            config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.3, goal_radius=0.05),
        )

        ctrl_cfg, tank_cfg, pf_cfg = build_ctrl_config(
            ControlCondition.PATH_DS_FULL,
            d_gain=ctrl_params.get("d_gain", 5.0),
            gravity_fn=grav_fn,
        )

        task_cfg, task_tank_cfg, task_pf_cfg = build_task_ctrl_config(
            condition.ctrl,
            d_gain=ctrl_params.get("d_gain", 5.0),
            gravity_fn=grav_fn,
            K_p=K_p, K_v=K_v, K_f=K_f, F_desired=F_desired,
        )

        tank = EnergyTank(tank_cfg)

        # ---- Simulation loop -------------------------------------------
        total_steps = approach_steps + slide_steps + perturb_steps + return_steps
        q    = env.q.copy()
        qdot = env.qdot.copy()

        t_slide     = 0.0
        contact_flags:  List[bool]  = []
        contact_forces: List[float] = []
        height_errors:  List[float] = []
        ee_positions:   List[np.ndarray] = []
        tank_levels:    List[float] = []
        clipped_steps:  int = 0
        powers_before:  List[float] = []
        powers_after:   List[float] = []
        beta_Rs:        List[float] = []

        prev_phase = "APPROACH"
        return_ds  = None
        circle_qs  = [q_approach]  # minimal RETURN target

        for i in range(total_steps):
            if i < approach_steps:
                phase = "APPROACH"
            elif i < approach_steps + slide_steps:
                phase = "WALL_SLIDE"
            elif i < approach_steps + slide_steps + perturb_steps:
                phase = "PERTURBATION"
            else:
                phase = "RETURN"

            if phase == "WALL_SLIDE" and prev_phase == "APPROACH":
                t_slide = 0.0
                tank = EnergyTank(task_tank_cfg)  # fresh tank for SLIDE

            if phase == "RETURN" and prev_phase == "PERTURBATION":
                return_ds = PathDS(
                    [env.q.copy(), q_approach],
                    config=DSConfig(K_c=3.0, K_r=0.5, K_n=0.3, goal_radius=0.05),
                )

            if phase in ("WALL_SLIDE", "PERTURBATION"):
                ee_pos, _ = env.ee_pose(q)
                contacts   = env.get_contact_forces()
                F_contact  = sum((np.asarray(c["force"]) for c in contacts), np.zeros(3))
                J          = env.jacobian(q)
                ref        = circle_on_plane_reference(t_slide, circle_cfg)

                if condition.ctrl == ControlCondition.JOINT_SPACE_PATH_ONLY:
                    # Use approach DS tangentially (joint-space only)
                    res = ctrl_step(q, qdot, approach_ds, tank, dt, config=ctrl_cfg)
                    tau = res.tau.copy()
                    beta_R       = res.beta_R
                    pf_clipped   = res.pf_clipped
                    pf_power_nom = res.pf_power_nom
                    pf_power_flt = res.pf_power_filtered
                else:
                    res = task_space_step(
                        q, qdot, ee_pos, ref, F_contact, J, tank, dt,
                        config=task_cfg, passivity_filter_cfg=task_pf_cfg,
                    )
                    tau = res.tau.copy()
                    beta_R       = res.beta_R
                    pf_clipped   = res.pf_clipped
                    pf_power_nom = res.pf_power_nom
                    pf_power_flt = res.pf_power_nom  # task_space_step doesn't expose filtered

                t_slide += dt
            else:
                ds_use = return_ds if (phase == "RETURN" and return_ds) else approach_ds
                res = ctrl_step(q, qdot, ds_use, tank, dt, config=ctrl_cfg)
                tau = res.tau.copy()
                beta_R       = res.beta_R
                pf_clipped   = res.pf_clipped
                pf_power_nom = res.pf_power_nom
                pf_power_flt = res.pf_power_filtered
                contacts     = env.get_contact_forces()
                ee_pos, _    = env.ee_pose(q)

            if phase == "PERTURBATION":
                tau[perturb_joint] += perturb_magnitude

            env.step(tau, dt=dt)
            q    = env.q.copy()
            qdot = env.qdot.copy()

            # Post-step metrics
            contacts_ps   = env.get_contact_forces()
            F_ps          = sum((np.asarray(c["force"]) for c in contacts_ps), np.zeros(3))
            F_n           = float(np.dot(F_ps, [0, 0, 1]))
            in_contact    = len(contacts_ps) > 0
            ee_pos_ps, _  = env.ee_pose(q)
            height_err    = float(ee_pos_ps[2] - circle_z)

            if phase in ("WALL_SLIDE", "PERTURBATION"):
                contact_flags.append(in_contact)
                contact_forces.append(max(0.0, F_n))
                height_errors.append(height_err)
                ee_positions.append(ee_pos_ps.copy())
                tank_levels.append(tank.energy)
                if pf_clipped:
                    clipped_steps += 1
                powers_before.append(pf_power_nom)
                powers_after.append(pf_power_flt)
                beta_Rs.append(beta_R)

            prev_phase = phase

        # ---- Contact metrics -------------------------------------------
        contact_established = any(contact_flags[:50]) if contact_flags else False
        cf_arr = np.array(contact_forces) if contact_forces else np.zeros(1)
        h_arr  = np.array(height_errors)  if height_errors  else np.zeros(1)

        # Circle tracking RMSE (in XY plane)
        circle_rmse = float("inf")
        radius_rmse = float("inf")
        arc_ratio   = 0.0
        n_slide     = len(ee_positions)
        if n_slide > 0:
            times = np.arange(n_slide) * dt
            refs  = np.array([circle_on_plane_reference(t, circle_cfg).x_d
                               for t in times])
            pos_arr = np.array(ee_positions)
            errs    = np.linalg.norm(pos_arr - refs, axis=1)
            circle_rmse = float(np.sqrt(np.mean(errs ** 2)))

            # Radius error
            dxy     = pos_arr[:, :2] - np.array([cx, cy])
            r_exec  = np.linalg.norm(dxy, axis=1)
            radius_rmse = float(np.sqrt(np.mean((r_exec - r) ** 2)))

            # Arc completion: how much of the circle did the EE actually sweep?
            angles = np.arctan2(dxy[:, 1], dxy[:, 0])
            angle_span = float(np.ptp(np.unwrap(angles)))
            arc_ratio  = min(1.0, abs(angle_span) / (2 * np.pi))

        tm.contact = ContactMetrics(
            contact_established=contact_established,
            contact_maintained_fraction=(np.mean(contact_flags) if contact_flags else 0.0),
            mean_contact_force=(float(np.mean(cf_arr[cf_arr > 0])) if np.any(cf_arr > 0) else 0.0),
            std_contact_force=(float(np.std(cf_arr[cf_arr > 0])) if np.sum(cf_arr > 0) > 1 else 0.0),
            mean_height_error=float(np.mean(np.abs(h_arr))),
            circle_tracking_rmse=circle_rmse,
            circle_radius_rmse=radius_rmse,
            arc_completion_ratio=arc_ratio,
            final_phase_progress=(t_slide / max(dt, slide_steps * dt)),
        )

        n_pass = len(tank_levels)
        tm.passivity = PassivityMetrics(
            n_violations=sum(1 for p in powers_before if p > 1e-6),
            clipped_ratio=(clipped_steps / max(1, n_pass)),
            mean_power_before=(float(np.mean(powers_before)) if powers_before else 0.0),
            mean_power_after=(float(np.mean(powers_after)) if powers_after else 0.0),
            min_tank_energy=(float(np.min(tank_levels)) if tank_levels else 0.0),
            final_tank_energy=(tank_levels[-1] if tank_levels else 0.0),
            beta_R_zero_fraction=(sum(1 for b in beta_Rs if b < 0.01) / max(1, n_pass)),
        )

    except Exception:
        tm.error = traceback.format_exc()

    tm.wall_time_s = time.perf_counter() - t_start
    return tm


# ---------------------------------------------------------------------------
# Contact passivity trial  (approach-method comparison)
# ---------------------------------------------------------------------------

def run_contact_passivity_trial(
    spec: ScenarioSpec,
    condition: TrialCondition,
    seed: int,
    trial_id: int = 0,
    dt: float = DT_CONTACT,
    approach_steps: int = 800,
    slide_steps: int = 1200,
    perturb_steps: int = 80,
    return_steps: int = 400,
    omega: float = 1.5,
    perturb_magnitude: float = 12.0,
    perturb_joint: int = 1,
    K_p: float = 5.0,
    K_v: float = 1.0,
    K_f: float = 2.0,
    F_desired: float = 3.0,
    circle_radius: Optional[float] = None,
) -> TrialMetrics:
    """
    Contact-passivity benchmark trial: compare approach methods on contact quality.

    APPROACH phase uses condition-specific navigation:
      IKCondition.VANILLA_DS   + any ctrl     -> straight PathDS (no planner)
      IKCondition.SINGLE/MULTI + any ctrl     -> BiRRT + PathDS
      any ik                   + VANILLA_DS_DIFFIK_MODULATION -> DiffIK task-space approach

    CONTACT phases (WALL_SLIDE, PERTURBATION, RETURN) always use task_space_step
    for a fair comparison of contact quality.

    New ContactMetrics fields are populated:
      time_to_first_contact_steps, impact_velocity_norm, peak_force_on_impact,
      n_contact_losses, normal_force_error_rmse, recovered_after_perturbation,
      time_to_recover_steps, integrated_positive_power, high_power_spike_count,
      contact_chatter_index.
    """
    t_start = time.perf_counter()
    tm = TrialMetrics(trial_id=trial_id, seed=seed, scenario=spec.name,
                      condition=condition.name)

    try:
        from src.simulation.panda_scene import load_panda_scene, apply_obstacle_friction
        from src.solver.ds.contact_ds import CircleContactConfig, circle_on_plane_reference
        from src.solver.controller.task_tracking import task_space_step

        viz = spec.visualization
        cx, cy   = float(viz["circle_center"][0]), float(viz["circle_center"][1])
        circle_z = float(viz.get("circle_z", 0.37))
        r        = circle_radius or float(viz.get("circle_radius", 0.08))

        _obs_friction = None
        for k, v in spec.obstacle_friction_map().items():
            if v is not None:
                _obs_friction = v
                break

        env = SimEnv(SimEnvConfig(
            obstacles=spec.obstacles_as_hjcd_dict(),
            timestep=dt,
            obstacle_friction=_obs_friction,
            ee_offset_body=np.array([0.0, 0.0, 0.212]),
        ))
        q_start = np.asarray(spec.q_start, dtype=float)
        env.set_state(q_start, np.zeros(len(q_start)))
        grav_fn = env.make_gravity_fn()

        # ---- Compute approach IK config -----------------------------------
        from src.visualization.wall_contact_demo import _jacobian_ik
        _approach_z   = circle_z - 0.010
        circle_start  = np.array([cx + r, cy, _approach_z])
        q_approach, _ = _jacobian_ik(env, q_start, circle_start)

        # ---- Controller configs for contact phase -------------------------
        ctrl_params = spec.controller or {}
        task_cfg, task_tank_cfg, task_pf_cfg = build_task_ctrl_config(
            ControlCondition.TASK_TRACKING_FULL,
            d_gain=ctrl_params.get("d_gain", 5.0),
            gravity_fn=grav_fn,
            K_p=K_p, K_v=K_v, K_f=K_f, F_desired=F_desired,
        )
        ctrl_cfg, tank_cfg, pf_cfg = build_ctrl_config(
            ControlCondition.PATH_DS_FULL,
            d_gain=ctrl_params.get("d_gain", 5.0),
            gravity_fn=grav_fn,
        )
        tank = EnergyTank(tank_cfg)

        # ---- Approach method selection ------------------------------------
        use_diffik = (condition.ctrl == ControlCondition.VANILLA_DS_DIFFIK_MODULATION)
        use_birrt  = (condition.ik in (IKCondition.SINGLE_IK_BEST,
                                       IKCondition.MULTI_IK_FULL,
                                       IKCondition.MULTI_IK_TOP_2,
                                       IKCondition.MULTI_IK_TOP_4))

        # Build approach DS / controller
        if use_diffik:
            from src.solver.controller.vanilla_ds_diffik import (
                VanillaDSDiffIKController, VanillaDSConfig, DiffIKConfig,
            )
            from src.solver.ds.modulation import ModulationConfig
            x_goal_approach, _ = env.ee_pose(q_approach)
            diffik_ds_cfg = VanillaDSConfig(pos_gain=2.0, max_task_speed=0.20, goal_tolerance=0.015)
            diffik_ik_cfg = DiffIKConfig(damping=1e-2, max_joint_speed=0.8, use_nullspace_posture=True)
            diffik_mod_cfg = ModulationConfig(enabled=False)
            diffik_ctrl = VanillaDSDiffIKController(
                ds_config=diffik_ds_cfg, diffik_config=diffik_ik_cfg,
                modulation_config=diffik_mod_cfg, q_nominal=q_start.copy(),
            )
        elif use_birrt:
            plan_col_fn = make_collision_fn(spec=spec, margin=0.0) \
                if spec.obstacles else (lambda q: True)
            plan_cfg_local = PlannerConfig(
                max_iterations=spec.planner.get("max_iterations", 8_000) if spec.planner else 8_000,
                step_size=spec.planner.get("step_size", 0.15) if spec.planner else 0.15,
                goal_bias=spec.planner.get("goal_bias", 0.15) if spec.planner else 0.15,
                seed=seed,
            )
            pr = plan(q_start, [q_approach], plan_col_fn, plan_cfg_local)
            approach_path = pr.waypoints if pr.success else [q_start, q_approach]
            approach_ds = PathDS(approach_path, config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.3, goal_radius=0.05))
        else:
            # Vanilla DS: straight-line PathDS, no planner
            approach_ds = PathDS([q_start, q_approach],
                                 config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.3, goal_radius=0.05))

        # ---- Circle config ------------------------------------------------
        circle_cfg = CircleContactConfig(
            center=np.array([cx, cy, circle_z]),
            radius=r, omega=omega,
            normal=np.array([0.0, 0.0, 1.0]),
            z_contact=circle_z,
        )

        # ---- Simulation loop ----------------------------------------------
        total_steps = approach_steps + slide_steps + perturb_steps + return_steps
        q    = env.q.copy()
        qdot = env.qdot.copy()

        t_slide       = 0.0
        first_contact_step: Optional[int] = None
        impact_vel    = 0.0
        prev_ee_pos   = None
        contact_flags:  List[bool]  = []
        contact_forces: List[float] = []
        normal_forces:  List[float] = []
        height_errors:  List[float] = []
        ee_positions:   List[np.ndarray] = []
        tank_levels:    List[float] = []
        powers_before:  List[float] = []
        powers_after:   List[float] = []
        beta_Rs:        List[float] = []
        clipped_steps   = 0
        prev_phase      = "APPROACH"
        return_ds: Optional[PathDS] = None

        D_imp = 5.0 * np.eye(len(q))

        for i in range(total_steps):
            if i < approach_steps:
                phase = "APPROACH"
            elif i < approach_steps + slide_steps:
                phase = "WALL_SLIDE"
            elif i < approach_steps + slide_steps + perturb_steps:
                phase = "PERTURBATION"
            else:
                phase = "RETURN"

            if phase == "WALL_SLIDE" and prev_phase == "APPROACH":
                t_slide = 0.0
                tank = EnergyTank(task_tank_cfg)

            if phase == "RETURN" and prev_phase == "PERTURBATION":
                return_ds = PathDS(
                    [env.q.copy(), q_approach],
                    config=DSConfig(K_c=3.0, K_r=0.5, K_n=0.3, goal_radius=0.05),
                )

            # ---- Compute torque -------------------------------------------
            if phase == "APPROACH":
                if use_diffik:
                    x_ee, _ = env.ee_pose(q)
                    J_pos   = env.jacobian(q)[:3, :]
                    qdot_des, _ = diffik_ctrl.step(q, x_goal_approach, J_pos, x_ee, obstacles=None)
                    G   = grav_fn(q)
                    tau = G - D_imp @ (qdot - qdot_des)
                else:
                    res = ctrl_step(q, qdot, approach_ds, tank, dt, config=ctrl_cfg)
                    tau = res.tau.copy()
                pf_clipped   = False
                pf_power_nom = 0.0
                pf_power_flt = 0.0
                beta_R       = 1.0
            elif phase in ("WALL_SLIDE", "PERTURBATION"):
                ee_pos, _ = env.ee_pose(q)
                contacts   = env.get_contact_forces()
                F_contact  = sum((np.asarray(c["force"]) for c in contacts), np.zeros(3))
                J          = env.jacobian(q)
                ref        = circle_on_plane_reference(t_slide, circle_cfg)
                res = task_space_step(
                    q, qdot, ee_pos, ref, F_contact, J, tank, dt,
                    config=task_cfg, passivity_filter_cfg=task_pf_cfg,
                )
                tau = res.tau.copy()
                beta_R       = res.beta_R
                pf_clipped   = res.pf_clipped
                pf_power_nom = res.pf_power_nom
                pf_power_flt = res.pf_power_nom
                t_slide += dt
            else:  # RETURN
                ds_use = return_ds if return_ds else approach_ds
                res = ctrl_step(q, qdot, ds_use, tank, dt, config=ctrl_cfg)
                tau = res.tau.copy()
                beta_R       = res.beta_R
                pf_clipped   = res.pf_clipped
                pf_power_nom = res.pf_power_nom
                pf_power_flt = res.pf_power_filtered

            if phase == "PERTURBATION":
                tau[perturb_joint] += perturb_magnitude

            # ---- Track EE velocity for impact detection -------------------
            ee_pos_pre, _ = env.ee_pose(q)
            env.step(tau, dt=dt)
            q    = env.q.copy()
            qdot = env.qdot.copy()

            # ---- Post-step metrics ----------------------------------------
            contacts_ps  = env.get_contact_forces()
            F_ps         = sum((np.asarray(c["force"]) for c in contacts_ps), np.zeros(3))
            F_n          = float(np.dot(F_ps, [0, 0, 1]))
            in_contact   = len(contacts_ps) > 0
            ee_pos_ps, _ = env.ee_pose(q)

            # Impact: first moment contact is detected
            if in_contact and first_contact_step is None:
                first_contact_step = i
                if prev_ee_pos is not None:
                    impact_vel = float(np.linalg.norm(ee_pos_ps - prev_ee_pos)) / dt
                else:
                    impact_vel = 0.0
            prev_ee_pos = ee_pos_ps.copy()

            if phase in ("WALL_SLIDE", "PERTURBATION"):
                height_err = float(ee_pos_ps[2] - circle_z)
                contact_flags.append(in_contact)
                contact_forces.append(max(0.0, F_n))
                normal_forces.append(F_n)
                height_errors.append(height_err)
                ee_positions.append(ee_pos_ps.copy())
                tank_levels.append(tank.energy)
                if pf_clipped:
                    clipped_steps += 1
                powers_before.append(pf_power_nom)
                powers_after.append(pf_power_flt)
                beta_Rs.append(beta_R)

            prev_phase = phase

        # ---- Post-loop metrics -------------------------------------------
        contact_established = any(contact_flags[:50]) if contact_flags else False
        cf_arr  = np.array(contact_forces) if contact_forces else np.zeros(1)
        nf_arr  = np.array(normal_forces)  if normal_forces  else np.zeros(1)
        h_arr   = np.array(height_errors)  if height_errors  else np.zeros(1)

        # Circle RMSE
        circle_rmse = float("inf")
        radius_rmse = float("inf")
        arc_ratio   = 0.0
        n_slide     = len(ee_positions)
        if n_slide > 0:
            times   = np.arange(n_slide) * dt
            refs    = np.array([circle_on_plane_reference(t, circle_cfg).x_d for t in times])
            pos_arr = np.array(ee_positions)
            errs    = np.linalg.norm(pos_arr - refs, axis=1)
            circle_rmse = float(np.sqrt(np.mean(errs ** 2)))
            dxy     = pos_arr[:, :2] - np.array([cx, cy])
            r_exec  = np.linalg.norm(dxy, axis=1)
            radius_rmse = float(np.sqrt(np.mean((r_exec - r) ** 2)))
            angles   = np.arctan2(dxy[:, 1], dxy[:, 0])
            arc_ratio = min(1.0, abs(float(np.ptp(np.unwrap(angles)))) / (2 * np.pi))

        # Impact force spike: max force in first 50 steps of contact phase
        peak_impact = 0.0
        if cf_arr is not None and len(cf_arr) >= 1:
            peak_impact = float(np.max(cf_arr[:50]))

        # Contact losses (drops after first establishment)
        n_contact_losses = 0
        if contact_flags:
            first_true = next((j for j, f in enumerate(contact_flags) if f), None)
            if first_true is not None:
                for j in range(first_true + 1, len(contact_flags)):
                    if contact_flags[j - 1] and not contact_flags[j]:
                        n_contact_losses += 1

        # Normal force RMSE
        normal_rmse = float("inf")
        if len(nf_arr) > 0 and np.any(np.array(contact_flags)):
            contact_idx = [j for j, f in enumerate(contact_flags) if f]
            if contact_idx:
                fn_contact = nf_arr[contact_idx]
                normal_rmse = float(np.sqrt(np.mean((fn_contact - F_desired) ** 2)))

        # Perturbation recovery: was contact re-established in RETURN phase?
        slide_end    = slide_steps + perturb_steps
        return_flags = contact_flags[slide_end:] if len(contact_flags) > slide_end else []
        recovered    = any(return_flags)
        recover_step = 0
        if recovered:
            recover_step = next((j for j, f in enumerate(return_flags) if f), 0)

        # Passivity / power metrics during contact
        pb_arr = np.array(powers_before) if powers_before else np.zeros(1)
        integrated_pos_power = float(np.sum(np.maximum(0.0, pb_arr)) * dt)
        high_power_count     = int(np.sum(np.abs(pb_arr) > 1.0))

        # Contact chatter: fraction of adjacent flag transitions
        chatter = 0.0
        if len(contact_flags) > 1:
            transitions = sum(1 for a, b in zip(contact_flags, contact_flags[1:]) if a != b)
            chatter     = transitions / max(1, len(contact_flags) - 1)

        tm.contact = ContactMetrics(
            contact_established=contact_established,
            contact_maintained_fraction=(float(np.mean(contact_flags)) if contact_flags else 0.0),
            mean_contact_force=(float(np.mean(cf_arr[cf_arr > 0])) if np.any(cf_arr > 0) else 0.0),
            std_contact_force=(float(np.std(cf_arr[cf_arr > 0])) if np.sum(cf_arr > 0) > 1 else 0.0),
            mean_height_error=float(np.mean(np.abs(h_arr))),
            circle_tracking_rmse=circle_rmse,
            circle_radius_rmse=radius_rmse,
            arc_completion_ratio=arc_ratio,
            final_phase_progress=(t_slide / max(dt, slide_steps * dt)),
            # New passivity-benchmark fields
            time_to_first_contact_steps=(first_contact_step or 0),
            impact_velocity_norm=(impact_vel if first_contact_step is not None else 0.0),
            peak_force_on_impact=peak_impact,
            n_contact_losses=n_contact_losses,
            normal_force_error_rmse=normal_rmse,
            recovered_after_perturbation=recovered,
            time_to_recover_steps=recover_step,
            integrated_positive_power=integrated_pos_power,
            high_power_spike_count=high_power_count,
            contact_chatter_index=chatter,
        )

        n_pass = len(tank_levels)
        tm.passivity = PassivityMetrics(
            n_violations=sum(1 for p in powers_before if p > 1e-6),
            clipped_ratio=(clipped_steps / max(1, n_pass)),
            mean_power_before=(float(np.mean(powers_before)) if powers_before else 0.0),
            mean_power_after=(float(np.mean(powers_after)) if powers_after else 0.0),
            min_tank_energy=(float(np.min(tank_levels)) if tank_levels else 0.0),
            final_tank_energy=(tank_levels[-1] if tank_levels else 0.0),
            beta_R_zero_fraction=(sum(1 for b in beta_Rs if b < 0.01) / max(1, n_pass)),
        )

    except Exception:
        tm.error = traceback.format_exc()

    tm.wall_time_s = time.perf_counter() - t_start
    return tm
