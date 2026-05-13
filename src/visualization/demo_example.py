"""
Phase 10 — Visualization / Demo Script

Runs one representative scenario end-to-end and produces:

  outputs/demo/<scenario>/
    summary.json          — pipeline metrics
    metrics.png           — 6-panel summary figure
    frame_*.png           — per-step snapshots (optional)
    animation.gif         — animated metrics (optional)

Usage::

    python -m src.visualization.demo_example
    python -m src.visualization.demo_example --scenario narrow_passage --seed 0
    python -m src.visualization.demo_example --scenario free_space --no-animation
    python -m src.visualization.demo_example --scenario contact_task --headless

Scenarios
---------
narrow_passage   (default) joint-space corridor; demonstrates multi-IK advantage
free_space       no obstacles; validates basic pipeline
contact_task     obstacle in workspace; contact force sensing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Scenario builders (mirrors benchmarks/scenarios.py)
# ---------------------------------------------------------------------------
from benchmarks.scenarios import (
    free_space_scenario,
    narrow_passage_scenario,
    contact_task_scenario,
)

# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------
from src.solver.planner.birrt import plan, PlannerConfig
from src.solver.ds.path_ds import PathDS, DSConfig
from src.solver.controller.impedance import (
    ControllerConfig,
    ControlResult,
    simulate,
)
from src.solver.tank.tank import EnergyTank, TankConfig
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.controller.contact_force import (
    ContactForceConfig,
    contact_force_control,
    extract_contact_normal_force,
)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
from src.visualization.plotting import (
    plot_ik_goals,
    plot_rrt_path,
    plot_executed_trajectory,
    plot_tank_energy,
    plot_passivity_metrics,
    plot_lyapunov,
    plot_distance_to_goal,
    plot_joint_trajectories,
    plot_contact_metrics,
)
from src.visualization.animation import save_trajectory_animation, save_frames


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS   = 7
DT         = 0.05
N_STEPS    = 300
GOAL_RADIUS = 0.05

_SCENARIO_MAP = {
    "narrow_passage": narrow_passage_scenario,
    "free_space":     free_space_scenario,
    "contact_task":   contact_task_scenario,
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    scenario: dict,
    seed: int,
    scenario_name: str,
) -> Dict:
    """
    Execute planning + control for one scenario/seed.

    Returns a rich dict with all components needed for visualization.
    """
    q_start = scenario["q_start"]

    # Determine goal set based on scenario type
    if scenario_name == "narrow_passage":
        Q_goals = scenario.get("Q_goals_multi", scenario.get("Q_goals", []))
    else:
        Q_goals = scenario.get("Q_goals", [])

    col_fn  = scenario.get("collision_fn", lambda q: True)

    # ---- 1. Filter IK goals -----------------------------------------------
    # For the demo we use a simple "all are safe" mask (no GPU IK required).
    # In a full pipeline this would be replaced by filter_safe_set().
    safe_mask = [True] * len(Q_goals)
    safe_goals = [g for g, s in zip(Q_goals, safe_mask) if s]

    # ---- 2. Plan -----------------------------------------------------------
    cfg_plan = PlannerConfig(
        max_iterations=8_000,
        step_size=0.15,
        goal_bias=0.15,
        seed=seed,
    )
    plan_result = plan(q_start, Q_goals, env=col_fn, config=cfg_plan)

    if not plan_result.success:
        return {
            "success": False,
            "plan_result": plan_result,
            "Q_goals": Q_goals,
            "safe_mask": safe_mask,
            "q_start": q_start,
            "results": [],
            "q_history": np.array([q_start]),
        }

    selected_idx = plan_result.goal_idx

    # ---- 3. DS + Controller -----------------------------------------------
    ds = PathDS(
        plan_result.path,
        config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.3, goal_radius=GOAL_RADIUS),
    )

    pf_cfg = PassivityFilterConfig()
    tank   = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0,
                                   epsilon_min=-0.05, epsilon_max=0.1))
    ctrl_cfg = ControllerConfig(
        d_gain=3.0,
        f_n_gain=0.0,
        orthogonalize_residual=True,
        alpha=0.5,
        passivity_filter=pf_cfg,
    )

    # simulate() uses a unit-mass Euler integrator internally.
    # We also capture q_history by tracking position via xdot_d integration.
    q = q_start.copy()
    qdot = np.zeros(N_JOINTS)
    results: List[ControlResult] = []
    q_history = [q.copy()]

    for _ in range(N_STEPS):
        res = simulate.__wrapped__(q, qdot, ds, tank, dt=DT, n_steps=1,
                                    config=ctrl_cfg)[0] \
              if hasattr(simulate, "__wrapped__") else None

        # Use simulate() for single steps to avoid re-init cost
        step_results = simulate(q, qdot, ds, tank, dt=DT, n_steps=1,
                                 config=ctrl_cfg)
        if not step_results:
            break
        r = step_results[0]
        results.append(r)

        # Unit-mass Euler: qdot += tau * dt; q += qdot * dt
        qdot = qdot + r.tau * DT
        q    = q    + qdot   * DT
        q_history.append(q.copy())

        # Re-init tank with updated energy (simulate() returns a fresh tank)
        # The tank was mutated in-place by simulate(); no re-init needed.

    q_history = np.array(q_history)
    q_goal    = Q_goals[selected_idx]
    converged = float(np.linalg.norm(q_history[-1] - q_goal)) < GOAL_RADIUS * 3

    return {
        "success":       True,
        "plan_result":   plan_result,
        "Q_goals":       Q_goals,
        "safe_mask":     safe_mask,
        "selected_idx":  selected_idx,
        "q_start":       q_start,
        "q_goal":        q_goal,
        "path":          plan_result.path,
        "results":       results,
        "q_history":     q_history,
        "converged":     converged,
    }


def _run_pipeline_simple(scenario: dict, seed: int, scenario_name: str) -> Dict:
    """
    Simplified pipeline runner using simulate() over the full horizon.
    Captures q_history by integrating from the controller output.
    """
    q_start = scenario["q_start"]

    if scenario_name == "narrow_passage":
        Q_goals = scenario.get("Q_goals_multi", scenario.get("Q_goals", []))
    else:
        Q_goals = scenario.get("Q_goals", [])

    col_fn  = scenario.get("collision_fn", lambda q: True)
    safe_mask = [True] * len(Q_goals)

    # Plan
    cfg_plan = PlannerConfig(max_iterations=8_000, step_size=0.15,
                             goal_bias=0.15, seed=seed)
    plan_result = plan(q_start, Q_goals, env=col_fn, config=cfg_plan)

    if not plan_result.success:
        return {"success": False, "plan_result": plan_result,
                "Q_goals": Q_goals, "safe_mask": safe_mask,
                "q_start": q_start, "results": [],
                "q_history": np.array([q_start])}

    selected_idx = plan_result.goal_idx
    q_goal = Q_goals[selected_idx]

    ds = PathDS(plan_result.path,
                config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.3,
                                goal_radius=GOAL_RADIUS))
    pf_cfg = PassivityFilterConfig()
    tank   = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0,
                                   epsilon_min=-0.05, epsilon_max=0.1))
    ctrl_cfg = ControllerConfig(d_gain=3.0, f_n_gain=0.0,
                                orthogonalize_residual=True, alpha=0.5,
                                passivity_filter=pf_cfg)

    results = simulate(q_start, np.zeros(N_JOINTS), ds, tank,
                       dt=DT, n_steps=N_STEPS, config=ctrl_cfg)

    # Reconstruct q_history from xdot_d integration (unit-mass Euler)
    q = q_start.copy()
    qdot = np.zeros(N_JOINTS)
    q_history = [q.copy()]
    for r in results:
        qdot = qdot + r.tau * DT
        q    = q    + qdot  * DT
        q_history.append(q.copy())
    q_history = np.array(q_history)

    final_goal_error = float(np.linalg.norm(q_history[-1] - q_goal))
    terminal_success = final_goal_error < GOAL_RADIUS * 3
    dists = [float(np.linalg.norm(qh - q_goal)) for qh in q_history]
    ever_in_goal = any(d < GOAL_RADIUS for d in dists)

    # Geometric path length (sum of ||Δq|| between consecutive waypoints)
    path_arr = np.array(plan_result.path)
    if len(path_arr) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1)))
    else:
        path_length = 0.0

    return {
        "success":           True,
        "plan_result":       plan_result,
        "Q_goals":           Q_goals,
        "safe_mask":         safe_mask,
        "selected_idx":      selected_idx,
        "q_start":           q_start,
        "q_goal":            q_goal,
        "path":              plan_result.path,
        "results":           results,
        "q_history":         q_history,
        "final_goal_error":  final_goal_error,
        "terminal_success":  terminal_success,
        "ever_in_goal":      ever_in_goal,
        "path_length":       path_length,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_summary_figure(data: dict, scenario_name: str, dt: float = DT) -> matplotlib.figure.Figure:
    """
    Build a 2×3 summary figure covering planning and control.

    Panels:
        [0,0] IK goals scatter
        [0,1] RRT planned path
        [0,2] Executed trajectory vs plan
        [1,0] Tank energy over time
        [1,1] Passivity filter metrics
        [1,2] Lyapunov V(q) / distance to goal
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Multi-IK DS Demo — scenario: {scenario_name}  "
        f"(seed {data.get('seed', 0)})",
        fontsize=12,
    )

    Q_goals     = data["Q_goals"]
    q_start     = data["q_start"]
    safe_mask   = data["safe_mask"]
    results     = data["results"]
    q_history   = data["q_history"]

    # ---- Row 0: Planning ---------------------------------------------------
    selected_idx = data.get("selected_idx", None)
    plot_ik_goals(axes[0, 0], Q_goals, q_start,
                  safe_mask=safe_mask, selected_idx=selected_idx,
                  title="IK Goal Set")

    if data["success"] and data.get("path"):
        plot_rrt_path(axes[0, 1], data["path"], q_start=q_start,
                      title="Planned RRT Path")
    else:
        axes[0, 1].text(0.5, 0.5, "Planning FAILED",
                        ha="center", va="center",
                        transform=axes[0, 1].transAxes,
                        fontsize=14, color="red")
        axes[0, 1].set_title("RRT Path")

    if data["success"] and len(q_history) > 1:
        plot_executed_trajectory(axes[0, 2], q_history,
                                  path=data.get("path"),
                                  title="Executed vs Planned")
    else:
        axes[0, 2].set_title("Executed Trajectory")

    # ---- Row 1: Controller metrics -----------------------------------------
    if results:
        plot_tank_energy(axes[1, 0], results, dt=dt)
        plot_passivity_metrics(axes[1, 1], results, dt=dt)

        q_goal = data.get("q_goal")
        if q_goal is not None and len(q_history) > 1:
            plot_distance_to_goal(axes[1, 2], q_history, q_goal,
                                  dt=dt, goal_radius=GOAL_RADIUS)
        else:
            plot_lyapunov(axes[1, 2], results, dt=dt)
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "No control data",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
def build_summary(data: dict, scenario_name: str) -> dict:
    """Build the required summary.json payload."""
    results   = data["results"]
    Q_goals   = data["Q_goals"]
    safe_mask = data["safe_mask"]

    n_ik_goals  = len(Q_goals)
    n_safe_goals = sum(safe_mask)

    if not results:
        return {
            "scenario":                scenario_name,
            "planning_success":        data["success"],
            "n_ik_goals":              n_ik_goals,
            "n_safe_goals":            n_safe_goals,
            "selected_goal_idx":       None,
            "planner_waypoint_count":  None,
            "planner_path_length":     None,
            "planner_iterations":      None,
            "final_goal_error":        None,
            "terminal_success":        False,
            "ever_entered_goal_region": False,
            "clipped_ratio":           None,
            "min_tank_energy":         None,
            "unhandled_violations":    None,
        }

    clipped    = sum(1 for r in results if r.pf_clipped)
    clip_ratio = clipped / len(results)
    min_tank   = min(r.tank_energy for r in results)
    unhandled  = sum(
        1 for r in results
        if r.pf_power_nom > 1e-10 and not r.pf_clipped
    )

    return {
        "scenario":                scenario_name,
        "planning_success":        data["success"],
        "n_ik_goals":              n_ik_goals,
        "n_safe_goals":            n_safe_goals,
        "selected_goal_idx":       data.get("selected_idx"),
        # Planner metrics — correct names per CLAUDE.md
        "planner_waypoint_count":  len(data.get("path", [])),
        "planner_path_length":     round(data.get("path_length", 0.0), 4),
        "planner_iterations":      data["plan_result"].iterations if data["success"] else None,
        # Goal completion — distinct names per CLAUDE.md
        "final_goal_error":        round(data.get("final_goal_error", float("nan")), 6),
        "terminal_success":        data.get("terminal_success", False),
        "ever_entered_goal_region": data.get("ever_in_goal", False),
        # Controller metrics
        "clipped_ratio":           round(clip_ratio, 4),
        "min_tank_energy":         round(float(min_tank), 6),
        "unhandled_violations":    unhandled,
        "n_steps":                 len(results),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-IK DS pipeline visualization demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIO_MAP.keys()),
        default="narrow_passage",
        help="Scenario to run",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the planner",
    )
    parser.add_argument(
        "--output-dir", default="outputs/demo",
        help="Root output directory",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Skip interactive display (always save files)",
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Skip GIF generation (faster)",
    )
    parser.add_argument(
        "--no-frames", action="store_true",
        help="Skip per-step PNG frames",
    )
    args = parser.parse_args(argv)

    # Use Agg backend when headless
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_name = args.scenario
    print(f"[demo] Scenario: {scenario_name}  seed={args.seed}")

    # ---- Build scenario ---------------------------------------------------
    builder = _SCENARIO_MAP[scenario_name]
    if scenario_name == "narrow_passage":
        scenario = builder(use_simenv=False)
    else:
        scenario = builder()

    # ---- Run pipeline -----------------------------------------------------
    print("[demo] Running pipeline …")
    data = _run_pipeline_simple(scenario, seed=args.seed,
                                 scenario_name=scenario_name)
    data["seed"] = args.seed

    if data["success"]:
        print(f"[demo] Planning OK — waypoints={len(data['path'])}, "
              f"path_length={data.get('path_length', 0.0):.3f} rad")
        print(f"[demo] Controller: {len(data['results'])} steps, "
              f"terminal_success={data.get('terminal_success')}, "
              f"final_error={data.get('final_goal_error', float('nan')):.4f} rad")
    else:
        print("[demo] WARNING — Planning FAILED. Showing partial output.")

    # ---- Output directory ------------------------------------------------
    out_dir = Path(args.output_dir) / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] Saving outputs to {out_dir.resolve()}")

    # ---- Summary JSON ----------------------------------------------------
    summary = build_summary(data, scenario_name)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[demo] summary.json saved → {summary_path}")
    print(f"       planning_success        = {summary['planning_success']}")
    print(f"       terminal_success        = {summary['terminal_success']}")
    print(f"       final_goal_error        = {summary['final_goal_error']}")
    print(f"       ever_entered_goal_region= {summary['ever_entered_goal_region']}")
    print(f"       clipped_ratio           = {summary['clipped_ratio']}")
    print(f"       min_tank_energy         = {summary['min_tank_energy']}")
    print(f"       unhandled_viol          = {summary['unhandled_violations']}")

    # ---- Metrics figure --------------------------------------------------
    print("[demo] Building summary figure …")
    fig = build_summary_figure(data, scenario_name, dt=DT)
    metrics_path = out_dir / "metrics.png"
    fig.savefig(metrics_path, dpi=120, bbox_inches="tight")
    print(f"[demo] metrics.png saved → {metrics_path}")

    if not args.headless:
        plt.show()
    plt.close(fig)

    # ---- Per-step frames -------------------------------------------------
    if not args.no_frames and data["results"]:
        print("[demo] Saving per-step frames …")
        frame_dir = out_dir / "frames"
        saved = save_frames(
            data["results"],
            data["q_history"],
            data["q_goal"],
            str(frame_dir),
            dt=DT,
            every_n=20,
            goal_radius=GOAL_RADIUS,
        )
        print(f"[demo] {len(saved)} frames saved → {frame_dir}")

    # ---- Animation -------------------------------------------------------
    if not args.no_animation and data["results"]:
        print("[demo] Building animation …")
        try:
            gif_path = str(out_dir / "animation.gif")
            save_trajectory_animation(
                data["results"],
                data["q_goal"],
                gif_path,
                dt=DT,
                fps=8,
                goal_radius=GOAL_RADIUS,
            )
            print(f"[demo] animation.gif saved → {gif_path}")
        except ImportError as exc:
            print(f"[demo] WARNING: animation skipped — {exc}")
        except Exception as exc:
            print(f"[demo] WARNING: animation failed — {exc}")

    print("[demo] Done.")
    return summary


if __name__ == "__main__":
    main()
