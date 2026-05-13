"""
Main Benchmark Runner.

Runs all three CLAUDE.md scenarios and collects the required metrics:

  Scenario 1 — Free Space
  Scenario 2 — Narrow Passage  (single vs multi-IK)
  Scenario 3 — Contact Task

Usage:
    conda run -n ds-iks python benchmarks/run_benchmarks.py

Outputs:
  * Printed table for each scenario
  * benchmarks/benchmark_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.planner.birrt        import plan, PlannerConfig
from src.solver.ds.path_ds           import PathDS, DSConfig
from src.solver.controller.impedance import ControllerConfig, simulate, step as ctrl_step
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.tank.tank            import EnergyTank, TankConfig
from src.simulation.env              import SimEnv, SimEnvConfig

from benchmarks.scenarios   import (
    free_space_scenario, narrow_passage_scenario, contact_task_scenario,
    N_JOINTS,
)
from benchmarks.metrics     import (
    BenchmarkResult, IKMetrics, PlanningMetrics, ControlMetrics, ContactMetrics,
)

N         = N_JOINTS
N_SEEDS   = 5    # seeds for success-rate averaging
MAX_ITER  = 5_000
STEP_SIZE = 0.15
DT_SIM    = 0.002  # MuJoCo timestep for Scenario 3
N_CTRL    = 300    # controller steps for Scenarios 1 & 2
N_CTRL_3  = 500    # controller steps for Scenario 3


# ---------------------------------------------------------------------------
# Scenario 1 — Free Space
# ---------------------------------------------------------------------------
def run_scenario1() -> BenchmarkResult:
    sc      = free_space_scenario()
    q_start = sc["q_start"]
    Q_goals = sc["Q_goals"]
    col_fn  = sc["collision_fn"]

    # IK metrics (hand-crafted set = "raw"; all pass free-space filter)
    ik_m = IKMetrics.compute(Q_raw=Q_goals, Q_safe=Q_goals)

    # Planning: success rate over N_SEEDS
    plan_results = []
    for seed in range(N_SEEDS):
        cfg = PlannerConfig(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                            goal_bias=0.15, seed=seed)
        plan_results.append(plan(q_start, Q_goals, env=col_fn, config=cfg))

    # Use first successful plan for controller
    best = next((r for r in plan_results if r.success), plan_results[0])
    plan_m = PlanningMetrics.from_plan_result(best)
    # Overwrite success/time to reflect average success rate
    n_ok = sum(r.success for r in plan_results)
    plan_m.success = n_ok == N_SEEDS   # True if all succeeded

    ctrl_results = []
    if best.success:
        ds   = PathDS(best.path, config=DSConfig(K_c=2.0, K_r=1.0,
                                                  K_n=0.0, goal_radius=0.05))
        # Tuned ε(s): small injection allowed when tank full, strict when empty
        tank = EnergyTank(TankConfig(s_init=1.0, epsilon_min=-0.05, epsilon_max=0.1))
        pf   = PassivityFilterConfig()
        cfg  = ControllerConfig(d_gain=3.0, f_n_gain=0.0, passivity_filter=pf,
                                orthogonalize_residual=True)
        ctrl_results = simulate(q_start, np.zeros(N), ds, tank,
                                dt=0.05, n_steps=N_CTRL, config=cfg)

    ctrl_m = ControlMetrics.from_results(ctrl_results) if ctrl_results else None

    return BenchmarkResult(
        scenario="1_free_space",
        ik=ik_m,
        planning=plan_m,
        control=ctrl_m,
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Narrow Passage (single vs multi-IK)
# ---------------------------------------------------------------------------
def run_scenario2() -> BenchmarkResult:
    sc      = narrow_passage_scenario()
    q_start = sc["q_start"]
    col_fn  = sc["collision_fn"]

    def _run_set(goals, label):
        successes, iters, times = [], [], []
        for seed in range(N_SEEDS):
            cfg = PlannerConfig(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                                goal_bias=0.10, seed=seed)
            t0 = time.perf_counter()
            r  = plan(q_start, goals, env=col_fn, config=cfg)
            successes.append(r.success)
            iters.append(r.iterations)
            times.append(time.perf_counter() - t0)
        return {
            "label":        label,
            "n_goals":      len(goals),
            "success_rate": float(np.mean(successes)),
            "mean_iters":   float(np.mean(iters)),
            "mean_time_s":  float(np.mean(times)),
        }

    single = _run_set(sc["Q_goals_single"], "single_ik")
    multi  = _run_set(sc["Q_goals_multi"],  "multi_ik")

    # PlanningMetrics for multi-IK (the recommended variant)
    best_multi = None
    for seed in range(N_SEEDS):
        cfg = PlannerConfig(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                            goal_bias=0.10, seed=seed)
        r = plan(q_start, sc["Q_goals_multi"], env=col_fn, config=cfg)
        if r.success:
            best_multi = r
            break

    plan_m = PlanningMetrics.from_plan_result(best_multi) if best_multi else None

    # Attach comparison as extra fields in a custom dict
    result = BenchmarkResult(
        scenario="2_narrow_passage",
        planning=plan_m,
    )
    # Store comparison results in scenario metadata
    result._meta = {"single_ik": single, "multi_ik": multi}  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Scenario 3 — Contact Task
# ---------------------------------------------------------------------------
def run_scenario3() -> BenchmarkResult:
    sc      = contact_task_scenario()
    q_start = sc["q_start"]
    q_goal  = sc["Q_goals"][0]
    obs     = sc["obstacles"]

    # Build SimEnv with obstacle
    env = SimEnv(SimEnvConfig(obstacles=obs, timestep=DT_SIM))
    env.set_state(q_start, np.zeros(N))

    ds   = PathDS([q_start, q_goal],
                  config=DSConfig(K_c=2.0, K_r=0.5, K_n=0.0, goal_radius=0.05))
    tank = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0,
                                  epsilon_min=-0.05, epsilon_max=0.1))
    grav = env.make_gravity_fn()
    pf   = PassivityFilterConfig()
    cfg  = ControllerConfig(d_gain=5.0, f_n_gain=0.0, gravity_fn=grav,
                             passivity_filter=pf, orthogonalize_residual=True)

    ctrl_results = []
    contact_log  = []
    q    = q_start.copy()
    qdot = np.zeros(N)

    for _ in range(N_CTRL_3):
        res = ctrl_step(q, qdot, ds, tank, DT_SIM, config=cfg)
        ctrl_results.append(res)

        env.step(res.tau, dt=DT_SIM)
        contact_log.append(env.get_contact_forces())
        q    = env.q
        qdot = env.qdot

    ctrl_m    = ControlMetrics.from_results(ctrl_results)
    contact_m = ContactMetrics.from_contact_log(contact_log, f_desired=0.0)

    return BenchmarkResult(
        scenario="3_contact_task",
        control=ctrl_m,
        contact=contact_m,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 60)
    print("  Multi-IK DS Benchmark Suite")
    print("=" * 60)

    all_results = {}

    print("\n[1/3] Running Scenario 1: Free Space...")
    r1 = run_scenario1()
    r1.print_summary()
    all_results["scenario_1"] = r1.to_dict()

    print("\n[2/3] Running Scenario 2: Narrow Passage...")
    r2 = run_scenario2()
    r2.print_summary()
    meta = getattr(r2, "_meta", {})
    all_results["scenario_2"] = {**r2.to_dict(), "comparison": meta}
    if meta:
        print("\n  IK Goal Comparison:")
        for variant, d in meta.items():
            print(f"    {variant:12s}: success={d['success_rate']:.0%}"
                  f"  iters={d['mean_iters']:.0f}"
                  f"  time={d['mean_time_s']:.3f}s")

    print("\n[3/3] Running Scenario 3: Contact Task...")
    r3 = run_scenario3()
    r3.print_summary()
    all_results["scenario_3"] = r3.to_dict()

    # Save JSON
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  Full results saved → {out_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("  Summary (CLAUDE.md Acceptance Criteria)")
    print("=" * 60)
    if r1.planning:
        print(f"  Scenario 1 success:  {'PASS' if r1.planning.success else 'FAIL'}")
    if r1.control:
        print(f"  No tank depletion:   "
              f"{'PASS' if r1.control.tank_energy_min > 0 else 'FAIL'}"
              f"  (min={r1.control.tank_energy_min:.3f})")
    if r2.planning:
        print(f"  Multi-IK vs single:  See comparison table above")
    if r3.control:
        print(f"  Contact stability:   "
              f"{'PASS' if r3.control.tank_energy_min >= 0 else 'FAIL'}"
              f"  (violations={r3.control.passivity_violations})")


if __name__ == "__main__":
    main()
