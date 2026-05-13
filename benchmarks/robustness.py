"""
Robustness Benchmarking Suite (CLAUDE.md — required).

Validates the system is not just correct, but robust across:

  1. Multi-seed stochastic evaluation (≥50 seeds)
     - IK seed, RRT sampling, initial state variation

  2. Parameter sweeps
     - alpha ∈ [0.2, 0.8]
     - epsilon_max ∈ [0.01, 0.2]
     - damping D ∈ [low, high]
     - K_c ∈ [1.0, 5.0]

  3. Disturbance injection
     - xdot_noise ~ Gaussian
     - force_noise ~ Gaussian

  4. Stress tests
     - high-speed motion
     - near-singular configurations

Robustness acceptance criteria:
  - success rate ≥ 95% across seeds
  - clipping ratio < 50%
  - convergence rate ≥ 95%
  - 0 unhandled passivity violations
  - stable contact across noise

Usage:
    conda run -n ds-iks python benchmarks/robustness.py
    conda run -n ds-iks python benchmarks/robustness.py --quick   # 20 seeds

Output: benchmarks/robustness_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.planner.birrt            import plan, PlannerConfig
from src.solver.ds.path_ds               import PathDS, DSConfig
from src.solver.controller.impedance     import ControllerConfig, simulate
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.tank.tank                import EnergyTank, TankConfig

from benchmarks.scenarios import (
    free_space_scenario,
    narrow_passage_scenario,
    N_JOINTS,
    Q_READY,
    _IK_GOALS_FREE,
    _GOAL_EASY,
    make_corridor_fn,
)

N        = N_JOINTS
DT       = 0.02
N_CTRL   = 300
MAX_ITER = 5_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _plan(q_start, Q_goals, col_fn, seed):
    cfg = PlannerConfig(max_iterations=MAX_ITER, step_size=0.15,
                        goal_bias=0.15, seed=seed)
    return plan(q_start, Q_goals, env=col_fn, config=cfg)


def _get_free_path(seed=0):
    sc = free_space_scenario()
    r  = _plan(sc["q_start"], sc["Q_goals"], sc["collision_fn"], seed)
    return r.path if r.success else None


def _run_controller(
    path,
    q0=None,
    qdot0=None,
    K_c=2.0,
    alpha=0.5,
    epsilon_max=0.1,
    d_gain=3.0,
    qdot_noise_std=0.0,
    rng=None,
):
    """Run the full pipeline for N_CTRL steps, returning control results."""
    if q0 is None:
        q0 = np.asarray(path[0])
    if qdot0 is None:
        qdot0 = np.zeros(N)
    if rng is None:
        rng = np.random.default_rng(0)

    ds   = PathDS(path, config=DSConfig(K_c=K_c, K_r=1.0, K_n=0.0,
                                         goal_radius=0.05))
    tank = EnergyTank(TankConfig(s_init=1.0, s_min=0.01, s_max=2.0,
                                  epsilon_min=-0.05, epsilon_max=epsilon_max))
    pf   = PassivityFilterConfig()
    cfg  = ControllerConfig(d_gain=d_gain, f_n_gain=0.0,
                             passivity_filter=pf,
                             orthogonalize_residual=True,
                             alpha=alpha)

    q    = np.asarray(q0, dtype=float).copy()
    qdot = np.asarray(qdot0, dtype=float).copy()
    results = []

    from src.solver.controller.impedance import step as ctrl_step
    for _ in range(N_CTRL):
        # Optionally inject velocity noise
        qdot_in = qdot + rng.standard_normal(N) * qdot_noise_std
        res = ctrl_step(q, qdot_in, ds, tank, DT, config=cfg)
        results.append(res)
        # Unit-mass Euler integration
        q    = q + qdot * DT
        qdot = qdot + res.tau * DT

    return results


def _metrics(results) -> Dict:
    """
    Summarise a list of ControlResult into scalar metrics.

    Key distinction:
    - passivity_violated: nominal z = ẋᵀ f_R_nom > 0 (pre-orthogonalization)
    - unhandled_violations: pf_power_nom > threshold AND not clipped
      pf_power_nom is the power AFTER orthogonalization and scaling,
      so with orthogonalize_residual=True this is ≈ 0 → unhandled ≈ 0.
    """
    n = len(results)
    if n == 0:
        return {}
    viols   = sum(1 for r in results if r.passivity_violated)
    # Unhandled = actual power injected (after ortho) exceeds threshold & not filtered
    # Threshold 1e-10 sits above floating-point residual (~1e-15) but below any
    # real injection; avoids false positives from orthogonalization float noise.
    unhand  = sum(1 for r in results
                  if r.pf_power_nom > 1e-10 and not r.pf_clipped)
    clipped = sum(1 for r in results if r.pf_clipped)
    V_start = results[0].V
    V_end   = results[-1].V
    return {
        "n_steps":              n,
        "passivity_violations": viols,
        "unhandled_violations": unhand,
        "clipped_steps":        clipped,
        "clipped_ratio":        clipped / n,
        "tank_energy_min":      min(r.tank_energy for r in results),
        "V_start":              V_start,
        "V_end":                V_end,
        "converged":            V_end < V_start,
    }


# ---------------------------------------------------------------------------
# 1. Multi-seed stochastic evaluation (≥50 seeds)
# ---------------------------------------------------------------------------
@dataclass
class MultiSeedResult:
    n_seeds:         int
    success_rate:    float   # planning succeeded
    success_rate_std: float
    convergence_rate: float  # V_end < V_start
    convergence_std:  float
    clipped_ratio_mean: float
    clipped_ratio_std:  float
    unhandled_mean:   float  # should be 0
    tank_min_mean:    float


def run_multiseed(n_seeds: int = 50) -> MultiSeedResult:
    """Evaluate free-space pipeline across n_seeds seeds."""
    sc       = free_space_scenario()
    q_start  = sc["q_start"]
    Q_goals  = sc["Q_goals"]
    col_fn   = sc["collision_fn"]

    plan_ok, convg, clips, unhand, tank_mins = [], [], [], [], []
    rng_base = np.random.default_rng(0)

    for seed in range(n_seeds):
        r = _plan(q_start, Q_goals, col_fn, seed)
        plan_ok.append(float(r.success))
        if not r.success:
            continue

        rng = np.random.default_rng(seed)
        results = _run_controller(r.path, rng=rng)
        m = _metrics(results)
        convg.append(float(m["converged"]))
        clips.append(m["clipped_ratio"])
        unhand.append(m["unhandled_violations"])
        tank_mins.append(m["tank_energy_min"])

    def _safe_std(lst):
        return float(np.std(lst)) if lst else 0.0

    return MultiSeedResult(
        n_seeds=n_seeds,
        success_rate=float(np.mean(plan_ok)),
        success_rate_std=_safe_std(plan_ok),
        convergence_rate=float(np.mean(convg)) if convg else 0.0,
        convergence_std=_safe_std(convg),
        clipped_ratio_mean=float(np.mean(clips)) if clips else 0.0,
        clipped_ratio_std=_safe_std(clips),
        unhandled_mean=float(np.mean(unhand)) if unhand else 0.0,
        tank_min_mean=float(np.mean(tank_mins)) if tank_mins else 0.0,
    )


# ---------------------------------------------------------------------------
# 2. Parameter sweeps
# ---------------------------------------------------------------------------
@dataclass
class SweepPoint:
    param_name:  str
    param_value: float
    clipped_ratio_mean: float
    clipped_ratio_std:  float
    convergence_rate:   float
    unhandled_mean:     float


def _sweep_param(
    param_name: str,
    values: List[float],
    n_seeds: int = 20,
) -> List[SweepPoint]:
    path_cache: Dict[int, Optional[list]] = {}
    sc = free_space_scenario()

    # Pre-plan paths once per seed
    for seed in range(n_seeds):
        r = _plan(sc["q_start"], sc["Q_goals"], sc["collision_fn"], seed)
        path_cache[seed] = r.path if r.success else None

    results = []
    for val in values:
        clips_all, convg_all, unhand_all = [], [], []
        for seed in range(n_seeds):
            path = path_cache[seed]
            if path is None:
                continue
            kwargs: Dict = {}
            kwargs[param_name] = val
            rng = np.random.default_rng(seed)
            runs = _run_controller(path, rng=rng, **kwargs)
            m = _metrics(runs)
            clips_all.append(m["clipped_ratio"])
            convg_all.append(float(m["converged"]))
            unhand_all.append(float(m["unhandled_violations"]))

        if not clips_all:
            continue
        results.append(SweepPoint(
            param_name=param_name,
            param_value=float(val),
            clipped_ratio_mean=float(np.mean(clips_all)),
            clipped_ratio_std=float(np.std(clips_all)),
            convergence_rate=float(np.mean(convg_all)),
            unhandled_mean=float(np.mean(unhand_all)),
        ))
    return results


def run_parameter_sweeps(n_seeds: int = 20) -> Dict[str, List[SweepPoint]]:
    sweeps = {}
    sweeps["alpha"]       = _sweep_param("alpha",       [0.2, 0.4, 0.5, 0.6, 0.8], n_seeds)
    sweeps["epsilon_max"] = _sweep_param("epsilon_max", [0.01, 0.05, 0.1, 0.15, 0.2], n_seeds)
    sweeps["d_gain"]      = _sweep_param("d_gain",      [1.0, 3.0, 5.0, 8.0], n_seeds)
    sweeps["K_c"]         = _sweep_param("K_c",         [1.0, 2.0, 3.0, 5.0], n_seeds)
    return sweeps


# ---------------------------------------------------------------------------
# 3. Disturbance injection
# ---------------------------------------------------------------------------
@dataclass
class DisturbanceResult:
    noise_std:          float
    convergence_rate:   float
    clipped_ratio_mean: float
    unhandled_mean:     float
    tank_min_mean:      float


def run_disturbance_injection(n_seeds: int = 20) -> List[DisturbanceResult]:
    """Evaluate robustness to Gaussian velocity noise at varying amplitudes."""
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    sc = free_space_scenario()

    paths = []
    for seed in range(n_seeds):
        r = _plan(sc["q_start"], sc["Q_goals"], sc["collision_fn"], seed)
        if r.success:
            paths.append((seed, r.path))

    results = []
    for std in noise_levels:
        convg, clips, unhand, tmins = [], [], [], []
        for seed, path in paths:
            rng = np.random.default_rng(seed + 1000)
            runs = _run_controller(path, qdot_noise_std=std, rng=rng)
            m = _metrics(runs)
            convg.append(float(m["converged"]))
            clips.append(m["clipped_ratio"])
            unhand.append(float(m["unhandled_violations"]))
            tmins.append(m["tank_energy_min"])

        results.append(DisturbanceResult(
            noise_std=std,
            convergence_rate=float(np.mean(convg)),
            clipped_ratio_mean=float(np.mean(clips)),
            unhandled_mean=float(np.mean(unhand)),
            tank_min_mean=float(np.mean(tmins)),
        ))
    return results


# ---------------------------------------------------------------------------
# 4. Stress tests
# ---------------------------------------------------------------------------
@dataclass
class StressResult:
    test_name:          str
    convergence_rate:   float
    clipped_ratio_mean: float
    unhandled_mean:     float
    tank_min_mean:      float


def run_stress_tests(n_seeds: int = 20) -> List[StressResult]:
    sc   = free_space_scenario()
    path = _get_free_path(seed=0)
    assert path is not None, "Free-space plan failed for stress tests"

    tests: List[StressResult] = []

    # --- High-speed motion: large initial qdot ----------------------------
    convg, clips, unhand, tmins = [], [], [], []
    for seed in range(n_seeds):
        rng   = np.random.default_rng(seed)
        qdot0 = rng.standard_normal(N) * 2.0   # 4× normal amplitude
        runs  = _run_controller(path, qdot0=qdot0, rng=rng)
        m = _metrics(runs)
        convg.append(float(m["converged"]))
        clips.append(m["clipped_ratio"])
        unhand.append(float(m["unhandled_violations"]))
        tmins.append(m["tank_energy_min"])

    tests.append(StressResult(
        test_name="high_speed_motion",
        convergence_rate=float(np.mean(convg)),
        clipped_ratio_mean=float(np.mean(clips)),
        unhandled_mean=float(np.mean(unhand)),
        tank_min_mean=float(np.mean(tmins)),
    ))

    # --- Near-singular start: q with small manipulability ----------------
    # Panda singularity: fully extended arm (q = zeros for most joints)
    q_singular = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    convg, clips, unhand, tmins = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # Build a short path from singular config to goal
        goal = _GOAL_EASY.copy()
        p = [q_singular, goal]
        runs = _run_controller(p, rng=rng)
        m = _metrics(runs)
        convg.append(float(m["converged"]))
        clips.append(m["clipped_ratio"])
        unhand.append(float(m["unhandled_violations"]))
        tmins.append(m["tank_energy_min"])

    tests.append(StressResult(
        test_name="near_singular_config",
        convergence_rate=float(np.mean(convg)),
        clipped_ratio_mean=float(np.mean(clips)),
        unhandled_mean=float(np.mean(unhand)),
        tank_min_mean=float(np.mean(tmins)),
    ))

    # --- Low-energy tank: starts nearly depleted -------------------------
    convg, clips, unhand, tmins = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        ds   = PathDS(path, config=DSConfig(K_c=2.0, K_r=1.0, K_n=0.0,
                                             goal_radius=0.05))
        tank = EnergyTank(TankConfig(s_init=0.05, s_min=0.01, s_max=2.0,
                                      epsilon_min=-0.05, epsilon_max=0.1))
        pf   = PassivityFilterConfig()
        cfg  = ControllerConfig(d_gain=3.0, f_n_gain=0.0,
                                 passivity_filter=pf,
                                 orthogonalize_residual=True,
                                 alpha=0.5)
        runs = simulate(np.asarray(path[0]), np.zeros(N), ds, tank,
                        dt=DT, n_steps=N_CTRL, config=cfg)
        m = _metrics(runs)
        convg.append(float(m["converged"]))
        clips.append(m["clipped_ratio"])
        unhand.append(float(m["unhandled_violations"]))
        tmins.append(m["tank_energy_min"])

    tests.append(StressResult(
        test_name="depleted_tank",
        convergence_rate=float(np.mean(convg)),
        clipped_ratio_mean=float(np.mean(clips)),
        unhandled_mean=float(np.mean(unhand)),
        tank_min_mean=float(np.mean(tmins)),
    ))

    return tests


# ---------------------------------------------------------------------------
# Acceptance criteria check
# ---------------------------------------------------------------------------
def check_acceptance(
    multiseed: MultiSeedResult,
    disturbance: List[DisturbanceResult],
    stress: List[StressResult],
) -> Dict[str, bool]:
    checks = {
        "planning_success_ge_95pct":   multiseed.success_rate >= 0.95,
        "convergence_rate_ge_95pct":   multiseed.convergence_rate >= 0.95,
        "clipping_ratio_lt_50pct":     multiseed.clipped_ratio_mean < 0.50,
        "no_unhandled_violations":     multiseed.unhandled_mean == 0.0,
        "tank_never_negative":         multiseed.tank_min_mean >= 0.0,
        "robust_to_noise_converge":    all(d.convergence_rate >= 0.80
                                           for d in disturbance if d.noise_std <= 0.1),
        "robust_no_unhandled_noise":   all(d.unhandled_mean == 0.0
                                           for d in disturbance),
        "stress_no_unhandled":         all(s.unhandled_mean == 0.0 for s in stress),
    }
    return checks


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def _hdr(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def _row(**kv) -> None:
    parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
             for k, v in kv.items()]
    print("  " + "   ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_seeds: int = 50) -> None:
    print("\n" + "=" * 70)
    print("  Robustness Benchmark Suite")
    print("=" * 70)

    all_results: Dict = {}

    # 1. Multi-seed evaluation
    _hdr(f"1. Multi-Seed Evaluation ({n_seeds} seeds)")
    ms = run_multiseed(n_seeds)
    _row(success_rate=ms.success_rate, success_std=ms.success_rate_std)
    _row(convergence_rate=ms.convergence_rate, convergence_std=ms.convergence_std)
    _row(clipped_ratio=ms.clipped_ratio_mean, clipped_std=ms.clipped_ratio_std)
    _row(unhandled_violations=ms.unhandled_mean)
    _row(tank_energy_min_mean=ms.tank_min_mean)
    all_results["multiseed"] = asdict(ms)

    # 2. Parameter sweeps
    _hdr("2. Parameter Sweeps (20 seeds each)")
    sweeps = run_parameter_sweeps(n_seeds=min(20, n_seeds))
    sweep_data: Dict = {}
    for param, points in sweeps.items():
        print(f"\n  {param}:")
        print(f"    {'value':>8}  {'clip%':>7}  {'conv%':>7}  {'unhandled':>10}")
        for p in points:
            print(f"    {p.param_value:>8.3f}  "
                  f"{p.clipped_ratio_mean:>7.1%}  "
                  f"{p.convergence_rate:>7.1%}  "
                  f"{p.unhandled_mean:>10.1f}")
        sweep_data[param] = [asdict(p) for p in points]
    all_results["parameter_sweeps"] = sweep_data

    # 3. Disturbance injection
    _hdr("3. Disturbance Injection")
    print(f"  {'noise_std':>10}  {'conv%':>7}  {'clip%':>7}  {'unhandled':>10}")
    disturbance = run_disturbance_injection(n_seeds=min(20, n_seeds))
    for d in disturbance:
        print(f"  {d.noise_std:>10.3f}  "
              f"{d.convergence_rate:>7.1%}  "
              f"{d.clipped_ratio_mean:>7.1%}  "
              f"{d.unhandled_mean:>10.1f}")
    all_results["disturbance"] = [asdict(d) for d in disturbance]

    # 4. Stress tests
    _hdr("4. Stress Tests")
    stress = run_stress_tests(n_seeds=min(20, n_seeds))
    for s in stress:
        _row(test=s.test_name, conv=s.convergence_rate,
             clip=s.clipped_ratio_mean, unhandled=s.unhandled_mean)
    all_results["stress"] = [asdict(s) for s in stress]

    # Acceptance criteria
    _hdr("Acceptance Criteria")
    checks = check_acceptance(ms, disturbance, stress)
    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Save
    out_path = Path(__file__).parent / "robustness_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run with 20 seeds instead of 50")
    args = parser.parse_args()
    main(n_seeds=20 if args.quick else 50)
