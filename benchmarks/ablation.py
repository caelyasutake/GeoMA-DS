"""
Ablation Studies (CLAUDE.md — updated, required).

Four comparisons as specified in the updated CLAUDE.md:

  1. No passivity filter vs passivity filter
     Metric: passivity violation count / rate, clipped steps

  2. Old contact force (unfiltered) vs filtered contact force
     Metric: force error, tau magnitude, contact power (before/after filter)

  3. Tank only vs tank + passivity filter
     Metric: passivity violations, clipped steps, tank energy min

  4. Old IK filter (no energy scoring) vs energy-aware IK filter
     Metric: mean approach energy of returned goals, goal ranking quality

Run with:
    conda run -n ds-iks python benchmarks/ablation.py

Output is a printed table plus a JSON file ``benchmarks/ablation_results.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.planner.birrt            import plan, PlannerConfig
from src.solver.ds.path_ds               import PathDS, DSConfig
from src.solver.controller.impedance     import ControllerConfig, simulate
from src.solver.controller.passivity_filter import PassivityFilterConfig
from src.solver.controller.contact_force import (
    ContactForceConfig,
    contact_force_control,
)
from src.solver.tank.tank                import EnergyTank, TankConfig
from src.solver.ik.filter                import (
    FilterConfig,
    RobotState,
    filter_safe_set,
    estimate_approach_energy,
    estimate_contact_energy,
    make_mujoco_jacobian_fn,
)

from benchmarks.scenarios import (
    free_space_scenario,
    narrow_passage_scenario,
    N_JOINTS,
    Q_READY,
    _IK_GOALS_FREE,
    _GOAL_EASY,
    _GOAL_HARD,
    make_corridor_fn,
)

N         = N_JOINTS
N_SEEDS   = 10
MAX_ITER  = 5_000
STEP_SIZE = 0.15
DT        = 0.02
N_CTRL    = 300


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------
def _plan_once(q_start, Q_goals, col_fn, seed=0):
    cfg = PlannerConfig(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                        goal_bias=0.15, seed=seed)
    return plan(q_start, Q_goals, env=col_fn, config=cfg)


def _ctrl_run(path, ds_cfg=None, tank_cfg=None, ctrl_cfg=None):
    if ds_cfg   is None: ds_cfg   = DSConfig(K_c=1.0, K_r=1.0, K_n=0.0, goal_radius=0.05)
    if tank_cfg is None: tank_cfg = TankConfig(s_init=1.0, s_min=0.01, s_max=2.0, lambda_=1.0)
    if ctrl_cfg is None: ctrl_cfg = ControllerConfig(d_gain=3.0, f_n_gain=0.0)
    ds   = PathDS(path, config=ds_cfg)
    tank = EnergyTank(tank_cfg)
    q0   = np.asarray(path[0])
    return simulate(q0, np.zeros(N), ds, tank, dt=DT, n_steps=N_CTRL, config=ctrl_cfg)


def _get_path():
    """Return a free-space path for controller ablations."""
    sc = free_space_scenario()
    r  = _plan_once(sc["q_start"], sc["Q_goals"], sc["collision_fn"], seed=0)
    assert r.success, "Free-space planning failed — cannot run controller ablations"
    return r.path


# =========================================================================
# Ablation 1: No passivity filter vs with passivity filter
# =========================================================================
@dataclass
class FilterAblationResult:
    condition:              str
    nominal_violations:     float   # mean steps where z > 0 (pre-filter)
    unhandled_violations:   float   # mean steps where z > 0 AND filter NOT clipping
                                    # = actual energy injection that slips through
    unhandled_rate:         float   # unhandled / n_steps
    clipped_steps:          float   # mean steps where pf_clipped=True
    mean_power_reduction:   float   # mean (power_nom - power_filtered) when clipped


def ablate_passivity_filter() -> List[FilterAblationResult]:
    """
    Compare controller WITHOUT passivity filter vs WITH passivity filter.

    Both conditions use the same path, tank, and damping.  The only
    difference is whether PassivityFilterConfig is set in ControllerConfig.

    Key metrics:
    - unhandled_violations: steps where z > 0 AND the filter did NOT clip.
      For no_filter: all nominal violations are unhandled.
      For with_filter: clipped steps are handled → unhandled = 0.
    - clipped_steps: how often the filter prevented energy injection.
    """
    path = _get_path()

    # High K_r, low damping → f_R dominates, violations more likely.
    ds_cfg   = DSConfig(K_c=0.5, K_r=2.0, K_n=0.0, goal_radius=0.05)
    tank_cfg = TankConfig(s_init=2.0, s_min=0.01, s_max=2.0, lambda_=1.0)

    conditions = [
        ("no_filter", ControllerConfig(d_gain=1.0, f_n_gain=0.0,
                                        passivity_filter=None)),
        ("with_filter", ControllerConfig(d_gain=1.0, f_n_gain=0.0,
                                          passivity_filter=PassivityFilterConfig())),
    ]

    rng = np.random.default_rng(0)
    results = []
    for label, ctrl_cfg in conditions:
        nom_viols, unhandled, clips, power_reds = [], [], [], []
        for seed in range(N_SEEDS):
            qdot0 = rng.standard_normal(N) * 0.3
            ds    = PathDS(path, config=ds_cfg)
            tank  = EnergyTank(tank_cfg)
            q0    = np.asarray(path[0])
            runs  = simulate(q0, qdot0, ds, tank, dt=DT, n_steps=N_CTRL,
                             config=ctrl_cfg)
            nom_viols.append(sum(1 for r in runs if r.passivity_violated))
            # Unhandled: z > 0 and NOT clipped by filter
            unhandled.append(sum(1 for r in runs
                                 if r.passivity_violated and not r.pf_clipped))
            clips.append(sum(1 for r in runs if r.pf_clipped))
            diffs = [r.pf_power_nom - r.pf_power_filtered
                     for r in runs if r.pf_clipped]
            power_reds.append(float(np.mean(diffs)) if diffs else 0.0)

        u = float(np.mean(unhandled))
        c = float(np.mean(clips))
        results.append(FilterAblationResult(
            condition=label,
            nominal_violations=float(np.mean(nom_viols)),
            unhandled_violations=u,
            unhandled_rate=u / N_CTRL,
            clipped_steps=c,
            mean_power_reduction=float(np.mean(power_reds)),
        ))
    return results


# =========================================================================
# Ablation 2: Old contact force (unfiltered) vs filtered contact force
# =========================================================================
@dataclass
class ContactAblationResult:
    condition:           str
    mean_tau_norm:       float   # mean ||τ_n|| over test cases
    mean_power_nom:      float   # mean ẋᵀ τ_n_nom
    mean_power_filtered: float   # mean ẋᵀ τ_n (after filter, if applied)
    clipped_rate:        float   # fraction of cases where filter clipped
    power_reduction:     float   # mean (power_nom - power_filtered)


def ablate_contact_force() -> List[ContactAblationResult]:
    """
    Compare old (unfiltered) contact force vs passivity-filtered contact force.

    Test protocol:
    - Random joint velocities (qdot) and a fixed contact scenario
    - Unfiltered: use_filter=False → τ_n = J_lin^T f_n_task directly
    - Filtered:   use_filter=True  → τ_n projected to satisfy ẋᵀ τ_n ≤ 0
    - Metric: power injected before/after, clip rate, torque norm
    """
    rng     = np.random.default_rng(42)
    jac_fn  = make_mujoco_jacobian_fn()
    n_hat   = np.array([0.0, 0.0, 1.0])
    F_des   = 5.0
    q0      = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    J       = jac_fn(q0)
    n_trials = 200

    conditions = [
        # Unfiltered: raw contact term, no power constraint
        ("unfiltered", ContactForceConfig(k_f=2.0, use_filter=False)),
        # Filtered: two-sided band [-ε_n, +ε_n] = [-0.1, +0.1]
        ("filtered",   ContactForceConfig(k_f=2.0, use_filter=True)),
    ]

    results = []
    for label, cfg in conditions:
        tau_norms, p_noms, p_filts, clips = [], [], [], []
        for _ in range(n_trials):
            qdot = rng.standard_normal(N) * 0.5
            tau_n, res = contact_force_control(
                F_contact=np.zeros(3), F_desired=F_des, n_hat=n_hat,
                jacobian=J, qdot=qdot, config=cfg,
            )
            tau_norms.append(float(np.linalg.norm(tau_n)))
            p_noms.append(res.power_nom)
            p_filts.append(res.power_filtered)
            clips.append(float(res.clipped))

        p_n  = float(np.mean(p_noms))
        p_f  = float(np.mean(p_filts))
        results.append(ContactAblationResult(
            condition=label,
            mean_tau_norm=float(np.mean(tau_norms)),
            mean_power_nom=p_n,
            mean_power_filtered=p_f,
            clipped_rate=float(np.mean(clips)),
            power_reduction=p_n - p_f,
        ))
    return results


# =========================================================================
# Ablation 3: Tank only vs tank + passivity filter
# =========================================================================
@dataclass
class TankFilterAblationResult:
    condition:              str
    unhandled_violations:   float   # z > 0 AND NOT clipped — actual energy injection
    unhandled_rate:         float   # unhandled / n_steps
    clipped_steps:          float   # steps handled by passivity filter
    tank_energy_min:        float   # how much tank energy was consumed
    converged_rate:         float


def ablate_tank_vs_tank_filter() -> List[TankFilterAblationResult]:
    """
    Compare:
      - tank_only:       EnergyTank active, passivity_filter=None
      - tank_and_filter: EnergyTank active, passivity_filter=PassivityFilterConfig()

    The filter is the proactive first line of defence (catches injection
    before the tank sees it); the tank is the reactive second line.
    Using both reduces UNHANDLED injections relative to tank alone.

    Metric: unhandled_violations = steps where z>0 and NOT filtered.
    - tank_only:       z>0 steps are unhandled until tank depletes → beta_R→0
    - tank_and_filter: filter clips z>0 steps proactively → fewer reach the tank
    """
    path = _get_path()

    ds_cfg   = DSConfig(K_c=0.5, K_r=2.0, K_n=0.0, goal_radius=0.05)
    tank_cfg = TankConfig(s_init=1.0, s_min=0.01, s_max=2.0, lambda_=1.0)

    conditions = [
        ("tank_only",
         ControllerConfig(d_gain=1.0, f_n_gain=0.0, passivity_filter=None)),
        ("tank_and_filter",
         ControllerConfig(d_gain=1.0, f_n_gain=0.0,
                          passivity_filter=PassivityFilterConfig())),
    ]

    rng = np.random.default_rng(7)
    results = []
    for label, ctrl_cfg in conditions:
        unhandled, clips, mins, convs = [], [], [], []
        for seed in range(N_SEEDS):
            qdot0 = rng.standard_normal(N) * 0.3
            ds    = PathDS(path, config=ds_cfg)
            tank  = EnergyTank(tank_cfg)
            q0    = np.asarray(path[0])
            runs  = simulate(q0, qdot0, ds, tank, dt=DT, n_steps=N_CTRL,
                             config=ctrl_cfg)
            unhandled.append(sum(1 for r in runs
                                 if r.passivity_violated and not r.pf_clipped))
            clips.append(sum(1 for r in runs if r.pf_clipped))
            mins.append(min(r.tank_energy for r in runs))
            convs.append(float(runs[-1].V < runs[0].V))

        u = float(np.mean(unhandled))
        c = float(np.mean(clips))
        results.append(TankFilterAblationResult(
            condition=label,
            unhandled_violations=u,
            unhandled_rate=u / N_CTRL,
            clipped_steps=c,
            tank_energy_min=float(np.mean(mins)),
            converged_rate=float(np.mean(convs)),
        ))
    return results


# =========================================================================
# Ablation 4: Old IK filter (no energy scoring) vs energy-aware IK filter
# =========================================================================
@dataclass
class IKFilterAblationResult:
    condition:           str
    n_safe:              float   # mean survivors
    mean_approach_energy:float   # mean E_approach of first returned goal
    mean_contact_energy: float   # mean E_contact of first returned goal
    first_goal_rank:     float   # mean rank of "best" goal (0 = optimal ordering)


def ablate_ik_filter() -> List[IKFilterAblationResult]:
    """
    Compare old FilterConfig (w_energy=0, w_contact=0, no sorting) vs
    energy-aware FilterConfig (w_energy=1, w_contact=0, sorted by approach energy).

    Test with a diverse set of IK goals at various distances from q_current.
    Metric: approach energy of the FIRST returned goal (lower is better).
    """
    rng = np.random.default_rng(99)
    q_curr = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    state  = RobotState(q=q_curr)

    # Generate random valid goals near the default config
    from src.solver.ik.filter import PANDA_Q_MIN, PANDA_Q_MAX
    n_goals_per_trial = 10
    n_trials = 20

    old_cfg = FilterConfig(
        manipulability_threshold=0.0,
        joint_limit_margin_threshold=-np.inf,
        collision_checker=None,
        jacobian_fn=None,
        w_energy=0.0,    # no energy-aware sorting
        w_contact=0.0,
    )
    new_cfg = FilterConfig(
        manipulability_threshold=0.0,
        joint_limit_margin_threshold=-np.inf,
        collision_checker=None,
        jacobian_fn=None,
        w_energy=1.0,    # sort by approach energy
        w_contact=0.0,
    )

    conditions = [
        ("old_filter",          old_cfg),
        ("energy_aware_filter", new_cfg),
    ]

    results = []
    for label, cfg in conditions:
        n_safes, e_approachs, e_contacts, ranks = [], [], [], []
        for _ in range(n_trials):
            # Random goals at various distances
            Q_raw = []
            for _ in range(n_goals_per_trial):
                q = rng.uniform(PANDA_Q_MIN * 0.5, PANDA_Q_MAX * 0.5)
                Q_raw.append(q)

            fres = filter_safe_set(Q_raw, state, tank_energy=1e9, config=cfg)
            if not fres.solutions:
                continue

            n_safes.append(fres.num_safe)
            first = fres.solutions[0]

            # Compute approach energy of the first goal
            e_app = estimate_approach_energy(first, q_curr)
            e_approachs.append(e_app)

            # Rank of the first goal in terms of approach energy
            # (0 = lowest energy = optimal)
            all_energies = [estimate_approach_energy(q, q_curr)
                            for q in fres.solutions]
            rank = sorted(all_energies).index(e_app)
            ranks.append(rank)

        results.append(IKFilterAblationResult(
            condition=label,
            n_safe=float(np.mean(n_safes)),
            mean_approach_energy=float(np.mean(e_approachs)),
            mean_contact_energy=0.0,   # contact weight disabled in both conditions
            first_goal_rank=float(np.mean(ranks)),
        ))
    return results


# =========================================================================
# Print & save
# =========================================================================
def _print_table(title: str, rows: list, fields: list) -> None:
    w = 24
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")
    header = "".join(f"{f:<{w}}" for f in fields)
    print(header)
    print("─" * len(header))
    for row in rows:
        d = asdict(row)
        line = "".join(
            f"{str(d[f])[:w-1]:<{w}}" if isinstance(d[f], str)
            else f"{d[f]:<{w}.4f}" if isinstance(d[f], float)
            else f"{d[f]:<{w}}"
            for f in fields
        )
        print(line)


def main() -> None:
    print("\n" + "=" * 72)
    print("  Ablation Studies (CLAUDE.md — updated)")
    print("=" * 72)

    all_results = {}

    print("\n[1/4] No passivity filter vs with passivity filter")
    ab1 = ablate_passivity_filter()
    _print_table(
        "Ablation 1: Passivity Filter Effect",
        ab1,
        ["condition", "nominal_violations", "unhandled_violations",
         "unhandled_rate", "clipped_steps"],
    )
    all_results["passivity_filter"] = [asdict(r) for r in ab1]

    print("\n[2/4] Old contact force (unfiltered) vs filtered contact force")
    ab2 = ablate_contact_force()
    _print_table(
        "Ablation 2: Contact Force Filtering",
        ab2,
        ["condition", "mean_tau_norm", "mean_power_nom",
         "mean_power_filtered", "clipped_rate"],
    )
    all_results["contact_force"] = [asdict(r) for r in ab2]

    print("\n[3/4] Tank only vs tank + passivity filter")
    ab3 = ablate_tank_vs_tank_filter()
    _print_table(
        "Ablation 3: Tank Only vs Tank + Filter",
        ab3,
        ["condition", "unhandled_violations", "unhandled_rate",
         "clipped_steps", "tank_energy_min"],
    )
    all_results["tank_vs_tank_filter"] = [asdict(r) for r in ab3]

    print("\n[4/4] Old IK filter vs energy-aware IK filter")
    ab4 = ablate_ik_filter()
    _print_table(
        "Ablation 4: IK Filter Energy Awareness",
        ab4,
        ["condition", "n_safe", "mean_approach_energy", "first_goal_rank"],
    )
    all_results["ik_filter"] = [asdict(r) for r in ab4]

    # Save JSON
    out_path = Path(__file__).parent / "ablation_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n  Results saved → {out_path}")

    # Key findings summary
    print("\n" + "=" * 72)
    print("  Key Findings")
    print("=" * 72)

    nf_u = ab1[0].unhandled_violations   # no_filter: all violations unhandled
    pf_u = ab1[1].unhandled_violations   # with_filter: filter catches them all
    pf_c = ab1[1].clipped_steps
    print(f"  [1] Filter: unhandled injections {nf_u:.1f} → {pf_u:.1f}"
          f"  (filter clipped {pf_c:.1f} steps / run)")

    uf_pw = ab2[0].mean_power_nom
    f_pw  = ab2[1].mean_power_filtered
    f_cr  = ab2[1].clipped_rate
    print(f"  [2] Contact: power nom={uf_pw:.3f} → filtered={f_pw:.3f}"
          f"  (clipped {f_cr:.0%} of steps)")

    to_u  = ab3[0].unhandled_violations
    tf_u  = ab3[1].unhandled_violations
    tf_c  = ab3[1].clipped_steps
    print(f"  [3] Tank+Filter: unhandled {to_u:.1f} → {tf_u:.1f}"
          f"  (filter pre-empted {tf_c:.1f} injections / run)")

    old_e = ab4[0].mean_approach_energy
    new_e = ab4[1].mean_approach_energy
    old_r = ab4[0].first_goal_rank
    new_r = ab4[1].first_goal_rank
    print(f"  [4] IK Filter: first-goal energy {old_e:.3f} → {new_e:.3f}"
          f"  (rank {old_r:.1f} → {new_r:.1f})")


if __name__ == "__main__":
    main()
