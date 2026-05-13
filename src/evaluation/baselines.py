"""
Condition definitions for the evaluation ablation study.

An evaluation condition is a combination of:
  - IKCondition:   how IK solutions are selected / filtered before planning
  - ControlCondition: which controller / passivity settings are used

Usage::

    from src.evaluation.baselines import IKCondition, ControlCondition, TrialCondition, make_condition

    cond = make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL)
    Q_goals = select_ik_goals(all_goals, cond.ik, seed=42)
    ctrl_cfg, tank_cfg, pf_cfg = build_controller_config(cond.ctrl, spec)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# IK conditions
# ---------------------------------------------------------------------------

class IKCondition(str, Enum):
    """How IK solutions are selected before handing to the planner."""
    MULTI_IK_FULL          = "multi_ik_full"          # all safe solutions
    SINGLE_IK_BEST         = "single_ik_best"         # top-ranked solution only
    SINGLE_IK_RANDOM       = "single_ik_random"       # random single solution
    MULTI_IK_TOP_2         = "multi_ik_top_2"         # top 2
    MULTI_IK_TOP_4         = "multi_ik_top_4"         # top 4
    MULTI_IK_TOP_8         = "multi_ik_top_8"         # top 8
    MULTI_IK_ENERGY_AWARE  = "multi_ik_energy_aware"  # energy-sorted (same as full, explicit label)
    VANILLA_DS             = "vanilla_ds"             # no BiRRT; straight-line DS to nearest IK goal


# ---------------------------------------------------------------------------
# Control conditions
# ---------------------------------------------------------------------------

class ControlCondition(str, Enum):
    """Which controller / passivity configuration is used."""
    # Planning-phase controllers
    PATH_DS_FULL        = "path_ds_full"           # DS + filter + tank (full system)
    PATH_DS_NO_TANK     = "path_ds_no_tank"        # DS + filter, tank disabled (β_R = 1 always)
    PATH_DS_NO_FILTER   = "path_ds_no_filter"      # DS + tank, passivity filter disabled
    PATH_DS_NO_MULTIK   = "path_ds_no_multik"      # single best IK + full DS controller
    WAYPOINT_PD         = "waypoint_pd_controller" # simple waypoint-following PD

    # Modulation-based controllers (replaces CBF with task-space modulation matrix)
    PATH_DS_MODULATION   = "path_ds_modulation"    # BiRRT + DS + modulation
    VANILLA_DS_MODULATION = "vanilla_ds_modulation" # no BiRRT, direct DS + modulation

    # Task-space DS + differential IK (no PathDS, no BiRRT)
    VANILLA_DS_DIFFIK_MODULATION = "vanilla_ds_diffik_modulation"

    # Contact-task controllers
    TASK_TRACKING_FULL         = "task_tracking_full"          # full contact circle controller
    TASK_TRACKING_NO_FORCE     = "task_tracking_no_force_reg"  # tangential only, K_f=0
    TASK_TRACKING_NO_PASSIVITY = "task_tracking_no_passivity"  # no tank/filter
    JOINT_SPACE_PATH_ONLY      = "joint_space_path_only"       # no task-space circle


# ---------------------------------------------------------------------------
# Composite condition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialCondition:
    """Composite condition identifier for a trial."""
    ik:   IKCondition
    ctrl: ControlCondition

    @property
    def name(self) -> str:
        return f"{self.ik.value}__{self.ctrl.value}"

    def __str__(self) -> str:
        return self.name


def make_condition(
    ik: IKCondition,
    ctrl: ControlCondition,
) -> TrialCondition:
    return TrialCondition(ik=ik, ctrl=ctrl)


# ---------------------------------------------------------------------------
# IK goal selection
# ---------------------------------------------------------------------------

def select_ik_goals(
    all_goals: List[np.ndarray],
    condition: IKCondition,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Apply the IK condition to select a subset of goals for the planner.

    Args:
        all_goals: Pre-filtered list of safe IK solutions (sorted best-first).
        condition: Which IK condition to apply.
        seed:      RNG seed for stochastic conditions.

    Returns:
        List of joint configs (possibly subset of all_goals).
    """
    if not all_goals:
        return []

    rng = random.Random(seed)

    if condition == IKCondition.SINGLE_IK_BEST:
        return [all_goals[0]]

    elif condition == IKCondition.SINGLE_IK_RANDOM:
        return [rng.choice(all_goals)]

    elif condition == IKCondition.MULTI_IK_TOP_2:
        return all_goals[:2]

    elif condition == IKCondition.MULTI_IK_TOP_4:
        return all_goals[:4]

    elif condition == IKCondition.MULTI_IK_TOP_8:
        return all_goals[:8]

    elif condition in (IKCondition.MULTI_IK_FULL, IKCondition.MULTI_IK_ENERGY_AWARE):
        return list(all_goals)

    return list(all_goals)


# ---------------------------------------------------------------------------
# Diversity metrics for IK sets
# ---------------------------------------------------------------------------

def ik_diversity_score(goals: List[np.ndarray]) -> float:
    """Mean pairwise L2 distance in joint space."""
    if len(goals) < 2:
        return 0.0
    dists = []
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            dists.append(float(np.linalg.norm(goals[i] - goals[j])))
    return float(np.mean(dists))


def ik_goal_spread(goals: List[np.ndarray]) -> float:
    """Mean per-joint standard deviation across the goal set."""
    if len(goals) < 2:
        return 0.0
    arr = np.stack(goals)            # (n_goals, n_joints)
    return float(np.mean(np.std(arr, axis=0)))


# ---------------------------------------------------------------------------
# Controller config factories
# ---------------------------------------------------------------------------

def build_ctrl_config(
    condition: ControlCondition,
    d_gain: float = 5.0,
    gravity_fn=None,
):
    """
    Build (ControllerConfig, TankConfig, PassivityFilterConfig) for a condition.

    Returns a plain dict of kwargs understood by the calling code.
    """
    from src.solver.controller.impedance import ControllerConfig
    from src.solver.tank.tank import TankConfig
    from src.solver.controller.passivity_filter import PassivityFilterConfig
    from src.solver.controller.cbf_filter import CBFConfig
    from src.solver.ds.modulation import ModulationConfig
    from src.solver.controller.hard_shield import HardShieldConfig

    pf_cfg = PassivityFilterConfig()
    tank_cfg = TankConfig(s_init=1.0, s_min=0.01, s_max=2.0)

    _modulation_conditions = (
        ControlCondition.PATH_DS_MODULATION,
        ControlCondition.VANILLA_DS_MODULATION,
        ControlCondition.VANILLA_DS_DIFFIK_MODULATION,
    )
    use_modulation = condition in _modulation_conditions

    # CBF on for standard conditions; disabled when modulation is the avoidance layer.
    cbf_cfg = CBFConfig(enabled=(not use_modulation), d_safe=0.03, d_buffer=0.05, alpha=8.0)
    mod_cfg = ModulationConfig(enabled=use_modulation) if use_modulation else None
    # Hard shield: always on for modulation conditions so contact is impossible.
    shield_cfg = HardShieldConfig(enabled=use_modulation, d_hard_min=0.01) if use_modulation else None

    if condition == ControlCondition.PATH_DS_NO_TANK:
        tank_cfg = TankConfig(s_init=2.0, s_min=0.0, s_max=1e9)

    elif condition == ControlCondition.PATH_DS_NO_FILTER:
        pf_cfg = PassivityFilterConfig(epsilon_min=-1e9, epsilon_max=-1e9)

    ctrl_cfg = ControllerConfig(
        d_gain=d_gain,
        f_n_gain=0.0,
        gravity_fn=gravity_fn,
        orthogonalize_residual=True,
        alpha=0.5,
        passivity_filter=pf_cfg,
        cbf=cbf_cfg,
        modulation=mod_cfg,
        hard_shield=shield_cfg,
    )

    return ctrl_cfg, tank_cfg, pf_cfg


def build_task_ctrl_config(
    condition: ControlCondition,
    d_gain: float = 5.0,
    gravity_fn=None,
    K_p: float = 5.0,
    K_v: float = 1.0,
    K_f: float = 2.0,
    F_desired: float = 3.0,
):
    """Build TaskTrackingConfig + TankConfig for a contact-task condition."""
    from src.solver.controller.task_tracking import TaskTrackingConfig
    from src.solver.tank.tank import TankConfig
    from src.solver.controller.passivity_filter import PassivityFilterConfig

    pf_cfg = PassivityFilterConfig()
    tank_cfg = TankConfig(s_init=1.0, s_min=0.01, s_max=2.0)

    if condition == ControlCondition.TASK_TRACKING_NO_FORCE:
        K_f = 0.0

    elif condition == ControlCondition.TASK_TRACKING_NO_PASSIVITY:
        tank_cfg = TankConfig(s_init=2.0, s_min=0.0, s_max=1e9)
        pf_cfg   = PassivityFilterConfig(epsilon_min=-1e9, epsilon_max=-1e9)

    task_cfg = TaskTrackingConfig(
        K_p=K_p, K_v=K_v, K_f=K_f, F_desired=F_desired,
        alpha_contact=1.0,
        damping=d_gain,
        lambda_reg=0.01,
        gravity_fn=gravity_fn,
        orthogonalize_residual=True,
        alpha=0.5,
        xdot_max=0.50,
        v_tan_max=0.25,
        v_norm_max=0.20,
    )

    return task_cfg, tank_cfg, pf_cfg


# ---------------------------------------------------------------------------
# Standard condition lists for experiments
# ---------------------------------------------------------------------------

PLANNING_IK_CONDITIONS = [
    IKCondition.MULTI_IK_FULL,
    IKCondition.SINGLE_IK_BEST,
    IKCondition.SINGLE_IK_RANDOM,
    IKCondition.MULTI_IK_TOP_2,
    IKCondition.MULTI_IK_TOP_4,
    IKCondition.MULTI_IK_ENERGY_AWARE,
]

PLANNING_CTRL_CONDITIONS = [
    ControlCondition.PATH_DS_FULL,
    ControlCondition.PATH_DS_NO_TANK,
    ControlCondition.PATH_DS_NO_FILTER,
    ControlCondition.WAYPOINT_PD,
    ControlCondition.PATH_DS_MODULATION,
    ControlCondition.VANILLA_DS_MODULATION,
    ControlCondition.VANILLA_DS_DIFFIK_MODULATION,
]

CONTACT_CTRL_CONDITIONS = [
    ControlCondition.TASK_TRACKING_FULL,
    ControlCondition.TASK_TRACKING_NO_FORCE,
    ControlCondition.TASK_TRACKING_NO_PASSIVITY,
    ControlCondition.JOINT_SPACE_PATH_ONLY,
]
