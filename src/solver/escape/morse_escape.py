"""Top-level Morse escape planner orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.solver.ds.escape_policy import EscapePolicy
from src.solver.ds.path_ds import PathDS
from src.solver.escape.escape_primitives import generate_candidates
from src.solver.escape.execution_adapter import AdaptedEscapeAction, adapt
from src.solver.escape.rollout_scorer import ScoredEscapeCandidate, score_all
from src.solver.ik.goal_selection import IKGoalInfo
from src.evaluation.metrics import MorseEscapeMetrics
from src.evaluation.stall_detection import StallDiagnostics, TrapDiagnostics


@dataclass
class NegativeCurvatureConfig:
    enabled: bool = False
    n_lanczos_iters: int = 12
    n_power_iters: int = 20
    curvature_threshold: float = -0.01
    hvp_eps: float = 1e-4


@dataclass
class MorseEscapeConfig:
    enabled: bool = True

    # Activation
    activate_on_trap: bool = True

    # Candidate generation
    num_random_dirs: int = 32
    use_joint_basis_dirs: bool = True
    use_ik_family_dirs: bool = True
    use_obstacle_tangents: bool = True
    use_clearance_gradient: bool = True
    max_candidates: int = 128

    # Rollout
    rollout_horizon: int = 30
    execute_prefix_len: int = 5
    escape_step_size: float = 0.05
    rollout_dt: float = 0.05
    collision_clearance_threshold: float = 0.0

    # Acceptance
    min_score_to_accept: float = 0.0
    min_clearance_improvement: float = 0.01
    max_energy_increase_allowed: float = 0.25

    # Scoring weights
    w_goal_progress: float = 5.0
    w_clearance_gain: float = 3.0
    w_final_clearance: float = 2.0
    w_collision: float = 1000.0
    w_joint_path_length: float = 0.1
    w_joint_limit_cost: float = 1.0
    w_energy_increase: float = 0.5

    # Fallback order
    fallback_to_existing_escape: bool = True
    fallback_to_birrt: bool = True

    # Reproducibility
    random_seed: Optional[int] = None

    # Negative curvature
    negative_curvature: NegativeCurvatureConfig = field(
        default_factory=NegativeCurvatureConfig
    )


@dataclass
class MorseEscapeResult:
    success: bool
    action: Optional[AdaptedEscapeAction]
    execute_prefix_qs: List[np.ndarray]
    escape_mode: Optional[str]
    metrics: MorseEscapeMetrics
    planning_time_s: float


class MorseEscapePlanner:
    def __init__(self, config: MorseEscapeConfig) -> None:
        self.config = config

    def plan_escape(
        self,
        q: np.ndarray,
        stall_diag: StallDiagnostics,
        trap_diag: TrapDiagnostics,
        ds: PathDS,
        clearance_fn: Callable[[np.ndarray], float],
        goal_error_fn: Callable[[np.ndarray], float],
        energy_fn: Callable[[np.ndarray], float],
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        ik_infos: List[IKGoalInfo],
        escape_policy: EscapePolicy,
        seed: Optional[int] = None,
    ) -> MorseEscapeResult:
        """Plan an escape from the current trapped state.

        Returns MorseEscapeResult. success=False means no valid candidate was found;
        the caller should use fallback logic.
        """
        t0 = time.perf_counter()
        metrics = MorseEscapeMetrics()

        if not self.config.enabled or not self.config.activate_on_trap:
            return MorseEscapeResult(
                success=False, action=None, execute_prefix_qs=[],
                escape_mode=None, metrics=metrics, planning_time_s=0.0,
            )

        effective_seed = seed if seed is not None else self.config.random_seed
        rng = np.random.default_rng(effective_seed)

        # 1. Generate candidates
        candidates = generate_candidates(
            q, stall_diag, trap_diag, jacobian_fn, clearance_fn,
            ik_infos, self.config, rng,
        )
        metrics.n_candidates_generated = len(candidates)
        metrics.activated = True

        if not candidates:
            elapsed = time.perf_counter() - t0
            metrics.planning_time_s = elapsed
            return MorseEscapeResult(
                success=False, action=None, execute_prefix_qs=[],
                escape_mode=None, metrics=metrics, planning_time_s=elapsed,
            )

        # 2. Score via rollout
        t_rollout = time.perf_counter()
        scored = score_all(
            q, candidates, ds, clearance_fn, goal_error_fn, energy_fn, self.config
        )
        metrics.rollout_eval_time_s = time.perf_counter() - t_rollout
        metrics.n_candidates_valid = sum(1 for s in scored if s.valid)

        # 3. Pick best valid candidate
        best: Optional[ScoredEscapeCandidate] = None
        for s in scored:
            if s.valid:
                best = s
                break

        if best is None:
            elapsed = time.perf_counter() - t0
            metrics.planning_time_s = elapsed
            return MorseEscapeResult(
                success=False, action=None, execute_prefix_qs=[],
                escape_mode=None, metrics=metrics, planning_time_s=elapsed,
            )

        # 4. Adapt to executable action
        action = adapt(best, escape_policy, self.config)

        # 5. Populate metrics
        metrics.selected_source_tag = best.candidate.source_tag
        metrics.selected_score = best.score
        metrics.execute_mode = action.mode
        nc_used = best.candidate.source_tag == "negative_curvature"
        metrics.negative_curvature_used = nc_used
        if nc_used:
            metrics.negative_curvature_eigenvalue = best.candidate.metadata.get("curvature")
        elapsed = time.perf_counter() - t0
        metrics.planning_time_s = elapsed
        prefix_qs = action.prefix_qs if action.prefix_qs is not None else []

        return MorseEscapeResult(
            success=True,
            action=action,
            execute_prefix_qs=prefix_qs,
            escape_mode=action.mode,
            metrics=metrics,
            planning_time_s=elapsed,
        )
