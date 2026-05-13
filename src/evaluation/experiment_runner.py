"""
Orchestrates N-trial experiments across conditions and scenarios.

Usage::

    runner = ExperimentRunner(output_dir=Path("outputs/eval"))
    results = runner.run(
        scenario_name="free_space",
        conditions=[
            make_condition(IKCondition.MULTI_IK_FULL, ControlCondition.PATH_DS_FULL),
            make_condition(IKCondition.SINGLE_IK_BEST, ControlCondition.PATH_DS_FULL),
        ],
        n_trials=50,
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.evaluation.baselines import ControlCondition, IKCondition, TrialCondition, make_condition
from src.evaluation.metrics import TrialMetrics, append_jsonl, load_jsonl
from src.evaluation.trial_runner import run_contact_trial, run_planning_trial
from src.scenarios.scenario_builders import (
    cluttered_tabletop_scenario,
    contact_task_scenario,
    free_space_scenario,
    narrow_passage_scenario,
    wall_contact_scenario,
)

try:
    from src.scenarios.scenario_builders import random_obstacle_field_scenario
except ImportError:
    random_obstacle_field_scenario = None

try:
    from src.scenarios.scenario_builders import u_shape_scenario
except ImportError:
    u_shape_scenario = None

try:
    from src.scenarios.scenario_builders import left_open_u_scenario
except ImportError:
    left_open_u_scenario = None

from src.scenarios.scenario_schema import ScenarioSpec


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_BUILDERS: Dict[str, Callable[..., ScenarioSpec]] = {
    "free_space":                   free_space_scenario,
    "narrow_passage":               narrow_passage_scenario,
    "cluttered_tabletop":           cluttered_tabletop_scenario,
    "contact_task":                 contact_task_scenario,
    "contact_circle":               wall_contact_scenario,
    "contact_circle_perturbation":  wall_contact_scenario,
}
if random_obstacle_field_scenario is not None:
    SCENARIO_BUILDERS["random_obstacle_field"] = random_obstacle_field_scenario
if u_shape_scenario is not None:
    SCENARIO_BUILDERS["u_shape"] = u_shape_scenario
if left_open_u_scenario is not None:
    SCENARIO_BUILDERS["left_open_u"] = left_open_u_scenario

CONTACT_SCENARIOS = {"contact_circle", "contact_circle_perturbation"}


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs a matrix of (scenario × condition) experiments over N trials each.

    Outputs:
        ``{output_dir}/per_trial_results.jsonl``  — one JSON line per trial
        ``{output_dir}/experiment_manifest.json`` — run configuration
    """

    def __init__(
        self,
        output_dir: Path = Path("outputs/eval"),
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._jsonl_path = self.output_dir / "per_trial_results.jsonl"

    # ------------------------------------------------------------------
    def run(
        self,
        scenario_name: str,
        conditions: List[TrialCondition],
        n_trials: int = 50,
        seed_offset: int = 0,
        trial_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[TrialMetrics]:
        """
        Run all (condition × trial) combinations for a single scenario.

        Seeds are assigned as ``seed_offset + trial_index`` for each trial,
        ensuring reproducibility and independence across conditions.

        Args:
            scenario_name: Key in SCENARIO_BUILDERS.
            conditions:    List of TrialCondition to test.
            n_trials:      Number of independent trials per condition.
            seed_offset:   Base seed (incremented per trial).
            trial_kwargs:  Extra kwargs forwarded to the trial runner.

        Returns:
            All TrialMetrics collected in this run.
        """
        if scenario_name not in SCENARIO_BUILDERS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(SCENARIO_BUILDERS)}"
            )

        trial_kwargs = trial_kwargs or {}
        is_contact   = scenario_name in CONTACT_SCENARIOS
        use_perturb  = scenario_name == "contact_circle_perturbation"

        spec = SCENARIO_BUILDERS[scenario_name]()

        all_results: List[TrialMetrics] = []
        total = len(conditions) * n_trials
        done  = 0

        for cond in conditions:
            for t in range(n_trials):
                seed = seed_offset + t
                trial_id = done

                t0 = time.perf_counter()
                try:
                    if is_contact:
                        kwargs = dict(trial_kwargs)
                        if use_perturb:
                            kwargs.setdefault("perturb_magnitude", 12.0)
                        else:
                            kwargs.setdefault("perturb_magnitude", 0.0)
                        result = run_contact_trial(
                            spec, cond, seed=seed, trial_id=trial_id, **kwargs
                        )
                    else:
                        result = run_planning_trial(
                            spec, cond, seed=seed, trial_id=trial_id, **trial_kwargs
                        )
                except Exception as exc:
                    import traceback
                    result = TrialMetrics(
                        trial_id=trial_id, seed=seed,
                        scenario=scenario_name, condition=cond.name,
                        error=traceback.format_exc(),
                        wall_time_s=time.perf_counter() - t0,
                    )

                append_jsonl(self._jsonl_path, result)
                all_results.append(result)
                done += 1

                if self.verbose:
                    status = "OK" if result.error is None else "ERR"
                    success = _quick_success(result)
                    print(
                        f"  [{done:4d}/{total}] {scenario_name} | "
                        f"{cond.name:50s} | seed={seed} | "
                        f"{status} success={success} "
                        f"({result.wall_time_s:.1f}s)"
                    )

        return all_results

    # ------------------------------------------------------------------
    def run_spec(
        self,
        spec: "ScenarioSpec",
        conditions: List[TrialCondition],
        n_trials: int = 50,
        seed_offset: int = 0,
        trial_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[TrialMetrics]:
        """
        Run all (condition × trial) combinations for a directly provided ScenarioSpec.

        Identical to ``run()`` but accepts a pre-built ScenarioSpec instead of
        looking one up by name.  Used for parameterised sweeps (e.g. the
        U-shape difficulty sweep) where the caller varies geometry between runs.

        Args:
            spec:        Pre-built ScenarioSpec.
            conditions:  List of TrialCondition to test.
            n_trials:    Number of independent trials per condition.
            seed_offset: Base seed (incremented per trial).
            trial_kwargs: Extra kwargs forwarded to the trial runner.

        Returns:
            All TrialMetrics collected in this run.
        """
        trial_kwargs = trial_kwargs or {}
        all_results: List[TrialMetrics] = []
        total = len(conditions) * n_trials
        done  = 0

        for cond in conditions:
            for t in range(n_trials):
                seed     = seed_offset + t
                trial_id = done
                t0 = time.perf_counter()
                try:
                    result = run_planning_trial(
                        spec, cond, seed=seed, trial_id=trial_id, **trial_kwargs
                    )
                except Exception:
                    import traceback as _tb
                    result = TrialMetrics(
                        trial_id=trial_id, seed=seed,
                        scenario=spec.name, condition=cond.name,
                        error=_tb.format_exc(),
                        wall_time_s=time.perf_counter() - t0,
                    )

                append_jsonl(self._jsonl_path, result)
                all_results.append(result)
                done += 1

                if self.verbose:
                    status  = "OK" if result.error is None else "ERR"
                    success = _quick_success(result)
                    print(
                        f"  [{done:4d}/{total}] {spec.name} | "
                        f"{cond.name:50s} | seed={seed} | "
                        f"{status} success={success} "
                        f"({result.wall_time_s:.1f}s)"
                    )

        return all_results

    # ------------------------------------------------------------------
    def run_matrix(
        self,
        scenarios: List[str],
        conditions: List[TrialCondition],
        n_trials: int = 50,
        trial_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[TrialMetrics]]:
        """
        Run all (scenario × condition) combinations.

        Returns:
            Dict mapping scenario_name → List[TrialMetrics].
        """
        all_by_scenario: Dict[str, List[TrialMetrics]] = {}
        for scen in scenarios:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Scenario: {scen}  ({len(conditions)} conditions × {n_trials} trials)")
                print(f"{'='*60}")
            results = self.run(
                scen, conditions, n_trials=n_trials, trial_kwargs=trial_kwargs
            )
            all_by_scenario[scen] = results
        return all_by_scenario

    # ------------------------------------------------------------------
    def load_results(self) -> List[TrialMetrics]:
        """Load all previously recorded results from the JSONL file."""
        if not self._jsonl_path.exists():
            return []
        return load_jsonl(self._jsonl_path)

    # ------------------------------------------------------------------
    def save_manifest(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration as a JSON manifest."""
        path = self.output_dir / "experiment_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Quick helpers
# ---------------------------------------------------------------------------

def _quick_success(tm: TrialMetrics) -> Optional[bool]:
    if tm.error:
        return False
    if tm.contact is not None:
        return tm.contact.contact_established
    if tm.execution is not None:
        return tm.execution.terminal_success
    if tm.plan is not None:
        return tm.plan.success
    return None


def build_ablation_conditions(
    ctrl: ControlCondition = ControlCondition.PATH_DS_FULL,
) -> List[TrialCondition]:
    """Standard IK ablation: vary IK condition, fix control condition."""
    from src.evaluation.baselines import PLANNING_IK_CONDITIONS
    return [make_condition(ik, ctrl) for ik in PLANNING_IK_CONDITIONS]


def build_baseline_conditions(
    ik: IKCondition = IKCondition.MULTI_IK_FULL,
) -> List[TrialCondition]:
    """Standard control baseline: fix IK, vary control condition."""
    from src.evaluation.baselines import PLANNING_CTRL_CONDITIONS
    return [make_condition(ik, ctrl) for ctrl in PLANNING_CTRL_CONDITIONS]


def build_contact_conditions(
    ik: IKCondition = IKCondition.MULTI_IK_FULL,
) -> List[TrialCondition]:
    """Contact task conditions."""
    from src.evaluation.baselines import CONTACT_CTRL_CONDITIONS
    return [make_condition(ik, ctrl) for ctrl in CONTACT_CTRL_CONDITIONS]
