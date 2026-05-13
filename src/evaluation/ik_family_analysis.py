"""
IK posture-family analysis for barrier-navigation scenarios.

Groups IK goal configurations by gross posture family (elbow forward vs
backward vs centre) using the mean Y-position of the elbow link during a
straight-line joint-space interpolation from q_start to each goal.

Usage::

    from src.evaluation.ik_family_analysis import analyse_ik_families
    from src.scenarios.scenario_builders import build_frontal_i_barrier_lr
    from src.solver.planner.collision import make_collision_fn

    spec   = build_frontal_i_barrier_lr("medium")
    col_fn = make_collision_fn(spec=spec)
    report = analyse_ik_families(spec, col_fn)
    print(report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.scenarios.scenario_schema import ScenarioSpec
from src.solver.planner.collision import _panda_link_positions


# Link index for the elbow (link4 in the 0-indexed Panda chain).
# _panda_link_positions returns [link1, link2, ..., link7].
_ELBOW_IDX = 3   # link4 = elbow

# Y-threshold for classifying a goal as "centre" (elbow stays near y=0)
_CENTRE_THRESH = 0.04   # metres


@dataclass
class IKFamilyReport:
    """Summary of IK posture families for one scenario configuration."""
    n_total:    int
    n_free:     int
    n_blocked:  int
    families:   Dict[str, dict] = field(default_factory=dict)
    # families format: {label: {"n_free": int, "n_blocked": int, "mean_elbow_y": float}}

    def __str__(self) -> str:
        lines = [
            f"IK Family Report  (total={self.n_total}, free={self.n_free}, "
            f"blocked={self.n_blocked})"
        ]
        for label, info in self.families.items():
            lines.append(
                f"  {label:<20} free={info['n_free']}  blocked={info['n_blocked']}"
                f"  mean_elbow_y={info['mean_elbow_y']:.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _mean_elbow_y(q_start: np.ndarray, q_goal: np.ndarray, n_steps: int = 10) -> float:
    """
    Estimate the mean Y-position of the elbow link along the straight-line
    joint-space path from q_start to q_goal.

    Used as a coarse proxy for whether the elbow swings forward (+Y),
    backward (-Y), or stays near centre during the traversal.
    """
    ys = []
    for t in np.linspace(0.0, 1.0, n_steps):
        q = q_start + t * (q_goal - q_start)
        positions = _panda_link_positions(q)
        ys.append(positions[_ELBOW_IDX][1])
    return float(np.mean(ys))


def _classify_family(mean_y: float) -> str:
    """Map mean elbow Y to a family label."""
    if mean_y > _CENTRE_THRESH:
        return "elbow_fwd"
    elif mean_y < -_CENTRE_THRESH:
        return "elbow_back"
    else:
        return "elbow_center"


def analyse_ik_families(
    spec: ScenarioSpec,
    col_fn: Callable[[np.ndarray], bool],
) -> IKFamilyReport:
    """
    Categorise IK goals by posture family and collision status.

    Args:
        spec:   ScenarioSpec (provides q_start and ik_goals).
        col_fn: Callable ``(q) -> bool`` where True = collision-free.

    Returns:
        IKFamilyReport with per-family free/blocked counts.
    """
    q_start   = np.asarray(spec.q_start, dtype=float)
    ik_goals  = [np.asarray(g, dtype=float) for g in spec.ik_goals]

    families: Dict[str, dict] = {}
    n_free    = 0
    n_blocked = 0

    for goal in ik_goals:
        is_free  = bool(col_fn(goal))
        mean_y   = _mean_elbow_y(q_start, goal)
        label    = _classify_family(mean_y)

        if label not in families:
            families[label] = {"n_free": 0, "n_blocked": 0, "mean_elbow_y": 0.0,
                               "_y_sum": 0.0, "_count": 0}
        families[label]["_y_sum"]  += mean_y
        families[label]["_count"]  += 1
        if is_free:
            families[label]["n_free"] += 1
            n_free += 1
        else:
            families[label]["n_blocked"] += 1
            n_blocked += 1

    # Compute per-family mean elbow Y and remove internal accumulators
    for label, info in families.items():
        info["mean_elbow_y"] = info["_y_sum"] / max(1, info["_count"])
        del info["_y_sum"]
        del info["_count"]

    return IKFamilyReport(
        n_total=len(ik_goals),
        n_free=n_free,
        n_blocked=n_blocked,
        families=families,
    )
