"""
HJCD-IK problem JSON generator.

Converts a canonical ``ScenarioSpec`` into the HJCD-IK JSON problem format
used by ``hjcdik.generate_solutions()``.

The generated JSON matches the schema in ``external/HJCD-IK/tests/*.json``:

    {
      "problems": {
        "<problem_set_name>": [
          {
            "collision_buffer_ik": float,
            "goal_ik": [],
            "goal_pose": {
              "frame": "panda_hand",
              "position_xyz": [x, y, z],
              "quaternion_wxyz": [w, x, y, z]
            },
            "obstacles": {
              "cuboid":   {name: {"dims": [...], "pose": [...]}},
              "cylinder": {name: {"radius": r, "height": h, "pose": [...]}},
              "sphere":   {name: {"radius": r, "pose": [...]}}
            },
            "start": [...],
            "world_frame": "panda_link0"
          }
        ]
      }
    }

Primary entry points
--------------------
build_problem_json(spec, ...)   — returns JSON string
save_problem_json(spec, path)   — writes JSON to file, returns dict
validate_problem_json_contains_obstacles(problem_json, spec)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.scenarios.scenario_schema import ScenarioSpec

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_WORLD_FRAME = "panda_link0"
_DEFAULT_EE_FRAME    = "panda_hand"
_PROBLEM_SET_NAME    = "multi_ik"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_problem_json(
    spec: ScenarioSpec,
    collision_buffer: float = 0.0,
    problem_set_name: str = _PROBLEM_SET_NAME,
    world_frame: str = _DEFAULT_WORLD_FRAME,
    ee_frame: str = _DEFAULT_EE_FRAME,
) -> str:
    """
    Build an HJCD-IK problem JSON string from a ScenarioSpec.

    Only collision-enabled obstacles are included (spec.collision_obstacles()).

    Args:
        spec:             Canonical scenario specification.
        collision_buffer: Safety margin added around obstacles (metres).
        problem_set_name: Key under "problems" in the JSON.
        world_frame:      Robot base frame name.
        ee_frame:         End-effector frame name.

    Returns:
        JSON string.

    Raises:
        ValueError: if spec has no target_pose.
    """
    if spec.target_pose is None:
        raise ValueError(
            f"Scenario {spec.name!r} has no target_pose — "
            "cannot generate HJCD-IK problem JSON without an EE target."
        )

    tp = spec.target_pose
    problem: Dict[str, Any] = {
        "collision_buffer_ik": float(collision_buffer),
        "goal_ik": [],
        "goal_pose": {
            "frame": ee_frame,
            "position_xyz": [float(v) for v in tp["position"]],
            "quaternion_wxyz": [float(v) for v in tp["quaternion_wxyz"]],
        },
        "obstacles": spec.obstacles_as_hjcd_dict(),
        "start": [float(v) for v in spec.q_start],
        "world_frame": world_frame,
    }

    doc = {"problems": {problem_set_name: [problem]}}
    return json.dumps(doc, indent=2)


def save_problem_json(
    spec: ScenarioSpec,
    path: str | Path,
    **kwargs,
) -> Dict:
    """
    Build and save the HJCD-IK problem JSON to ``path``.

    Returns the parsed dict (for downstream use).
    """
    json_str = build_problem_json(spec, **kwargs)
    parsed   = json.loads(json_str)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json_str)
    return parsed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_problem_json_contains_obstacles(
    problem_json: str | Dict,
    spec: ScenarioSpec,
) -> None:
    """
    Verify that every collision-enabled obstacle in ``spec`` appears in the
    generated HJCD-IK problem JSON.

    Args:
        problem_json: JSON string or already-parsed dict.
        spec:         Canonical scenario specification.

    Raises:
        AssertionError: if any collision obstacle is missing.
    """
    if isinstance(problem_json, str):
        data = json.loads(problem_json)
    else:
        data = problem_json

    # Navigate to the first problem entry
    problems_root = data.get("problems", {})
    all_problems  = []
    for problems_list in problems_root.values():
        all_problems.extend(problems_list)

    if not all_problems:
        raise AssertionError("Problem JSON contains no problem entries.")

    problem = all_problems[0]
    obs_in_json: Dict = problem.get("obstacles", {})

    # Flatten all obstacle names present in JSON
    json_obs_names = set()
    for type_dict in obs_in_json.values():
        json_obs_names.update(type_dict.keys())

    # Check every collision-enabled obstacle appears by name
    missing = []
    for obs in spec.collision_obstacles():
        if obs.name not in json_obs_names:
            missing.append(obs.name)

    if missing:
        raise AssertionError(
            f"These collision obstacles from spec {spec.name!r} are missing "
            f"from the HJCD-IK problem JSON: {missing}\n"
            f"JSON obstacle names: {sorted(json_obs_names)}"
        )
