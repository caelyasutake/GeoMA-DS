"""
Phase 1: HJCD-IK Wrapper

Generates a diverse set of IK solutions using HJCD-IK.
JSON problem definitions are built programmatically from target_pose,
robot_config, and env_config, matching the format in external/HJCD-IK/tests/*.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

import hjcdik

# ---------------------------------------------------------------------------
# Default robot configuration (Panda arm)
# ---------------------------------------------------------------------------
DEFAULT_ROBOT_CONFIG: Dict[str, Any] = {
    "start": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "world_frame": "panda_link0",
    "end_effector_frame": "panda_hand",
    "collision_buffer_ik": 0.0,
}

PROBLEM_SET_NAME = "multi_ik"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class IKMetadata:
    """Metadata stored alongside IK solutions."""
    num_raw: int                      # solutions returned by HJCD-IK
    num_unique: int                   # after deduplication
    num_final: int                    # after any post-processing
    pos_errors: np.ndarray            # per-solution position error (m)
    ori_errors: np.ndarray            # per-solution orientation error (rad)
    cluster_labels: np.ndarray        # cluster id per solution (-1 = noise)
    solve_time_s: float               # wall-clock solve time
    collision_free: bool


@dataclass
class IKResult:
    solutions: List[np.ndarray]       # list of joint configs, shape (n_joints,)
    metadata: IKMetadata


# ---------------------------------------------------------------------------
# JSON problem builder
# ---------------------------------------------------------------------------
def _build_problems_json(
    target_pose: Dict[str, Any],
    robot_config: Dict[str, Any],
    env_config: Optional[Dict[str, Any]],
) -> str:
    """
    Build a JSON string matching the HJCD-IK problem format.

    target_pose must have:
        "position": [x, y, z]
        "quaternion_wxyz": [w, x, y, z]

    robot_config must have:
        "start": list of joint angles
        "world_frame": str
        "end_effector_frame": str
        "collision_buffer_ik": float

    env_config may have:
        "obstacles": {
            "cuboid": { name: {"dims": [...], "pose": [...]} },
            "cylinder": { name: {"radius": float, "height": float, "pose": [...]} }
        }
    """
    rc = {**DEFAULT_ROBOT_CONFIG, **robot_config}

    problem = {
        "collision_buffer_ik": float(rc.get("collision_buffer_ik", 0.0)),
        "goal_ik": [],
        "goal_pose": {
            "frame": rc["end_effector_frame"],
            "position_xyz": [float(v) for v in target_pose["position"]],
            "quaternion_wxyz": [float(v) for v in target_pose["quaternion_wxyz"]],
        },
        "obstacles": (env_config or {}).get("obstacles", {}),
        "start": [float(v) for v in rc["start"]],
        "world_frame": rc["world_frame"],
    }

    doc = {"problems": {PROBLEM_SET_NAME: [problem]}}
    return json.dumps(doc)


def _parse_target_pose(target_pose) -> Dict[str, Any]:
    """Accept either a dict or a flat list [x,y,z,qw,qx,qy,qz]."""
    if isinstance(target_pose, (list, np.ndarray)) and len(target_pose) == 7:
        arr = [float(v) for v in target_pose]
        return {"position": arr[:3], "quaternion_wxyz": arr[3:]}
    if isinstance(target_pose, dict):
        return target_pose
    raise ValueError(
        "target_pose must be a 7-element list [x,y,z,qw,qx,qy,qz] or a dict "
        "with 'position' and 'quaternion_wxyz' keys."
    )


def _flat_target(target_pose_dict: Dict[str, Any]) -> List[float]:
    """Return [x, y, z, qw, qx, qy, qz] for HJCD-IK."""
    pos = list(target_pose_dict["position"])
    quat = list(target_pose_dict["quaternion_wxyz"])
    return pos + quat


# ---------------------------------------------------------------------------
# Post-processing: deduplication
# ---------------------------------------------------------------------------
def _deduplicate(
    configs: np.ndarray,
    pos_errors: np.ndarray,
    ori_errors: np.ndarray,
    threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove near-duplicate joint configurations using pairwise L2 distance.
    Keeps the solution with the smallest position error when duplicates exist.
    threshold: joint-space L2 distance below which two configs are considered identical.
    """
    if len(configs) == 0:
        return configs, pos_errors, ori_errors

    # Sort by pos_error ascending so we keep the best of each duplicate group
    order = np.argsort(pos_errors)
    configs = configs[order]
    pos_errors = pos_errors[order]
    ori_errors = ori_errors[order]

    keep = np.ones(len(configs), dtype=bool)
    dists = cdist(configs, configs)

    for i in range(len(configs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(configs)):
            if keep[j] and dists[i, j] < threshold:
                keep[j] = False

    return configs[keep], pos_errors[keep], ori_errors[keep]


# ---------------------------------------------------------------------------
# Post-processing: clustering
# ---------------------------------------------------------------------------
def _cluster(configs: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster joint configs with k-means (or trivial assignment if too few points).
    Returns integer label array of length len(configs).
    """
    if len(configs) == 0:
        return np.array([], dtype=int)

    k = min(n_clusters, len(configs))
    if k <= 1:
        return np.zeros(len(configs), dtype=int)

    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(configs.astype(np.float64), k, minit="points", seed=0)
    return labels


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def solve_batch(
    target_pose,
    robot_config: Optional[Dict[str, Any]] = None,
    env_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 2000,
    num_solutions: int = 10,
    dedup_threshold: float = 1e-3,
    n_clusters: int = 5,
) -> IKResult:
    """
    Generate a diverse set of IK solutions for target_pose.

    Args:
        target_pose: 7-element list [x,y,z,qw,qx,qy,qz] OR dict with
                     'position' and 'quaternion_wxyz' keys.
        robot_config: Robot configuration dict. Defaults to Panda arm defaults.
        env_config:   Environment config with 'obstacles' key. None = no obstacles.
        batch_size:   Number of random IK restarts (larger = more diverse solutions).
        num_solutions: Number of solutions to request from HJCD-IK.
        dedup_threshold: Joint-space L2 distance for deduplication.
        n_clusters:   Number of clusters for post-processing.

    Returns:
        IKResult with .solutions (list of np.ndarray) and .metadata.

    Raises:
        RuntimeError: If HJCD-IK returns zero solutions.
    """
    robot_config = robot_config or {}
    target_dict = _parse_target_pose(target_pose)
    flat_target = _flat_target(target_dict)

    has_obstacles = bool(
        env_config and env_config.get("obstacles")
    )
    collision_free = has_obstacles

    # Always build problems_json so HJCD-IK uses the correct EE frame
    # (panda_hand, not its internal default which may target a different frame).
    # When there are no obstacles the problem still carries robot/EE config.
    problems_json = _build_problems_json(target_dict, robot_config, env_config)

    t0 = time.perf_counter()
    raw = hjcdik.generate_solutions(
        flat_target,
        batch_size=batch_size,
        num_solutions=num_solutions,
        collision_free=collision_free,
        problems_json_text=problems_json,
        problem_set_name=PROBLEM_SET_NAME,
        problem_idx=0,
    )
    solve_time = time.perf_counter() - t0

    count = int(raw.get("count", 0))
    if count == 0:
        raise RuntimeError(
            f"HJCD-IK returned 0 solutions for target {flat_target}. "
            "Consider increasing batch_size or relaxing collision constraints."
        )

    configs = raw["joint_config"][:count]        # (count, n_joints)
    pos_err = raw["pos_errors"][:count]
    ori_err = raw["ori_errors"][:count]

    # Deduplicate
    configs_u, pos_err_u, ori_err_u = _deduplicate(
        configs, pos_err, ori_err, threshold=dedup_threshold
    )

    # Cluster
    labels = _cluster(configs_u, n_clusters=n_clusters)

    solutions = [configs_u[i] for i in range(len(configs_u))]

    metadata = IKMetadata(
        num_raw=count,
        num_unique=len(configs_u),
        num_final=len(solutions),
        pos_errors=pos_err_u,
        ori_errors=ori_err_u,
        cluster_labels=labels,
        solve_time_s=solve_time,
        collision_free=collision_free,
    )

    return IKResult(solutions=solutions, metadata=metadata)
