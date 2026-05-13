"""
Shared cache helpers for MB diagnostic scripts.

All three scripts (eval_mb_failure_breakdown, debug_mb_bookshelf_scene,
audit_collision_spheres) use the same output directory layout and metadata
fingerprinting logic implemented here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_CLEARANCE_FN_VERSION = "1"


def cache_dir(save_dir: str | Path, set_name: str, prob_idx: int) -> Path:
    """Return the cache directory for one problem."""
    return Path(save_dir) / f"{set_name}_{prob_idx:04d}"


def _sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def obstacle_hash(raw_obs: dict) -> str:
    """Stable hash of the raw obstacle dict (sorted keys)."""
    return _sha256_short(json.dumps(raw_obs, sort_keys=True))


def problem_spec_hash(spec) -> str:
    """Stable hash of the ScenarioSpec geometry (start config + target + obstacles)."""
    d = {
        "q_start": spec.q_start.tolist(),
        "target": spec.target_pose,
        "obstacles": [
            {"name": o.name, "type": o.type,
             "pos": list(o.position), "size": list(o.size),
             "orient": list(o.orientation_wxyz)}
            for o in spec.obstacles
        ],
    }
    return _sha256_short(json.dumps(d, sort_keys=True))


def robot_model_hash() -> str:
    """Stable hash of the robot collision model (_LINK_RADII + _MJCF_BODIES)."""
    from src.solver.planner.collision import _LINK_RADII, _MJCF_BODIES
    s = repr(sorted(_LINK_RADII.items())) + repr(_MJCF_BODIES)
    return _sha256_short(s)


def _git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def build_metadata(
    set_name: str,
    prob_idx: int,
    seed: int,
    spec,
    raw_obs: dict,
    hjcd_batch: int,
    hjcd_nsol: int,
    planner_max_iter: int,
    planner_step_size: float,
) -> Dict[str, Any]:
    """Build the full metadata dict for writing to metadata.json."""
    return {
        "set_name":              set_name,
        "prob_idx":              prob_idx,
        "seed":                  seed,
        "hjcd_batch":            hjcd_batch,
        "hjcd_nsol":             hjcd_nsol,
        "obstacle_hash":         obstacle_hash(raw_obs),
        "problem_spec_hash":     problem_spec_hash(spec),
        "robot_model_hash":      robot_model_hash(),
        "mb_target_frame":       "panda_grasptarget",
        "target_offset_m":       [0.0, 0.0, 0.105],
        "collision_backend":     "sphere_swept_obb",
        "clearance_fn_version":  _CLEARANCE_FN_VERSION,
        "ik_collision_free":     True,
        "planner_max_iter":      planner_max_iter,
        "planner_step_size":     planner_step_size,
        "code_git_commit":       _git_commit(),
    }


def ik_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the IK-validity fingerprint from a metadata dict."""
    keys = [
        "set_name", "prob_idx", "seed", "hjcd_batch", "hjcd_nsol",
        "obstacle_hash", "problem_spec_hash", "robot_model_hash",
        "mb_target_frame", "target_offset_m", "clearance_fn_version",
    ]
    return {k: meta[k] for k in keys}


def birrt_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the BiRRT-validity fingerprint (superset of IK fingerprint)."""
    fp = ik_fingerprint(meta)
    fp["planner_max_iter"] = meta["planner_max_iter"]
    fp["planner_step_size"] = meta["planner_step_size"]
    return fp


def write_metadata(cache_d: Path, fields: Dict[str, Any]) -> None:
    cache_d.mkdir(parents=True, exist_ok=True)
    with open(cache_d / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)


def read_metadata(cache_d: Path) -> Optional[Dict[str, Any]]:
    p = cache_d / "metadata.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_fingerprint(
    cache_d: Path,
    fingerprint: Dict[str, Any],
    artifact_name: str,
    force_cache: bool = False,
) -> bool:
    """
    Compare fingerprint against stored metadata.json.

    Returns True if cache is valid.
    Returns True (with warning) if stale but force_cache=True.
    Prints error and calls sys.exit(1) if stale and force_cache=False.
    Returns False (no exit) if metadata.json does not exist.
    """
    meta = read_metadata(cache_d)
    if meta is None:
        return False

    for k, v in fingerprint.items():
        stored = meta.get(k)
        if stored != v:
            msg = (
                f"[STALE CACHE] {artifact_name}: mismatch on '{k}': "
                f"expected {v!r}, got {stored!r}. "
                f"Pass --force-cache to override."
            )
            if force_cache:
                print(msg)
                return True
            print(msg)
            sys.exit(1)
    return True
