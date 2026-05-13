"""Translates a scored escape candidate into an executable action.

Never generates new motion — only adapts pre-validated rollout prefixes
or maps to an existing EscapePolicy mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.solver.escape.rollout_scorer import ScoredEscapeCandidate
from src.solver.ds.escape_policy import EscapePolicy


@dataclass
class AdaptedEscapeAction:
    mode: str                            # "prefix" | "escape_policy"
    prefix_qs: Optional[List[np.ndarray]] = None
    escape_policy_mode: Optional[str] = None    # e.g. "ESCAPE_CLEARANCE"
    escape_policy_params: Optional[dict] = None


# Mapping from source tag to preferred EscapePolicy fallback mode name.
# Uppercase strings match the name-based lookup in trial_runner.py:
#   EscapeMode[action.escape_policy_mode]  (not value-based EscapeMode(...))
_TAG_TO_MODE = {
    "clearance_grad":      "ESCAPE_CLEARANCE",
    "tangent":             "ESCAPE_CLEARANCE",
    "ik_family":           "BRIDGE_TARGET",
    "nullspace":           "ESCAPE_CLEARANCE",
    "random":              "ESCAPE_CLEARANCE",
    "joint_basis":         "BACKTRACK",
    "negative_curvature":  "ESCAPE_CLEARANCE",
}


def adapt(
    scored: ScoredEscapeCandidate,
    escape_policy: EscapePolicy,
    config,
) -> AdaptedEscapeAction:
    """Translate a ScoredEscapeCandidate into an executable action.

    Priority:
    1. If valid prefix (pre-checked by RolloutScorer), use it directly.
    2. Otherwise, map source_tag to the most natural EscapePolicy mode.
    """
    tag = scored.candidate.source_tag

    # Use pre-validated prefix if available
    if scored.valid and scored.execute_prefix_qs:
        return AdaptedEscapeAction(
            mode="prefix",
            prefix_qs=list(scored.execute_prefix_qs),  # copy reference, not new motion
        )

    # Fall back to EscapePolicy mode
    mode_name = _TAG_TO_MODE.get(tag, "ESCAPE_CLEARANCE")
    params: dict = {}

    if mode_name == "BRIDGE_TARGET" and tag == "ik_family":
        params["family_label"] = scored.candidate.metadata.get("family_label", "")
        params["preferred_escape_dir"] = scored.candidate.direction.copy()

    if mode_name == "ESCAPE_CLEARANCE":
        params["preferred_escape_dir"] = scored.candidate.direction.copy()

    return AdaptedEscapeAction(
        mode="escape_policy",
        escape_policy_mode=mode_name,
        escape_policy_params=params,
    )
