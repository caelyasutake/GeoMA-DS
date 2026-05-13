"""
IK coverage expansion — fills missing homotopy classes in the attractor set.

Core framing:
    global homotopy diversity = task-equivalent IK attractor coverage
    local motion               = differential-geometric DS shaping

When HJCD-IK returns solutions that are all in the same homotopy class (e.g.,
all left-side for a cross barrier), this module generates additional solutions
in the missing classes using biased-seed iterative differential IK.

Two classification systems are supported:
  - Elbow family: elbow_fwd / elbow_back / elbow_center (universal)
  - Gate window:  upper-left / upper-right / lower-left / lower-right
                  (cross-barrier-specific, requires barrier geometry)

No BiRRT, no MuJoCo, no HJCD-IK dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.solver.planner.collision import _panda_link_positions, _panda_hand_transform

# Panda joint limits (duplicated from filter.py to avoid MuJoCo import)
_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8975, -0.0175, -2.8973])
_Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8975,  3.7525,  2.8973])

# panda_grasptarget offset in hand frame (0.105m along hand Z)
_GT_OFFSET = np.array([0.0, 0.0, 0.105])

# Elbow classification threshold (matches factory._CENTRE_THRESH)
_CENTRE_THRESH = 0.04
_ELBOW_IDX     = 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoverageConfig:
    """Parameters for the IK coverage expansion sampler."""
    # Differential IK — targets panda_grasptarget (the correct HJCD-IK EE frame)
    ik_max_iter:      int   = 200     # iterations per seed
    ik_alpha:         float = 0.30    # damped Newton step size
    ik_tol:           float = 5e-3    # grasptarget convergence threshold (m)
    ik_min_clearance: float = 0.005   # minimum clearance for a valid goal (m)

    # Final acceptance: grasptarget must be this close to target after convergence.
    # Tighter than ik_tol to reject near-misses from early stopping.
    grasptarget_accept_tol: float = 0.010   # metres

    # Seed budget
    n_seeds_per_class:   int = 120    # seeds to try per missing class
    max_goals_per_class: int = 2      # accept at most this many per class
    seed:                int = 42     # RNG seed for reproducibility

    # Deduplication (joint space)
    dedup_threshold: float = 0.10     # min distance from any existing goal (rad)

    # Target classes
    # Families to try for (empty = skip family expansion)
    target_families: List[str] = field(default_factory=lambda: [
        "elbow_fwd", "elbow_back", "elbow_center"
    ])
    # Windows to try for (empty = skip window expansion)
    # Set by build_geo_multi_attractor_ds when a cross-style barrier is detected.
    target_windows: List[str] = field(default_factory=list)

    verbose: bool = True


# ---------------------------------------------------------------------------
# Elbow family classification
# ---------------------------------------------------------------------------

def classify_elbow_family(q_goal: np.ndarray) -> str:
    """Classify elbow family of a goal config using link3 Y position."""
    elbow_y = float(_panda_link_positions(q_goal)[_ELBOW_IDX][1])
    if elbow_y > _CENTRE_THRESH:
        return "elbow_fwd"
    elif elbow_y < -_CENTRE_THRESH:
        return "elbow_back"
    return "elbow_center"


# ---------------------------------------------------------------------------
# Gate-window classification (cross barrier)
# ---------------------------------------------------------------------------

def make_window_label_fn(
    x_post:      float = 0.48,
    z_mid:       float = 0.45,
    x_post_half: float = 0.02,
    z_bar_half:  float = 0.02,
    n_interp:    int   = 30,
) -> Callable[[np.ndarray, np.ndarray], str]:
    """
    Return a window-label function for a cross-style barrier.

    The label is derived by interpolating q_start → q_goal in joint space,
    then finding the first arm link (indices 2-6) that crosses y = 0 (the
    barrier plane).  That link's (X, Z) position determines the window:

        X < x_post - x_post_half  →  left
        X > x_post + x_post_half  →  right
        Z > z_mid  + z_bar_half   →  upper
        Z < z_mid  - z_bar_half   →  lower

    Args:
        x_post:      barrier X position (center of center_post)
        z_mid:       barrier Z midpoint
        x_post_half: half-thickness of center_post in X
        z_bar_half:  half-thickness of horiz_bar in Z
        n_interp:    interpolation steps for crossing search
    """
    x_lo = x_post - x_post_half
    x_hi = x_post + x_post_half
    z_lo = z_mid  - z_bar_half
    z_hi = z_mid  + z_bar_half

    def label_fn(q_goal: np.ndarray, q_start: np.ndarray) -> str:
        qs = [q_start + (t / (n_interp - 1)) * (q_goal - q_start)
              for t in range(n_interp)]
        link_positions = [_panda_link_positions(q) for q in qs]

        for link_idx in range(2, 7):
            for t in range(1, n_interp):
                y_prev = float(link_positions[t - 1][link_idx][1])
                y_curr = float(link_positions[t][link_idx][1])
                if y_prev < 0.0 <= y_curr:
                    frac   = -y_prev / (y_curr - y_prev + 1e-12)
                    lp_a   = link_positions[t - 1][link_idx]
                    lp_b   = link_positions[t][link_idx]
                    x_cross = float(lp_a[0]) * (1 - frac) + float(lp_b[0]) * frac
                    z_cross = float(lp_a[2]) * (1 - frac) + float(lp_b[2]) * frac

                    if x_cross < x_lo:
                        lr = "left"
                    elif x_cross > x_hi:
                        lr = "right"
                    else:
                        return "thru-post"

                    if z_cross > z_hi:
                        ud = "upper"
                    elif z_cross < z_lo:
                        ud = "lower"
                    else:
                        return f"thru-bar-{lr}"

                    return f"{ud}-{lr}"
        return "unknown"

    return label_fn


# ---------------------------------------------------------------------------
# Grasptarget FK helper and Jacobian
# ---------------------------------------------------------------------------

def _grasptarget_pos(q: np.ndarray) -> np.ndarray:
    """panda_grasptarget world position = panda_hand + R_hand @ [0,0,0.105]."""
    hp, R = _panda_hand_transform(q)
    return np.array(hp, dtype=float) + R @ _GT_OFFSET


def _grasptarget_jacobian_fd(q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    3x7 finite-difference Jacobian of panda_grasptarget w.r.t. q.

    Uses central differences. Accounts for both hand translation and
    hand-frame rotation (which affects where grasptarget lands).
    """
    J = np.zeros((3, 7))
    for j in range(7):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        J[:, j] = (_grasptarget_pos(qp) - _grasptarget_pos(qm)) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------------
# Differential IK targeting panda_grasptarget
# ---------------------------------------------------------------------------

def _diff_ik_grasptarget(
    q_seed:                np.ndarray,
    target_grasptarget_pos: np.ndarray,
    clearance_fn:          Callable[[np.ndarray], float],
    config:                CoverageConfig,
) -> tuple[Optional[np.ndarray], str]:
    """
    Iterative differential IK from q_seed targeting panda_grasptarget.

    HJCD-IK solved for panda_grasptarget (0.105m past panda_hand along hand Z).
    Differential IK must target the same frame so expanded attractors are
    valid task solutions.

    Returns:
        (q, reason) where reason in:
          "accepted"           -- converged and grasptarget within accept_tol
          "grasptarget_miss"   -- converged but grasptarget error > accept_tol
          "collision"          -- clearance below min_clearance
          "joint_limit"        -- clipped outside joint limits
          "diverged"           -- did not converge within max_iter
    """
    q   = q_seed.copy()
    eps = 1e-4
    for _ in range(config.ik_max_iter):
        gt_pos = _grasptarget_pos(q)
        err    = target_grasptarget_pos - gt_pos
        if np.linalg.norm(err) < config.ik_tol:
            q = np.clip(q, _Q_MIN, _Q_MAX)
            if clearance_fn(q) < config.ik_min_clearance:
                return None, "collision"
            final_err = float(np.linalg.norm(target_grasptarget_pos - _grasptarget_pos(q)))
            if final_err > config.grasptarget_accept_tol:
                return None, "grasptarget_miss"
            return q, "accepted"
        J   = _grasptarget_jacobian_fd(q, eps=eps)
        dq  = np.linalg.pinv(J) @ err * config.ik_alpha
        q   = np.clip(q + dq, _Q_MIN, _Q_MAX)
    return None, "diverged"


def _is_sufficiently_different(
    q: np.ndarray,
    existing: List[np.ndarray],
    threshold: float,
) -> bool:
    """True if q is at least threshold rad from every config in existing."""
    for q_ex in existing:
        if float(np.linalg.norm(q - q_ex)) < threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Biased seed generation
# ---------------------------------------------------------------------------

def _seed_for_family(family: str, rng: np.random.Generator) -> np.ndarray:
    """
    Random seed biased toward the target elbow family.

    Elbow family is controlled primarily by q1 (shoulder elevation):
      - elbow_fwd   : q1 positive  → elbow on the forward/positive-Y side
      - elbow_back  : q1 negative  → elbow on the back/negative-Y side
      - elbow_center: q1 near zero
    """
    q = rng.uniform(_Q_MIN, _Q_MAX)
    if family == "elbow_fwd":
        q[1] = rng.uniform(0.3, 1.7)
    elif family == "elbow_back":
        q[1] = rng.uniform(-1.7, -0.3)
    else:  # elbow_center
        q[1] = rng.uniform(-0.25, 0.25)
    return q


def _seed_for_window(window: str, rng: np.random.Generator) -> np.ndarray:
    """
    Random seed biased toward the target gate-window class.

    Empirical ranges for the frontal_cross_barrier_easy goal
    at hand position [0.48, 0.24, 0.45]:

      upper-right : q0 ∈ [0.5, 2.3],  q1 ∈ [-0.5, 0.8]
      lower-right : q0 ∈ [1.5, 2.9],  q1 ∈ [-1.8, -0.3]
      upper-left  : q0 ∈ [1.2, 2.0],  q1 ∈ [0.8, 1.8]
      lower-left  : q0 ∈ [-2.0, -1.0], q1 ∈ [-1.8, -0.5]
    """
    q = rng.uniform(_Q_MIN, _Q_MAX)

    if window == "upper-right":
        q[0] = rng.uniform(0.5, 2.3)
        q[1] = rng.uniform(-0.5, 0.8)
    elif window == "lower-right":
        q[0] = rng.uniform(1.5, 2.9)
        q[1] = rng.uniform(-1.8, -0.3)
    elif window == "upper-left":
        q[0] = rng.uniform(1.2, 2.0)
        q[1] = rng.uniform(0.8, 1.8)
    elif window == "lower-left":
        q[0] = rng.uniform(-2.0, -1.0)
        q[1] = rng.uniform(-1.8, -0.5)
    # else: fully random (for unknown windows)

    return q


# ---------------------------------------------------------------------------
# Coverage expansion
# ---------------------------------------------------------------------------

def expand_ik_coverage(
    existing_goals:         List[np.ndarray],
    target_grasptarget_pos: np.ndarray,
    clearance_fn:           Callable[[np.ndarray], float],
    q_start:                np.ndarray,
    config:                 CoverageConfig,
    window_label_fn:        Optional[Callable[[np.ndarray, np.ndarray], str]] = None,
) -> Tuple[List[np.ndarray], dict]:
    """
    Generate IK solutions for homotopy classes missing from existing_goals.

    Checks:
    1. Elbow family coverage (elbow_fwd, elbow_back, elbow_center).
    2. Gate-window coverage (upper-right, lower-right, etc.) if window_label_fn
       is provided and config.target_windows is non-empty.

    Args:
        existing_goals:          Current IK solutions (list of (7,) arrays).
        target_grasptarget_pos:  World position that panda_grasptarget must reach.
                                 Must match the HJCD-IK target frame (= red sphere).
        clearance_fn:            (q) -> float signed clearance function.
        q_start:                 Start configuration (for window label interpolation).
        config:                  CoverageConfig with budgets and class targets.
        window_label_fn:         Optional (q_goal, q_start) -> str for window labels.

    Returns:
        (new_goals, report) where:
          new_goals — additional valid configs filling missing classes
          report    — dict with coverage diagnostics
    """
    rng = np.random.default_rng(config.seed)

    # --- Classify existing ---
    existing_families = [classify_elbow_family(q) for q in existing_goals]
    existing_windows  = (
        [window_label_fn(q, q_start) for q in existing_goals]
        if window_label_fn is not None else []
    )

    covered_families = set(existing_families)
    covered_windows  = set(existing_windows)

    missing_families = [f for f in config.target_families if f not in covered_families]
    missing_windows  = (
        [w for w in config.target_windows if w not in covered_windows]
        if config.target_windows else []
    )

    report: dict = {
        "n_existing":        len(existing_goals),
        "covered_families":  sorted(covered_families),
        "covered_windows":   sorted(covered_windows),
        "missing_families":  sorted(missing_families),
        "missing_windows":   sorted(missing_windows),
        "family_histogram":  {f: existing_families.count(f) for f in sorted(covered_families)},
        "window_histogram":  {w: existing_windows.count(w)  for w in sorted(covered_windows)},
        "expansion_log":     [],
        "n_new_goals":       0,
    }

    if not missing_families and not missing_windows:
        report["expansion_log"].append("all classes already covered — no expansion needed")
        return [], report

    new_goals: List[np.ndarray] = []
    all_goals = list(existing_goals)   # pool for dedup checks

    def _run_expansion(
        missing_classes: List[str],
        seed_fn: Callable[[str, np.random.Generator], np.ndarray],
        label_fn: Callable[[np.ndarray], str],
        kind: str,
    ) -> None:
        for cls in missing_classes:
            found    = 0
            attempts = 0
            goal_details: List[dict] = []
            for _ in range(config.n_seeds_per_class):
                q_seed           = seed_fn(cls, rng)
                q_result, reason = _diff_ik_grasptarget(
                    q_seed, target_grasptarget_pos, clearance_fn, config
                )
                attempts += 1

                if q_result is not None:
                    actual = label_fn(q_result)
                    gt_err = float(np.linalg.norm(
                        target_grasptarget_pos - _grasptarget_pos(q_result)
                    ))
                    hp = np.array(_panda_link_positions(q_result)[-1], dtype=float)
                    hand_err = float(np.linalg.norm(
                        target_grasptarget_pos - hp
                    ))
                    if actual == cls:
                        if _is_sufficiently_different(q_result, all_goals, config.dedup_threshold):
                            new_goals.append(q_result)
                            all_goals.append(q_result)
                            found += 1
                            goal_details.append({
                                "reason":        "accepted",
                                "family":        actual,
                                "grasptarget_err": gt_err,
                                "hand_err":      hand_err,
                                "clearance":     float(clearance_fn(q_result)),
                            })
                            if found >= config.max_goals_per_class:
                                break
                        else:
                            goal_details.append({
                                "reason":        "dedup_reject",
                                "family":        actual,
                                "grasptarget_err": gt_err,
                                "hand_err":      hand_err,
                                "clearance":     float(clearance_fn(q_result)),
                            })
                    else:
                        goal_details.append({
                            "reason":        f"wrong_class({actual})",
                            "family":        actual,
                            "grasptarget_err": gt_err,
                            "hand_err":      hand_err,
                            "clearance":     float(clearance_fn(q_result)),
                        })

            entry = {
                "kind":         kind,
                "class":        cls,
                "attempts":     attempts,
                "found":        found,
                "goal_details": goal_details,
            }
            report["expansion_log"].append(entry)

    # --- Expand by elbow family ---
    if missing_families:
        _run_expansion(
            missing_families,
            seed_fn  = _seed_for_family,
            label_fn = classify_elbow_family,
            kind     = "family",
        )

    # --- Expand by window ---
    if missing_windows and window_label_fn is not None:
        _run_expansion(
            missing_windows,
            seed_fn  = _seed_for_window,
            label_fn = lambda q: window_label_fn(q, q_start),
            kind     = "window",
        )

    report["n_new_goals"] = len(new_goals)

    # Updated coverage after expansion
    all_families = existing_families + [classify_elbow_family(q) for q in new_goals]
    all_windows  = (
        existing_windows + [window_label_fn(q, q_start) for q in new_goals]
        if window_label_fn is not None else existing_windows
    )
    report["final_families"] = sorted(set(all_families))
    report["final_windows"]  = sorted(set(all_windows))

    return new_goals, report


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_coverage_report(
    existing_goals: List[np.ndarray],
    new_goals:      List[np.ndarray],
    q_start:        np.ndarray,
    report:         dict,
    window_label_fn: Optional[Callable[[np.ndarray, np.ndarray], str]] = None,
) -> None:
    """Print a formatted coverage report to stdout."""
    sep  = "=" * 70
    dash = "-" * 70

    print()
    print(sep)
    print("IK Coverage Report")
    print(sep)
    print(f"  Existing goals      : {report['n_existing']}")
    print(f"  New goals generated : {report['n_new_goals']}")
    print(f"  Total goals         : {report['n_existing'] + report['n_new_goals']}")

    print()
    print("  Elbow family histogram (existing):")
    for fam, cnt in sorted(report.get("family_histogram", {}).items()):
        bar = "=" * cnt
        print(f"    {fam:<16} {cnt:>3}  {bar}")
    print(f"  Missing families    : {report.get('missing_families', [])}")

    if report.get("covered_windows") or report.get("missing_windows"):
        print()
        print("  Window histogram (existing):")
        for win, cnt in sorted(report.get("window_histogram", {}).items()):
            bar = "=" * cnt
            print(f"    {win:<16} {cnt:>3}  {bar}")
        print(f"  Missing windows     : {report.get('missing_windows', [])}")

    if report.get("expansion_log"):
        print()
        print("  Expansion log:")
        for entry in report["expansion_log"]:
            if isinstance(entry, str):
                print(f"    {entry}")
            else:
                ok = "OK" if entry["found"] > 0 else "FAILED"
                print(f"    [{ok}] {entry['kind']}/{entry['class']:<16} "
                      f"found={entry['found']}  attempts={entry['attempts']}")
                for d in entry.get("goal_details", []):
                    gt_e = d.get("grasptarget_err", float("nan"))
                    h_e  = d.get("hand_err",        float("nan"))
                    cl   = d.get("clearance",        float("nan"))
                    rsn  = d.get("reason", "?")
                    print(f"      [{rsn:<22}] gt_err={gt_e:.4f} m  "
                          f"hand_err={h_e:.4f} m  cl={cl:.4f} m")

    print()
    print(f"  Final families      : {report.get('final_families', report.get('covered_families', []))}")
    if report.get("final_windows") or report.get("covered_windows"):
        print(f"  Final windows       : {report.get('final_windows', report.get('covered_windows', []))}")

    # Per-goal table for new goals — grasptarget_err shown per-goal via expansion_log
    if new_goals:
        print()
        print("  New goals detail (accepted candidates):")
        print(f"  {'idx':>4}  {'family':<14}  {'window':<14}  {'gt_err(m)':>9}")
        print(f"  {'-'*58}")
        # Collect accepted entries from expansion_log in order
        accepted_details: List[dict] = []
        for entry in report.get("expansion_log", []):
            if isinstance(entry, dict):
                for d in entry.get("goal_details", []):
                    if d.get("reason") == "accepted":
                        accepted_details.append(d)
        for i, q in enumerate(new_goals):
            fam = classify_elbow_family(q)
            win = window_label_fn(q, q_start) if window_label_fn else "n/a"
            gt_e = accepted_details[i]["grasptarget_err"] if i < len(accepted_details) else float("nan")
            print(f"  {len(existing_goals)+i:>4}  {fam:<14}  {win:<14}  {gt_e:>9.4f}")
    print(sep)
    print()
