"""
Phase 2: IK Filtering (Updated — energy-aware scoring)

Constructs a safe IK set from raw HJCD-IK solutions by applying:
  - collision checking (dependency-injected)
  - manipulability thresholding (via MuJoCo Jacobian)
  - joint-limit margin
  - energy-need estimation vs. current tank level
    * estimate_energy_need   — approach / path cost
    * estimate_contact_energy — energy expected during contact phase
    * estimate_approach_energy — alias with direction-weighting

Energy-aware scoring sorts surviving solutions by a composite score so the
planner's first goal is the most energy-efficient one:

    J_i = w_energy  * estimate_approach_energy(q_i, q_current)
        + w_contact * estimate_contact_energy(q_i, n_hat, F_desired)

Main interface:
    Q_safe = filter_safe_set(Q_IK, state, tank_energy, config)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Panda joint limits (from panda.urdf)
# ---------------------------------------------------------------------------
PANDA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8975, -0.0175, -2.8973])
PANDA_Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8975,  3.7525,  2.8973])

# Path to Panda URDF bundled with HJCD-IK
_HJCD_ROOT = Path(__file__).resolve().parents[3] / "external" / "HJCD-IK"
_PANDA_URDF = _HJCD_ROOT / "include" / "test_urdf" / "panda.urdf"

# Body name used for Jacobian computation (last revolute body; fixed-joint
# descendants like panda_hand are merged into it by MuJoCo)
_EE_BODY_NAME = "panda_link7"


# ---------------------------------------------------------------------------
# Kinematics helper — MuJoCo-based Jacobian
# ---------------------------------------------------------------------------
def _load_kinematics_model(urdf_path: Path = _PANDA_URDF):
    """
    Load the Panda URDF into MuJoCo, stripping mesh references so it works
    without the binary STL files.  Returns (MjModel, ee_body_id).
    """
    text = urdf_path.read_text(encoding="utf-8")
    text = re.sub(r'<visual>.*?</visual>', '', text, flags=re.DOTALL)
    text = re.sub(r'<collision>.*?</collision>', '', text, flags=re.DOTALL)
    model = mujoco.MjModel.from_xml_string(text)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY_NAME)
    if ee_id < 0:
        raise RuntimeError(f"End-effector body '{_EE_BODY_NAME}' not found in model.")
    return model, ee_id


def make_mujoco_jacobian_fn(
    urdf_path: Path = _PANDA_URDF,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a Jacobian function (q) -> J (6 x nv) backed by MuJoCo.
    The model is loaded once and reused across calls.

    Returns:
        jacobian_fn: callable that accepts a joint-config array and returns
                     the 6×n end-effector Jacobian [jacp; jacr].
    """
    mj_model, ee_id = _load_kinematics_model(urdf_path)
    mj_data = mujoco.MjData(mj_model)
    nv = mj_model.nv

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    def jacobian_fn(q: np.ndarray) -> np.ndarray:
        mj_data.qpos[:nv] = q[:nv]
        mujoco.mj_kinematics(mj_model, mj_data)
        mujoco.mj_comPos(mj_model, mj_data)
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, ee_id)
        return np.vstack([jacp, jacr])

    return jacobian_fn


# ---------------------------------------------------------------------------
# Individual filter criteria
# ---------------------------------------------------------------------------
def is_collision_free(
    q: np.ndarray,
    collision_checker: Optional[Callable[[np.ndarray], bool]],
) -> bool:
    """
    Return True if q is collision-free.

    Args:
        q: Joint configuration, shape (n_joints,).
        collision_checker: Callable (q) -> bool. If None, the check is
            skipped and True is returned (conservative: assume valid).
    """
    if collision_checker is None:
        return True
    return bool(collision_checker(q))


def manipulability(
    q: np.ndarray,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]],
) -> float:
    """
    Compute Yoshikawa manipulability index: w = sqrt(det(J J^T)).

    Args:
        q: Joint configuration.
        jacobian_fn: Callable (q) -> J (6 x n). If None, returns inf
            (criterion is skipped).

    Returns:
        Manipulability scalar >= 0.  Returns inf when jacobian_fn is None.
    """
    if jacobian_fn is None:
        return float("inf")
    J = jacobian_fn(q)
    val = np.linalg.det(J @ J.T)
    return float(np.sqrt(max(0.0, val)))


def joint_limit_margin(
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> float:
    """
    Minimum distance (rad) between q and any joint limit.

    Returns a negative value if q violates a limit.
    """
    lower = q - q_min
    upper = q_max - q
    return float(np.minimum(lower, upper).min())


def estimate_energy_need(
    q: np.ndarray,
    q_current: np.ndarray,
    scale: float = 1.0,
) -> float:
    """
    Approximate energy required to reach q from q_current.

    Uses a scaled kinetic-energy surrogate:
        E = 0.5 * scale * ||q - q_current||^2

    Args:
        q:         Target joint configuration.
        q_current: Current joint configuration.
        scale:     Scalar weight (tune to match physical units).

    Returns:
        Estimated energy (non-negative scalar).
    """
    return float(0.5 * scale * np.sum((np.asarray(q) - np.asarray(q_current)) ** 2))


def estimate_approach_energy(
    q: np.ndarray,
    q_current: np.ndarray,
    scale: float = 1.0,
    direction_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate the energy needed to move from q_current to goal q,
    optionally weighting individual joints by their expected contribution.

    Formula:
        E_approach = 0.5 * scale * sum_i( w_i * (q_i - q_current_i)^2 )

    When direction_weights is None this reduces to estimate_energy_need.

    Args:
        q:                 Target joint configuration.
        q_current:         Current joint configuration.
        scale:             Global scaling factor.
        direction_weights: Per-joint weights, shape (n_joints,).  None → all 1.

    Returns:
        Estimated approach energy (non-negative scalar).
    """
    q       = np.asarray(q,       dtype=float)
    q_curr  = np.asarray(q_current, dtype=float)
    delta   = q - q_curr
    if direction_weights is not None:
        w = np.asarray(direction_weights, dtype=float)
        return float(0.5 * scale * float(w @ (delta ** 2)))
    return float(0.5 * scale * float(delta @ delta))


def estimate_contact_energy(
    q: np.ndarray,
    n_hat: Optional[np.ndarray] = None,
    F_desired: float = 0.0,
    k_f: float = 1.0,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """
    Estimate the energy that will be injected during the contact phase when
    the robot reaches goal configuration q.

    Simple surrogate:
        E_contact = 0.5 * ||τ_n_nom||^2
    where τ_n_nom = J_lin^T (k_f * F_desired * n_hat) is the joint-space
    force needed to track the desired contact force.

    When jacobian_fn is None (or n_hat is None / F_desired == 0) returns 0,
    which means the contact criterion is inactive.

    Args:
        q:           Target joint configuration.
        n_hat:       Contact surface outward normal (3,). None → returns 0.
        F_desired:   Desired normal force magnitude. 0 → returns 0.
        k_f:         Contact force gain (same as ContactForceConfig.k_f).
        jacobian_fn: Callable (q) -> J (6×n).  None → returns 0.

    Returns:
        Estimated contact energy (non-negative scalar).
    """
    if jacobian_fn is None or n_hat is None or F_desired == 0.0:
        return 0.0

    n_hat = np.asarray(n_hat, dtype=float)
    n_hat_u = n_hat / (np.linalg.norm(n_hat) + 1e-12)

    J = np.asarray(jacobian_fn(q), dtype=float)   # (6, n)
    J_lin = J[:3, :]                               # (3, n)
    f_n_task = k_f * float(F_desired) * n_hat_u   # (3,)
    tau_n_nom = J_lin.T @ f_n_task                  # (n,)

    return float(0.5 * float(tau_n_nom @ tau_n_nom))


# ---------------------------------------------------------------------------
# Filter configuration and state
# ---------------------------------------------------------------------------
@dataclass
class FilterConfig:
    """Parameters controlling which IK solutions survive filtering."""

    # Joint limits — default to Panda limits
    q_min: np.ndarray = field(default_factory=lambda: PANDA_Q_MIN.copy())
    q_max: np.ndarray = field(default_factory=lambda: PANDA_Q_MAX.copy())

    # Manipulability: solutions below this threshold are near-singular
    manipulability_threshold: float = 0.01

    # Minimum distance (rad) to any joint limit
    joint_limit_margin_threshold: float = 0.05

    # Scale applied to energy estimate
    energy_scale: float = 1.0

    # Energy-aware scoring weights (0 = disabled)
    # Surviving solutions are sorted ascending by the composite score:
    #   J_i = w_energy * E_approach(q_i) + w_contact * E_contact(q_i)
    w_energy:  float = 1.0   # weight for approach energy
    w_contact: float = 0.0   # weight for contact energy (off by default)

    # Contact parameters used by estimate_contact_energy (needed when w_contact > 0)
    contact_n_hat:     Optional[np.ndarray] = None   # contact surface normal (3,)
    contact_F_desired: float = 0.0                   # desired contact force magnitude
    contact_k_f:       float = 1.0                   # contact force gain

    # Dependency-injected functions (None = skip that criterion)
    collision_checker: Optional[Callable[[np.ndarray], bool]] = None
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class RobotState:
    """Snapshot of robot state needed for filtering."""
    q: np.ndarray   # current joint configuration, shape (n_joints,)


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    solutions: List[np.ndarray]       # surviving joint configs
    rejected_collision: int           # count rejected by collision check
    rejected_manipulability: int      # count rejected by manipulability
    rejected_joint_limits: int        # count rejected by joint-limit margin
    rejected_energy: int              # count rejected by energy budget

    @property
    def num_safe(self) -> int:
        return len(self.solutions)


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def filter_safe_set(
    Q_IK: List[np.ndarray],
    state: RobotState,
    tank_energy: float,
    config: Optional[FilterConfig] = None,
) -> FilterResult:
    """
    Filter raw IK solutions to construct a safe IK set Q_safe.

    Criteria applied in order:
        1. is_collision_free   — remove configs that collide
        2. manipulability      — remove near-singular configs
        3. joint_limit_margin  — remove configs close to joint limits
        4. estimate_energy_need — remove configs too costly given tank_energy

    Args:
        Q_IK:        List of joint configs from HJCD-IK wrapper.
        state:       Current robot state (at minimum: state.q for energy est.).
        tank_energy: Available energy in the tank (scalar).  Solutions whose
                     estimated energy cost exceeds this are rejected.
        config:      FilterConfig; uses safe defaults when None.

    Returns:
        FilterResult with the surviving solutions and per-criterion reject counts.

    Raises:
        ValueError: If Q_IK is empty.
    """
    if len(Q_IK) == 0:
        raise ValueError("Q_IK is empty — no IK solutions to filter.")

    if config is None:
        config = FilterConfig()

    n_col = n_man = n_lim = n_en = 0
    safe: List[np.ndarray] = []

    for q in Q_IK:
        q = np.asarray(q, dtype=float)

        # 1. Collision
        if not is_collision_free(q, config.collision_checker):
            n_col += 1
            continue

        # 2. Manipulability
        m = manipulability(q, config.jacobian_fn)
        if m < config.manipulability_threshold:
            n_man += 1
            continue

        # 3. Joint-limit margin
        margin = joint_limit_margin(q, config.q_min, config.q_max)
        if margin < config.joint_limit_margin_threshold:
            n_lim += 1
            continue

        # 4. Energy budget
        e = estimate_energy_need(q, state.q, config.energy_scale)
        if e > tank_energy:
            n_en += 1
            continue

        safe.append(q)

    # 5. Energy-aware scoring: sort surviving solutions by composite score
    #    J_i = w_energy * E_approach(q_i) + w_contact * E_contact(q_i)
    #    Lower score = more energy-efficient (preferred).
    if safe and (config.w_energy != 0.0 or config.w_contact != 0.0):
        def _score(q: np.ndarray) -> float:
            s = 0.0
            if config.w_energy != 0.0:
                s += config.w_energy * estimate_approach_energy(
                    q, state.q, scale=config.energy_scale
                )
            if config.w_contact != 0.0:
                s += config.w_contact * estimate_contact_energy(
                    q,
                    n_hat=config.contact_n_hat,
                    F_desired=config.contact_F_desired,
                    k_f=config.contact_k_f,
                    jacobian_fn=config.jacobian_fn,
                )
            return s

        safe.sort(key=_score)

    return FilterResult(
        solutions=safe,
        rejected_collision=n_col,
        rejected_manipulability=n_man,
        rejected_joint_limits=n_lim,
        rejected_energy=n_en,
    )
