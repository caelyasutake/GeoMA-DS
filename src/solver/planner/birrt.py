"""
Phase 3: Multi-Goal Bi-directional RRT Planner

Tree S (start): grows from q_start.
Tree G (goals): initialised with ALL IK goal configurations.
Planning terminates when the two trees are connected (any goal reached).

Instrumentation logged per call:
  - nodes_explored   (total nodes across both trees)
  - time_to_solution (wall clock, seconds)
  - collision_checks (number of is_collision_free calls)
  - iterations       (RRT main-loop iterations)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.solver.ik.filter import PANDA_Q_MAX, PANDA_Q_MIN

# ---------------------------------------------------------------------------
# Internal tree structure
# ---------------------------------------------------------------------------
class _Tree:
    """Lightweight RRT tree stored as parallel arrays for speed."""

    def __init__(self) -> None:
        self._qs: List[np.ndarray] = []
        self._parents: List[int] = []

    # ------------------------------------------------------------------
    def add(self, q: np.ndarray, parent: int) -> int:
        idx = len(self._qs)
        self._qs.append(q.copy())
        self._parents.append(parent)
        return idx

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._qs)

    # ------------------------------------------------------------------
    def nearest(self, q: np.ndarray) -> Tuple[int, float]:
        """Return (index, distance) of the node closest to q (L2 in joint space)."""
        if not self._qs:
            raise RuntimeError("Tree is empty.")
        qs = np.stack(self._qs)
        dists = np.linalg.norm(qs - q, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    # ------------------------------------------------------------------
    def path_to_root(self, idx: int) -> List[np.ndarray]:
        """Return the path [root, ..., node[idx]] in forward order."""
        nodes: List[np.ndarray] = []
        while idx != -1:
            nodes.append(self._qs[idx])
            idx = self._parents[idx]
        nodes.reverse()
        return nodes

    # ------------------------------------------------------------------
    def root_of(self, idx: int) -> int:
        """Return the index of the root ancestor of node idx."""
        while self._parents[idx] != -1:
            idx = self._parents[idx]
        return idx


# ---------------------------------------------------------------------------
# Planner configuration and result
# ---------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    """Tuning parameters for the bi-directional RRT."""

    max_iterations: int = 10_000
    step_size: float = 0.1          # max extension step (rad)
    goal_bias: float = 0.10         # probability of sampling a goal directly
    q_min: np.ndarray = field(default_factory=lambda: PANDA_Q_MIN.copy())
    q_max: np.ndarray = field(default_factory=lambda: PANDA_Q_MAX.copy())
    seed: int = 0

    # Clearance-adaptive step size
    min_step_size:             float = 0.02   # floor step size near obstacles (rad)
    clearance_step_scale:      float = 1.0    # weight of clearance scaling (0 = off)
    clearance_step_threshold:  float = 0.10   # clearance below which scaling begins (m)

    # Gaussian near-node sampling (narrow passage finder)
    # With probability gaussian_bias, sample near a random existing tree node
    # instead of uniformly.  Helps find narrow corridors around obstacles.
    gaussian_bias: float = 0.20   # fraction of samples that are near-node Gaussian
    gaussian_std:  float = 0.15   # std of per-joint Gaussian noise (rad)

    # Path shortcutting (post-processing after a path is found)
    shortcut_iterations:       int   = 300    # random shortcut attempts; 0 = off
    shortcut_n_check:          int   = 12     # collision-check resolution along shortcut
    shortcut_clearance_margin: float = 0.005  # min clearance required along shortcut (m)


@dataclass
class PlanResult:
    """Outcome of a plan() call."""

    success: bool
    path: Optional[List[np.ndarray]]  # joint-space path q_start → q_goal
    goal_idx: Optional[int]           # index into Q_goals that was reached
    nodes_explored: int               # |tree_S| + |tree_G| at termination
    time_to_solution: float           # wall clock (seconds)
    collision_checks: int             # total is_collision_free calls
    iterations: int                   # main-loop iterations performed


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------
def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    """Return a point at most step_size away from q_from toward q_to."""
    delta = q_to - q_from
    dist = float(np.linalg.norm(delta))
    if dist <= step_size:
        return q_to.copy()
    return q_from + (delta / dist) * step_size


def shortcut_path(
    path: List[np.ndarray],
    collision_fn: Callable[[np.ndarray], bool],
    rng: np.random.Generator,
    n_iterations: int = 300,
    n_check: int = 12,
    clearance_fn: Optional[Callable[[np.ndarray], float]] = None,
    clearance_margin: float = 0.005,
) -> List[np.ndarray]:
    """
    Post-process a joint-space path by collapsing straight-line shortcuts.

    For each random pair (i, j) with i < j, checks whether the straight-line
    segment path[i] → path[j] is collision-free (and above clearance_margin
    if clearance_fn is provided).  If so, replaces the intermediate waypoints.

    Returns a shorter path with the same endpoints.
    """
    path = list(path)
    for _ in range(n_iterations):
        n = len(path)
        if n <= 2:
            break
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        if j - i <= 1:
            continue
        a, b = path[i], path[j]
        ts = np.linspace(0.0, 1.0, n_check + 2)[1:-1]
        ok = True
        for t in ts:
            q_mid = a + t * (b - a)
            if not collision_fn(q_mid):
                ok = False
                break
            if clearance_fn is not None and clearance_fn(q_mid) < clearance_margin:
                ok = False
                break
        if ok:
            path = path[:i + 1] + path[j:]
    return path


def _adaptive_step(
    q: np.ndarray,
    config: "PlannerConfig",
    clearance_fn: Optional[Callable[[np.ndarray], float]],
) -> float:
    """
    Return a step size scaled by obstacle clearance at q.

    When clearance < clearance_step_threshold, linearly interpolates from
    step_size down to min_step_size.  Returns config.step_size when no
    clearance function is provided or clearance_step_scale == 0.
    """
    if clearance_fn is None or config.clearance_step_scale == 0.0:
        return config.step_size
    cl = clearance_fn(q)
    if cl >= config.clearance_step_threshold:
        return config.step_size
    # Linear blend: at cl=threshold → step_size; at cl=0 → min_step_size
    t = float(np.clip(cl / config.clearance_step_threshold, 0.0, 1.0))
    scaled = config.min_step_size + t * (config.step_size - config.min_step_size)
    return max(config.min_step_size, scaled * config.clearance_step_scale
               + config.step_size * (1.0 - config.clearance_step_scale))


def _extend(
    tree: _Tree,
    q_rand: np.ndarray,
    config: "PlannerConfig",
    collision_fn: Callable[[np.ndarray], bool],
    col_count: List[int],
    clearance_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Extend tree one step toward q_rand.

    Step size is reduced near obstacles when clearance_fn is provided.
    Returns (q_new, node_idx) on success, None if blocked by collision.
    """
    near_idx, _ = tree.nearest(q_rand)
    q_near = tree._qs[near_idx]
    step = _adaptive_step(q_near, config, clearance_fn)
    q_new = _steer(q_near, q_rand, step)

    col_count[0] += 1
    if not collision_fn(q_new):
        return None

    new_idx = tree.add(q_new, near_idx)
    return q_new, new_idx


def _connect(
    tree: _Tree,
    q_target: np.ndarray,
    config: "PlannerConfig",
    collision_fn: Callable[[np.ndarray], bool],
    col_count: List[int],
    clearance_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Tuple[Optional[int], bool]:
    """
    Greedily extend tree toward q_target until it is reached or blocked.

    Step size adapts to obstacle clearance at each intermediate node.
    Returns (last_node_idx, reached) where:
      - last_node_idx: index of the last node added (None if no progress at all)
      - reached: True if q_target was added to the tree
    """
    near_idx, near_dist = tree.nearest(q_target)

    # Already in tree
    if near_dist < 1e-9:
        return near_idx, True

    last_idx: Optional[int] = None
    cur_idx = near_idx

    while True:
        q_cur = tree._qs[cur_idx]
        step = _adaptive_step(q_cur, config, clearance_fn)
        q_new = _steer(q_cur, q_target, step)

        col_count[0] += 1
        if not collision_fn(q_new):
            break  # blocked

        new_idx = tree.add(q_new, cur_idx)
        last_idx = new_idx
        cur_idx = new_idx

        dist_remaining = float(np.linalg.norm(q_new - q_target))
        if dist_remaining < 1e-9:
            return last_idx, True  # reached exactly

    return last_idx, False


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def plan(
    q_start,
    Q_goals: List[np.ndarray],
    env=None,
    config: Optional[PlannerConfig] = None,
    clearance_fn: Optional[Callable[[np.ndarray], float]] = None,
    scaffold_waypoints: Optional[List[np.ndarray]] = None,
    scaffold_bias: float = 0.15,
) -> PlanResult:
    """
    Multi-goal bi-directional RRT with Gaussian sampling, adaptive step size,
    scaffold warm-starting, and post-plan path shortcutting.

    Args:
        q_start:          Start joint configuration, shape (n_joints,).
        Q_goals:          IK goal configurations (list of arrays).
        env:              Collision checker — callable (q) -> bool (True = free),
                          or object with ``is_collision_free``, or None.
        config:           PlannerConfig; safe defaults when None.
        clearance_fn:     Optional (q) -> float min obstacle clearance.
                          Enables adaptive step size near obstacles.
        scaffold_waypoints: Extra joint configs (e.g. waypoints from a previous
                          path) used as additional bias samples during planning.
                          They are NOT added as tree_g roots — goal_idx semantics
                          are preserved.  None = disabled.
        scaffold_bias:    Probability of sampling from scaffold_waypoints when
                          they are provided.  Default 0.15.

    Returns:
        PlanResult.  On success, path[0] = q_start, path[-1] ≈ a Q_goals entry.
        The path is post-processed by shortcut_path when shortcut_iterations > 0.

    Raises:
        ValueError: If Q_goals is empty.
    """
    if not Q_goals:
        raise ValueError("Q_goals is empty — no goals to plan toward.")

    if config is None:
        config = PlannerConfig()

    # Resolve collision function
    if env is None:
        collision_fn: Callable = lambda _q: True
    elif callable(env):
        collision_fn = env
    elif hasattr(env, "is_collision_free"):
        collision_fn = env.is_collision_free
    else:
        collision_fn = lambda _q: True

    rng = np.random.default_rng(config.seed)
    q_start = np.asarray(q_start, dtype=float)
    goals = [np.asarray(g, dtype=float) for g in Q_goals]
    n_goals = len(goals)

    # Initialise trees
    tree_s = _Tree()   # start tree
    tree_g = _Tree()   # goal tree — roots at indices 0 … n_goals-1
    tree_s.add(q_start, parent=-1)
    for qg in goals:
        tree_g.add(qg, parent=-1)

    col_count = [0]
    t0 = time.perf_counter()

    n_dof = len(q_start)
    scaffolds = [np.asarray(w, dtype=float) for w in scaffold_waypoints] \
                if scaffold_waypoints else []

    for iteration in range(config.max_iterations):

        # ---- Sample -------------------------------------------------------
        r = rng.random()
        cumulative = 0.0

        # Cumulative probability thresholds for sample type selection
        p_scaffold = scaffold_bias if scaffolds else 0.0
        p_goal     = p_scaffold + config.goal_bias
        p_gaussian = p_goal + config.gaussian_bias

        if r < p_scaffold:
            q_rand = scaffolds[int(rng.integers(len(scaffolds)))]
        elif r < p_goal:
            q_rand = goals[int(rng.integers(n_goals))]
        elif r < p_gaussian:
            src_tree = tree_s if iteration % 2 == 0 else tree_g
            rand_node = src_tree._qs[int(rng.integers(len(src_tree)))]
            q_rand = rand_node + rng.standard_normal(n_dof) * config.gaussian_std
            q_rand = np.clip(q_rand, config.q_min, config.q_max)
        else:
            q_rand = rng.uniform(config.q_min, config.q_max)

        # ---- Alternate which tree extends ---------------------------------
        if iteration % 2 == 0:
            # Extend tree_s, then try to connect tree_g
            ext = _extend(tree_s, q_rand, config, collision_fn, col_count, clearance_fn)
            if ext is None:
                continue
            q_new, idx_s = ext

            conn_idx, reached = _connect(
                tree_g, q_new, config, collision_fn, col_count, clearance_fn
            )
            if reached and conn_idx is not None:
                path_s = tree_s.path_to_root(idx_s)     # [q_start, …, q_new]
                path_g = tree_g.path_to_root(conn_idx)  # [q_goal,  …, q_new]
                goal_idx = tree_g.root_of(conn_idx)
                full_path = path_s + list(reversed(path_g))[1:]
                if config.shortcut_iterations > 0:
                    full_path = shortcut_path(
                        full_path, collision_fn, rng,
                        n_iterations=config.shortcut_iterations,
                        n_check=config.shortcut_n_check,
                        clearance_fn=clearance_fn,
                        clearance_margin=config.shortcut_clearance_margin,
                    )
                return PlanResult(
                    success=True,
                    path=full_path,
                    goal_idx=goal_idx,
                    nodes_explored=len(tree_s) + len(tree_g),
                    time_to_solution=time.perf_counter() - t0,
                    collision_checks=col_count[0],
                    iterations=iteration + 1,
                )

        else:
            # Extend tree_g, then try to connect tree_s
            ext = _extend(tree_g, q_rand, config, collision_fn, col_count, clearance_fn)
            if ext is None:
                continue
            q_new, idx_g = ext

            conn_idx, reached = _connect(
                tree_s, q_new, config, collision_fn, col_count, clearance_fn
            )
            if reached and conn_idx is not None:
                path_s = tree_s.path_to_root(conn_idx)  # [q_start, …, q_new]
                path_g = tree_g.path_to_root(idx_g)     # [q_goal,  …, q_new]
                goal_idx = tree_g.root_of(idx_g)
                full_path = path_s + list(reversed(path_g))[1:]
                if config.shortcut_iterations > 0:
                    full_path = shortcut_path(
                        full_path, collision_fn, rng,
                        n_iterations=config.shortcut_iterations,
                        n_check=config.shortcut_n_check,
                        clearance_fn=clearance_fn,
                        clearance_margin=config.shortcut_clearance_margin,
                    )
                return PlanResult(
                    success=True,
                    path=full_path,
                    goal_idx=goal_idx,
                    nodes_explored=len(tree_s) + len(tree_g),
                    time_to_solution=time.perf_counter() - t0,
                    collision_checks=col_count[0],
                    iterations=iteration + 1,
                )

    # Planning failed
    return PlanResult(
        success=False,
        path=None,
        goal_idx=None,
        nodes_explored=len(tree_s) + len(tree_g),
        time_to_solution=time.perf_counter() - t0,
        collision_checks=col_count[0],
        iterations=config.max_iterations,
    )
