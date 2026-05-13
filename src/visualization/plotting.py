"""
Reusable plotting utilities for the Multi-IK DS pipeline.

Functions
---------
plot_ik_goals            — joint-space scatter of IK candidates
plot_rrt_path            — joint-space RRT path overlay
plot_executed_trajectory — executed vs planned trajectory
plot_tank_energy         — tank energy over time
plot_passivity_metrics   — residual power and clipping events
plot_contact_metrics     — contact force vs desired over time
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C_START    = "#2196F3"   # blue
_C_GOAL_ALL = "#BDBDBD"   # grey
_C_SAFE     = "#4CAF50"   # green
_C_SELECTED = "#F44336"   # red
_C_PATH     = "#FF9800"   # orange
_C_EXEC     = "#9C27B0"   # purple
_C_TANK     = "#00BCD4"   # cyan
_C_CLIP     = "#F44336"   # red


# ---------------------------------------------------------------------------
# Planning plots
# ---------------------------------------------------------------------------
def plot_ik_goals(
    ax: Axes,
    Q_goals: Sequence[np.ndarray],
    q_start: np.ndarray,
    safe_mask: Optional[Sequence[bool]] = None,
    selected_idx: Optional[int] = None,
    dim1: int = 0,
    dim2: int = 1,
    title: str = "IK Goal Set",
) -> None:
    """
    Scatter plot of IK candidate configurations in a 2-D joint projection.

    Args:
        ax:           Matplotlib Axes to draw on.
        Q_goals:      List of goal configurations, each shape (n,).
        q_start:      Start configuration, shape (n,).
        safe_mask:    Boolean mask, same length as Q_goals (True = safe).
        selected_idx: Index into Q_goals that the planner actually used.
        dim1, dim2:   Joint indices for the two axes.
        title:        Subplot title.
    """
    if safe_mask is None:
        safe_mask = [True] * len(Q_goals)

    for i, q in enumerate(Q_goals):
        color = _C_GOAL_ALL
        zorder = 2
        if safe_mask[i]:
            color = _C_SAFE
            zorder = 3
        if selected_idx is not None and i == selected_idx:
            color = _C_SELECTED
            zorder = 4
        ax.scatter(
            q[dim1], q[dim2],
            c=color, s=80, zorder=zorder,
            edgecolors="k", linewidths=0.5,
        )

    ax.scatter(
        q_start[dim1], q_start[dim2],
        c=_C_START, s=120, marker="*", zorder=5,
        edgecolors="k", linewidths=0.5, label="Start",
    )

    legend_handles = [
        mpatches.Patch(color=_C_START,    label="Start"),
        mpatches.Patch(color=_C_GOAL_ALL, label="All IK goals"),
        mpatches.Patch(color=_C_SAFE,     label="Safe goals"),
    ]
    if selected_idx is not None:
        legend_handles.append(
            mpatches.Patch(color=_C_SELECTED, label="Selected goal")
        )
    ax.legend(handles=legend_handles, fontsize=7, loc="best")
    ax.set_xlabel(f"q[{dim1}] (rad)")
    ax.set_ylabel(f"q[{dim2}] (rad)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_rrt_path(
    ax: Axes,
    path: List[np.ndarray],
    q_start: Optional[np.ndarray] = None,
    dim1: int = 0,
    dim2: int = 1,
    title: str = "RRT Path",
    label: str = "Planned path",
    color: str = _C_PATH,
) -> None:
    """
    Draw the planned RRT waypoints as a connected line in joint space.

    Args:
        ax:     Matplotlib Axes.
        path:   List of waypoints (each shape (n,)).
        q_start: Highlight start position (optional).
        dim1, dim2: Joint indices for the axes.
        title:  Subplot title.
        label:  Line label for legend.
        color:  Line/marker colour.
    """
    if not path:
        return
    xs = [q[dim1] for q in path]
    ys = [q[dim2] for q in path]
    ax.plot(xs, ys, "-o", color=color, markersize=4, linewidth=1.5,
            label=label, zorder=3)
    ax.scatter(xs[0],  ys[0],  c=_C_START,    s=100, marker="*",
               zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter(xs[-1], ys[-1], c=_C_SELECTED, s=100, marker="D",
               zorder=5, edgecolors="k", linewidths=0.5, label="Goal")
    ax.legend(fontsize=7)
    ax.set_xlabel(f"q[{dim1}] (rad)")
    ax.set_ylabel(f"q[{dim2}] (rad)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_executed_trajectory(
    ax: Axes,
    q_history: np.ndarray,
    path: Optional[List[np.ndarray]] = None,
    dim1: int = 0,
    dim2: int = 1,
    title: str = "Executed Trajectory",
) -> None:
    """
    Overlay the executed trajectory on the planned path in joint space.

    Args:
        ax:        Matplotlib Axes.
        q_history: Array of shape (T, n) — joint positions over time.
        path:      Planned waypoints (optional overlay).
        dim1, dim2: Joint indices for the axes.
        title:     Subplot title.
    """
    q_history = np.asarray(q_history)
    if path is not None:
        xs = [q[dim1] for q in path]
        ys = [q[dim2] for q in path]
        ax.plot(xs, ys, "--", color=_C_PATH, linewidth=1.5,
                label="Planned", alpha=0.6, zorder=2)

    ax.plot(
        q_history[:, dim1], q_history[:, dim2],
        color=_C_EXEC, linewidth=1.5, label="Executed", zorder=3,
    )
    ax.scatter(
        q_history[0, dim1], q_history[0, dim2],
        c=_C_START, s=100, marker="*", zorder=5,
        edgecolors="k", linewidths=0.5,
    )
    ax.scatter(
        q_history[-1, dim1], q_history[-1, dim2],
        c=_C_EXEC, s=80, marker="D", zorder=5,
        edgecolors="k", linewidths=0.5, label="Final",
    )
    ax.legend(fontsize=7)
    ax.set_xlabel(f"q[{dim1}] (rad)")
    ax.set_ylabel(f"q[{dim2}] (rad)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Metrics plots
# ---------------------------------------------------------------------------
def plot_tank_energy(
    ax: Axes,
    results: list,
    dt: float = 0.05,
    title: str = "Tank Energy",
) -> None:
    """
    Line plot of energy tank level over time.

    Args:
        ax:      Matplotlib Axes.
        results: List of ControlResult objects from simulate().
        dt:      Controller timestep (s).
        title:   Subplot title.
    """
    times = [i * dt for i in range(len(results))]
    energies = [r.tank_energy for r in results]
    ax.plot(times, energies, color=_C_TANK, linewidth=1.5)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Depletion threshold")
    ax.fill_between(times, energies, 0, alpha=0.15, color=_C_TANK)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tank energy s")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_passivity_metrics(
    ax: Axes,
    results: list,
    dt: float = 0.05,
    title: str = "Passivity Filter Metrics",
) -> None:
    """
    Plot residual power (before/after filter) and mark clipping events.

    Args:
        ax:      Matplotlib Axes.
        results: List of ControlResult from simulate().
        dt:      Controller timestep (s).
        title:   Subplot title.
    """
    times   = [i * dt for i in range(len(results))]
    p_nom   = [r.pf_power_nom    for r in results]
    p_filt  = [r.pf_power_filtered for r in results]
    clipped = [r.pf_clipped      for r in results]

    ax.plot(times, p_nom,  color="steelblue", linewidth=1.0,
            label="Power nom", alpha=0.7)
    ax.plot(times, p_filt, color="darkorange", linewidth=1.0,
            label="Power filtered")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    # Mark clipping events
    clip_t = [t for t, c in zip(times, clipped) if c]
    if clip_t:
        ax.scatter(clip_t, [0.0] * len(clip_t),
                   color=_C_CLIP, s=10, zorder=5, label="Clipped", marker="|")

    clip_ratio = sum(clipped) / max(len(clipped), 1) * 100
    ax.set_title(f"{title}  (clip {clip_ratio:.1f}%)", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (ẋᵀ f_R)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_lyapunov(
    ax: Axes,
    results: list,
    dt: float = 0.05,
    title: str = "Lyapunov V(q)",
) -> None:
    """
    Plot the Lyapunov function value over time to show convergence.

    Args:
        ax:      Matplotlib Axes.
        results: List of ControlResult from simulate().
        dt:      Controller timestep (s).
        title:   Subplot title.
    """
    times = [i * dt for i in range(len(results))]
    vs    = [r.V for r in results]
    ax.plot(times, vs, color="mediumseagreen", linewidth=1.5)
    ax.fill_between(times, vs, min(vs), alpha=0.1, color="mediumseagreen")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V(q)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_contact_metrics(
    ax: Axes,
    force_history: Optional[list] = None,
    desired_force: Optional[float] = None,
    dt: float = 0.05,
    title: str = "Contact Forces",
) -> None:
    """
    Plot contact force norm over time vs desired.

    Args:
        ax:            Matplotlib Axes.
        force_history: List of np.ndarray (3,) contact forces per step.
        desired_force: Target normal force magnitude (dashed reference).
        dt:            Controller timestep (s).
        title:         Subplot title.
    """
    if not force_history:
        ax.text(0.5, 0.5, "No contact data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_title(title, fontsize=9)
        return

    times  = [i * dt for i in range(len(force_history))]
    norms  = [float(np.linalg.norm(f)) for f in force_history]
    ax.plot(times, norms, color="tomato", linewidth=1.5, label="|F_contact|")
    if desired_force is not None:
        ax.axhline(desired_force, color="k", linestyle="--", linewidth=0.8,
                   label=f"F_desired = {desired_force:.1f} N")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force norm (N)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_joint_trajectories(
    ax: Axes,
    q_history: np.ndarray,
    dt: float = 0.05,
    title: str = "Joint Trajectories",
) -> None:
    """
    Plot all joint angles over time.

    Args:
        ax:        Matplotlib Axes.
        q_history: Array shape (T, n_joints).
        dt:        Controller timestep (s).
        title:     Subplot title.
    """
    q_history = np.asarray(q_history)
    T, n = q_history.shape
    times = np.arange(T) * dt
    cmap = plt.get_cmap("tab10")
    for j in range(n):
        ax.plot(times, q_history[:, j], color=cmap(j % 10),
                linewidth=1.0, label=f"q[{j}]", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)


def plot_distance_to_goal(
    ax: Axes,
    q_history: np.ndarray,
    q_goal: np.ndarray,
    dt: float = 0.05,
    goal_radius: float = 0.05,
    title: str = "Distance to Goal",
) -> None:
    """
    Plot ‖q(t) − q*‖ over time.

    Args:
        ax:          Matplotlib Axes.
        q_history:   Array shape (T, n).
        q_goal:      Goal configuration, shape (n,).
        dt:          Controller timestep (s).
        goal_radius: Acceptance radius (dashed reference line).
        title:       Subplot title.
    """
    q_history = np.asarray(q_history)
    times = np.arange(len(q_history)) * dt
    dists = [float(np.linalg.norm(q - q_goal)) for q in q_history]
    ax.plot(times, dists, color="coral", linewidth=1.5)
    ax.axhline(goal_radius, color="k", linestyle="--", linewidth=0.8,
               alpha=0.7, label=f"Goal radius {goal_radius:.2f}")
    ax.fill_between(times, dists, goal_radius,
                    where=[d > goal_radius for d in dists],
                    alpha=0.1, color="coral")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("‖q − q*‖ (rad)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
