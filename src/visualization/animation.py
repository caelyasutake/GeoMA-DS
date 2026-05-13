"""
Animation utilities for the Multi-IK DS pipeline.

Creates animated GIFs or sequences of PNG frames showing joint trajectories
and key metrics evolving over time.

Functions
---------
save_trajectory_animation   — produce animated GIF of execution metrics
save_frames                 — save individual PNG frames for each timestep
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.visualization.plotting import (
    plot_tank_energy,
    plot_passivity_metrics,
    plot_distance_to_goal,
    plot_joint_trajectories,
)


def save_trajectory_animation(
    results: list,
    q_goal: np.ndarray,
    output_path: str,
    dt: float = 0.05,
    fps: int = 10,
    goal_radius: float = 0.05,
) -> str:
    """
    Create an animated GIF that shows metrics evolving frame-by-frame.

    Each frame covers steps [0 … k] for k ∈ {0, 1, …, T-1}.
    To keep file size reasonable the animation is down-sampled to at most
    ``fps`` frames per simulated second.

    Args:
        results:     List of ControlResult from simulate().
        q_goal:      Goal configuration, shape (n,).
        output_path: Destination path (should end in ``.gif``).
        dt:          Controller timestep (s).
        fps:         Frames per second in the animation (before GIF quantization).
        goal_radius: Goal acceptance radius for the distance plot.

    Returns:
        Absolute path to the saved GIF.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        raise ImportError("Pillow is required for GIF export: pip install Pillow")

    T = len(results)
    # How many simulation steps between animation frames
    sim_fps  = 1.0 / dt                    # steps per simulated second
    step_gap = max(1, int(sim_fps / fps))

    q_history = np.array([r.xdot_d for r in results])   # proxy; use q if available

    # Build data arrays upfront
    times    = np.arange(T) * dt
    energies = np.array([r.tank_energy for r in results])
    p_nom    = np.array([r.pf_power_nom for r in results])
    p_filt   = np.array([r.pf_power_filtered for r in results])
    dists    = np.array([float(np.linalg.norm(r.xdot_d - q_goal)) for r in results])

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("DS Controller Execution (animated)", fontsize=11)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_tank = fig.add_subplot(gs[0, 0])
    ax_pass = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])

    frame_indices = list(range(0, T, step_gap)) + [T - 1]

    def _draw_frame(k: int) -> None:
        for ax in [ax_tank, ax_pass, ax_dist, ax_beta]:
            ax.clear()

        t_k = times[: k + 1]

        # Tank energy
        ax_tank.plot(t_k, energies[: k + 1], color="#00BCD4", linewidth=1.5)
        ax_tank.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax_tank.set_xlim(0, times[-1])
        ax_tank.set_ylim(bottom=0)
        ax_tank.set_xlabel("t (s)", fontsize=8)
        ax_tank.set_ylabel("Tank s", fontsize=8)
        ax_tank.set_title("Tank Energy", fontsize=9)
        ax_tank.grid(True, alpha=0.3)

        # Passivity power
        ax_pass.plot(t_k, p_nom[: k + 1],  color="steelblue",  linewidth=1.0,
                     label="nom", alpha=0.7)
        ax_pass.plot(t_k, p_filt[: k + 1], color="darkorange", linewidth=1.0,
                     label="filt")
        ax_pass.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax_pass.set_xlim(0, times[-1])
        ax_pass.set_xlabel("t (s)", fontsize=8)
        ax_pass.set_ylabel("ẋᵀ f_R", fontsize=8)
        ax_pass.set_title("Residual Power", fontsize=9)
        ax_pass.legend(fontsize=6)
        ax_pass.grid(True, alpha=0.3)

        # Distance to goal
        ax_dist.plot(t_k, dists[: k + 1], color="coral", linewidth=1.5)
        ax_dist.axhline(goal_radius, color="k", linestyle="--",
                        linewidth=0.7, alpha=0.7)
        ax_dist.set_xlim(0, times[-1])
        ax_dist.set_xlabel("t (s)", fontsize=8)
        ax_dist.set_ylabel("‖q−q*‖", fontsize=8)
        ax_dist.set_title("Distance to Goal", fontsize=9)
        ax_dist.grid(True, alpha=0.3)

        # Beta_R
        betas = [r.beta_R for r in results[: k + 1]]
        ax_beta.plot(t_k, betas, color="mediumseagreen", linewidth=1.5)
        ax_beta.set_xlim(0, times[-1])
        ax_beta.set_ylim(-0.05, 1.1)
        ax_beta.set_xlabel("t (s)", fontsize=8)
        ax_beta.set_ylabel("β_R", fontsize=8)
        ax_beta.set_title("Tank Gate β_R", fontsize=9)
        ax_beta.grid(True, alpha=0.3)

        fig.suptitle(
            f"DS Controller Execution — t = {times[k]:.2f} s / {times[-1]:.2f} s",
            fontsize=11,
        )

    anim = FuncAnimation(
        fig,
        lambda frame: _draw_frame(frame_indices[frame]),
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=False,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(Path(output_path).resolve())


def save_frames(
    results: list,
    q_history: np.ndarray,
    q_goal: np.ndarray,
    output_dir: str,
    dt: float = 0.05,
    every_n: int = 10,
    goal_radius: float = 0.05,
) -> List[str]:
    """
    Save a PNG snapshot every ``every_n`` steps showing metrics up to that point.

    Args:
        results:    List of ControlResult from simulate().
        q_history:  Array shape (T, n) of joint positions.
        q_goal:     Goal configuration, shape (n,).
        output_dir: Directory to write frames into.
        dt:         Controller timestep (s).
        every_n:    Interval between saved frames.
        goal_radius: Goal acceptance radius.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = len(results)
    saved = []

    for k in range(0, T, every_n):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(f"Step {k}/{T}  t={k * dt:.2f} s", fontsize=11)

        plot_tank_energy(axes[0, 0], results[: k + 1], dt=dt)
        plot_passivity_metrics(axes[0, 1], results[: k + 1], dt=dt)
        plot_distance_to_goal(axes[1, 0], q_history[: k + 1], q_goal,
                              dt=dt, goal_radius=goal_radius)
        plot_joint_trajectories(axes[1, 1], q_history[: k + 1], dt=dt)

        plt.tight_layout()
        path = output_dir / f"frame_{k:05d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        saved.append(str(path))

    return saved
