"""
Visualisation utilities for MDP results.

Provides two plots:
  1. Value function heatmap — how "good" each cell is
  2. Policy arrows — which direction the agent should move from each cell
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.create_maze import MazeData
from src.mdp import MDPResult, ACTIONS


# Arrow direction vectors for policy visualisation
_ARROW = {
    "UP":    ( 0, -0.3),
    "DOWN":  ( 0,  0.3),
    "LEFT":  (-0.3,  0),
    "RIGHT": ( 0.3,  0),
}


def plot_value_function(maze: MazeData, result: MDPResult, title: str = "Value Function"):
    """
    Plot the value function as a heatmap overlaid on the maze.
    Brighter cells have higher value (closer to goal under optimal policy).
    """
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])

    # Build a 2D array of values — walls get NaN so they render separately
    value_grid = np.full((rows, cols), np.nan)
    for (r, c), v in result.values.items():
        value_grid[r, c] = v

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw walls in dark grey
    wall_grid = np.where(np.array(grid) == 1, 1.0, np.nan)
    ax.imshow(wall_grid, cmap="gray_r", vmin=0, vmax=1)

    # Overlay value heatmap on free cells
    im = ax.imshow(value_grid, cmap="YlGnBu", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, label="V(s)")

    # Mark start and goal
    ax.scatter(maze.start[1], maze.start[0], c="green", s=120, zorder=5, label="Start")
    ax.scatter(maze.goal[1],  maze.goal[0],  c="red",   s=120, zorder=5, label="Goal")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_policy(maze: MazeData, result: MDPResult, title: str = "Policy"):
    """
    Plot the policy as directional arrows overlaid on the maze.
    Each free cell shows an arrow indicating the best action.
    """
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw maze (walls = black, free = white)
    ax.imshow(np.array(grid), cmap="binary")

    # Draw policy arrows
    for (r, c), action in result.policy.items():
        if action is None:
            continue  # goal cell — no arrow
        dx, dy = _ARROW[action]
        ax.annotate(
            "",
            xy=(c + dx, r + dy),
            xytext=(c, r),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
        )

    # Overlay solution path
    if result.path:
        ys = [p[0] for p in result.path]
        xs = [p[1] for p in result.path]
        ax.plot(xs, ys, c="orange", linewidth=2, zorder=4, label="Path")

    # Mark start and goal
    ax.scatter(maze.start[1], maze.start[0], c="green", s=120, zorder=5, label="Start")
    ax.scatter(maze.goal[1],  maze.goal[0],  c="red",   s=120, zorder=5, label="Goal")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_mdp_results(maze: MazeData, result: MDPResult, algorithm: str, size: str):
    """Convenience wrapper — plots both value function and policy side by side."""
    plot_value_function(maze, result, title=f"{algorithm} Value Function — {size}")
    plot_policy(maze, result, title=f"{algorithm} Policy — {size}")