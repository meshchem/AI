import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np

from src.create_maze import MazeData
from src.mdp_algorithms import ValueIterationResult


# Arrow direction vectors for policy visualization
arrow = {
    "up":    (0, -0.3),
    "down":  (0,  0.3),
    "left":  (-0.3,  0),
    "right": (0.3,  0),
}


# Plot the value function as a heatmap overlaid on the maze.
# Brighter cells have higher value (closer to goal under optimal policy).
def plot_value_function(maze: MazeData, result: ValueIterationResult, title: str = "Value Function", save_path: str = None):
  
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
    ax.scatter(maze.start[1], maze.start[0], c="green", s=100, zorder=5, label="Start")
    ax.scatter(maze.goal[1],  maze.goal[0],  c="red",   s=100, zorder=5, label="Goal")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()  # Close after saving
    else:
        plt.show()

# Plot the policy as directional arrows overlaid on the maze.
# Each free cell shows an arrow indicating the best action.
def plot_policy(maze: MazeData, result: ValueIterationResult, title: str = "Policy", save_path: str = None):
   
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw maze (walls = black, free = white)
    ax.imshow(np.array(grid), cmap="binary")

    # Draw policy arrows
    for (r, c), action in result.policy.items():
        if action is None:
            continue  # goal node —> no arrow
        dx, dy = arrow[action]
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
    ax.scatter(maze.start[1], maze.start[0], c="green", s=100, zorder=5, label="Start")
    ax.scatter(maze.goal[1],  maze.goal[0],  c="red",   s=100, zorder=5, label="Goal")

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

#  plots both value function and policy side by side
def plot_mdp_results(maze: MazeData, result: ValueIterationResult, algorithm: str, size: str, save_dir: str = None):
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        value_path = os.path.join(save_dir, f"{algorithm.replace(' ', '_').lower()}_value_{size}.png")
        policy_path = os.path.join(save_dir, f"{algorithm.replace(' ', '_').lower()}_policy_{size}.png")
        
        plot_value_function(maze, result, title=f"{algorithm} Value Function — {size}", save_path=value_path)
        plot_policy(maze, result, title=f"{algorithm} Policy — {size}", save_path=policy_path)
    else:
        plot_value_function(maze, result, title=f"{algorithm} Value Function — {size}")
        plot_policy(maze, result, title=f"{algorithm} Policy — {size}")


