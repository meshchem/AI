import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from typing import Dict, Optional, List
from copy import deepcopy

from src.create_maze import MazeData, Coord, get_neighbours


# Arrow direction vectors for policy visualisation
_ARROW_SCALE = 0.35
_ACTIONS = {
    "UP":    ( 0, -_ARROW_SCALE),
    "DOWN":  ( 0,  _ARROW_SCALE),
    "LEFT":  (-_ARROW_SCALE,  0),
    "RIGHT": ( _ARROW_SCALE,  0),
}


def _compute_iteration_snapshots(maze: MazeData, algorithm: str, max_snapshots: int = 100):
    """
    Re-run the MDP algorithm and capture snapshots at each iteration.
    
    Returns list of (iteration, values, policy, converged) tuples.
    """
    from src.old_mdp.mdp import (
        GAMMA, LIVING_REWARD, GOAL_REWARD, THETA, ACTIONS,
        _free_cells, _bellman_value
    )
    
    grid = maze.grid
    goal = maze.goal
    free = _free_cells(grid)
    
    snapshots = []
    
    if algorithm == "value_iteration":
        # Value Iteration with snapshots
        V = {s: 0.0 for s in free}
        V[goal] = GOAL_REWARD
        
        iteration = 0
        while True:
            delta = 0.0
            new_V = V.copy()
            
            for state in free:
                if state == goal:
                    continue
                best = max(
                    _bellman_value(state, action, grid, goal, V)
                    for action in ACTIONS
                )
                delta = max(delta, abs(best - V[state]))
                new_V[state] = best
            
            V = new_V
            iteration += 1
            
            # Extract policy
            policy = {}
            for state in free:
                if state == goal:
                    policy[state] = None
                else:
                    policy[state] = max(
                        ACTIONS,
                        key=lambda a: _bellman_value(state, a, grid, goal, V)
                    )
            
            converged = delta < THETA
            snapshots.append((iteration, deepcopy(V), deepcopy(policy), converged))
            
            if converged or iteration >= max_snapshots:
                break
    
    elif algorithm == "policy_iteration":
        # Policy Iteration with snapshots
        policy = {s: "UP" for s in free}
        policy[goal] = None
        V = {s: 0.0 for s in free}
        V[goal] = GOAL_REWARD
        
        iteration = 0
        while True:
            # Policy Evaluation
            eval_iters = 0
            while True:
                delta = 0.0
                new_V = V.copy()
                for state in free:
                    if state == goal:
                        continue
                    action = policy[state]
                    v = _bellman_value(state, action, grid, goal, V)
                    delta = max(delta, abs(v - V[state]))
                    new_V[state] = v
                V = new_V
                eval_iters += 1
                if delta < THETA:
                    break
            
            # Policy Improvement
            policy_stable = True
            for state in free:
                if state == goal:
                    continue
                old_action = policy[state]
                best_action = max(
                    ACTIONS,
                    key=lambda a: _bellman_value(state, a, grid, goal, V)
                )
                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False
            
            iteration += 1
            snapshots.append((iteration, deepcopy(V), deepcopy(policy), policy_stable))
            
            if policy_stable or iteration >= max_snapshots:
                break
    
    return snapshots


def animate_mdp(
    maze: MazeData,
    algorithm: str,  # "value_iteration" or "policy_iteration"
    output_file: str = None,
    fps: int = 2,
    show: bool = True,
):
    """
    Create an animated visualization of MDP value/policy convergence.

    Parameters:
        maze: MazeData with grid, start, goal
        algorithm: "value_iteration" or "policy_iteration"
        output_file: if provided, save as MP4 (requires ffmpeg)
        fps: frames per second (slower than search animations)
        show: whether to display the animation interactively
    """
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])
    
    # Compute snapshots
    print(f"Computing {algorithm} snapshots...")
    snapshots = _compute_iteration_snapshots(maze, algorithm, max_snapshots=100)
    print(f"  Captured {len(snapshots)} iterations")
    
    # Set up figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    algo_name = "Value Iteration" if algorithm == "value_iteration" else "Policy Iteration"
    fig.suptitle(f"{algo_name} — Iteration 0", fontsize=16)
    
    # --- Left plot: Value Function Heatmap ---
    ax1.set_title("Value Function V(s)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Draw walls
    wall_grid = np.where(np.array(grid) == 1, 1.0, np.nan)
    ax1.imshow(wall_grid, cmap="gray_r", vmin=0, vmax=1)
    
    # Placeholder for value heatmap
    value_grid = np.full((rows, cols), np.nan)
    value_im = ax1.imshow(value_grid, cmap="YlGnBu", interpolation="nearest", alpha=0.8)
    cbar1 = plt.colorbar(value_im, ax=ax1, fraction=0.046)
    
    # Mark start and goal
    ax1.scatter(maze.start[1], maze.start[0], s="100", c="green")
    ax1.scatter(maze.goal[1], maze.goal[0], c="red", s=100,  marker="*")
    
    # --- Right plot: Policy Arrows ---
    ax2.set_title("Policy π(s)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Draw maze
    ax2.imshow(np.array(grid), cmap="binary")
    
    # Mark start and goal
    ax2.scatter(maze.start[1], maze.start[0], c="green", s=100, zorder=5, marker="s", edgecolors="black")
    ax2.scatter(maze.goal[1], maze.goal[0], c="red", s=100, zorder=5, marker="*", edgecolors="black")
    
    # Containers for dynamic elements
    arrow_artists = []
    value_texts = []
    
    # init animation with empty state
    def init():
        return []
    
    # update frame
    def update(frame_idx):
        nonlocal value_im
        
        # Clear previous arrows and text
        for artist in arrow_artists:
            artist.remove()
        arrow_artists.clear()
        
        for text in value_texts:
            text.remove()
        value_texts.clear()
        
        # Get current snapshot
        if frame_idx >= len(snapshots):
            frame_idx = len(snapshots) - 1
        
        iteration, values, policy, converged = snapshots[frame_idx]
        
        # Update title
        status = "CONVERGED" if converged else "Running..."
        fig.suptitle(f"{algo_name} — Iteration {iteration} ({status})", fontsize=16)
        
        # Update value heatmap
        value_grid = np.full((rows, cols), np.nan)
        for (r, c), v in values.items():
            value_grid[r, c] = v
        
        value_im.set_data(value_grid)
        
        # Update colorbar limits
        valid_values = [v for v in values.values() if v is not None]
        if valid_values:
            vmin, vmax = min(valid_values), max(valid_values)
            value_im.set_clim(vmin, vmax)
        
        # Update policy arrows
        for (r, c), action in policy.items():
            if action is None:
                continue  # goal cell
            
            dx, dy = _ACTIONS[action]
            arrow = FancyArrowPatch(
                (c, r),
                (c + dx, r + dy),
                arrowstyle="->",
                color="steelblue",
                lw=1.5,
                mutation_scale=15,
                zorder=3,
            )
            ax2.add_patch(arrow)
            arrow_artists.append(arrow)
        
        # Optionally show value text on cells (can be cluttered for large mazes)
        # Uncomment if you want numeric values displayed:
        for (r, c), v in values.items():
            if (r, c) != maze.goal and abs(v) < 10:  # skip goal and extreme values
                text = ax1.text(c, r, f"{v:.2f}", ha="center", va="center",
                               fontsize=6, color="black", zorder=4)
                value_texts.append(text)
        
        return [value_im] + arrow_artists + value_texts
    
    # Total frames = all snapshots + hold final frame
    total_frames = len(snapshots) + 20
    
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=1000 // fps,
        blit=False,
        repeat=True,
    )
    
    if output_file:
        print(f"Saving animation to {output_file}...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(output_file, writer=writer)
        print(f"Saved: {output_file}")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return anim


def animate_mdp_comparison(
    maze: MazeData,
    output_dir: str = "videos",
    fps: int = 2,
):
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for algo in ["value_iteration", "policy_iteration"]:
        output_file = f"{output_dir}/{algo}_mdp.mp4"
        animate_mdp(
            maze,
            algorithm=algo,
            output_file=output_file,
            fps=fps,
            show=False,
        )