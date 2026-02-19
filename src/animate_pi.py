import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from typing import Dict, Optional, List, Tuple
from copy import deepcopy

from src.create_maze import MazeData, Coord


arrow_size = 0.35
arrows = {
    "up":    (0, -arrow_size),
    "down":  (0,  arrow_size),
    "left":  (-arrow_size, 0),
    "right": (arrow_size, 0),
}


def policy_iteration_frames(
    maze: MazeData,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_frames: int = 100
) -> List[Tuple[int, Dict, Dict, bool]]:

    from src.mdp_algorithms import get_states, get_next_state, get_reward, actions
    
    states = get_states(maze)
    goal = maze.goal
    
    # Initialize policy 
    policy = {s: "down" for s in states}
    policy[goal] = None
    
    # Initialize value function
    V = {s: 0.0 for s in states}
    V[goal] = 1  # goal_reward
    
    frames = []
    iteration = 0
    
    while True:
        while True:
            delta = 0
            
            for s in states:
                if s == goal:
                    continue
                
                v_old = V[s]
                
                # Follow current policy
                action = policy[s]
                s_next = get_next_state(maze, s, action)
                reward = get_reward(maze, s, action)
                
                # V^π(s) = R(s,π(s)) + γ * V^π(s')
                V[s] = reward + gamma * V[s_next]
                
                delta = max(delta, abs(v_old - V[s]))
            
            # Check if values converged for this policy
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        
        for s in states:
            if s == goal:
                continue
            
            old_action = policy[s]
            
            # Find best action according to current values
            best_action = None
            best_value = float('-inf')
            
            for a in actions:
                s_next = get_next_state(maze, s, a)
                reward = get_reward(maze, s, a)
                q_value = reward + gamma * V[s_next]
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            
            policy[s] = best_action
            
            # Check if policy changed
            if best_action != old_action:
                policy_stable = False
        
        iteration += 1
        
        # Save frame
        frames.append((iteration, deepcopy(V), deepcopy(policy), policy_stable))
        
        # Check if policy converged
        if policy_stable or iteration >= max_frames:
            break
    
    return frames


def animate_policy_iteration(
    maze: MazeData,
    gamma: float = 0.9,
    theta: float = 1e-6,
    output_dir: str = "videos",
    fps: int = 2,
    show: bool = True,
):
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/policy_iteration_mdp.mp4"
  
    grid = maze.grid
    rows, cols = len(grid), len(grid[0])
    
    # Compute frames
    print(f"Generating Policy Iteration frames")
    frames = policy_iteration_frames(maze, gamma, theta)
    print(f"  Captured {len(frames)} iterations")
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Policy Iteration, k = 0", fontsize=16)
    
    # 1st plot: Heatmap
    ax1.set_title("Value Function V(s)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Draw walls
    wall_grid = np.where(np.array(grid) == 1, 1.0, np.nan)
    ax1.imshow(wall_grid, cmap="gray_r", vmin=0, vmax=1)
    
    # Placeholder for values
    value_grid = np.full((rows, cols), np.nan)
    value_im = ax1.imshow(value_grid, cmap="YlGnBu", interpolation="nearest", alpha=0.8)
    cbar1 = plt.colorbar(value_im, ax=ax1, fraction=0.046)
    
    # Mark start and goal
    ax1.scatter(maze.start[1], maze.start[0], c="green", s=100, zorder=5, label="Start")
    ax1.scatter(maze.goal[1], maze.goal[0], c="red", s=100, zorder=5, marker="*", label="Goal")
    
    # 2nd plot: Arrows 
    ax2.set_title("Policy π(s)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Draw maze
    ax2.imshow(np.array(grid), cmap="binary")
    
    # Mark start and goal
    ax2.scatter(maze.start[1], maze.start[0], c="green", s=100, zorder=5, label="Start")
    ax2.scatter(maze.goal[1], maze.goal[0], c="red", s=100, zorder=5, marker="*", label="Goal")
    
    # Containers for dynamic elements
    arrow_artists = []
    value_texts = []
    
    # init animation
    def init():
        return []
    
    # update frame with new values and policy 
    def update(frame_idx):
        nonlocal value_im
        
        # Clear previous arrows and texts
        for artist in arrow_artists:
            artist.remove()
        arrow_artists.clear()
        
        for text in value_texts:
            text.remove()
        value_texts.clear()
        
        # Get current frame
        if frame_idx >= len(frames):
            frame_idx = len(frames) - 1
        
        iteration, values, policy, converged = frames[frame_idx]
        
        # Update title
        status = "Function has converges" if converged else "Iterating"
        fig.suptitle(f"Policy Iteration, k = {iteration} ({status})", fontsize=16)
        
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
        
        # Draw value text on cells
        for (r, c), v in values.items():
            if (r, c) != maze.goal and abs(v) < 10:  # skip goal and extreme values
                text = ax1.text(c, r, f"{v:.2f}", ha="center", va="center",
                               fontsize=3, color="black", zorder=4)
                value_texts.append(text)
        
        # Draw policy arrows
        for (r, c), action in policy.items():
            if action is None:
                continue  # goal
            
            dx, dy = arrows[action]
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
        
        return [value_im] + arrow_artists + value_texts
    
    # Total frames
    total_frames = len(frames) + 20  # hold final frame
    
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
        print(f"Saving animation to: {output_file}")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(output_file, writer=writer)
        print(f"Saved: {output_file}")
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return anim
