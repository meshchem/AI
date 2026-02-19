from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from create_maze import MazeData, Coord, load_maze


# Algorithm parameters
gamma = 0.9               # discount factor
theta = 1e-6              # convergence threshold

# Rewards
step_cost = -1            # normal direction_deltas
goal_reward = 1       # reaching goal

# actions (lowercase) - redirection_deltasd "stay"
actions = ["up", "down", "left", "right"]
direction_deltas = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_states(maze: MazeData) -> List[Coord]:
    states = []
    for r in range(len(maze.grid)):
        for c in range(len(maze.grid[0])):
            if maze.grid[r][c] == 0:  # no wall
                states.append((r, c))
    return states

#   Get next state after taking action a from state s.
def get_next_state(maze: MazeData, s: Coord, a: str) -> Coord:
    dr, dc = direction_deltas[a]
    next_r, next_c = s[0] + dr, s[1] + dc
    
    rows, cols = len(maze.grid), len(maze.grid[0])
    
    # Check if next position is valid (in bounds and not a wall)
    if 0 <= next_r < rows and 0 <= next_c < cols and maze.grid[next_r][next_c] == 0:
        return (next_r, next_c)
    else:
        # Hit wall or boundary - stay in current state
        return s

# 
def get_reward(maze: MazeData, s: Coord, a: str) -> float:

    next_state = get_next_state(maze, s, a)
    
    # Check if we reached the goal
    if next_state == maze.goal:
        return goal_reward  # +1
    
    # Normal direction_deltas (even if bounced off wall)
    return step_cost  # -1


# maze: MazeData object containing grid, start, goal
# gamma: discount factor (default 0.9)
# theta: convergence threshold (default 1e-6)
def value_iteration_with_walls(maze: MazeData, gamma=gamma, theta=theta) -> Tuple[Dict[Coord, str], Dict[Coord, float]]:

    # Get all states
    states = get_states(maze)
    
    # Initialize value function to 0 for all states
    V = {s: 0.0 for s in states}
    
    # Set goal value to the goal reward
    V[maze.goal] = goal_reward
    
    iterations = 0
    
    # Value Iteration: iterate until convergence
    while True:
        delta = 0
        
        # For each state
        for s in states:
            if s == maze.goal:
                # Terminal state - value stays at goal reward
                continue
            
            v_old = V[s]
            
            # Compute value for each action
            action_values = []
            for a in actions:
                # Get next state (might be same state if hit wall)
                s_next = get_next_state(maze, s, a)
                
                # Get immediate reward
                reward = get_reward(maze, s, a)
                
                # Q(s,a) = R(s,a) + γ * V(s')
                q_value = reward + gamma * V[s_next]
                action_values.append(q_value)
            
            # V(s) = max_a Q(s,a)
            V[s] = max(action_values)
            
            # Track maximum change
            delta = max(delta, abs(v_old - V[s]))
        
        iterations += 1
        
        # Check convergence
        if delta < theta:
            break
        
        # Safety check
        if iterations > 10000:
            print("Warning: Max iterations reached, stopping early")
            break
    
    # Extract optimal policy
    policy = {}
    for s in states:
        if s == maze.goal:
            policy[s] = None  # No action needed at goal
            continue
        
        # Find action with highest Q-value
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
    
    print(f"Value Iteration converged in {iterations} iterations")

    # policy: Dict mapping state -> best action
    # V: Dict mapping state -> value
    
    return policy, V

# Follow the policy from start to goal to extract the path.
def extract_path(maze: MazeData, policy: Dict[Coord, str], max_steps: int = 1000) -> List[Coord]:

    path = [maze.start]
    current = maze.start
    visited = {maze.start}
    
    steps = 0
    while current != maze.goal and steps < max_steps:
        action = policy.get(current)
        if action is None:
            break
        
        # Take action
        next_state = get_next_state(maze, current, action)
        
        # Check for loops
        if next_state == current:
            # Stuck (all actions lead to walls)
            break
        
        if next_state in visited:
            print("Warning: Cycle detected in policy")
            break
        
        path.append(next_state)
        visited.add(next_state)
        current = next_state
        steps += 1
    
    return path if current == maze.goal else []


def prettify_policy(policy: Dict[Coord, Optional[str]], grid: List[List[int]]) -> str:
    rows, cols = len(grid), len(grid[0])
    
    arrow_map = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
        None: "G",
    }
    
    lines = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            if grid[r][c] == 1:  # Wall
                line += "█"
            else:
                action = policy.get((r, c))
                line += arrow_map.get(action, " ")
        lines.append(line)
    
    return "\n".join(lines)

# Print a sample of state values to understand the value function.
def print_value_heatmap(V: Dict[Coord, float], grid: List[List[int]], num_samples: int = 10):
    print("\nSample State Values (sorted by value):")
    print(f"{'State':<15} | {'Value':<15} | {'Description'}")
    print("-" * 50)
    
    # Sort by value
    sorted_states = sorted(V.items(), key=lambda x: x[1], reverse=True)
    
    # Print top values
    for state, value in sorted_states[:num_samples]:
        desc = "GOAL" if value == goal_reward else ""
        print(f"{str(state):<15} | {value:<15.2f} | {desc}")


# Example usage
# if __name__ == '__main__':
#     # Load a maze
#     maze = load_maze("/Users/mariameshi/Documents/year_5/AI/mazes/maze_19x19_seed42.json")

#     print(f"Maze size: {len(maze.grid)}x{len(maze.grid[0])}")
#     print(f"Start: {maze.start}")
#     print(f"Goal: {maze.goal}")
#     print(f"\nReward Structure:")
#     print(f"  Normal direction_deltas: {step_cost}")
#     print(f"  Goal: {goal_reward}")
#     print()
    
#     # Run Value Iteration
#     policy, value_function = value_iteration_with_walls(maze, gamma=gamma, theta=theta)
    
#     # Extract path
#     path = extract_path(maze, policy)
    
#     print(f"\nResults:")
#     print(f"  Path length: {len(path)} steps")
#     print(f"  Start value: {value_function[maze.start]:.2f}")
#     print(f"  Goal value: {value_function[maze.goal]:.2f}")
    
#     # Calculate expected total reward if following policy
#     expected_reward = value_function[maze.start]
#     print(f"  Expected total reward: {expected_reward:.2f}")
    
#     print("\nOptimal Policy Visualization:")
#     print(prettify_policy(policy, maze.grid))
    
#     # Show value samples
#     print_value_heatmap(value_function, maze.grid)
    
#     # Path preview
#     if path:
#         print(f"\nPath (first 10 steps): {path[:10]}")
#     else:
#         print("\nWarning: No valid path found!")