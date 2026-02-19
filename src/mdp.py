from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.create_maze import MazeData, Coord, load_maze

@dataclass
class ValueIterationResult:
    policy: Dict[Coord, str]
    values: Dict[Coord, float]
    path: List[Coord]
    path_length: int
    nodes_explored: int
    iterations: int


# Algorithm parameters
gamma  = 0.9    # discount factor
theta  = 1e-4   # convergence threshold — loose enough to be fast, tight enough for optimal paths
K_EVAL = 10     # modified policy iteration: max inner evaluation sweeps per outer iteration
                # K=1 behaves like value iteration, K=∞ is pure policy iteration, 10 is a good sweet spot

# Rewards
step_cost = -0.04     
goal_reward = 1      

# actions 
actions = ["up", "down", "left", "right"]
action_deltas = {
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
    dr, dc = action_deltas[a]
    next_r, next_c = s[0] + dr, s[1] + dc
    
    rows, cols = len(maze.grid), len(maze.grid[0])
    
    # Check if next position is valid (no a wall)
    if 0 <= next_r < rows and 0 <= next_c < cols and maze.grid[next_r][next_c] == 0:
        return (next_r, next_c)
    else:
        return s

# 
def get_reward(maze: MazeData, s: Coord, a: str) -> float:

    next_state = get_next_state(maze, s, a)
    
    # Check if goal was reached
    if next_state == maze.goal:
        return goal_reward  # +1
    
    # Normal action_deltas costs
    return step_cost 


# maze: MazeData object containing grid, start, goal
# gamma: discount factor (default 0.9)
# theta: convergence threshold (default 1e-6)
def value_iteration(maze: MazeData, gamma=gamma, theta=theta) -> Tuple[Dict[Coord, str], Dict[Coord, float]]:

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
                # final state, no actions needed
                continue
            
            v_old = V[s]
            
            # Compute value for each action
            action_values = []
            for a in actions:
                # Get next state
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
        
        # Iteration check
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

    path = extract_path(maze, policy)
    
    return ValueIterationResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        nodes_explored=len(V),
        iterations=iterations,
    )


def policy_iteration(maze: MazeData, gamma=gamma, theta=theta) -> ValueIterationResult:
   
    # Get all states
    states = get_states(maze)
    
    # Initialize policy (everyone goes down initially)
    policy = {s: "down" for s in states}
    policy[maze.goal] = None
    
    # Initialize value function
    V = {s: 0.0 for s in states}
    V[maze.goal] = goal_reward
    
    iterations = 0
    
    # Policy Iteration: alternate evaluation and improvement
    while True:
        # Policy Evaluation
        for _ in range(K_EVAL):
            delta = 0
            
            for s in states:
                if s == maze.goal:
                    continue
                
                v_old = V[s]
                
                # Follow current policy
                action = policy[s]
                s_next = get_next_state(maze, s, action)
                reward = get_reward(maze, s, action)
                
                # V^π(s) = R(s,π(s)) + γ * V^π(s')
                V[s] = reward + gamma * V[s_next]
                
                delta = max(delta, abs(v_old - V[s]))
            
            # Early exit if values already converged within K_EVAL sweeps
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        
        for s in states:
            if s == maze.goal:
                continue
            
            old_action = policy[s]
            
            # best action according to current values
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
        
        iterations += 1
        
        # Check if policy converged
        if policy_stable:
            break
        
        # Iteration check
        if iterations > 1000:
            print("Warning: Max iterations reached, stopping early")
            break
    
    print(f"Policy Iteration converged in {iterations} iterations")
    
    path = extract_path(maze, policy)
    
    return ValueIterationResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        nodes_explored=len(V),
        iterations=iterations,
    )

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
            print("Cycle detected in policy")
            break
        
        path.append(next_state)
        visited.add(next_state)
        current = next_state
        steps += 1
    
    return path if current == maze.goal else []


def draw_policy(policy: Dict[Coord, Optional[str]], grid: List[List[int]]) -> str:
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