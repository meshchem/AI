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
    delta_history: List[float] = None


# Algorithm parameters
discount_factor  = 0.9      # determines how much the agent prioritizes future rewards over immediate reward
theta  = 0.0001              # convergence threshold
k_eval = 20                 # max inner evaluation sweeps per outer iteration
                            
# Rewards
step_cost = -0.04     
goal_reward = 1      

# actions 
actions = ["up", "down", "left", "right"]
directions = {
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
    dr, dc = directions[a]
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
    
    # Normal directions costs
    return step_cost 

def get_reward(maze: MazeData, s: Coord, a: str, step_cost=step_cost) -> float:
    
    next_state = get_next_state(maze, s, a)
    # Check if goal was reached
    if next_state == maze.goal:
        return goal_reward
    
     # Normal directions costs
    return step_cost


def value_iteration(maze: MazeData, discount_factor=discount_factor, theta=theta, step_cost=step_cost) -> ValueIterationResult:

    states = get_states(maze)
    
    V = {s: 0.0 for s in states}
    V[maze.goal] = goal_reward
    
    iterations = 0
    delta_history = []
    
    while True:
        delta = 0
        
        for s in states:
            if s == maze.goal:
                continue
            
            v_old = V[s]
            
            action_values = []
            for a in actions:
                s_next = get_next_state(maze, s, a)
                reward = get_reward(maze, s, a)
                q_value = reward + discount_factor * V[s_next]
                action_values.append(q_value)
            
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        
        delta_history.append(delta)  # track delta each iteration
        iterations += 1
        
        if delta < theta:
            break
        
        if iterations > 10000:
            print("Max iterations reached!")
            break
    
    policy = {}
    for s in states:
        if s == maze.goal:
            policy[s] = None
            continue
        
        best_action = None
        best_value = float('-inf')
        
        for a in actions:
            s_next = get_next_state(maze, s, a)
            reward = get_reward(maze, s, a)
            q_value = reward + discount_factor * V[s_next]
            
            if q_value > best_value:
                best_value = q_value
                best_action = a
        
        policy[s] = best_action
    
    # print(f"Value Iteration converged in {iterations} iterations")

    path = extract_path(maze, policy)
    
    return ValueIterationResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        nodes_explored=len(V),
        iterations=iterations,
        delta_history=delta_history,
    )

def policy_iteration(maze: MazeData, discount_factor=discount_factor, theta=theta, k_eval=k_eval, step_cost=step_cost) -> ValueIterationResult:
   
    states = get_states(maze)
    
    policy = {s: "down" for s in states}
    policy[maze.goal] = None
    
    V = {s: 0.0 for s in states}
    V[maze.goal] = goal_reward
    
    iterations = 0
    delta_history = []
    
    while True:
        # Policy Evaluation
        eval_delta = 0
        for _ in range(k_eval):
            delta = 0
            
            for s in states:
                if s == maze.goal:
                    continue
                
                v_old = V[s]
                action = policy[s]
                s_next = get_next_state(maze, s, action)
                reward = get_reward(maze, s, action)
                V[s] = reward + discount_factor * V[s_next]
                delta = max(delta, abs(v_old - V[s]))
                eval_delta = max(eval_delta, delta)
            
            if delta < theta:
                break
        
        delta_history.append(eval_delta)  # track max delta across evaluation passes
        
        # Policy Improvement
        policy_stable = True
        
        for s in states:
            if s == maze.goal:
                continue
            
            old_action = policy[s]
            
            best_action = None
            best_value = float('-inf')
            
            for a in actions:
                s_next = get_next_state(maze, s, a)
                reward = get_reward(maze, s, a)
                q_value = reward + discount_factor * V[s_next]
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            
            policy[s] = best_action
            
            if best_action != old_action:
                policy_stable = False
        
        iterations += 1
        
        if policy_stable:
            break
        
        if iterations > 1000:
            print("Max iterations reached!")
            break
    
    # print(f"Policy Iteration converged in {iterations} iterations")
    
    path = extract_path(maze, policy)
    
    return ValueIterationResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        nodes_explored=len(V),
        iterations=iterations,
        delta_history=delta_history,
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
            print("Cycle detected")
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