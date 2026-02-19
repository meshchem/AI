from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.create_maze import MazeData, Coord

# ---------------------------------------------------------------------------
# Constants (defaults - can be overridden in function calls)
# ---------------------------------------------------------------------------

DEFAULT_GAMMA = 0.99          # discount factor
DEFAULT_LIVING_REWARD = -0.04 # reward at each non-terminal step
DEFAULT_GOAL_REWARD = 1.0     # reward on reaching the goal
DEFAULT_THETA = 1e-6          # convergence threshold

# Default transition probabilities - DETERMINISTIC
# Agent always moves in intended direction (no drift/noise)
DEFAULT_PROB_STRAIGHT = 1.0
DEFAULT_PROB_LEFT = 0.0
DEFAULT_PROB_RIGHT = 0.0

# Cardinal directions: (dr, dc)
ACTIONS = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
}

# For each intended action, the relative drift directions
# (straight, left, right) with probabilities (0.8, 0.1, 0.1)
# "left" and "right" are relative to the intended direction of travel
_LEFT_OF = {
    "UP":    "LEFT",
    "LEFT":  "DOWN",
    "DOWN":  "RIGHT",
    "RIGHT": "UP",
}
_RIGHT_OF = {
    "UP":    "RIGHT",
    "RIGHT": "DOWN",
    "DOWN":  "LEFT",
    "LEFT":  "UP",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MDPResult:
    policy: Dict[Coord, Optional[str]]  # best action at each free cell
    values: Dict[Coord, float]          # V(s) for each free cell
    path: List[Coord]                   # path from start to goal via policy
    path_length: int                    # number of steps in path
    iterations: int                     # iterations until convergence
    nodes_explored: int                 # free cells considered (all of them)


# ---------------------------------------------------------------------------
# Shared MDP helpers
# ---------------------------------------------------------------------------

def _free_cells(grid: List[List[int]]) -> List[Coord]:
    """Return all non-wall cells."""
    return [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == 0
    ]


def _transition(
    grid: List[List[int]],
    state: Coord,
    action: str,
    prob_straight: float = DEFAULT_PROB_STRAIGHT,
    prob_left: float = DEFAULT_PROB_LEFT,
    prob_right: float = DEFAULT_PROB_RIGHT,
) -> List[Tuple[float, Coord]]:
    """
    Return a list of (probability, next_state) pairs for taking
    `action` from `state` under the stochastic transition model.

    If the agent would move into a wall, it stays in place.
    
    Parameters:
        prob_straight: probability of moving in intended direction (default 0.8)
        prob_left: probability of drifting left (default 0.1)
        prob_right: probability of drifting right (default 0.1)
    """
    rows, cols = len(grid), len(grid[0])
    outcomes = []

    for prob, direction in [
        (prob_straight, action),
        (prob_left, _LEFT_OF[action]),
        (prob_right, _RIGHT_OF[action]),
    ]:
        dr, dc = ACTIONS[direction]
        nr, nc = state[0] + dr, state[1] + dc

        # Stay in place if the move goes into a wall or out of bounds
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            next_state = (nr, nc)
        else:
            next_state = state

        outcomes.append((prob, next_state))

    return outcomes


def _extract_path(
    policy: Dict[Coord, Optional[str]],
    start: Coord,
    goal: Coord,
    max_steps: int = 10_000,
) -> List[Coord]:
    """
    Follow the greedy policy from start to goal.
    Returns an empty list if the goal is unreachable within max_steps.
    """
    path = [start]
    current = start
    visited = {start}

    while current != goal and len(path) < max_steps:
        action = policy.get(current)
        if action is None:
            break
        dr, dc = ACTIONS[action]
        next_cell = (current[0] + dr, current[1] + dc)

        if next_cell in visited:
            break  # caught in a loop — policy failed to reach goal

        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell

    return path if current == goal else []


def _bellman_value(
    state: Coord,
    action: str,
    grid: List[List[int]],
    goal: Coord,
    V: Dict[Coord, float],
    gamma: float = DEFAULT_GAMMA,
    living_reward: float = DEFAULT_LIVING_REWARD,
    goal_reward: float = DEFAULT_GOAL_REWARD,
    prob_straight: float = DEFAULT_PROB_STRAIGHT,
    prob_left: float = DEFAULT_PROB_LEFT,
    prob_right: float = DEFAULT_PROB_RIGHT,
) -> float:
    """
    Compute the Bellman expected value for one (state, action) pair.

    Q(s, a) = Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]
    """
    q = 0.0
    for prob, next_state in _transition(grid, state, action, prob_straight, prob_left, prob_right):
        if next_state == goal:
            reward = goal_reward
        else:
            reward = living_reward
        q += prob * (reward + gamma * V[next_state])
    return q


# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(
    maze: MazeData,
    gamma: float = DEFAULT_GAMMA,
    living_reward: float = DEFAULT_LIVING_REWARD,
    goal_reward: float = DEFAULT_GOAL_REWARD,
    theta: float = DEFAULT_THETA,
    prob_straight: float = DEFAULT_PROB_STRAIGHT,
    prob_left: float = DEFAULT_PROB_LEFT,
    prob_right: float = DEFAULT_PROB_RIGHT,
) -> MDPResult:
    """
    Value Iteration for MDP maze solving.

    Repeatedly applies the Bellman optimality update to every free cell
    until values converge (max change < theta):

        V(s) ← max_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]

    Once converged, extract the greedy policy:

        π(s) = argmax_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]

    Parameters:
        gamma: discount factor (default 0.99)
        living_reward: reward for non-terminal steps (default -0.04)
        goal_reward: reward at goal (default 1.0)
        theta: convergence threshold (default 1e-6)
        prob_straight: P(intended direction) (default 0.8)
        prob_left: P(drift left) (default 0.1)
        prob_right: P(drift right) (default 0.1)

    Time complexity  : O(iterations · |S| · |A|)
    Space complexity : O(|S|)
    """
    grid = maze.grid
    goal = maze.goal
    free = _free_cells(grid)

    # Initialise all state values to 0
    V: Dict[Coord, float] = {s: 0.0 for s in free}
    V[goal] = goal_reward  # terminal state has fixed value

    iterations = 0

    while True:
        delta = 0.0
        new_V = V.copy()

        for state in free:
            if state == goal:
                continue  # terminal — value stays fixed

            best = max(
                _bellman_value(state, action, grid, goal, V, gamma, living_reward, goal_reward,
                             prob_straight, prob_left, prob_right)
                for action in ACTIONS
            )

            delta = max(delta, abs(best - V[state]))
            new_V[state] = best

        V = new_V
        iterations += 1

        if delta < theta:
            break

    # Extract greedy policy from converged values
    policy: Dict[Coord, Optional[str]] = {}
    for state in free:
        if state == goal:
            policy[state] = None
            continue
        policy[state] = max(
            ACTIONS,
            key=lambda a: _bellman_value(state, a, grid, goal, V, gamma, living_reward, goal_reward,
                                        prob_straight, prob_left, prob_right)
        )

    path = _extract_path(policy, maze.start, goal)

    return MDPResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        iterations=iterations,
        nodes_explored=len(free),
    )


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------

def policy_iteration(
    maze: MazeData,
    gamma: float = DEFAULT_GAMMA,
    living_reward: float = DEFAULT_LIVING_REWARD,
    goal_reward: float = DEFAULT_GOAL_REWARD,
    theta: float = DEFAULT_THETA,
    prob_straight: float = DEFAULT_PROB_STRAIGHT,
    prob_left: float = DEFAULT_PROB_LEFT,
    prob_right: float = DEFAULT_PROB_RIGHT,
) -> MDPResult:
    """
    Policy Iteration for MDP maze solving.

    Alternates between two steps until the policy stops changing:

        1. Policy Evaluation — compute V^π by solving the linear system:
               V^π(s) = Σ P(s'|s,π(s)) · [ R(s,a,s') + γ · V^π(s') ]
           (solved iteratively here rather than via matrix inversion)

        2. Policy Improvement — update policy greedily w.r.t. V^π:
               π(s) ← argmax_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V^π(s') ]

    Policy iteration typically converges in far fewer iterations than
    value iteration, though each iteration is more expensive.

    Parameters:
        gamma: discount factor (default 0.99)
        living_reward: reward for non-terminal steps (default -0.04)
        goal_reward: reward at goal (default 1.0)
        theta: convergence threshold (default 1e-6)
        prob_straight: P(intended direction) (default 0.8)
        prob_left: P(drift left) (default 0.1)
        prob_right: P(drift right) (default 0.1)

    Time complexity  : O(iterations · |S| · |A|)
    Space complexity : O(|S|)
    """
    grid = maze.grid
    goal = maze.goal
    free = _free_cells(grid)

    # Initialise with a uniform policy (everyone goes UP to start)
    policy: Dict[Coord, Optional[str]] = {
        s: "UP" for s in free
    }
    policy[goal] = None

    V: Dict[Coord, float] = {s: 0.0 for s in free}
    V[goal] = goal_reward

    iterations = 0

    while True:
        # --- Step 1: Policy Evaluation ---
        # Iteratively compute V^π until it converges
        while True:
            delta = 0.0
            new_V = V.copy()

            for state in free:
                if state == goal:
                    continue
                action = policy[state]
                v = _bellman_value(state, action, grid, goal, V, gamma, living_reward, goal_reward,
                                 prob_straight, prob_left, prob_right)
                delta = max(delta, abs(v - V[state]))
                new_V[state] = v

            V = new_V
            if delta < theta:
                break

        # --- Step 2: Policy Improvement ---
        policy_stable = True

        for state in free:
            if state == goal:
                continue

            old_action = policy[state]
            best_action = max(
                ACTIONS,
                key=lambda a: _bellman_value(state, a, grid, goal, V, gamma, living_reward, goal_reward,
                                            prob_straight, prob_left, prob_right)
            )

            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False

        iterations += 1

        if policy_stable:
            break

    path = _extract_path(policy, maze.start, goal)

    return MDPResult(
        policy=policy,
        values=V,
        path=path,
        path_length=len(path),
        iterations=iterations,
        nodes_explored=len(free),
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def prettify_policy(policy: Dict[Coord, Optional[str]], grid: List[List[int]]) -> str:
    """
    Create a visual representation of the policy using arrow symbols.
    
    Returns a multi-line string showing the policy as a grid with:
        ↑ ↓ ← → for directional moves
        G for goal
        █ for walls
    """
    rows, cols = len(grid), len(grid[0])
    
    arrow_map = {
        "UP": "↑",
        "DOWN": "↓",
        "LEFT": "←",
        "RIGHT": "→",
        None: "G",  # Goal
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



# """
# MDP-based maze solving using Value Iteration and Policy Iteration.

# Models the maze as a Markov Decision Process with stochastic transitions:
#     - 80% chance of moving in the intended direction
#     - 10% chance of drifting left (relative to intended direction)
#     - 10% chance of drifting right (relative to intended direction)
#     -  0% chance of moving backward

# Rewards:
#     - +1.0  on reaching the goal
#     - LIVING_REWARD at every non-terminal step (default -0.04)

# Discount factor γ = 0.99 (agent cares strongly about future rewards).
# """

# from dataclasses import dataclass
# from typing import List, Tuple, Dict, Optional
# import numpy as np

# from src.create_maze import MazeData, Coord

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------

# GAMMA = 0.99          # discount factor
# LIVING_REWARD = -0.04 # reward at each non-terminal step
# GOAL_REWARD = 1.0     # reward on reaching the goal
# THETA = 1e-6          # convergence threshold for value iteration

# # Cardinal directions: (dr, dc)
# ACTIONS = {
#     "UP":    (-1,  0),
#     "DOWN":  ( 1,  0),
#     "LEFT":  ( 0, -1),
#     "RIGHT": ( 0,  1),
# }

# # For each intended action, the relative drift directions
# # (straight, left, right) with probabilities (0.8, 0.1, 0.1)
# # "left" and "right" are relative to the intended direction of travel
# _LEFT_OF = {
#     "UP":    "LEFT",
#     "LEFT":  "DOWN",
#     "DOWN":  "RIGHT",
#     "RIGHT": "UP",
# }
# _RIGHT_OF = {
#     "UP":    "RIGHT",
#     "RIGHT": "DOWN",
#     "DOWN":  "LEFT",
#     "LEFT":  "UP",
# }


# # ---------------------------------------------------------------------------
# # Result dataclass
# # ---------------------------------------------------------------------------

# @dataclass
# class MDPResult:
#     policy: Dict[Coord, Optional[str]]  # best action at each free cell
#     values: Dict[Coord, float]          # V(s) for each free cell
#     path: List[Coord]                   # path from start to goal via policy
#     path_length: int                    # number of steps in path
#     iterations: int                     # iterations until convergence
#     nodes_explored: int                 # free cells considered (all of them)

# # ---------------------------------------------------------------------------
# # Shared MDP helpers
# # ---------------------------------------------------------------------------

# def _free_cells(grid: List[List[int]]) -> List[Coord]:
#     return [
#         (r, c)
#         for r in range(len(grid))
#         for c in range(len(grid[0]))
#         if grid[r][c] == 0
#     ]


# def _transition(
#     grid: List[List[int]],
#     state: Coord,
#     action: str,
# ) -> List[Tuple[float, Coord]]:
#     rows, cols = len(grid), len(grid[0])
#     outcomes = []

#     for prob, direction in [
#         (0.8, action),
#         (0.1, _LEFT_OF[action]),
#         (0.1, _RIGHT_OF[action]),
#     ]:
#         dr, dc = ACTIONS[direction]
#         nr, nc = state[0] + dr, state[1] + dc

#         # Stay in place if the move goes into a wall or out of bounds
#         if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
#             next_state = (nr, nc)
#         else:
#             next_state = state

#         outcomes.append((prob, next_state))

#     return outcomes


# def _extract_path(
#     policy: Dict[Coord, Optional[str]],
#     start: Coord,
#     goal: Coord,
#     max_steps: int = 10_000,
# ) -> List[Coord]:
#     """
#     Follow the greedy policy from start to goal.
#     Returns an empty list if the goal is unreachable within max_steps.
#     """
#     path = [start]
#     current = start
#     visited = {start}

#     while current != goal and len(path) < max_steps:
#         action = policy.get(current)
#         if action is None:
#             break
#         dr, dc = ACTIONS[action]
#         next_cell = (current[0] + dr, current[1] + dc)

#         if next_cell in visited:
#             break  # caught in a loop — policy failed to reach goal

#         path.append(next_cell)
#         visited.add(next_cell)
#         current = next_cell

#     return path if current == goal else []


# def _bellman_value(
#     state: Coord,
#     action: str,
#     grid: List[List[int]],
#     goal: Coord,
#     V: Dict[Coord, float],
# ) -> float:
#     """
#     Compute the Bellman expected value for one (state, action) pair.

#     Q(s, a) = Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]
#     """
#     q = 0.0
#     for prob, next_state in _transition(grid, state, action):
#         if next_state == goal:
#             reward = GOAL_REWARD
#         else:
#             reward = LIVING_REWARD
#         q += prob * (reward + GAMMA * V[next_state])
#     return q


# # ---------------------------------------------------------------------------
# # Value Iteration
# # ---------------------------------------------------------------------------

# def value_iteration(maze: MazeData) -> MDPResult:
#     """
#     Value Iteration for MDP maze solving.

#     Repeatedly applies the Bellman optimality update to every free cell
#     until values converge (max change < THETA):

#         V(s) ← max_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]

#     Once converged, extract the greedy policy:

#         π(s) = argmax_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V(s') ]

#     Convergence is guaranteed for finite MDPs with γ < 1.

#     Time complexity  : O(iterations · |S| · |A|)
#     Space complexity : O(|S|)
#     """
#     grid = maze.grid
#     goal = maze.goal
#     free = _free_cells(grid)

#     # Initialise all state values to 0
#     V: Dict[Coord, float] = {s: 0.0 for s in free}
#     V[goal] = GOAL_REWARD  # terminal state has fixed value

#     iterations = 0

#     while True:
#         delta = 0.0
#         new_V = V.copy()

#         for state in free:
#             if state == goal:
#                 continue  # terminal — value stays fixed

#             best = max(
#                 _bellman_value(state, action, grid, goal, V)
#                 for action in ACTIONS
#             )

#             delta = max(delta, abs(best - V[state]))
#             new_V[state] = best

#         V = new_V
#         iterations += 1

#         if delta < THETA:
#             break

#     # Extract greedy policy from converged values
#     policy: Dict[Coord, Optional[str]] = {}
#     for state in free:
#         if state == goal:
#             policy[state] = None
#             continue
#         policy[state] = max(
#             ACTIONS,
#             key=lambda a: _bellman_value(state, a, grid, goal, V)
#         )

#     path = _extract_path(policy, maze.start, goal)

#     return MDPResult(
#         policy=policy,
#         values=V,
#         path=path,
#         path_length=len(path),
#         iterations=iterations,
#         nodes_explored=len(free),
#     )


# # ---------------------------------------------------------------------------
# # Policy Iteration
# # ---------------------------------------------------------------------------

# def policy_iteration(maze: MazeData) -> MDPResult:
#     """
#     Policy Iteration for MDP maze solving.

#     Alternates between two steps until the policy stops changing:

#         1. Policy Evaluation — compute V^π by solving the linear system:
#                V^π(s) = Σ P(s'|s,π(s)) · [ R(s,a,s') + γ · V^π(s') ]
#            (solved iteratively here rather than via matrix inversion)

#         2. Policy Improvement — update policy greedily w.r.t. V^π:
#                π(s) ← argmax_a Σ P(s'|s,a) · [ R(s,a,s') + γ · V^π(s') ]

#     Policy iteration typically converges in far fewer iterations than
#     value iteration, though each iteration is more expensive.

#     Time complexity  : O(iterations · |S| · |A|)
#     Space complexity : O(|S|)
#     """
#     grid = maze.grid
#     goal = maze.goal
#     free = _free_cells(grid)

#     # Initialise with a uniform policy (everyone goes UP to start)
#     policy: Dict[Coord, Optional[str]] = {
#         s: "UP" for s in free
#     }
#     policy[goal] = None

#     V: Dict[Coord, float] = {s: 0.0 for s in free}
#     V[goal] = GOAL_REWARD

#     iterations = 0

#     while True:
#         # --- Step 1: Policy Evaluation ---
#         # Iteratively compute V^π until it converges
#         while True:
#             delta = 0.0
#             new_V = V.copy()

#             for state in free:
#                 if state == goal:
#                     continue
#                 action = policy[state]
#                 v = _bellman_value(state, action, grid, goal, V)
#                 delta = max(delta, abs(v - V[state]))
#                 new_V[state] = v

#             V = new_V
#             if delta < THETA:
#                 break

#         # --- Step 2: Policy Improvement ---
#         policy_stable = True

#         for state in free:
#             if state == goal:
#                 continue

#             old_action = policy[state]
#             best_action = max(
#                 ACTIONS,
#                 key=lambda a: _bellman_value(state, a, grid, goal, V)
#             )

#             policy[state] = best_action
#             if best_action != old_action:
#                 policy_stable = False

#         iterations += 1

#         if policy_stable:
#             break

#     path = _extract_path(policy, maze.start, goal)

#     return MDPResult(
#         policy=policy,
#         values=V,
#         path=path,
#         path_length=len(path),
#         iterations=iterations,
#         nodes_explored=len(free),
#     )