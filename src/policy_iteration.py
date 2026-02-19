from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.create_maze import MazeData, Coord, load_maze

# ─────────────────────────────────────────────
#  Algorithm parameters
# ─────────────────────────────────────────────
gamma     = 0.9    # Discount factor:  how much future rewards are worth vs immediate ones
                   #   → 0.9 means a reward 10 steps away is worth 0.9^10 ≈ 35% of its face value
                   #   → higher γ = cares more about the future = slower convergence
theta     = 1e-4   # Convergence threshold: stop inner evaluation when max state change < theta
                   #   → 1e-4 is looser than 1e-6; path quality is identical but converges faster
K_EVAL    = 10     # Modified policy iteration: max inner evaluation steps per outer iteration
                   #   → Instead of running evaluation to full convergence (pure policy iteration)
                   #     we do at most K steps. K=1 is equivalent to value iteration.
                   #     K=∞ is pure policy iteration. K=10 is a good sweet spot.

# ─────────────────────────────────────────────
#  Rewards & actions
# ─────────────────────────────────────────────
step_cost   = -0.04
goal_reward =  1.0

actions = ["up", "down", "left", "right"]
action_deltas = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}


# ─────────────────────────────────────────────
#  Result container  (same shape as before)
# ─────────────────────────────────────────────
@dataclass
class ValueIterationResult:
    policy:         Dict[Coord, str]
    values:         Dict[Coord, float]
    path:           List[Coord]
    path_length:    int
    nodes_explored: int
    iterations:     int


# ─────────────────────────────────────────────
#  Helpers  (unchanged from your original)
# ─────────────────────────────────────────────
def get_states(maze: MazeData) -> List[Coord]:
    return [
        (r, c)
        for r in range(len(maze.grid))
        for c in range(len(maze.grid[0]))
        if maze.grid[r][c] == 0
    ]


def get_next_state(maze: MazeData, s: Coord, a: str) -> Coord:
    dr, dc = action_deltas[a]
    nr, nc = s[0] + dr, s[1] + dc
    rows, cols = len(maze.grid), len(maze.grid[0])
    if 0 <= nr < rows and 0 <= nc < cols and maze.grid[nr][nc] == 0:
        return (nr, nc)
    return s  # bump into wall → stay


def get_reward(maze: MazeData, s: Coord, a: str) -> float:
    return goal_reward if get_next_state(maze, s, a) == maze.goal else step_cost


# ─────────────────────────────────────────────
#  Heuristic initial policy
#  Point every state roughly toward the goal
#  using Manhattan-distance direction.
#  This starts us much closer to optimal so
#  policy improvement needs fewer rounds.
# ─────────────────────────────────────────────
def heuristic_policy(maze: MazeData, states: List[Coord]) -> Dict[Coord, Optional[str]]:
    """
    For each state, pick the action that moves us closest to the goal
    (Manhattan distance). Walls are ignored here — this is just a warm start.
    """
    gr, gc = maze.goal
    policy: Dict[Coord, Optional[str]] = {}

    for s in states:
        if s == maze.goal:
            policy[s] = None
            continue

        sr, sc = s
        dr = gr - sr   # positive  → goal is below
        dc = gc - sc   # positive  → goal is to the right

        # Pick the dominant axis first; break ties by the other axis
        if abs(dr) >= abs(dc):
            policy[s] = "down" if dr >= 0 else "up"
        else:
            policy[s] = "right" if dc >= 0 else "left"

    return policy


# ─────────────────────────────────────────────
#  Q-value helper  (avoids repeated inline code)
# ─────────────────────────────────────────────
def q_value(maze: MazeData, V: Dict[Coord, float], s: Coord, a: str) -> float:
    """Q(s, a) = R(s, a) + γ · V(s')"""
    s_next = get_next_state(maze, s, a)
    return get_reward(maze, s, a) + gamma * V[s_next]


def best_action(maze: MazeData, V: Dict[Coord, float], s: Coord) -> str:
    """Return the greedy-best action from state s given values V."""
    return max(actions, key=lambda a: q_value(maze, V, s, a))


# ─────────────────────────────────────────────
#  Modified Policy Iteration
#
#  Key ideas vs your original:
#   1. Heuristic warm-start  → fewer outer iterations
#   2. Modified evaluation   → run at most K inner steps instead of full
#      convergence; avoids wasting time evaluating a policy we're about
#      to change anyway
#   3. Single-pass improvement check  → detect stability while improving,
#      no second loop needed
# ─────────────────────────────────────────────
def policy_iteration(maze: MazeData, gamma: float = gamma, theta: float = theta) -> ValueIterationResult:

    states = get_states(maze)

    # ── 1. Initialisation ────────────────────
    # Heuristic policy: points toward goal rather than blindly "down"
    policy = heuristic_policy(maze, states)

    # Values start at 0 everywhere; goal is fixed at +1
    V: Dict[Coord, float] = {s: 0.0 for s in states}
    V[maze.goal] = goal_reward

    outer_iterations  = 0   # counts full policy-improvement rounds
    total_eval_sweeps = 0   # counts individual evaluation sweeps (diagnostic)

    # ── 2. Policy Iteration loop ─────────────
    while True:
        # ── 2a. Modified Policy Evaluation ───
        # Run at most K_EVAL sweeps of Bellman evaluation for the current policy.
        # We don't need exact V^π — just a good enough estimate to improve on.
        for _ in range(K_EVAL):
            delta = 0.0

            for s in states:
                if s == maze.goal:
                    continue

                v_old = V[s]
                a     = policy[s]

                # Bellman evaluation: V(s) ← R(s,π(s)) + γ·V(s')
                V[s]  = q_value(maze, V, s, a)

                delta = max(delta, abs(v_old - V[s]))

            total_eval_sweeps += 1

            # Early exit if values already converged within K sweeps
            if delta < theta:
                break

        # ── 2b. Policy Improvement ───────────
        # Update the policy greedily using the (approximate) values above.
        # Track whether anything changed.
        policy_stable = True

        for s in states:
            if s == maze.goal:
                continue

            old_action = policy[s]
            new_action = best_action(maze, V, s)
            policy[s]  = new_action

            if new_action != old_action:
                policy_stable = False   # at least one state changed → keep going

        outer_iterations += 1

        # ── 2c. Convergence check ─────────────
        if policy_stable:
            # Policy didn't change at all → we're at the optimum
            break

        if outer_iterations > 1000:
            print("Warning: max outer iterations reached")
            break

    print(
        f"Policy Iteration converged in {outer_iterations} outer iterations "
        f"({total_eval_sweeps} evaluation sweeps total, K_EVAL={K_EVAL})"
    )

    path = extract_path(maze, policy)

    return ValueIterationResult(
        policy         = policy,
        values         = V,
        path           = path,
        path_length    = len(path),
        nodes_explored = len(V),
        iterations     = outer_iterations,
    )


# ─────────────────────────────────────────────
#  Path extraction  (unchanged)
# ─────────────────────────────────────────────
def extract_path(maze: MazeData, policy: Dict[Coord, str], max_steps: int = 1000) -> List[Coord]:
    path    = [maze.start]
    current = maze.start
    visited = {maze.start}

    while current != maze.goal and len(path) < max_steps:
        action = policy.get(current)
        if action is None:
            break

        next_state = get_next_state(maze, current, action)

        if next_state == current:
            break  # stuck against a wall

        if next_state in visited:
            print("Cycle detected in policy — maze may be unsolvable with current params")
            break

        path.append(next_state)
        visited.add(next_state)
        current = next_state

    return path if current == maze.goal else []


# ─────────────────────────────────────────────
#  Visualisation helpers  (unchanged)
# ─────────────────────────────────────────────
def draw_policy(policy: Dict[Coord, Optional[str]], grid: List[List[int]]) -> str:
    arrow_map = {"up": "↑", "down": "↓", "left": "←", "right": "→", None: "G"}
    rows, cols = len(grid), len(grid[0])
    lines = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            if grid[r][c] == 1:
                line += "█"
            else:
                line += arrow_map.get(policy.get((r, c)), " ")
        lines.append(line)
    return "\n".join(lines)


def print_value_heatmap(V: Dict[Coord, float], grid: List[List[int]], num_samples: int = 10):
    print("\nSample State Values (sorted by value):")
    print(f"{'State':<15} | {'Value':<15} | {'Description'}")
    print("-" * 50)
    for state, value in sorted(V.items(), key=lambda x: x[1], reverse=True)[:num_samples]:
        desc = "GOAL" if value == goal_reward else ""
        print(f"{str(state):<15} | {value:<15.2f} | {desc}")