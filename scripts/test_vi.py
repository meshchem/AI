from src.create_maze import load_maze
from src.mdp_algorithms import value_iteration, prettify_policy
import matplotlib.pyplot as plt
import numpy as np



# """
# Value Iteration Testing and Analysis

# Tests to demonstrate understanding:
# 1. Convergence with different gamma values
# 2. Effect of living reward on path selection
# 3. Iteration count vs maze size
# 4. Value function propagation visualization
# """

# from src.create_maze import load_maze
# from src.old_mdp.mdp import value_iteration, prettify_policy
# import matplotlib.pyplot as plt
# import numpy as np


# def test_gamma_sensitivity():
#     """
#     Test 1: How does gamma affect convergence?
    
#     What this tests:
#     - Lower gamma → faster convergence (agent is myopic)
#     - Higher gamma → slower convergence (agent plans far ahead)
#     - Gamma near 1.0 → values can be much larger
#     """
#     print("\n" + "="*80)
#     print("VALUE ITERATION TEST 1: Gamma Sensitivity")
#     print("="*80)
    
#     maze = load_maze("mazes/maze_19x19_seed42.json")
#     gammas = [0.5, 0.7, 0.9, 0.95, 0.99]
    
#     print(f"{'Gamma':<10} | {'Iterations':<12} | {'Path Length':<12} | {'Max Value':<12} | {'Min Value':<12}")
#     print("-"*80)
    
#     for gamma in gammas:
#         result = value_iteration(maze, gamma=gamma)
#         max_val = max(result.values.values())
#         min_val = min(v for v in result.values.values() if v != 0)
        
#         print(f"{gamma:<10.2f} | {result.iterations:<12} | {result.path_length:<12} | {max_val:<12.4f} | {min_val:<12.4f}")
    
#     print("\nKey Observations:")
#     print("  - Higher gamma → more iterations needed")
#     print("  - Higher gamma → larger value magnitudes")
#     print("  - All find same path length (deterministic)")


# def test_living_reward_impact():
#     """
#     Test 2: How does living reward affect behavior?
    
#     What this tests:
#     - Large penalty → agent desperate to reach goal quickly
#     - Small penalty → agent doesn't mind taking longer paths
#     - Zero penalty → no incentive to minimize steps
#     """
#     print("\n" + "="*80)
#     print("VALUE ITERATION TEST 2: Living Reward Impact")
#     print("="*80)
    
#     maze = load_maze("mazes/maze_19x19_seed42.json")
#     living_rewards = [-1.0, -0.4, -0.04, -0.01, 0.0]
    
#     print(f"{'Living Reward':<15} | {'Iterations':<12} | {'Path Length':<12} | {'Goal Value':<12}")
#     print("-"*80)
    
#     for lr in living_rewards:
#         result = value_iteration(maze, living_reward=lr)
#         goal_val = result.values[maze.goal]
        
#         print(f"{lr:<15.2f} | {result.iterations:<12} | {result.path_length:<12} | {goal_val:<12.4f}")
    
#     print("\nKey Observations:")
#     print("  - Living reward affects value function scale")
#     print("  - Path length same (deterministic, all paths optimal)")
#     print("  - Negative reward = cost per step")


# def test_maze_size_scaling():
#     """
#     Test 3: How does algorithm scale with maze size?
    
#     What this tests:
#     - Iterations vs number of states
#     - Time complexity empirically
#     - Convergence rate with problem size
#     """
#     print("\n" + "="*80)
#     print("VALUE ITERATION TEST 3: Maze Size Scaling")
#     print("="*80)
    
#     maze_files = [
#         ("mazes/maze_9x9_seed42.json", "9x9"),
#         ("mazes/maze_19x19_seed42.json", "19x19"),
#         ("mazes/maze_19x19_seed42.json", "19x19"),
#     ]
    
#     print(f"{'Maze Size':<12} | {'States':<10} | {'Iterations':<12} | {'Path Length':<12}")
#     print("-"*80)
    
#     for filepath, size in maze_files:
#         maze = load_maze(filepath)
#         result = value_iteration(maze)
#         num_states = result.nodes_explored
        
#         print(f"{size:<12} | {num_states:<10} | {result.iterations:<12} | {result.path_length:<12}")
    
#     print("\nKey Observations:")
#     print("  - More states → more iterations needed")
#     print("  - Roughly linear relationship")
#     print("  - Each iteration visits all states")


# def test_convergence_threshold():
#     """
#     Test 4: How does convergence threshold affect accuracy?
    
#     What this tests:
#     - Tighter threshold → more iterations
#     - Looser threshold → faster but less precise
#     - Tradeoff between speed and accuracy
#     """
#     print("\n" + "="*80)
#     print("VALUE ITERATION TEST 4: Convergence Threshold")
#     print("="*80)
    
#     maze = load_maze("mazes/maze_19x19_seed42.json")
#     thresholds = [1e-2, 1e-4, 1e-6, 1e-8]
    
#     print(f"{'Threshold':<12} | {'Iterations':<12} | {'Path Length':<12} | {'Max Value':<12}")
#     print("-"*80)
    
#     for theta in thresholds:
#         result = value_iteration(maze, theta=theta)
#         max_val = max(result.values.values())
        
#         print(f"{theta:<12.0e} | {result.iterations:<12} | {result.path_length:<12} | {max_val:<12.6f}")
    
#     print("\nKey Observations:")
#     print("  - Tighter threshold → more iterations")
#     print("  - Path quality similar (all find optimal)")
#     print("  - Default 1e-6 is good balance")


# def test_policy_visualization():
#     """
#     Test 5: Visualize the computed policy
    
#     What this tests:
#     - Policy arrows point toward goal
#     - Forms a flow field
#     - Every state has an action
#     """
#     print("\n" + "="*80)
#     print("VALUE ITERATION TEST 5: Policy Visualization")
#     print("="*80)
    
#     maze = load_maze("mazes/maze_19x19_seed42.json")
#     result = value_iteration(maze)
    
#     print("\nComputed Policy (arrows show best action at each state):")
#     print(prettify_policy(result.policy, maze.grid))
    
#     print(f"\nPath Length: {result.path_length}")
#     print(f"Iterations: {result.iterations}")
    
#     print("\nKey Observations:")
#     print("  - All arrows point generally toward goal")
#     print("  - Forms a 'flow field' of optimal actions")
#     print("  - Policy is complete (every free cell has action)")


# def main():
#     """Run all Value Iteration tests."""
#     print("\n" + "="*80)
#     print("VALUE ITERATION COMPREHENSIVE TESTING")
#     print("="*80)
    
#     test_gamma_sensitivity()
#     test_living_reward_impact()
#     test_maze_size_scaling()
#     test_convergence_threshold()
#     test_policy_visualization()
    
#     print("\n" + "="*80)
#     print("VALUE ITERATION TESTING COMPLETE")
#     print("="*80)
#     print("\nKey Algorithm Properties Demonstrated:")
#     print("  ✓ Convergence guaranteed for γ < 1")
#     print("  ✓ Finds optimal policy")
#     print("  ✓ Scales with number of states")
#     print("  ✓ Sensitive to gamma and living reward")
#     print("  ✓ Convergence threshold controls precision")


# if __name__ == "__main__":
#     main()