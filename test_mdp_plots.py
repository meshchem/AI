from src.create_maze import load_maze
from src.mdp_algorithms import value_iteration, policy_iteration
from src.mdp_plots import plot_mdp_results

maze = load_maze("mazes/maze_15x15_seed42.json")

vi_result = value_iteration(maze)
pi_result = policy_iteration(maze)

# Individual plots
plot_mdp_results(maze, vi_result, "Value Iteration", "15x15")
plot_mdp_results(maze, pi_result, "Policy Iteration", "15x15")
