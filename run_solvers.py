from src.create_maze import load_maze, plot_maze
from src.dfs import dfs
from src.bfs import bfs
from src.astar import astar_manhattan, astar_euclidean
from src.old_mdp.mdp import value_iteration, policy_iteration
from src.mdp_plots import plot_mdp_results



mazes = {
    # "5x5  ":   load_maze("mazes/maze_5x5_seed2104.json"),
    # "8x8": load_maze("mazes/maze_8x8_seed2104.json"),
    # "10x10": load_maze("mazes/maze_10x10_seed2104.json"),
    # "9x9  ":   load_maze("mazes/maze_9x9_seed42.json"),
    # "15x15": load_maze("mazes/maze_15x15_seed42.json"),
    # "19x19": load_maze("mazes/maze_19x19_seed42.json"),
     "9x9  ":   load_maze("mazes/maze_9x9_seed1337.json"),
    "15x15": load_maze("mazes/maze_15x15_seed1337.json"),
    "19x19": load_maze("mazes/maze_19x19_seed1337.json"),
    
   
}
# DFS
for size, maze in mazes.items():
    title = f"Depth First Search"
    result = dfs(maze)
    plot_maze(maze.grid, maze.start, maze.goal, path=result.path, title=(f"{title}, {size}"))

# BFS
for size, maze in mazes.items():
    title = f"Bredth First Search"
    result = bfs(maze)
    plot_maze(maze.grid, maze.start, maze.goal, path=result.path, title=(f"{title}, {size}"))

# A* Manhattan
for size, maze in mazes.items():
    title = f"A* Manhattan"
    result = astar_manhattan(maze)
    plot_maze(maze.grid, maze.start, maze.goal, path=result.path, title=(f"{title}, {size}"))

# A* Euclidean
for size, maze in mazes.items():
    title = f"A* Euclidean"
    result = astar_euclidean(maze)
    plot_maze(maze.grid, maze.start, maze.goal, path=result.path, title=(f"{title}, {size}"))



for size, maze in mazes.items():
    vi_result = value_iteration(maze)
    plot_mdp_results(maze, vi_result, "Value Iteration", size)

    pi_result = policy_iteration(maze)
    plot_mdp_results(maze, pi_result, "Policy Iteration", size)

# Collect all results first
algorithms = {
    "DFS": dfs,
    "BFS": bfs,
    "A* Manhattan": astar_manhattan,
    "A* Euclidean": astar_euclidean,
    "Value Iteration": value_iteration,
    "Policy Iteration": policy_iteration,
}

results = {}
for algo_name, algo_fn in algorithms.items():
    results[algo_name] = {}
    for size, maze in mazes.items():
        results[algo_name][size] = algo_fn(maze)

# Print a table for each maze size
for size in mazes:
    print("\n" + "="*80)
    print(f"MAZE SIZE: {size}")
    print("="*80)
    print(f"{'Algorithm':<20} | {'Path Length':<12} | {'Nodes Explored':<15} | {'Iterations':<12}")
    print("-"*80)
    
    for algo_name in algorithms:
        result = results[algo_name][size]
        
        # Handle different result types
        if hasattr(result, 'iterations'):
            # MDP result
            iterations = result.iterations
        else:
            # Search result
            iterations = "--"
        
        print(f"{algo_name:<20} | {result.path_length:<12} | {result.nodes_explored:<15} | {iterations:<12}")
    
    print("="*80)
