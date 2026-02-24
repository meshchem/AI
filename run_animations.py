import os
import time

from src.create_maze import load_maze, plot_maze
from src.dfs import dfs
from src.bfs import bfs
from src.astar import astar_manhattan, astar_euclidean
from src.animate_search import animate_search_algorithms

from src.mdp_algorithms import value_iteration, policy_iteration
from src.animate_vi import animate_value_iteration
from src.animate_pi import animate_policy_iteration

MAZE_FILES = {
        "9x9":   "mazes/maze_9x9_seed67.json",
        "15x15": "mazes/maze_15x15_seed67.json",
        "19x19": "mazes/maze_19x19_seed67.json",
        "29x29": "mazes/maze_29x29_seed67.json",
        "39x39": "mazes/maze_39x39_seed67.json",
        # "99x99": "mazes/maze_99x99_seed67.json",
    }

CSV_PATH    = "results/results.csv"
PLOT_DIR   = "plots/seed67"
VIDEO_DIR  = "videos"



#  Helper function
def run_timed(fn, maze):
    t0 = time.perf_counter()
    result = fn(maze)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return result, elapsed_ms


def main():
    
    mazes = {}

    for size, path in MAZE_FILES.items():
        if os.path.exists(path):
            mazes[size] = load_maze(path)
        else:
            print(f"run save_mazes first")
    
    algorithms = {
        "DFS": dfs,
        "BFS": bfs,
        "A* Manhattan": astar_manhattan,
        "A* Euclidean": astar_euclidean,
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
    }
    
    search_algorithms = ["DFS", "BFS", "A* Manhattan", "A* Euclidean"]

    print("Running all algorithms\n")
    results = {name: {} for name in algorithms}
    timings = {name: {} for name in algorithms}   # ms, keyed same as results

    for name, fn in algorithms.items():
        for size, maze in mazes.items():
            print(f"  {name:<20}  {size}...", end=" ", flush=True)
            result, ms = run_timed(fn, maze)
            results[name][size] = result
            timings[name][size] = ms
            print(f"{ms:.1f} ms")

   
    # Generating animation for each algorithm
    for size, maze in mazes.items():
        print(f"\nCreating algorithm animations")
        
        # Search algorithm animations
        search_results = {name: results[name][size] for name in search_algorithms}
        animate_search_algorithms(
            maze,
            search_results,
            output_dir=f"videos/{size}",
            fps=5,
        )
        
        # MDP animations
        # value_iteration
        animate_value_iteration(
            maze,
            discount_factor=0.9,
            theta=1e-6,
            output_dir=f"videos/{size}",
            fps=2,
            show=False,  
        )

         # policy iteration
        animate_policy_iteration(
            maze,
            discount_factor=0.9,
            theta=1e-6,
            output_dir=f"videos/{size}",
            fps=2,
            show=True,
        )


if __name__ == "__main__":
    main()