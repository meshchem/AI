import os
import time
import csv

from src.create_maze import load_maze, plot_maze
from src.dfs import dfs
from src.bfs import bfs
from src.astar import astar_manhattan, astar_euclidean
from src.animate_search import animate_search_algorithms

from src.mdp_algorithms import value_iteration, policy_iteration
from src.mdp_plots import plot_mdp_results
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

CSV_COL = [
    "algorithm",
    "maze_size",
    "path_length",
    "nodes_explored",
    "iterations",       # "--" for search algorithms
    "runtime_ms",       # wall-clock time in milliseconds
]


#  Helper functions
def run_timed(fn, maze):
    t0 = time.perf_counter()
    result = fn(maze)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return result, elapsed_ms


def save_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {path}")


def print_summary(results, timings, algorithms, mazes):
    for size in mazes:
        print("\n" + "=" * 90)
        print(f"  MAZE SIZE: {size}")
        print("=" * 90)
        print(
            f"{'Algorithm':<20} | {'Path Length':<12} | "
            f"{'Nodes Explored':<15} | {'Iterations':<12} | {'Time (ms)':<10}"
        )
        print("-" * 90)
        for name in algorithms:
            result = results[name][size]
            ms     = timings[name][size]
            iters  = result.iterations if hasattr(result, "iterations") else "--"
            print(
                f"{name:<20} | {result.path_length:<12} | "
                f"{result.nodes_explored:<15} | {iters:<12} | {ms:<10.3f}"
            )
        print("=" * 90)



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

    # Console summary table
    print_summary(results, timings, algorithms, mazes)

    # CSV export 
    csv_rows = []
    for name in algorithms:
        for size in mazes:
            result = results[name][size]
            iters  = result.iterations if hasattr(result, "iterations") else "--"
            csv_rows.append({
                "algorithm":      name,
                "maze_size":      size,
                "path_length":    result.path_length,
                "nodes_explored": result.nodes_explored,
                "iterations":     iters,
                "runtime_ms":     round(timings[name][size], 3),
            })

    save_csv(csv_rows, CSV_PATH)

    # Search algorithm plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    for algo_name in search_algorithms:
        for size, maze in mazes.items():
            result    = results[algo_name][size]
            safe_name = algo_name.replace(" ", "_").replace("*", "star").lower()
            save_path = f"{PLOT_DIR}/{safe_name}_{size}.png"
            plot_maze(
                maze.grid, maze.start, maze.goal,
                path=result.path,
                title=f"{algo_name} — {size}",
                save_path=save_path,
            )

    # MDP plots
    for size, maze in mazes.items():
        vi_result = results["Value Iteration"][size]
        pi_result = results["Policy Iteration"][size]

        plot_mdp_results(maze, vi_result, "Value Iteration",  size, save_dir=PLOT_DIR)
        plot_mdp_results(maze, pi_result, "Policy Iteration", size, save_dir=PLOT_DIR)


if __name__ == "__main__":
    main()