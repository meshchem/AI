"""
Script to pre-generate mazes and save them to the mazes/ directory.

Run once from the project root:
    python scripts/generate_mazes.py

All four search algorithms should load from these files to ensure
they are compared on identical mazes.
"""

import os
from src.maze_generator import generate_maze, save_maze, print_maze, plot_maze

# --- Configuration ---
SIZES = [
    (5, 5),
    (10, 10),
    (15, 15),
]
SEEDS = [42, 123, 999]
OUTPUT_DIR = "mazes"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated = []

    for rows, cols in SIZES:
        for seed in SEEDS:
            maze = generate_maze(rows, cols, seed=seed)

            # Sanity checks — catch any generation issues early
            assert maze.grid[maze.start[0]][maze.start[1]] == 0, \
                f"Start is a wall! (seed={seed}, size={rows}x{cols})"
            assert maze.grid[maze.goal[0]][maze.goal[1]] == 0, \
                f"Goal is a wall! (seed={seed}, size={rows}x{cols})"

            filename = f"maze_{rows}x{cols}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            save_maze(maze, filepath)

            generated.append((filename, rows, cols, seed))
            print(f"Generated: {filename}")
            print(f"  Grid size : {len(maze.grid)} x {len(maze.grid[0])} (internal)")
            print(f"  Start     : {maze.start}")
            print(f"  Goal      : {maze.goal}")
            print_maze(maze.grid, maze.start, maze.goal)
            plot_maze(maze.grid, maze.start, maze.goal)
            print()

    print(f"Done. {len(generated)} maze(s) saved to '{OUTPUT_DIR}/':")
    for filename, rows, cols, seed in generated:
        print(f"  {filename}")


if __name__ == "__main__":
    main()