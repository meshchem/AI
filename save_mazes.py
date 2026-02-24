import os
from src.create_maze import generate_maze, generate_imperfect_maze, save_maze, print_maze

# Config
SIZES = [
    (5, 5),
    (8, 8),
    (10, 10),
    (15, 15),
    (20, 20),
]

SEEDS = [67]
OUTPUT_DIR = "mazes"
WALL_REMOVAL_PROB = 0.2


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated = []

    for rows, cols in SIZES:
        for seed in SEEDS:
           
            #  Maze with multiple paths)
            imperfect_maze = generate_imperfect_maze(rows, cols, seed=seed, wall_removal_prob= WALL_REMOVAL_PROB)

            filename_imp = f"maze_{rows*2-1}x{cols*2-1}_seed{seed}.json"
            filepath_imp = os.path.join(OUTPUT_DIR, filename_imp)
            save_maze(imperfect_maze, filepath_imp)

            generated.append((filename_imp, rows, cols, seed, "imperfect"))
            print(f"Generated: {filename_imp} (15% walls removed)")
            print(f"  Grid size : {len(imperfect_maze.grid)} x {len(imperfect_maze.grid[0])} (internal)")
            print_maze(imperfect_maze.grid, imperfect_maze.start, imperfect_maze.goal)
            print()

    print(f"Done. {len(generated)} maze(s) saved to '{OUTPUT_DIR}/':")
    for filename, rows, cols, seed, maze_type in generated:
        print(f"  {filename}")


if __name__ == "__main__":
    main()