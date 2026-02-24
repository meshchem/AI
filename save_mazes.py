import os
from src.create_maze import generate_maze, generate_imperfect_maze, save_maze, print_maze

# Config
SIZES = [
    # (5, 5),
    # (8, 8),
    # (10, 10),
    # (15, 15),
    # (20, 20),
    (50, 50),
]
SEEDS = [67]
OUTPUT_DIR = "mazes"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated = []

    for rows, cols in SIZES:
        for seed in SEEDS:
           
        #    # Maze with one unique path (perfect maze)
        #     maze = generate_maze(rows, cols, seed=seed)

        #     # Start/goal check
        #     assert maze.grid[maze.start[0]][maze.start[1]] == 0, \
        #         f"Start is a wall! (seed={seed}, size={rows}x{cols})"
        #     assert maze.grid[maze.goal[0]][maze.goal[1]] == 0, \
        #         f"Goal is a wall! (seed={seed}, size={rows}x{cols})"

        #     filename = f"maze_{rows}x{cols}_seed{seed}_perfect.json"
        #     filepath = os.path.join(OUTPUT_DIR, filename)
        #     save_maze(maze, filepath)

        #     generated.append((filename, rows, cols, seed, "perfect"))
        #     print(f"Generated: {filename}")
        #     print(f"  Grid size : {len(maze.grid)} x {len(maze.grid[0])} (internal)")
        #     print(f"  Start     : {maze.start}")
        #     print(f"  Goal      : {maze.goal}")
        #     print_maze(maze.grid, maze.start, maze.goal)
        #     print()

            #  Maze with multiple paths (potentially)
            imperfect_maze = generate_imperfect_maze(rows, cols, seed=seed, wall_removal_prob=0.2)

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