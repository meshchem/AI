from src.create_maze import generate_maze, generate_imperfect_maze
from src.create_maze import print_maze
from src.create_maze import plot_maze


maze = imperfect_maze = generate_imperfect_maze(8, 8, seed=42, wall_removal_prob=0.2)
plot_maze(maze.grid, maze.start, maze.goal)
# print_maze(maze.grid, maze.start, maze.goal)

# maze = generate_maze(15, 15)
# maze  = generate_maze(50, 50)
# maze = generate_maze(100, 100)

print("Rows:", len(maze.grid))
print("Cols:", len(maze.grid[0]))
print("Start:", maze.start)
print("Goal:", maze.goal)

assert maze.grid[maze.start[0]][maze.start[1]] == 0, "Start is a wall!"
assert maze.grid[maze.goal[0]][maze.goal[1]] == 0, "Goal is a wall!"