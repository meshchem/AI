from src.maze_generator import generate_maze
from src.maze_generator import print_maze
from src.maze_generator import plot_maze


# maze = generate_maze(8, 8, seed=42)
maze = generate_maze(15, 15)
print_maze(maze.grid, maze.start, maze.goal)
plot_maze(maze.grid, maze.start, maze.goal)

# maze = generate_maze(15, 15)
# maze  = generate_maze(50, 50)
# maze = generate_maze(100, 100)

print("Rows:", len(maze.grid))
print("Cols:", len(maze.grid[0]))
print("Start:", maze.start)
print("Goal:", maze.goal)