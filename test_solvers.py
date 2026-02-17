from src.maze_generator import load_maze, plot_maze
from src.dfs import dfs
from src.bfs import bfs


mazes = {
    "5x5  ":   load_maze("mazes/maze_5x5_seed42.json"),
    "10x10": load_maze("mazes/maze_10x10_seed42.json"),
    "15x15": load_maze("mazes/maze_15x15_seed42.json"),
}

print("\nDFS: ")
for size, maze in mazes.items():
    result = dfs(maze)
    # plot_maze(maze.grid, maze.start, maze.goal, path=result.path)
    print(f"  {size} | path length: {result.path_length} | nodes explored: {result.nodes_explored}")


print("\nBFS: ")
for size, maze in mazes.items():
    result = bfs(maze)
    # plot_maze(maze.grid, maze.start, maze.goal, path=result.path)
    print(f"  {size} | path length: {result.path_length} | nodes explored: {result.nodes_explored}")
