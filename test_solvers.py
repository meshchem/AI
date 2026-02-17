from src.maze_generator import load_maze, plot_maze
from src.dfs import dfs
from src.bfs import bfs

# mazes
maze5 = load_maze("mazes/maze_5x5_seed42.json")
maze10 = load_maze("mazes/maze_10x10_seed42.json")
maze15 = load_maze("mazes/maze_15x15_seed42.json")


# dfs
# dfs_result_5 = dfs(maze5)
# dfs_result_10 = dfs(maze10)
# dfs_result_15 = dfs(maze15)

# plot_maze(maze5.grid, maze5.start, maze5.goal, path=dfs_result_5.path)
# plot_maze(maze10.grid, maze10.start, maze10.goal, path=dfs_result_10.path)
# plot_maze(maze15.grid, maze15.start, maze15.goal, path=dfs_result_15.path)

# print("Path length:", dfs_result_5.path_length)
# print("Nodes explored:", dfs_result_10.nodes_explored)
# print("Path:", dfs_result_15.path)

# dfs

bfs_result_5 = bfs(maze5)
bfs_result_10 = bfs(maze10)
bfs_result_15 = bfs(maze15)

plot_maze(maze5.grid, maze5.start, maze5.goal, path=bfs_result_5.path)
plot_maze(maze10.grid, maze10.start, maze10.goal, path=bfs_result_10.path)
plot_maze(maze15.grid, maze15.start, maze15.goal, path=bfs_result_15.path)


print("Path length:", bfs_result_5.path_length)
print("Nodes explored:", bfs_result_10.nodes_explored)
print("Path:", bfs_result_15.path)