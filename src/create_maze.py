"""
Maze generation uses the 'mazelib' library:
https://github.com/john-science/mazelib
Licensed under MIT License.

Only the maze generation component is reused.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from mazelib import Maze
from mazelib.generate.Prims import Prims

Coord = Tuple[int, int]


@dataclass
class MazeData:
    grid: List[List[int]]  # 0 = free, 1 = wall
    start: Coord
    goal: Coord
    seed: int


def generate_maze(rows: int, cols: int, seed: Optional[int] = None) -> MazeData:
    if seed is None:
        seed = random.randint(0, 1_000_000)

    random.seed(seed)

    m = Maze()
    m.generator = Prims(rows, cols)
    m.generate()

    grid = [[int(cell) for cell in row] for row in m.grid]

    start = (1, 1)
    goal = (len(grid) - 2, len(grid[0]) - 2)

    return MazeData(grid=grid, start=start, goal=goal, seed=seed)

#  randomly removes a percentage of internal walls to create loops and alternative routes.
def generate_imperfect_maze(
    rows: int,
    cols: int,
    seed: Optional[int] = None,
    wall_removal_prob: float = 0.15
) -> MazeData:
   
    # Generate perfect maze 
    maze = generate_maze(rows, cols, seed)
    grid = maze.grid

    # Identify removable walls (internal walls, not borders)
    removable_walls = []
    for r in range(2, len(grid) - 2):
        for c in range(2, len(grid[0]) - 2):
            if grid[r][c] == 1:  # it's a wall
                # Only remove if there is free cells on opposite sides
                # only horizontal or vertical walls, no corners
                has_h_passage = (grid[r][c-1] == 0 and grid[r][c+1] == 0)
                has_v_passage = (grid[r-1][c] == 0 and grid[r+1][c] == 0)
                if has_h_passage or has_v_passage:
                    removable_walls.append((r, c))

    # Remove a random subset of walls
    num_to_remove = int(len(removable_walls) * wall_removal_prob)
    walls_to_remove = random.sample(removable_walls, num_to_remove)

    for r, c in walls_to_remove:
        grid[r][c] = 0

    return MazeData(grid=grid, start=maze.start, goal=maze.goal, seed=seed)


#  Return all passable (non-wall) neighbours of a cell.
#  4-directional movement
def get_neighbours(grid: List[List[int]], cell: Coord) -> List[Coord]:
    r, c = cell
    rows, cols = len(grid), len(grid[0])
    neighbours = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbours.append((nr, nc))
    return neighbours

# Visualise maze using ASCII:
def print_maze(grid: List[List[int]], start: Coord = None, goal: Coord = None):
    for r, row in enumerate(grid):
        line = ""
        for c, cell in enumerate(row):
            if (r, c) == start:
                line += "S"
            elif (r, c) == goal:
                line += "G"
            else:
                line += "#" if cell else "."
        print(line)

# Visualise maze using matplotlib:
def plot_maze(grid, start=None, goal=None, path=None, title=None, save_path: str = None):
  
    arr = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="binary")  # black = wall, white = free

    if start:
        plt.scatter(start[1], start[0], c="green", s=100, zorder=3, label="Start")

    if goal:
        plt.scatter(goal[1], goal[0], c="red", s=100, zorder=3, label="Goal",  marker="*")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, c="blue", linewidth=2, label="Path")

    if title:
        plt.title(title)

    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

#  Save a maze to a JSON file (stores grid, start, goal, and seed for reproducability)
def save_maze(maze: MazeData, filepath: str):
    data = {
        "grid": maze.grid,
        "start": list(maze.start),
        "goal": list(maze.goal),
        "seed": maze.seed,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# load saved maze
def load_maze(filepath: str) -> MazeData:
    with open(filepath) as f:
        data = json.load(f)
    return MazeData(
        grid=data["grid"],
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
        seed=data["seed"],
    )



