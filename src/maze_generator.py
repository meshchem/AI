"""
Maze generation uses the 'mazelib' library:
https://github.com/john-science/mazelib
Licensed under MIT License.

Only the maze generation component is reused.
All search and MDP algorithms are original implementations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import matplotlib.pyplot as plt
import numpy as np
import json

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
    """
    Generate a maze using mazelib (MIT License).
    Maze size is parameterised (not hardcoded).
    """

    if seed is None:
        seed = random.randint(0, 1_000_000)

    random.seed(seed)

    m = Maze()
    m.generator = Prims(rows, cols)
    m.generate()

    grid = m.grid

    start = (1, 1)
    goal = (len(grid) - 2, len(grid[0]) - 2)

    return MazeData(grid=grid, start=start, goal=goal, seed=seed)

def get_neighbours(grid, cell: Coord) -> List[Coord]:
    r, c = cell
    rows, cols = len(grid), len(grid[0])
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            result.append((nr, nc))
    return result

def save_maze(maze: MazeData, filepath: str):
    """
    Save a maze to a JSON file.

    Stores grid, start, goal, and seed so the exact maze can be
    reproduced or loaded later for reproducible algorithm comparisons.
    """
    data = {
        "grid": maze.grid,
        "start": list(maze.start),
        "goal": list(maze.goal),
        "seed": maze.seed,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_maze(filepath: str) -> MazeData:
    """Load a maze from a JSON file produced by save_maze()."""
    with open(filepath) as f:
        data = json.load(f)
    return MazeData(
        grid=data["grid"],
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
        seed=data["seed"],
    )



def plot_maze(grid, start=None, goal=None, path=None):
    arr = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="binary")  # black=wall, white=free

    if start:
        plt.scatter(start[1], start[0], c="green", s=100, label="Start")

    if goal:
        plt.scatter(goal[1], goal[0], c="red", s=100, label="Goal")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys)

    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()


def print_maze(grid, start=None, goal=None, path=None):
    """
    ASCII visualisation of maze.
    1 = wall (#)
    0 = free (.)
    """
    for r in range(len(grid)):
        row_str = ""
        for c in range(len(grid[0])):
            if start and (r, c) == start:
                row_str += "S"
            elif goal and (r, c) == goal:
                row_str += "G"
            elif path and (r, c) in path:
                row_str += "*"
            elif grid[r][c] == 1:
                row_str += "#"
            else:
                row_str += "."
        print(row_str)

def save_maze(maze: MazeData, filepath: str):
    data = {
        "grid": [list(map(int, row)) for row in maze.grid],
        "start": list(maze.start),
        "goal": list(maze.goal),
        "seed": maze.seed
    }
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_maze(filepath: str) -> MazeData:
    with open(filepath) as f:
        data = json.load(f)
    return MazeData(
        grid=data["grid"],
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
        seed=data["seed"]
    )