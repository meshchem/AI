from dataclasses import dataclass
from collections import deque
from typing import List
from src.create_maze import MazeData, Coord, get_neighbours


@dataclass
class SearchResult:
    path: List[Coord]         # solution path from start to goal, empty if not found
    visited: List[Coord]      # every node explored, in order (for visualisation)
    path_length: int          # number of steps in the solution path
    nodes_explored: int       # total nodes popped from the frontier


def bfs(maze: MazeData) -> SearchResult:
    start = maze.start
    goal = maze.goal
    grid = maze.grid

    # Queue holds the current path from start to the node being explored.
    # Each entry is a list of coords representing the path so far.
    frontier: deque[List[Coord]] = deque([[start]])

    visited: List[Coord] = []
    explored = set()
    explored.add(start)

    while frontier:
        current_path = frontier.popleft()      # FIFO — take the earliest added current_path
        current_node = current_path[-1]          # the node at the end of this path

        visited.append(current_node)

        if current_node == goal:
            return SearchResult(
                path=current_path,
                visited=visited,
                path_length=len(current_path),
                nodes_explored=len(visited),
            )

        for neighbour in get_neighbours(grid, current_node):
            if neighbour not in explored:
                explored.add(neighbour)
                frontier.append(current_path + [neighbour])  # extend path to this neighbour

    # No path found
    return SearchResult(
        path=[],
        visited=visited,
        path_length=0,
        nodes_explored=len(visited),
    )


