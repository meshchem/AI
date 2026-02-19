from dataclasses import dataclass
import heapq
from typing import List, Dict
import math

from src.create_maze import MazeData, Coord, get_neighbours
from src.dfs import SearchResult


@dataclass
class SearchResult:
    path: List[Coord]         # solution path from start to goal, empty if not found
    visited: List[Coord]      # every node explored, in order (for visualisation)
    path_length: int          # number of steps in the solution path
    nodes_explored: int       # total nodes popped from the frontier


    # Heuristics: 
    # - Manhattan distance: h(n) = |x1 - x2| + |y1 - y2|
    # - Euclidean distance: h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a: Coord, b: Coord) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(maze: MazeData, heuristic) -> SearchResult:

    start = maze.start
    goal = maze.goal
    grid = maze.grid

    # Priority queue entries: (f_score, g_score, node)
    # g_score is included to break ties — prefer nodes closer to start
    h_start = heuristic(start, goal)
    heap = [(h_start, 0, start)]

    # g_score: cheapest known cost from start to each node
    g_score: Dict[Coord, float] = {start: 0}

    # came_from: reconstructs the final path 
    came_from: Dict[Coord, Coord] = {}

    visited: List[Coord] = []
    seen = set()

    while heap:
        f, g, current = heapq.heappop(heap)

        if current in seen:
            continue

        seen.add(current)
        visited.append(current)

        if current == goal:
            # Reconstruct path by walking back through came_from
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            return SearchResult(
                path=path,
                visited=visited,
                path_length=len(path),
                nodes_explored=len(visited),
            )

        for neighbour in get_neighbours(grid, current):
            tentative_g = g + 1  # each step costs 1

            if neighbour not in g_score or tentative_g < g_score[neighbour]:
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                f_score = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(heap, (f_score, tentative_g, neighbour))

    # No path found
    return SearchResult(
        path=[],
        visited=visited,
        path_length=0,
        nodes_explored=len(visited),
    )


def astar_manhattan(maze: MazeData) -> SearchResult:
    return astar(maze, manhattan)


def astar_euclidean(maze: MazeData) -> SearchResult:
    return astar(maze, euclidean)
