from dataclasses import dataclass
import heapq
from typing import List, Dict
import math

from src.create_maze import MazeData, Coord, get_neighbours
from src.dfs import SearchResult


@dataclass
class SearchResult:
    path: List[Coord]        
    visited: List[Coord]     
    path_length: int         
    nodes_explored: int       

   
    
# Heuristics: 
#   Manhattan distance: h(n) = |x1 - x2| + |y1 - y2|
def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

 #  Euclidean distance: h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
def euclidean(a: Coord, b: Coord) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(maze: MazeData, heuristic) -> SearchResult:

    start = maze.start
    goal = maze.goal
    grid = maze.grid

    # Priority queue entries: (f_n, g_n, node)
    # g_n is included to break ties,  prefer nodes closer to start
    h_start = heuristic(start, goal)
    heap = [(h_start, 0, start)]

    # g_n: cheapest known cost from start to each node
    g_n: Dict[Coord, float] = {start: 0}

    # trace_path: reconstructs the final path 
    trace_path: Dict[Coord, Coord] = {}

    visited: List[Coord] = []
    explored = set()

    while heap:
        f, g, current = heapq.heappop(heap)

        if current in explored:
            continue

        explored.add(current)
        visited.append(current)

        if current == goal:
            # Reconstruct path by walking back through trace_path
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = trace_path[node]
            path.append(start)
            path.reverse()

            return SearchResult(
                path=path,
                visited=visited,
                path_length=len(path),
                nodes_explored=len(visited),
            )

        for neighbour in get_neighbours(grid, current):
            candidate_g = g + 1  # each step costs 1

            if neighbour not in g_n or candidate_g < g_n[neighbour]:
                g_n[neighbour] = candidate_g
                trace_path[neighbour] = current
                f_n = candidate_g + heuristic(neighbour, goal)
                heapq.heappush(heap, (f_n, candidate_g, neighbour))

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
