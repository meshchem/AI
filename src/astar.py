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

    """
    A* Search maze solver.

    Uses a priority queue (min-heap) to explore the maze, prioritising nodes
    with the lowest estimated total cost (g + h) to reach the goal.
    Guarantees the shortest path if the heuristic is admissible (never overestimates).
    Time complexity  : O(E) in the worst case (if heuristic is poor, behaves like BFS)
    Space complexity : O(V) in the worst case (if heuristic is poor, explores all nodes)

    Heuristics: 
    - Manhattan distance: h(n) = |x1 - x2| + |y1 - y2|
    - Euclidean distance: h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    
    - A* with Manhattan distance heuristic is often used for grid-based pathfinding, as it provides a good balance between accuracy and computational efficiency, especially when movement is restricted to four directions (up, down, left, right).
    - A* with Euclidean distance heuristic can be more accurate for pathfinding in continuous spaces or when diagonal movement is allowed, but it may be computationally more expensive than Manhattan distance.
    
    - Dijkstra's algorithm is a special case of A* where h(n) = 0 for all n, meaning it explores nodes based solely on g(n) (the cost from the start node to n).
    - Greedy Best-First Search is a special case of A* where g(n) = 0 for all n, meaning it explores nodes based solely on h(n) (the heuristic estimate to the goal).
    
    - A* with Dijkstra's algorithm (h(n) = 0) will explore all nodes in the order of their g(n) values, effectively performing a uniform-cost search that guarantees the shortest path but may be inefficient if the search space is large and the goal is far from the start.
    - A* with Greedy Best-First Search (g(n) = 0) will explore nodes based solely on their heuristic estimates to the goal, which can lead to faster exploration but may not guarantee the shortest path if the heuristic is not accurate or admissible.


    """


def _manhattan(a: Coord, b: Coord) -> int:
    """
    Manhattan distance heuristic.

    Counts the total horizontal + vertical steps between two cells.
    This is the 'correct' heuristic for grid mazes with 4-directional
    movement — it never overestimates, so A* is guaranteed to find
    the shortest path (admissible + consistent).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _euclidean(a: Coord, b: Coord) -> float:
    """
    Euclidean distance heuristic.

    Straight-line distance between two cells.
    Still admissible (never overestimates) for a 4-directional grid
    because the straight-line distance is always <= the actual step
    cost. However it is less informed than Manhattan — it underestimates
    more, so A* will explore more nodes than with Manhattan.
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _astar(maze: MazeData, heuristic) -> SearchResult:
    """
    Core A* implementation, shared by both heuristic variants.

    Explores nodes in order of f(n) = g(n) + h(n) where:
        g(n) = exact cost from start to n (number of steps)
        h(n) = heuristic estimate from n to goal

    Time complexity  : O(V log V)  due to priority queue operations
    Space complexity : O(V)
    """
    start = maze.start
    goal = maze.goal
    grid = maze.grid

    # Priority queue entries: (f_score, g_score, node)
    # g_score is included to break ties — prefer nodes closer to start
    h_start = heuristic(start, goal)
    heap = [(h_start, 0, start)]

    # g_score: cheapest known cost from start to each node
    g_score: Dict[Coord, float] = {start: 0}

    # came_from: used to reconstruct the path at the end
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
            tentative_g = g + 1  # each step has cost 1

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
    return _astar(maze, _manhattan)


def astar_euclidean(maze: MazeData) -> SearchResult:
    return _astar(maze, _euclidean)
