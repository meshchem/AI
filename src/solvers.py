from dataclasses import dataclass
from typing import List, Tuple

Coord = Tuple[int, int]

@dataclass
class SearchResult:
    path: List[Coord]        # the solution path from start to goal
    visited: List[Coord]     # all nodes explored (for visualisation)
    path_length: int         # len(path)
    nodes_explored: int      # len(visited)