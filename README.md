# CS7IS2 – Artificial Intelligence  
## Assignment 1 – Search and MDP Algorithms in Maze Environments

Author: Maria Meshcheryakova  
Student Number: 21366427  

---

## Overview

This project implements and compares classical search algorithms and Markov Decision Process (MDP) algorithms in randomly generated maze environments.

Implemented algorithms:

Search:
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- A* (Manhattan heuristic)
- A* (Euclidean heuristic)

MDP:
- Value Iteration
- Policy Iteration

Maze generation uses the `mazelib` library (MIT License).
All search and MDP implementations are original.

---

##  Project Structure

src/
    maze_generator.py
    bfs.py
    dfs.py
    astar.py
    value_iteration.py
    policy_iteration.py
analyse.py 
run_all.py
save_mazes.py
test_maze.py
test_mdp_plots.py
test_mdp.py
test_mdp.py
test_solvers.py
requirements.txt
README.md


---

## Setup Instructions

### 1. Create virtual environment

python -m venv venv
source venv/bin/activate  

### 2. Instal Requirements

pip install -r requirements.txt

### 3. Create and save mazes

python3 save_mazes.py

in the config insert the sizes of mazes you want to create
! matlib will produce a solvable maze of size (2r-1),(2c-1)!

# Config
SIZES = [
    (5,5),
    (25, 25),
    (50, 50),
]

Adjust the WALL_REMOVAL_PROB variable to add more paths through the maze
WALL_REMOVAL_PROB = 0 (one path through the maze)


### 4. Run algorithms to solve mazes

In run_algorithms.py, insert mazes you want to solve

e.g.

MAZE_FILES = {
        # "Size_label": "mazes/file_name.json",
        e.g.: 
        "9x9":   "mazes/maze_9x9_seed67.json",
        "15x15":   "mazes/maze_15x15_seed67.json",
    }

python3 run_algorithms.py

this will produce a csv results file and plots for each maze size 

### 4. Create animations

In run_animations.py, insert mazes you want to solve

e.g.

MAZE_FILES = {
        # "Size_label": "mazes/file_name.json",
        e.g.: 
        "9x9":   "mazes/maze_9x9_seed67.json",
        "15x15":   "mazes/maze_15x15_seed67.json",
    }

python3 run_animations.py

Will produce a video animations of each algorithms solving the maze 

### 6. Analyse algorithms

In analyse.py, insert mazes you want to solve:

e.g. 
MAZE_FILES = {
    # "Size_label": "mazes/file_name.json",
        e.g.: 
        "9x9":   "mazes/maze_9x9_seed67.json",
        "15x15":   "mazes/maze_15x15_seed67.json",
}

python3 analyse.py 

This will produce comparison plots of different evaluation metrics between the search algorithms

