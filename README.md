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
test_maze.py
requirements.txt
README.md


---

## Setup Instructions

### 1. Create virtual environment

python -m venv venv
source venv/bin/activate  

### 2. Instal Requirements


pip install -r requirements.txt

