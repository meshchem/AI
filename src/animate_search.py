import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from typing import List
from src.create_maze import MazeData, Coord

    # Parameters:
    #     maze: 
    #     visited: nodes explored in order (from SearchResult.visited)
    #     path: final solution path (from SearchResult.path)
    #     algorithm_name: e.g. "BFS", "DFS", "A* Manhattan"
    #     output_file: if provided, save as MP4 (requires ffmpeg)
    #     fps: frames per second for the animation
    #     show: whether to display the animation interactively
  

def animate_search(
    maze: MazeData,             #  grid, start, goal
    visited: List[Coord],       #  nodes explored in order
    path: List[Coord],          #  final path
    algorithm_name: str,        
    output_file: str = None,    
    fps: int = 10,              # frames per second
    show: bool = True,          
):
    
    grid = maze.grid
    # rows, cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Drawing maze (walls = black, free = white)
    ax.imshow(np.array(grid), cmap="binary", interpolation="nearest")

    # Mark start and goal
    ax.scatter(maze.start[1], maze.start[0], c="green", s=150, zorder=5, label="Start")
    ax.scatter(maze.goal[1], maze.goal[0], c="red", s=150, zorder=5, label="Goal", marker="*")

    ax.set_title(f"{algorithm_name} — Exploring...", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")

    # Container for visited nodes
    visited_nodes = []

    # Pre-compute path set for fast lookup
    path_set = set(path) if path else set()

    # initialize animation with an empty frame
    def init():
        return []

    # update each frame
    def update(frame):
        # Clear previous visited nodes
        for artist in visited_nodes:
            artist.remove()
        visited_nodes.clear()

        # Draw all nodes visited up to this frame
        for i in range(min(frame + 1, len(visited))):
            node = visited[i]
            r, c = node

            # Color based on whether it's on the final path
            if node in path_set:
                color = "lightgreen"
                alpha = 0.6
            else:
                color = "#FF9800"  
                alpha = 0.4

            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1, 1,
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)
            visited_nodes.append(rect)

        # Update title with progress
        visited_count = min(frame + 1, len(visited))
        ax.set_title(
            f"{algorithm_name} - Visited: {visited_count}/{len(visited)}",
            fontsize=14,
        )

        # Final path
        # if frame >= len(visited) and path:
        #     ys = [p[0] for p in path]
        #     xs = [p[1] for p in path]
        #     line, = ax.plot(xs, ys, c="blue", linewidth=3, zorder=4, label="Path")
        #     visited_nodes.append(line)
        #     ax.set_title(
        #         f"{algorithm_name} — Complete! Path length: {len(path)}",
        #         fontsize=14,
        #     )
        #     ax.legend(loc="upper right")

        # return visited_nodes

    total_frames = len(visited) + 30  # hold final state for 30 frames

    animation_object = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=1000 // fps,
        blit=False,
        repeat=True,
    )

    if output_file:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        animation_object.save(output_file, writer=writer)
        print(f"Animation directory: {output_file}")

    if show:
        plt.tight_layout()
        plt.show()

    return animation_object


def animate_search_algorithms(
    maze: MazeData,
    results: dict,
    output_dir: str = "videos",
    fps: int = 10,
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, result in results.items():
        output_file = os.path.join(
            output_dir,
            f"{name.replace(' ', '_').lower()}_sol.mp4"
        )
        animate_search(
            maze,
            visited=result.visited,
            path=result.path,
            algorithm_name=name,
            output_file=output_file,
            fps=fps,
            show=False, 
        )