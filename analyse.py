import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.create_maze import load_maze
from src.dfs import dfs
from src.bfs import bfs
from src.astar import astar_manhattan, astar_euclidean
from src.mdp_algorithms import value_iteration, policy_iteration

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MAZE_FILES = {
    "9×9":   "mazes/maze_9x9_seed67.json",
    "15×15": "mazes/maze_15x15_seed67.json",
    "19×19": "mazes/maze_19x19_seed67.json",
    "29×29": "mazes/maze_29x29_seed67.json",
    "39x39": "mazes/maze_39x39_seed67.json",
    # "99x99": "mazes/maze_99x99_seed67.json",
}

FILE_DIR = "plots/comparison"

SEARCH_ALG = ["DFS", "BFS", "A* Manhattan", "A* Euclidean"]
MDP_ALG    = ["Value Iteration", "Policy Iteration"]
ALL_ALGS   = SEARCH_ALG + MDP_ALG

ALGO_FUNC = {
    "DFS":              dfs,
    "BFS":              bfs,
    "A* Manhattan":     astar_manhattan,
    "A* Euclidean":     astar_euclidean,
    "Value Iteration":  value_iteration,
    "Policy Iteration": policy_iteration,
}

ALG_COLOURS = {
    "DFS":              "#e15759",
    "BFS":              "#f28e2b",
    "A* Manhattan":     "#4e79a7",
    "A* Euclidean":     "#76b7b2",
    "Value Iteration":  "#59a14f",
    "Policy Iteration": "#b07aa1",
}

MAZE_COLOURS = {
    "9×9":   "#4e79a7",
    "15×15": "#f28e2b",
    "19×19": "#e15759",
    "29×29": "#76b7b2",
    "39x39": "#59a14f",
    "99x99": "#59a14a",
}

MARKERS = {
    "DFS":              "o",
    "BFS":              "s",
    "A* Manhattan":     "^",
    "A* Euclidean":     "D",
    "Value Iteration":  "P",
    "Policy Iteration": "X",
}

DISCOUNT_FACTORS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
THRESHOLDS = [0.1, 0.01, 0.001, 0.0001, 0.00001]
K_VALS = [1, 2, 5, 10, 20, 50, 100]
STEP_COSTS = [-0.01, -0.04, -0.1, -0.5, -1.0, -2.0, 1, 5]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def timed(fn, maze):
    t0 = time.perf_counter()
    result = fn(maze)
    return result, (time.perf_counter() - t0) * 1000


def savefig(name):
    os.makedirs(FILE_DIR, exist_ok=True)
    path = os.path.join(FILE_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ─────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────
def collect_results(mazes):
    results = {name: {} for name in ALL_ALGS}
    timings = {name: {} for name in ALL_ALGS}

    for name, fn in ALGO_FUNC.items():
        for size, maze in mazes.items():
            print(f"  {name:<20} {size}:    ", end=" ", flush=True)
            r, ms = timed(fn, maze)
            results[name][size] = r
            timings[name][size] = ms
            print(f"{ms:.1f} ms")

    return results, timings


def collect_discount_factor_results(mazes):
    vi_data, pi_data = {}, {}
    for size, maze in mazes.items():
        vi_rows, pi_rows = [], []
        for g in DISCOUNT_FACTORS:
            vi_r, vi_ms = timed(lambda m, g=g: value_iteration(m, discount_factor=g), maze)
            pi_r, pi_ms = timed(lambda m, g=g: policy_iteration(m, discount_factor=g), maze)
            vi_rows.append({"discount_factor": g, "iterations": vi_r.iterations, "runtime_ms": vi_ms})
            pi_rows.append({"discount_factor": g, "iterations": pi_r.iterations, "runtime_ms": pi_ms})
        vi_data[size] = vi_rows
        pi_data[size] = pi_rows
    return vi_data, pi_data


def collect_threshold_results(mazes):
    vi_data, pi_data = {}, {}
    for size, maze in mazes.items():
        vi_rows, pi_rows = [], []
        for t in THRESHOLDS:
            vi_r, vi_ms = timed(lambda m, t=t: value_iteration(m, theta=t), maze)
            pi_r, pi_ms = timed(lambda m, t=t: policy_iteration(m, theta=t), maze)
            vi_rows.append({"threshold": t, "iterations": vi_r.iterations, "runtime_ms": vi_ms})
            pi_rows.append({"threshold": t, "iterations": pi_r.iterations, "runtime_ms": pi_ms})
        vi_data[size] = vi_rows
        pi_data[size] = pi_rows
    return vi_data, pi_data


def collect_k_eval_results(mazes):
    data = {}
    for size, maze in mazes.items():
        rows = []
        for k in K_VALS:
            r, ms = timed(lambda m, k=k: policy_iteration(m, k_eval=k), maze)
            rows.append({"k": k, "iterations": r.iterations, "runtime_ms": ms})
        data[size] = rows
    return data

def collect_step_cost_results(maze):
    vi_rows, pi_rows = [], []

    for cost in STEP_COSTS:
        vi_r, vi_ms = timed(lambda m, c=cost: value_iteration(m, step_cost=c), maze)
        pi_r, pi_ms = timed(lambda m, c=cost: policy_iteration(m, step_cost=c), maze)

        vi_rows.append({
            "step_cost": cost,
            "iterations": vi_r.iterations,
            "path_length": vi_r.path_length,
            "runtime_ms": vi_ms
        })

        pi_rows.append({
            "step_cost": cost,
            "iterations": pi_r.iterations,
            "path_length": pi_r.path_length,
            "runtime_ms": pi_ms
        })

    return vi_rows, pi_rows

# ─────────────────────────────────────────────
# Plots (only all-sizes / multi-maze plots)
# ─────────────────────────────────────────────
def plot_nodes_explored(results, sizes):
    print("Plotting: nodes explored vs maze size")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sizes))

    for name in ALL_ALGS:
        vals = [results[name][s].nodes_explored for s in sizes]
        ax.plot(x, vals, marker=MARKERS[name], label=name,
                color=ALG_COLOURS[name], linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    style_ax(ax, "Maze size", "Nodes explored", "Nodes Explored vs Maze Size")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("nodes_explored.png")


def plot_runtime(timings, sizes):
    print("Plotting: runtime vs maze size")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sizes))

    for name in ALL_ALGS:
        vals = [timings[name][s] for s in sizes]
        ax.plot(x, vals, marker=MARKERS[name], label=name,
                color=ALG_COLOURS[name], linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    style_ax(ax, "Maze size", "Runtime (ms)", "Runtime vs Maze Size")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    savefig("runtime.png")


def plot_path_length(results, sizes):
    print("Plotting: path length vs maze size")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sizes))

    for name in ALL_ALGS:
        vals = [results[name][s].path_length for s in sizes]
        ax.plot(x, vals, marker=MARKERS[name], label=name,
                color=ALG_COLOURS[name], linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    style_ax(ax, "Maze size", "Path length (steps)", "Path Length vs Maze Size")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("path_length.png")


def plot_mdp_iterations(results, sizes):
    print("Plotting: MDP iterations vs maze size")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sizes))

    for name in MDP_ALG:
        vals = [results[name][s].iterations for s in sizes]
        ax.plot(x, vals, marker=MARKERS[name], label=name,
                color=ALG_COLOURS[name], linewidth=2, markersize=8)
        for xi, v in zip(x, vals):
            ax.annotate(str(v), (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    style_ax(ax, "Maze size", "Iterations to convergence", "MDP Convergence: Iterations vs Maze Size")
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("mdp_iterations.png")


def plot_discount_factor_convergence_all_sizes(vi_data, pi_data, sizes):
    print("Plotting: discount_factor vs convergence (all maze sizes)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Effect of Discount Factor γ on MDP Convergence", fontsize=13, fontweight="bold")

    ax_vi_iter, ax_vi_rt = axes[0]
    ax_pi_iter, ax_pi_rt = axes[1]

    g_vals = [r["discount_factor"] for r in list(vi_data.values())[0]]

    for size in sizes:
        vi_iters    = [r["iterations"] for r in vi_data[size]]
        vi_runtimes = [r["runtime_ms"] for r in vi_data[size]]
        pi_iters    = [r["iterations"] for r in pi_data[size]]
        pi_runtimes = [r["runtime_ms"] for r in pi_data[size]]

        ax_vi_iter.plot(g_vals, vi_iters,    color=MAZE_COLOURS[size], marker="o", linewidth=2, markersize=6, label=size)
        ax_vi_rt.plot(  g_vals, vi_runtimes, color=MAZE_COLOURS[size], marker="o", linewidth=2, markersize=6, label=size)
        ax_pi_iter.plot(g_vals, pi_iters,    color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)
        ax_pi_rt.plot(  g_vals, pi_runtimes, color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)

    style_ax(ax_vi_iter, "γ", "Iterations",   "Value Iteration: Iterations vs γ")
    style_ax(ax_vi_rt,   "γ", "Runtime (ms)", "Value Iteration: Runtime vs γ")
    style_ax(ax_pi_iter, "γ", "Iterations",   "Policy Iteration: Iterations vs γ")
    style_ax(ax_pi_rt,   "γ", "Runtime (ms)", "Policy Iteration: Runtime vs γ")

    for ax in [ax_vi_iter, ax_vi_rt, ax_pi_iter, ax_pi_rt]:
        ax.set_xticks(g_vals)
        ax.legend(fontsize=8, title="Maze size")

    plt.tight_layout()
    savefig("discount_factor_convergence.png")


def plot_threshold_convergence_all_sizes(vi_data, pi_data, sizes):
    print("Plotting: threshold vs convergence (all maze sizes)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Effect of Convergence Threshold θ on MDP Convergence", fontsize=13, fontweight="bold")

    ax_vi_iter, ax_vi_rt = axes[0]
    ax_pi_iter, ax_pi_rt = axes[1]

    t_vals = [r["threshold"] for r in list(vi_data.values())[0]]

    for size in sizes:
        vi_iters    = [r["iterations"] for r in vi_data[size]]
        vi_runtimes = [r["runtime_ms"] for r in vi_data[size]]
        pi_iters    = [r["iterations"] for r in pi_data[size]]
        pi_runtimes = [r["runtime_ms"] for r in pi_data[size]]

        ax_vi_iter.plot(t_vals, vi_iters,    color=MAZE_COLOURS[size], marker="o", linewidth=2, markersize=6, label=size)
        ax_vi_rt.plot(  t_vals, vi_runtimes, color=MAZE_COLOURS[size], marker="o", linewidth=2, markersize=6, label=size)
        ax_pi_iter.plot(t_vals, pi_iters,    color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)
        ax_pi_rt.plot(  t_vals, pi_runtimes, color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)

    style_ax(ax_vi_iter, "θ", "Iterations",   "Value Iteration: Iterations vs θ")
    style_ax(ax_vi_rt,   "θ", "Runtime (ms)", "Value Iteration: Runtime vs θ")
    style_ax(ax_pi_iter, "θ", "Iterations",   "Policy Iteration: Iterations vs θ")
    style_ax(ax_pi_rt,   "θ", "Runtime (ms)", "Policy Iteration: Runtime vs θ")

    for ax in [ax_vi_iter, ax_vi_rt, ax_pi_iter, ax_pi_rt]:
        ax.set_xscale("log")
        ax.set_xticks(t_vals)
        ax.set_xticklabels([str(t) for t in t_vals], rotation=45)
        ax.legend(fontsize=8, title="Maze size")

    plt.tight_layout()
    savefig("threshold_convergence.png")


def plot_k_eval_all_sizes(data, sizes):
    print("Plotting: K_EVAL vs convergence (all maze sizes)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effect of K Evaluation Passes on Policy Iteration", fontsize=13, fontweight="bold")

    k_vals = [r["k"] for r in list(data.values())[0]]

    for size in sizes:
        iters    = [r["iterations"] for r in data[size]]
        runtimes = [r["runtime_ms"] for r in data[size]]

        ax1.plot(k_vals, iters,    color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)
        ax2.plot(k_vals, runtimes, color=MAZE_COLOURS[size], marker="X", linewidth=2, markersize=6, label=size)

    style_ax(ax1, "K (inner evaluation passes)", "Outer iterations to convergence", "Outer Iterations vs K")
    style_ax(ax2, "K (inner evaluation passes)", "Runtime (ms)", "Runtime vs K")

    for ax in [ax1, ax2]:
        ax.set_xticks(k_vals)
        ax.legend(fontsize=8, title="Maze size")

    plt.tight_layout()
    savefig("k_eval_policy_iteration.png")


def plot_step_cost(vi_rows, pi_rows):
    print("Plotting: step cost vs path length and iterations")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effect of Step Cost on MDP Algorithms",
                 fontsize=13, fontweight="bold")

    c_vals = [r["step_cost"] for r in vi_rows]

    for rows, name in [(vi_rows, "Value Iteration"),
                       (pi_rows, "Policy Iteration")]:

        lengths = [r["path_length"] for r in rows]
        iters   = [r["iterations"] for r in rows]

        ax1.plot(
            c_vals, lengths,
            marker=MARKERS[name],
            color=ALG_COLOURS[name],
            linewidth=2,
            markersize=8,
            label=name
        )

        ax2.plot(
            c_vals, iters,
            marker=MARKERS[name],
            color=ALG_COLOURS[name],
            linewidth=2,
            markersize=8,
            label=name
        )

    style_ax(ax1, "Step Cost", "Path length (steps)",
             "Path Length vs Step Cost")
    ax1.set_xticks(c_vals)
    ax1.set_xticklabels([str(c) for c in c_vals], rotation=45)
    ax1.legend(fontsize=9)

    style_ax(ax2, "Step Cost", "Iterations to convergence",
             "Iterations vs Step Cost")
    ax2.set_xticks(c_vals)
    ax2.set_xticklabels([str(c) for c in c_vals], rotation=45)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    savefig("step_cost.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    mazes = {}
    for size, path in MAZE_FILES.items():
        if os.path.exists(path):
            mazes[size] = load_maze(path)
        else:
            print(f"[skip] {path} not found")

    if not mazes:
        print("No maze files found, run save_mazes.py first.")
        return

    sizes = list(mazes.keys())

    print("\nCollecting results:")
    results, timings = collect_results(mazes)

    print("\nCollecting sensitivity data (all maze sizes)")
    vi_df_data, pi_df_data = collect_discount_factor_results(mazes)
    vi_th_data, pi_th_data = collect_threshold_results(mazes)
    k_data = collect_k_eval_results(mazes)
    vi_step_rows, pi_step_rows = collect_step_cost_results(mazes[sizes[-1]])


    print("\nGenerating plots")
    plot_nodes_explored(results, sizes)
    plot_runtime(timings, sizes)
    plot_path_length(results, sizes)
    plot_mdp_iterations(results, sizes)
    plot_discount_factor_convergence_all_sizes(vi_df_data, pi_df_data, sizes)
    plot_threshold_convergence_all_sizes(vi_th_data, pi_th_data, sizes)
    plot_k_eval_all_sizes(k_data, sizes)
    plot_step_cost(vi_step_rows, pi_step_rows)

    print(f"\nAll plots saved to {FILE_DIR}/")


if __name__ == "__main__":
    main()