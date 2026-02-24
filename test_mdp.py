import time
from src.create_maze import load_maze
from src.mdp_algorithms import value_iteration, policy_iteration, draw_policy


def timer():
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000


def header(n, title):
    print(f"\nTest {n}: {title}")
    print("-" * 50)


def row(label, value):
    print(f"  {label:<30} {value}")


# ─────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────

def test_basic_functionality():
    header(1, "Basic Functionality")

    maze = load_maze("mazes/maze_9x9_seed67.json")

    t = timer(); vi = value_iteration(maze);  vi_ms = t()
    t = timer(); pi = policy_iteration(maze); pi_ms = t()

    row("VI iterations",  vi.iterations)
    row("VI path length", vi.path_length)
    row("VI time (ms)",   f"{vi_ms:.2f}")
    row("PI iterations",  pi.iterations)
    row("PI path length", pi.path_length)
    row("PI time (ms)",   f"{pi_ms:.2f}")


def test_optimal_policy():
    header(2, "Optimal Policy Consistency")

    maze = load_maze("mazes/maze_15x15_seed1337.json")

    t = timer(); vi = value_iteration(maze);  vi_ms = t()
    t = timer(); pi = policy_iteration(maze); pi_ms = t()

    row("VI path length",      vi.path_length)
    row("PI path length",      pi.path_length)
    row("Path lengths match",  vi.path_length == pi.path_length)
    row("Policies identical",  vi.policy == pi.policy)
    row("VI time (ms)",        f"{vi_ms:.2f}")
    row("PI time (ms)",        f"{pi_ms:.2f}")
    # row("VI Policy",            f"{vi.policy}")
    # row("VI Policy",            f"{pi.policy}")

    if vi.policy != pi.policy:
        print("\n  VI policy:")
        print(draw_policy(vi.policy, maze.grid))
        print("\n  PI policy:")
        print(draw_policy(pi.policy, maze.grid))


def test_convergence_speed():
    header(3, "Convergence Speed")

    maze_files = [
        ("mazes_1/maze_9x9_seed1337.json",  "9x9"),
        ("mazes_1/maze_15x15_seed1337.json", "15x15"),
        ("mazes_1/maze_19x19_seed1337.json", "19x19"),
    ]

    print(f"  {'Maze':<8} {'VI iters':<10} {'PI iters':<10} {'VI ms':<10} {'PI ms':<10} {'Speedup'}")
    print(f"  {'-'*60}")

    for filepath, size in maze_files:
        maze = load_maze(filepath)
        t = timer(); vi = value_iteration(maze);  vi_ms = t()
        t = timer(); pi = policy_iteration(maze); pi_ms = t()
        speedup = vi_ms / pi_ms if pi_ms > 0 else float("inf")
        print(
            f"  {size:<8} {vi.iterations:<10} {pi.iterations:<10} "
            f"{vi_ms:<10.2f} {pi_ms:<10.2f} {speedup:.2f}x"
        )


def test_value_function_properties():
    header(4, "Value Function Properties")

    maze = load_maze("mazes/maze_15x15_seed1337.json")
    t = timer(); result = value_iteration(maze); elapsed = t()

    goal_value  = result.values[maze.goal]
    start_value = result.values[maze.start]
    max_value   = max(result.values.values())

    row("Goal value",               f"{goal_value:.4f}")
    row("Start value",              f"{start_value:.4f}")
    row("Goal has max value",       goal_value == max_value)
    row("Start value < goal value", start_value < goal_value)
    row("All values ≤ goal value",  all(v <= goal_value for v in result.values.values()))
    row("Time (ms)",                f"{elapsed:.2f}")


def test_gamma_sensitivity():
    header(5, "Gamma Sensitivity")

    maze   = load_maze("mazes/maze_29x29_seed67.json")
    gammas = [0.5, 0.7, 0.9, 0.95, 0.99]

    print(f"  {'Gamma':<8} {'Iterations':<12} {'Path length':<14} {'Start value':<14} {'Time (ms)'}")
    print(f"  {'-'*60}")

    for g in gammas:
        t = timer(); result = value_iteration(maze, gamma=g); elapsed = t()
        print(
            f"  {g:<8.2f} {result.iterations:<12} {result.path_length:<14} "
            f"{result.values[maze.start]:<14.4f} {elapsed:.2f}"
        )


def test_path_validity():
    header(6, "Path Validity")

    maze = load_maze("mazes_1/maze_15x15_seed1337.json")

    for name, fn in [("Value Iteration", value_iteration), ("Policy Iteration", policy_iteration)]:
        t = timer(); result = fn(maze); elapsed = t()

        print(f"  {name}")
        row("  Starts at start",    result.path[0] == maze.start)
        row("  Ends at goal",       result.path[-1] == maze.goal)
        row("  No wall cells",      all(maze.grid[r][c] == 0 for r, c in result.path))
        row("  No repeated states", len(result.path) == len(set(result.path)))
        row("  Path length",        result.path_length)
        row("  Time (ms)",          f"{elapsed:.2f}")


def test_policy_completeness():
    header(7, "Policy Completeness")

    maze = load_maze("mazes/maze_15x15_seed1337.json")
    t = timer(); result = value_iteration(maze); elapsed = t()

    free_cells = sum(1 for row in maze.grid for cell in row if cell == 0)
    valid_actions = {"up", "down", "left", "right", None}

    row("Free cells in maze", free_cells)
    row("Cells in policy",    len(result.policy))
    row("Policy complete",    free_cells == len(result.policy))
    row("All actions valid",  all(a in valid_actions for a in result.policy.values()))
    row("Time (ms)",          f"{elapsed:.2f}")


def test_determinism():
    header(8, "Determinism")

    maze = load_maze("mazes/maze_15x15_seed1337.json")
    t = timer(); r1 = value_iteration(maze); ms1 = t()
    t = timer(); r2 = value_iteration(maze); ms2 = t()

    row("Same iterations", r1.iterations == r2.iterations)
    row("Same path",       r1.path == r2.path)
    row("Same policy",     r1.policy == r2.policy)
    row("Run 1 (ms)",      f"{ms1:.2f}")
    row("Run 2 (ms)",      f"{ms2:.2f}")


def test_visual_output():
    header(9, "Visual Policy Output")

    maze = load_maze("mazes/maze_9x9_seed1337.json")
    t = timer(); result = value_iteration(maze); elapsed = t()

    print(draw_policy(result.policy, maze.grid))
    row("Path length", result.path_length)
    row("Iterations",  result.iterations)
    row("Start value", f"{result.values[maze.start]:.4f}")
    row("Time (ms)",   f"{elapsed:.2f}")


# ─────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────

def main():
    print("MDP Algorithm Tests")
    print("=" * 50)

    tests = [
        # test_basic_functionality,
        # test_optimal_policy,
        # test_convergence_speed,
        # test_value_function_properties,
        test_gamma_sensitivity,
        # test_path_validity,
        # test_policy_completeness,
        # test_determinism,
        # test_visual_output,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  error: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed}/{len(tests)} passed", end="")
    print("" if failed else "  (all passed)")
    if failed:
        print(f"  {failed} failed")


if __name__ == "__main__":
    main()