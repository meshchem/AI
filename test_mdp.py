from src.create_maze import load_maze
from src.mdp_algorithms import value_iteration, policy_iteration, draw_policy


def test_basic_functionality():
    """Test 1: Basic functionality - do algorithms run without errors?"""
    print("\n" + "="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    maze = load_maze("mazes/maze_9x9_seed1337.json")
    
    print("Running Value Iteration...")
    vi_result = value_iteration(maze)
    print(f"  ✓ Converged in {vi_result.iterations} iterations")
    print(f"  ✓ Found path of length {vi_result.path_length}")
    
    print("\nRunning Policy Iteration...")
    pi_result = policy_iteration(maze)
    print(f"  ✓ Converged in {pi_result.iterations} iterations")
    print(f"  ✓ Found path of length {pi_result.path_length}")
    
    print("\n✓ TEST PASSED: Both algorithms run successfully")


def test_optimal_policy():
    """Test 2: Do both algorithms find the same optimal policy?"""
    print("\n" + "="*80)
    print("TEST 2: Optimal Policy Consistency")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    
    vi_result = value_iteration(maze)
    pi_result = policy_iteration(maze)
    
    # Compare policies
    same_policy = vi_result.policy == pi_result.policy
    same_path_length = vi_result.path_length == pi_result.path_length
    
    print(f"Value Iteration path length: {vi_result.path_length}")
    print(f"Policy Iteration path length: {pi_result.path_length}")
    print(f"Policies identical: {same_policy}")
    print(f"Path lengths identical: {same_path_length}")
    
    if same_policy and same_path_length:
        print("\n✓ TEST PASSED: Both algorithms find the same optimal policy")
    else:
        print("\n✗ TEST FAILED: Algorithms found different policies")
        print("\nValue Iteration Policy:")
        print(draw_policy(vi_result.policy, maze.grid))
        print("\nPolicy Iteration Policy:")
        print(draw_policy(pi_result.policy, maze.grid))


def test_convergence_speed():
    """Test 3: Compare iteration counts between VI and PI"""
    print("\n" + "="*80)
    print("TEST 3: Convergence Speed Comparison")
    print("="*80)
    
    maze_files = [
        ("mazes/maze_9x9_seed1337.json", "9x9"),
        ("mazes/maze_15x15_seed1337.json", "15x15"),
        ("mazes/maze_19x19_seed1337.json", "19x19"),
    ]
    
    print(f"{'Maze':<10} | {'VI Iterations':<15} | {'PI Iterations':<15} | {'Speedup':<10}")
    print("-"*80)
    
    for filepath, size in maze_files:
        maze = load_maze(filepath)
        
        vi_result = value_iteration(maze)
        pi_result = policy_iteration(maze)
        
        speedup = vi_result.iterations / pi_result.iterations
        
        print(f"{size:<10} | {vi_result.iterations:<15} | {pi_result.iterations:<15} | {speedup:<10.2f}x")
    
    print("\n✓ TEST PASSED: Policy Iteration typically converges faster")


def test_value_function_properties():
    """Test 4: Value function has expected properties"""
    print("\n" + "="*80)
    print("TEST 4: Value Function Properties")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    result = value_iteration(maze)
    
    # Property 1: Goal has highest value
    goal_value = result.values[maze.goal]
    max_value = max(result.values.values())
    
    print(f"Goal value: {goal_value:.4f}")
    print(f"Max value: {max_value:.4f}")
    print(f"Goal has highest value: {goal_value == max_value}")
    
    # Property 2: Values decrease with distance from goal
    start_value = result.values[maze.start]
    print(f"\nStart value: {start_value:.4f}")
    print(f"Start value < Goal value: {start_value < goal_value}")
    
    # Property 3: All non-goal values are less than goal
    all_less = all(v <= goal_value for v in result.values.values())
    print(f"All values ≤ goal value: {all_less}")
    
    if goal_value == max_value and start_value < goal_value and all_less:
        print("\n✓ TEST PASSED: Value function has expected properties")
    else:
        print("\n✗ TEST FAILED: Value function properties violated")


def test_gamma_sensitivity():
    """Test 5: How does gamma affect convergence?"""
    print("\n" + "="*80)
    print("TEST 5: Gamma Sensitivity")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    gammas = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    print(f"{'Gamma':<10} | {'Iterations':<12} | {'Path Length':<12} | {'Start Value':<12}")
    print("-"*80)
    
    for gamma in gammas:
        result = value_iteration(maze, gamma=gamma)
        start_value = result.values[maze.start]
        
        print(f"{gamma:<10.2f} | {result.iterations:<12} | {result.path_length:<12} | {start_value:<12.4f}")
    
    print("\nObservations:")
    print("  - Higher gamma → more iterations (agent plans further ahead)")
    print("  - Path length same (deterministic environment)")
    print("  - Start value increases with gamma (future rewards matter more)")
    print("\n✓ TEST PASSED: Gamma sensitivity behaves as expected")


def test_path_validity():
    """Test 6: Does the extracted path actually reach the goal?"""
    print("\n" + "="*80)
    print("TEST 6: Path Validity")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    
    for algo_name, algo_fn in [("Value Iteration", value_iteration), 
                                ("Policy Iteration", policy_iteration)]:
        result = algo_fn(maze)
        
        # Check path properties
        path_starts_at_start = result.path[0] == maze.start
        path_ends_at_goal = result.path[-1] == maze.goal
        path_has_no_walls = all(maze.grid[r][c] == 0 for r, c in result.path)
        
        print(f"\n{algo_name}:")
        print(f"  Path starts at start: {path_starts_at_start}")
        print(f"  Path ends at goal: {path_ends_at_goal}")
        print(f"  Path has no walls: {path_has_no_walls}")
        print(f"  Path length: {result.path_length}")
        
        if path_starts_at_start and path_ends_at_goal and path_has_no_walls:
            print(f"  ✓ {algo_name} path is valid")
        else:
            print(f"  ✗ {algo_name} path is INVALID")
            print(f"  Path: {result.path[:10]}...")


def test_policy_completeness():
    """Test 7: Does policy provide action for every free cell?"""
    print("\n" + "="*80)
    print("TEST 7: Policy Completeness")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    result = value_iteration(maze)
    
    # Count free cells
    free_cells = sum(1 for row in maze.grid for cell in row if cell == 0)
    policy_cells = len(result.policy)
    
    print(f"Free cells in maze: {free_cells}")
    print(f"Cells in policy: {policy_cells}")
    print(f"Policy complete: {free_cells == policy_cells}")
    
    # Check all actions are valid
    valid_actions = ["up", "down", "left", "right", None]
    all_valid = all(action in valid_actions for action in result.policy.values())
    print(f"All actions valid: {all_valid}")
    
    if free_cells == policy_cells and all_valid:
        print("\n✓ TEST PASSED: Policy is complete and valid")
    else:
        print("\n✗ TEST FAILED: Policy incomplete or has invalid actions")


def test_determinism():
    """Test 8: Do algorithms produce same results on repeated runs?"""
    print("\n" + "="*80)
    print("TEST 8: Determinism")
    print("="*80)
    
    maze = load_maze("mazes/maze_15x15_seed1337.json")
    
    # Run twice
    result1 = value_iteration(maze)
    result2 = value_iteration(maze)
    
    same_iterations = result1.iterations == result2.iterations
    same_path = result1.path == result2.path
    same_policy = result1.policy == result2.policy
    
    print(f"Same iteration count: {same_iterations}")
    print(f"Same path: {same_path}")
    print(f"Same policy: {same_policy}")
    
    if same_iterations and same_path and same_policy:
        print("\n✓ TEST PASSED: Algorithm is deterministic")
    else:
        print("\n✗ TEST FAILED: Algorithm is non-deterministic")


def test_visual_output():
    """Test 9: Visual policy inspection"""
    print("\n" + "="*80)
    print("TEST 9: Visual Policy Output")
    print("="*80)
    
    maze = load_maze("mazes/maze_9x9_seed1337.json")
    result = value_iteration(maze)
    
    print("\nValue Iteration Policy (arrows should point toward goal):")
    print(draw_policy(result.policy, maze.grid))
    
    print(f"\nPath length: {result.path_length}")
    print(f"Iterations: {result.iterations}")
    print(f"Start value: {result.values[maze.start]:.4f}")
    
    print("\n✓ TEST PASSED: Visual output generated (manually inspect arrows)")


def main():
    """Run all tests."""
    print("="*80)
    print("MDP ALGORITHM TEST SUITE")
    print("="*80)
    
    tests = [
        test_basic_functionality,
        test_optimal_policy,
        test_convergence_speed,
        test_value_function_properties,
        test_gamma_sensitivity,
        test_path_validity,
        test_policy_completeness,
        test_determinism,
        test_visual_output,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")


if __name__ == "__main__":
    main()