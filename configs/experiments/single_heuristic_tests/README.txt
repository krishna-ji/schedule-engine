SINGLE HEURISTIC TEST TEMPLATES
================================

This folder contains 25 pre-configured test templates, one for each heuristic.
Each file has exactly ONE heuristic enabled and all others disabled.

USAGE
-----

To test a specific heuristic:

1. Copy the desired test file content
2. Paste it into: configs/experiments/heuristic_testing.py
3. Run: uv run heuristic-testing --test --name "test-name"

EXAMPLE: Testing Kempe Chain
-----------------------------

1. Copy content from: test_kempe_chain.py
2. Paste into: configs/experiments/heuristic_testing.py (replace HeuristicTestingTestConfig class)
3. Run:
   uv run heuristic-testing --test --name "kempe-chain-r01"

4. Results saved to:
   output/f-heuristic-testing/evaluation_<timestamp>/


AVAILABLE TESTS (25 total)
---------------------------

Construction (3):
  test_largest_degree_first.py
  test_most_constrained_first.py
  test_earliest_deadline_first.py

Perturbation (5):
  test_random_swap.py
  test_temporal_shift.py
  test_room_shuffle.py
  test_instructor_reassign.py
  test_multi_perturbation.py

Improvement (3):
  test_kempe_chain.py
  test_ejection_chain.py
  test_variable_depth_search.py

Diversity (4):
  test_distance_preserving_crossover.py
  test_crowding_mutation.py
  test_niching_selection.py
  test_adaptive_diversity_maintenance.py

Meta (4):
  test_variable_neighborhood_descent.py
  test_iterated_local_search.py
  test_adaptive_large_neighborhood.py
  test_guided_local_search.py

Repair (6):
  test_exhaustive_repair.py
  test_greedy_repair.py
  test_igls_repair.py
  test_lns_repair.py
  test_memetic_repair.py
  test_selective_repair.py


QUICK WORKFLOW
--------------

# Test 3 different heuristics in sequence:

# 1. Kempe Chain
Copy test_kempe_chain.py → configs/experiments/heuristic_testing.py
uv run heuristic-testing --test --name "kempe-chain"

# 2. Random Swap
Copy test_random_swap.py → configs/experiments/heuristic_testing.py
uv run heuristic-testing --test --name "random-swap"

# 3. Variable Neighborhood Descent
Copy test_variable_neighborhood_descent.py → configs/experiments/heuristic_testing.py
uv run heuristic-testing --test --name "vnd"

# Compare results:
uv run list-experiments


NOTES
-----

- Repair heuristics have repair_enabled: bool = True
- All other heuristics have repair_enabled: bool = False
- Test profile: 30 gens, 10 pop (~2-5 min)
- Prod profile: 2000 gens, 200 pop (~1-3 hours)
