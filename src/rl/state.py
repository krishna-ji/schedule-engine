"""
State representation for RL environment.

Converts a schedule (Individual) into a fixed-size numerical feature vector
that the RL agent can use for decision-making.

State Vector (5 dimensions):
1. norm_hard_violations: [0.0-1.0] Hard violations / max_possible_conflicts
2. norm_soft_violations: [0.0-1.0] Soft violations / theoretical_max
3. fitness_delta: [-inf, +inf] Change from previous iteration
4. norm_stagnation: [0.0-1.0] Iterations_since_improvement / 100
5. progress: [0.0-1.0] Current_iteration / max_iterations

TODO (Phase 2):
- Implement StateCalculator class
- Normalization utilities
- State history tracking
"""

# Placeholder for Phase 2 implementation
