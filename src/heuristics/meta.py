"""
Meta-Heuristics - High-Level Search Strategies

Provides high-level meta-heuristic strategies that orchestrate combinations
of lower-level heuristics for more sophisticated search behavior.

Meta-heuristics coordinate:
1. When to apply which heuristic
2. How to balance exploration vs exploitation
3. How to escape local optima systematically

Strategies:
1. Variable Neighborhood Descent: Systematic neighborhood exploration
2. Iterated Local Search: Perturbation + local search cycles
3. Adaptive Large Neighborhood: Dynamic destroy-repair patterns

Architecture:
- Decorator-based registration with @meta_heuristic
- Orchestrates construction/perturbation/improvement heuristics
- Adaptive parameter control
- Can wrap GA for hybrid algorithms

Usage:
    from src.heuristics.meta import variable_neighborhood_descent

    # Apply VND to improve solution
    improved = variable_neighborhood_descent(individual, context)
"""

import copy
import random

from src.domain.types import SchedulingContext
from src.domain.gene import SessionGene

# Import other heuristic categories for orchestration
from src.heuristics import improvement, perturbation
from src.heuristics.registry import meta_heuristic

# ================
# VARIABLE NEIGHBORHOOD DESCENT (Systematic neighborhood exploration)
# ================


@meta_heuristic(
    name="variable_neighborhood_descent",
    description="Systematically explore multiple neighborhoods until local optimum",
    priority=1,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def variable_neighborhood_descent(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_neighborhoods: int = 3,
    max_iterations: int = 5,
) -> int:
    """
    Variable Neighborhood Descent (VND) meta-heuristic.

    Systematically explores different neighborhoods in sequence:
    1. Start with neighborhood N1
    2. Apply improvement heuristic for N1
    3. If improved, restart from N1
    4. If not improved, move to N2
    5. Continue until all neighborhoods explored without improvement

    Neighborhoods correspond to different improvement operators:
    - N1: Kempe chain moves
    - N2: Ejection chain moves
    - N3: Variable depth search

    Args:
        individual: Individual to improve
        context: Scheduling context
        max_neighborhoods: Number of neighborhoods to explore
        max_iterations: Max iterations per neighborhood

    Returns:
        Total number of improvements made
    """
    neighborhood_operators = [
        improvement.kempe_chain,
        improvement.ejection_chain,
        improvement.variable_depth_search,
    ]

    # Limit to available neighborhoods
    max_neighborhoods = min(max_neighborhoods, len(neighborhood_operators))

    total_improvements = 0
    current_neighborhood = 0

    while current_neighborhood < max_neighborhoods:
        operator = neighborhood_operators[current_neighborhood]

        # Apply operator
        improvements = operator(individual, context, max_iterations=max_iterations)

        if improvements > 0:
            # Improvement found - restart from first neighborhood
            total_improvements += improvements
            current_neighborhood = 0
        else:
            # No improvement - move to next neighborhood
            current_neighborhood += 1

    return total_improvements


# ================
# ITERATED LOCAL SEARCH (Perturbation + local search cycles)
# ================


@meta_heuristic(
    name="iterated_local_search",
    description="Alternate between perturbation and local search for global optimization",
    priority=2,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def iterated_local_search(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 5,
    perturbation_strength: float = 0.3,
) -> int:
    """
    Iterated Local Search (ILS) meta-heuristic.

    Alternates between perturbation and local search:
    1. Apply local search to current solution (improve)
    2. Perturb solution to escape local optimum
    3. Apply local search to perturbed solution
    4. Accept if better than best found
    5. Repeat

    Algorithm:
    - Local search: Variable neighborhood descent
    - Perturbation: Temporal shift + room shuffle

    Args:
        individual: Individual to improve
        context: Scheduling context
        num_iterations: Number of ILS iterations
        perturbation_strength: Perturbation intensity (0-1)

    Returns:
        Number of iterations that found improvements
    """
    # Save best solution
    best_individual = [copy.copy(gene) for gene in individual]
    best_fitness = _simple_fitness(best_individual, context)

    improvements = 0

    for _iteration in range(num_iterations):
        # Local search phase
        variable_neighborhood_descent(individual, context, max_neighborhoods=2)

        # Evaluate current solution
        current_fitness = _simple_fitness(individual, context)

        # Update best if improved
        if current_fitness < best_fitness:
            best_individual = [copy.copy(gene) for gene in individual]
            best_fitness = current_fitness
            improvements += 1

        # Perturbation phase (escape local optimum)
        perturbation.temporal_shift(
            individual, context, probability=perturbation_strength
        )
        perturbation.room_shuffle(
            individual, context, probability=perturbation_strength * 0.5
        )

    # Restore best solution
    for i, gene in enumerate(best_individual):
        individual[i] = copy.copy(gene)

    # Invalidate fitness
    if hasattr(individual, "fitness"):
        del individual.fitness.values

    return improvements


# ================
# ADAPTIVE LARGE NEIGHBORHOOD SEARCH (Dynamic destroy-repair)
# ================


@meta_heuristic(
    name="adaptive_large_neighborhood",
    description="Adaptive destroy-repair with dynamic neighborhood sizing",
    priority=3,
    enabled_by_default=True,
    requires_population=False,
    modifies_individual=True,
)
def adaptive_large_neighborhood(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 10,
    initial_destroy_rate: float = 0.3,
) -> int:
    """
    Adaptive Large Neighborhood Search (ALNS) meta-heuristic.

    Dynamically adjusts destroy and repair operators based on performance:
    1. Destroy part of solution (remove/randomize genes)
    2. Repair using construction heuristics
    3. Accept if improved
    4. Adapt destroy rate based on success

    Operators track their success rates and are selected adaptively.

    Args:
        individual: Individual to improve
        context: Scheduling context
        num_iterations: Number of ALNS iterations
        initial_destroy_rate: Initial fraction of solution to destroy

    Returns:
        Number of improving iterations
    """

    # Track operator performance
    operator_scores = {
        "temporal_destroy": 1.0,
        "room_destroy": 1.0,
        "random_destroy": 1.0,
    }

    destroy_rate = initial_destroy_rate
    improvements = 0

    # Save best solution
    best_individual = [copy.copy(gene) for gene in individual]
    best_fitness = _simple_fitness(best_individual, context)

    for _iteration in range(num_iterations):
        # Select destroy operator (roulette wheel based on scores)
        destroy_operator = _select_operator_adaptive(operator_scores)

        # Destroy part of solution
        destroyed_indices = _destroy_solution(
            individual, destroy_rate, destroy_operator
        )

        # Repair using construction heuristic
        _repair_solution(individual, destroyed_indices, context)

        # Evaluate
        current_fitness = _simple_fitness(individual, context)

        # Update scores and accept decision
        if current_fitness < best_fitness:
            # Major improvement
            operator_scores[destroy_operator] += 3.0
            best_individual = [copy.copy(gene) for gene in individual]
            best_fitness = current_fitness
            improvements += 1
            destroy_rate = max(0.1, destroy_rate * 0.9)  # Reduce destroy rate

        elif current_fitness < best_fitness * 1.05:
            # Minor improvement
            operator_scores[destroy_operator] += 1.0
            destroy_rate = min(0.5, destroy_rate * 1.1)  # Increase destroy rate

        else:
            # No improvement - revert
            for i, gene in enumerate(best_individual):
                individual[i] = copy.copy(gene)
            destroy_rate = min(0.5, destroy_rate * 1.05)  # Slightly increase

        # Decay scores (prevent stagnation)
        for key in operator_scores:
            operator_scores[key] *= 0.95

    # Restore best solution
    for i, gene in enumerate(best_individual):
        individual[i] = copy.copy(gene)

    # Invalidate fitness
    if hasattr(individual, "fitness"):
        del individual.fitness.values

    return improvements


# ================
# GUIDED LOCAL SEARCH (Penalty-based guidance)
# ================


@meta_heuristic(
    name="guided_local_search",
    description="Local search guided by dynamic penalties on solution features",
    priority=4,
    enabled_by_default=False,  # Advanced feature
    requires_population=False,
    modifies_individual=True,
)
def guided_local_search(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 10,
    penalty_factor: float = 0.1,
) -> int:
    """
    Guided Local Search (GLS) meta-heuristic.

    Augments local search with penalties on solution features:
    - When stuck in local optimum, add penalties to "bad" features
    - Penalties guide search away from previously explored regions
    - Eventually finds different local optima

    Features tracked:
    - Specific time slot assignments
    - Room assignments
    - Instructor assignments

    Args:
        individual: Individual to improve
        context: Scheduling context
        num_iterations: Number of GLS iterations
        penalty_factor: Penalty weight factor

    Returns:
        Number of improving iterations
    """
    # Feature penalties
    time_penalties: dict[tuple[str, int], float] = {}
    room_penalties: dict[tuple[str, str], float] = {}
    instructor_penalties: dict[tuple[str, str], float] = {}

    improvements = 0
    best_fitness = _simple_fitness(individual, context)

    for _iteration in range(num_iterations):
        # Apply local search
        local_improvements = variable_neighborhood_descent(
            individual, context, max_neighborhoods=2, max_iterations=3
        )

        # Evaluate
        current_fitness = _simple_fitness_with_penalties(
            individual,
            context,
            time_penalties,
            room_penalties,
            instructor_penalties,
            penalty_factor,
        )

        if local_improvements > 0:
            improvements += 1
            # Update best fitness (used for tracking)
            best_fitness = current_fitness  # noqa: F841
        else:
            # Stuck - add penalties to current features
            for gene in individual:
                key_time = (gene.course_id, gene.time_quantum)
                key_room = (gene.course_id, gene.room_id)
                key_instructor = (gene.course_id, gene.instructor_id)

                time_penalties[key_time] = time_penalties.get(key_time, 0) + 1
                room_penalties[key_room] = room_penalties.get(key_room, 0) + 1
                instructor_penalties[key_instructor] = (
                    instructor_penalties.get(key_instructor, 0) + 1
                )

    # Invalidate fitness
    if hasattr(individual, "fitness"):
        del individual.fitness.values

    return improvements


# ================
# HELPER FUNCTIONS
# ================


def _simple_fitness(individual: list[SessionGene], context: SchedulingContext) -> float:
    """Simple fitness approximation (lower is better)."""
    violations = 0

    # Count time conflicts
    time_usage = {}
    for gene in individual:
        # course = context.courses[(gene.course_id, gene.course_type)]  # Unused
        time_range = tuple(
            range(gene.time_quantum, gene.time_quantum + gene.duration_quanta)
        )

        for group_id in gene.group_ids:
            key = (group_id, time_range)
            if key in time_usage:
                violations += 1
            time_usage[key] = True

        # Instructor conflicts
        key = (gene.instructor_id, time_range)
        if key in time_usage:
            violations += 1
        time_usage[key] = True

        # Room conflicts
        key = (gene.room_id, time_range)
        if key in time_usage:
            violations += 1
        time_usage[key] = True

    return violations


def _simple_fitness_with_penalties(
    individual: list[SessionGene],
    context: SchedulingContext,
    time_penalties: dict,
    room_penalties: dict,
    instructor_penalties: dict,
    penalty_factor: float,
) -> float:
    """Fitness with GLS penalties."""
    base_fitness = _simple_fitness(individual, context)

    penalty = 0.0
    for gene in individual:
        key_time = (gene.course_id, gene.time_quantum)
        key_room = (gene.course_id, gene.room_id)
        key_instructor = (gene.course_id, gene.instructor_id)

        penalty += time_penalties.get(key_time, 0)
        penalty += room_penalties.get(key_room, 0)
        penalty += instructor_penalties.get(key_instructor, 0)

    return base_fitness + penalty * penalty_factor


def _select_operator_adaptive(operator_scores: dict[str, float]) -> str:
    """Select operator using roulette wheel selection based on scores."""
    total_score = sum(operator_scores.values())

    if total_score == 0:
        return random.choice(list(operator_scores.keys()))

    rand_val = random.uniform(0, total_score)
    cumulative = 0.0

    for operator, score in operator_scores.items():
        cumulative += score
        if cumulative >= rand_val:
            return operator

    return list(operator_scores.keys())[-1]


def _destroy_solution(
    individual: list[SessionGene], destroy_rate: float, destroy_operator: str
) -> list[int]:
    """
    Destroy part of solution by randomizing genes.

    Returns indices of destroyed genes.
    """
    from src.io.time_system import QuantumTimeSystem

    num_destroy = int(len(individual) * destroy_rate)
    destroy_indices = random.sample(range(len(individual)), num_destroy)

    # Randomize destroyed genes based on operator
    for idx in destroy_indices:
        gene = individual[idx]

        if destroy_operator == "temporal_destroy":
            # Randomize time (ensure session fits within valid range)
            time_system = QuantumTimeSystem()
            all_quanta = time_system.get_all_operating_quanta()
            session_duration = gene.duration_quanta
            # Only select quanta where the full session fits
            valid_start_quanta = [
                q
                for q in all_quanta
                if q + session_duration <= time_system.total_quanta
            ]
            if valid_start_quanta:
                gene.time_quantum = random.choice(valid_start_quanta)
            elif all_quanta:
                # Fallback: use earliest quantum if no perfect fit
                gene.time_quantum = min(all_quanta)

        elif destroy_operator == "room_destroy":
            # Keep time, randomize room (simplified)
            pass  # Would need context access

        elif destroy_operator == "random_destroy":
            # Randomize everything (simplified)
            pass

    return destroy_indices


def _repair_solution(
    individual: list[SessionGene],
    destroyed_indices: list[int],
    context: SchedulingContext,
) -> None:
    """
    Repair destroyed genes using construction heuristics.

    This is simplified - full implementation would use proper construction.
    """
    # Simple repair: for each destroyed gene, find valid assignment
    from src.io.time_system import QuantumTimeSystem

    time_system = QuantumTimeSystem()
    all_quanta = time_system.get_all_operating_quanta()

    for idx in destroyed_indices:
        gene = individual[idx]
        course = context.courses[(gene.course_id, gene.course_type)]

        # Find valid time (ensure session fits within operating hours)
        # Use gene.duration_quanta to respect the actual session length
        session_duration = gene.duration_quanta
        valid_times = [
            t for t in all_quanta if t + session_duration <= time_system.total_quanta
        ]

        if valid_times:
            gene.time_quantum = random.choice(valid_times)
        elif all_quanta:
            # Fallback: use first available quantum if no perfect fit
            gene.time_quantum = min(all_quanta)

        # Ensure valid room (simplified - just pick any room with matching features)
        compatible_rooms = [
            r_id
            for r_id, room in context.rooms.items()
            if room.is_suitable_for_course_type(course.required_room_features)
        ]

        if compatible_rooms:
            gene.room_id = random.choice(compatible_rooms)
        elif context.rooms:
            # Fallback: pick any available room if no perfect match
            gene.room_id = random.choice(list(context.rooms.keys()))

        # Ensure qualified instructor
        if course.qualified_instructor_ids:
            gene.instructor_id = random.choice(course.qualified_instructor_ids)
