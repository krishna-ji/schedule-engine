"""
Diversity Heuristics - Maintain Population Diversity

Provides operators that maintain and enhance population diversity to prevent
premature convergence and explore diverse regions of the search space.

Diversity heuristics are useful for:
1. Preventing premature convergence
2. Maintaining exploration vs exploitation balance
3. Niching and speciation in multi-objective optimization

Strategies:
1. Distance Preserving Crossover: Crossover that maintains parent distance
2. Crowding Mutation: Mutation favoring less-explored regions
3. Niching Selection: Selection promoting diverse individuals

Architecture:
- Decorator-based registration with @diversity_heuristic
- Requires population access (operates on multiple individuals)
- Returns diversity metrics or modified population
- Integrated with GA selection/variation operators

Usage:
    from src.heuristics.diversity import distance_preserving_crossover

    # Apply diversity-preserving crossover
    offspring1, offspring2 = distance_preserving_crossover(parent1, parent2, context)
"""

import copy
import random
from collections import defaultdict

from src.domain.types import SchedulingContext
from src.domain.gene import SessionGene
from src.heuristics.registry import diversity_heuristic

# ================
# DISTANCE PRESERVING CROSSOVER
# ================


@diversity_heuristic(
    name="distance_preserving_crossover",
    description="Crossover operator that maintains phenotypic distance between parents",
    priority=1,
    enabled_by_default=True,
    requires_population=True,
    modifies_individual=False,
)
def distance_preserving_crossover(
    parent1: list[SessionGene],
    parent2: list[SessionGene],
    context: SchedulingContext,
    preserve_distance: float = 0.7,
) -> tuple[list[SessionGene], list[SessionGene]]:
    """
    Crossover that preserves distance between parents.

    Standard crossover can create offspring very similar to one parent,
    reducing diversity. This operator ensures offspring maintain a
    minimum distance from both parents.

    Algorithm:
    1. Perform standard crossover
    2. Calculate distance from offspring to each parent
    3. If too similar, inject diversity through mutation
    4. Return diverse offspring

    Args:
        parent1: First parent individual
        parent2: Second parent individual
        context: Scheduling context
        preserve_distance: Minimum normalized distance to maintain (0-1)

    Returns:
        Tuple of two offspring individuals
    """
    # Create offspring via one-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)

    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    # Make deep copies to avoid reference issues
    offspring1 = [copy.copy(gene) for gene in offspring1]
    offspring2 = [copy.copy(gene) for gene in offspring2]

    # Validate all genes after crossover (SessionGene.__post_init__ handles bounds checking)
    for gene in offspring1:
        gene.__post_init__()  # Re-validate quantum ranges

    for gene in offspring2:
        gene.__post_init__()  # Re-validate quantum ranges

    # Calculate distances
    dist_off1_p1 = _calculate_individual_distance(offspring1, parent1)
    dist_off1_p2 = _calculate_individual_distance(offspring1, parent2)
    dist_off2_p1 = _calculate_individual_distance(offspring2, parent1)
    dist_off2_p2 = _calculate_individual_distance(offspring2, parent2)

    # Inject diversity if too similar
    if min(dist_off1_p1, dist_off1_p2) < preserve_distance:
        _inject_diversity(offspring1, context, intensity=0.2)

    if min(dist_off2_p1, dist_off2_p2) < preserve_distance:
        _inject_diversity(offspring2, context, intensity=0.2)

    # Invalidate fitness on offspring to force re-evaluation
    # Note: Fitness is attached by DEAP/RL environment after heuristic returns
    # Just ensure offspring are clean copies

    return offspring1, offspring2


# ================
# CROWDING MUTATION
# ================


@diversity_heuristic(
    name="crowding_mutation",
    description="Mutation that favors less-explored regions of search space",
    priority=2,
    enabled_by_default=True,
    requires_population=True,
    modifies_individual=True,
)
def crowding_mutation(
    individual: list[SessionGene],
    population: list[list[SessionGene]],
    context: SchedulingContext,
    intensity: float = 0.3,
) -> int:
    """
    Mutation that moves individuals away from crowded regions.

    Analyzes population to find common patterns (time slots, room assignments)
    and mutates away from these over-represented choices.

    Algorithm:
    1. Identify crowded regions (common time slots, rooms, instructors)
    2. Mutate genes that use over-represented values
    3. Favor under-represented alternatives

    Args:
        individual: Individual to mutate
        population: Current population (for crowding analysis)
        context: Scheduling context
        intensity: Mutation intensity (0-1)

    Returns:
        Number of genes mutated
    """
    # Analyze population for crowding
    time_usage: dict[int, int] = defaultdict(int)
    room_usage: dict[str, int] = defaultdict(int)
    instructor_usage: dict[str, int] = defaultdict(int)

    for ind in population:
        for gene in ind:
            time_usage[gene.time_quantum] += 1
            room_usage[gene.room_id] += 1
            instructor_usage[gene.instructor_id] += 1

    # Calculate average usage
    avg_time_usage = sum(time_usage.values()) / len(time_usage) if time_usage else 0
    avg_room_usage = sum(room_usage.values()) / len(room_usage) if room_usage else 0
    avg_instructor_usage = (
        sum(instructor_usage.values()) / len(instructor_usage)
        if instructor_usage
        else 0
    )

    mutations = 0

    for gene in individual:
        if random.random() > intensity:
            continue

        # Mutate if in crowded region
        if time_usage[gene.time_quantum] > avg_time_usage * 1.5:
            # Move to less-used time
            from src.io.time_system import QuantumTimeSystem

            time_system = QuantumTimeSystem()
            all_quanta = time_system.get_all_operating_quanta()
            under_used_times = [t for t in all_quanta if time_usage[t] < avg_time_usage]

            if under_used_times:
                course = context.courses.get((gene.course_id, gene.course_type))
                if not course:
                    continue
                # Use gene.duration_quanta (actual session length)
                valid_times = [
                    t
                    for t in under_used_times
                    if t + gene.duration_quanta <= time_system.total_quanta
                ]

                if valid_times:
                    gene.time_quantum = random.choice(valid_times)
                    mutations += 1

        if room_usage[gene.room_id] > avg_room_usage * 1.5:
            # Find less-used room
            course = context.courses.get((gene.course_id, gene.course_type))
            if not course:
                continue
            compatible_rooms = [
                r_id
                for r_id, room in context.rooms.items()
                if room.is_suitable_for_course_type(course.required_room_features)
                and room_usage[r_id] < avg_room_usage
            ]

            if compatible_rooms:
                gene.room_id = random.choice(compatible_rooms)
                mutations += 1

        if instructor_usage[gene.instructor_id] > avg_instructor_usage * 1.5:
            # Find less-used instructor
            course = context.courses.get((gene.course_id, gene.course_type))
            if not course:
                continue
            under_used_instructors = [
                i_id
                for i_id in course.qualified_instructor_ids
                if instructor_usage[i_id] < avg_instructor_usage
            ]

            if under_used_instructors:
                gene.instructor_id = random.choice(under_used_instructors)
                mutations += 1

    # Invalidate fitness
    if hasattr(individual, "fitness") and mutations > 0:
        del individual.fitness.values

    return mutations


# ================
# NICHING SELECTION
# ================


@diversity_heuristic(
    name="niching_selection",
    description="Selection operator that promotes diverse individuals (fitness sharing)",
    priority=3,
    enabled_by_default=True,
    requires_population=True,
    modifies_individual=False,
)
def niching_selection(
    individual: list[SessionGene],
    population: list[list[SessionGene]],
    context: SchedulingContext,
    niche_radius: float = 0.3,
) -> list[SessionGene]:
    """
    Selection that promotes diverse individuals through fitness sharing.

    For RL compatibility, this operates on a single individual:
    - If individual is in a crowded region, replace with more diverse alternative
    - Otherwise return original

    Algorithm:
    1. Calculate how crowded the current individual's region is
    2. If too crowded, select a more isolated individual from population
    3. Otherwise keep original

    Args:
        individual: Individual to potentially replace
        population: Current population
        context: Scheduling context
        niche_radius: Radius for niche definition

    Returns:
        Either original individual or more diverse alternative
    """
    if len(population) <= 1:
        return individual

    # Find individual's index in population
    try:
        ind_idx = next(i for i, ind in enumerate(population) if ind is individual)
    except StopIteration:
        # Individual not in population - return as-is
        return individual

    # Calculate pairwise distances
    distances = {}
    for i, ind1 in enumerate(population):
        for j, ind2 in enumerate(population[i + 1 :], start=i + 1):
            dist = _calculate_individual_distance(ind1, ind2)
            distances[(i, j)] = dist

    # Calculate niche counts (how crowded each individual is)
    niche_counts = defaultdict(float)

    for i in range(len(population)):
        count = 0.0
        for j in range(len(population)):
            if i == j:
                continue

            # Get distance
            key = (min(i, j), max(i, j))
            dist = distances.get(key, 1.0)

            # Sharing function
            if dist < niche_radius:
                count += 1.0 - (dist / niche_radius)

        niche_counts[i] = max(count, 1.0)  # Avoid division by zero

    # Check if current individual is in a crowded region
    current_crowding = niche_counts[ind_idx]
    avg_crowding = sum(niche_counts.values()) / len(niche_counts)

    # If significantly more crowded than average, select a more isolated individual
    if current_crowding > avg_crowding * 1.3:
        # Find least crowded individual
        least_crowded_idx = min(range(len(population)), key=lambda i: niche_counts[i])
        return population[least_crowded_idx]

    # Otherwise keep original
    return individual


# ================
# ADAPTIVE DIVERSITY MAINTENANCE
# ================


@diversity_heuristic(
    name="adaptive_diversity_maintenance",
    description="Dynamically adjust diversity based on convergence state",
    priority=4,
    enabled_by_default=False,  # Advanced feature
    requires_population=True,
    modifies_individual=True,
)
def adaptive_diversity_maintenance(
    individual: list[SessionGene],
    population: list[list[SessionGene]],
    context: SchedulingContext,
    generation: int,
    diversity_threshold: float = 0.2,
) -> list[SessionGene]:
    """
    Adaptively maintain diversity based on population convergence.

    For RL compatibility, operates on single individual:
    - Measures population diversity
    - If too low, injects diversity into given individual
    - Otherwise returns individual unchanged

    Args:
        individual: Individual to potentially diversify
        population: Current population
        context: Scheduling context
        generation: Current generation number
        diversity_threshold: Minimum diversity to maintain

    Returns:
        Modified or original individual
    """
    # Calculate population diversity
    diversity = _calculate_population_diversity(population)

    # If diversity too low, inject diversity into this individual
    if diversity < diversity_threshold:
        modified = [copy.copy(gene) for gene in individual]
        _inject_diversity(modified, context, intensity=0.4)
        return modified

    # Otherwise return unchanged
    return individual


# ================
# HELPER FUNCTIONS
# ================


def _calculate_individual_distance(
    ind1: list[SessionGene], ind2: list[SessionGene]
) -> float:
    """
    Calculate normalized distance between two individuals.

    Distance based on:
    - Time slot differences
    - Room differences
    - Instructor differences

    Returns normalized distance in [0, 1]
    """
    if len(ind1) != len(ind2):
        return 1.0

    differences = 0
    total_comparisons = len(ind1) * 3  # time, room, instructor

    for gene1, gene2 in zip(ind1, ind2, strict=True):
        if gene1.time_quantum != gene2.time_quantum:
            differences += 1
        if gene1.room_id != gene2.room_id:
            differences += 1
        if gene1.instructor_id != gene2.instructor_id:
            differences += 1

    return differences / total_comparisons if total_comparisons > 0 else 0.0


def _calculate_population_diversity(population: list[list[SessionGene]]) -> float:
    """
    Calculate overall population diversity.

    Computes average pairwise distance between all individuals.
    Returns normalized diversity in [0, 1]
    """
    if len(population) < 2:
        return 1.0

    total_distance = 0.0
    comparisons = 0

    for i, ind1 in enumerate(population):
        for ind2 in population[i + 1 :]:
            total_distance += _calculate_individual_distance(ind1, ind2)
            comparisons += 1

    return total_distance / comparisons if comparisons > 0 else 0.0


def _inject_diversity(
    individual: list[SessionGene], context: SchedulingContext, intensity: float
) -> None:
    """
    Inject diversity into individual through random mutations.

    Args:
        individual: Individual to modify
        context: Scheduling context
        intensity: Mutation intensity (0-1)
    """
    for gene in individual:
        if random.random() > intensity:
            continue

        mutation_type = random.choice(["time", "room", "instructor"])

        if mutation_type == "time":
            course = context.courses.get((gene.course_id, gene.course_type))
            if not course:
                continue
            from src.io.time_system import QuantumTimeSystem

            time_system = QuantumTimeSystem()
            all_quanta = time_system.get_all_operating_quanta()
            valid_times = [
                t
                for t in all_quanta
                if t + gene.duration_quanta <= time_system.total_quanta
            ]
            if valid_times:
                gene.time_quantum = random.choice(valid_times)

        elif mutation_type == "room":
            course = context.courses.get((gene.course_id, gene.course_type))
            if not course:
                continue
            compatible_rooms = [
                r_id
                for r_id, room in context.rooms.items()
                if room.is_suitable_for_course_type(course.required_room_features)
            ]
            if compatible_rooms:
                gene.room_id = random.choice(compatible_rooms)

        elif mutation_type == "instructor":
            course = context.courses.get((gene.course_id, gene.course_type))
            if not course:
                continue
            if len(course.qualified_instructor_ids) > 1:
                alternatives = [
                    i
                    for i in course.qualified_instructor_ids
                    if i != gene.instructor_id
                ]
                if alternatives:
                    gene.instructor_id = random.choice(alternatives)

    # Invalidate fitness
    if hasattr(individual, "fitness"):
        del individual.fitness.values
