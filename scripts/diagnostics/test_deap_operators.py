"""
Diagnostic script to verify DEAP operator tuple handling.

Tests that crossover and mutation operators properly return and unpack tuples
to prevent GPU evaluation failures caused by tuple corruption.
"""

from deap import base, creator, tools
from src.ga.sessiongene import SessionGene
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual
from src.core.types import SchedulingContext
import random

# Setup DEAP types
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_mock_individual():
    """Create a mock individual with SessionGenes."""
    genes = [
        SessionGene(
            course_id="TEST101",
            course_type="theory",
            instructor_id=f"INST{i}",
            group_ids=[f"GRP{i}"],
            room_id=f"ROOM{i}",
            start_quanta=i * 10,
            num_quanta=2,
        )
        for i in range(5)
    ]
    ind = creator.Individual(genes)
    ind.fitness.values = (0.0, 0.0)
    return ind


def create_mock_context():
    """Create minimal SchedulingContext for testing."""
    from src.entities.course import Course
    from src.entities.instructor import Instructor
    from src.entities.group import Group
    from src.entities.room import Room

    context = SchedulingContext(
        courses={
            ("TEST101", "theory"): Course(
                course_id="TEST101",
                course_code="TEST101",
                course_name="Test Course",
                course_type="theory",
                quanta_per_week=2,
            )
        },
        instructors={
            f"INST{i}": Instructor(
                instructor_id=f"INST{i}",
                name=f"Instructor {i}",
                is_full_time=True,
                qualified_courses=[("TEST101", "theory")],
            )
            for i in range(10)
        },
        groups={
            f"GRP{i}": Group(
                group_id=f"GRP{i}", group_name=f"Group {i}", student_count=30
            )
            for i in range(10)
        },
        rooms={
            f"ROOM{i}": Room(room_id=f"ROOM{i}", room_name=f"Room {i}", capacity=50)
            for i in range(10)
        },
        available_quanta=set(range(70)),
    )
    return context


def test_crossover_operator():
    """Test that crossover operator properly handles tuples."""
    print("\n" + "=" * 60)
    print("TEST 1: Crossover Operator Tuple Handling")
    print("=" * 60)

    toolbox = base.Toolbox()
    toolbox.register("mate", crossover_course_group_aware, cx_prob=0.5)

    # Create two individuals
    ind1 = create_mock_individual()
    ind2 = create_mock_individual()

    print(f"Before crossover:")
    print(f"  ind1 type: {type(ind1)}, len: {len(ind1)}")
    print(f"  ind1[0] type: {type(ind1[0])}, has course_id: {hasattr(ind1[0], 'course_id')}")
    print(f"  ind2 type: {type(ind2)}, len: {len(ind2)}")
    print(f"  ind2[0] type: {type(ind2[0])}, has course_id: {hasattr(ind2[0], 'course_id')}")

    # Call crossover
    result = toolbox.mate(ind1, ind2)

    print(f"\nAfter crossover (result):")
    print(f"  result type: {type(result)}")
    print(f"  result length: {len(result)}")

    # Unpack tuple (correct DEAP pattern)
    child1, child2 = result

    print(f"\nAfter unpacking:")
    print(f"  child1 type: {type(child1)}, len: {len(child1)}")
    print(f"  child1[0] type: {type(child1[0])}, has course_id: {hasattr(child1[0], 'course_id')}")
    print(f"  child2 type: {type(child2)}, len: {len(child2)}")
    print(f"  child2[0] type: {type(child2[0])}, has course_id: {hasattr(child2[0], 'course_id')}")

    # Verify genes are SessionGene objects
    assert isinstance(
        child1[0], SessionGene
    ), f"child1[0] should be SessionGene, got {type(child1[0])}"
    assert isinstance(
        child2[0], SessionGene
    ), f"child2[0] should be SessionGene, got {type(child2[0])}"

    print("\n✓ Crossover test PASSED: Children contain SessionGene objects")
    return True


def test_mutation_operator():
    """Test that mutation operator properly handles tuples."""
    print("\n" + "=" * 60)
    print("TEST 2: Mutation Operator Tuple Handling")
    print("=" * 60)

    toolbox = base.Toolbox()
    context = create_mock_context()
    toolbox.register("mutate", mutate_individual, context=context, mut_prob=1.0, guided=False)

    # Create individual
    ind = create_mock_individual()

    print(f"Before mutation:")
    print(f"  ind type: {type(ind)}, len: {len(ind)}")
    print(f"  ind[0] type: {type(ind[0])}, has course_id: {hasattr(ind[0], 'course_id')}")

    # Call mutation
    result = toolbox.mutate(ind)

    print(f"\nAfter mutation (result):")
    print(f"  result type: {type(result)}")
    print(f"  result length: {len(result)}")

    # Unpack tuple (correct DEAP pattern)
    mutant = result[0]

    print(f"\nAfter unpacking:")
    print(f"  mutant type: {type(mutant)}, len: {len(mutant)}")
    print(f"  mutant[0] type: {type(mutant[0])}, has course_id: {hasattr(mutant[0], 'course_id')}")

    # Verify genes are SessionGene objects
    assert isinstance(
        mutant[0], SessionGene
    ), f"mutant[0] should be SessionGene, got {type(mutant[0])}"

    print("\n✓ Mutation test PASSED: Mutant contains SessionGene objects")
    return True


def test_operator_chain():
    """Test full operator chain: selection -> crossover -> mutation."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Operator Chain (Selection -> Crossover -> Mutation)")
    print("=" * 60)

    toolbox = base.Toolbox()
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", crossover_course_group_aware, cx_prob=0.5)
    toolbox.register("clone", lambda x: creator.Individual(list(x)))

    context = create_mock_context()
    toolbox.register("mutate", mutate_individual, context=context, mut_prob=1.0, guided=False)

    # Create population
    population = [create_mock_individual() for _ in range(10)]

    print(f"Initial population: {len(population)} individuals")
    print(f"  population[0] type: {type(population[0])}")
    print(f"  population[0][0] type: {type(population[0][0])}")

    # Selection
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    print(f"\nAfter selection + clone: {len(offspring)} offspring")
    print(f"  offspring[0] type: {type(offspring[0])}")
    print(f"  offspring[0][0] type: {type(offspring[0][0])}")

    # Crossover (mimic _parallel_crossover)
    cxpb = 0.7
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < cxpb:
            result = toolbox.mate(offspring[i], offspring[i + 1])
            offspring[i], offspring[i + 1] = result  # CRITICAL: Unpack tuple
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values

    print(f"\nAfter crossover:")
    print(f"  offspring[0] type: {type(offspring[0])}")
    print(f"  offspring[0][0] type: {type(offspring[0][0])}")

    # Mutation (mimic _parallel_mutation)
    mutpb = 0.2
    for i in range(len(offspring)):
        if random.random() < mutpb:
            result = toolbox.mutate(offspring[i])
            offspring[i] = result[0]  # CRITICAL: Unpack tuple
            del offspring[i].fitness.values

    print(f"\nAfter mutation:")
    print(f"  offspring[0] type: {type(offspring[0])}")
    print(f"  offspring[0][0] type: {type(offspring[0][0])}")

    # Verify all genes are SessionGene objects
    for idx, ind in enumerate(offspring):
        for gene_idx, gene in enumerate(ind):
            assert isinstance(
                gene, SessionGene
            ), f"offspring[{idx}][{gene_idx}] should be SessionGene, got {type(gene)}"

    print("\n✓ Operator chain test PASSED: All genes remain SessionGene objects")
    return True


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("DEAP OPERATOR TUPLE HANDLING DIAGNOSTIC")
    print("=" * 70)
    print("\nThis test verifies that DEAP genetic operators properly return and")
    print("unpack tuples to prevent GPU evaluation failures.")
    print()

    random.seed(42)  # Reproducible tests

    try:
        test_crossover_operator()
        test_mutation_operator()
        test_operator_chain()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nDEAP operators correctly handle tuple unpacking.")
        print("If GPU errors persist, check:")
        print("  1. Process has been restarted to load fixes")
        print("  2. No other code paths modify individuals")
        print("  3. GPU evaluator receives correct individual objects")
        print()

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
