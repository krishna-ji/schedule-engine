from src.workflows.standard_run import load_input_data
from src.ga.population import generate_course_group_aware_population
from src.decoder.individual_decoder import decode_individual
from src.constraints.hard import course_completeness


def test_course_completeness_zero_in_initialized_population():
    """Population initialization must fully allocate all required course quanta."""
    _, context = load_input_data("data")
    population = generate_course_group_aware_population(1, context, parallel=False)

    assert population, "Expected at least one initialized individual"

    sessions = decode_individual(
        population[0],
        context.courses,
        context.instructors,
        context.groups,
        context.rooms,
    )

    assert course_completeness(sessions, context.courses) == 0
