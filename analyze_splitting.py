"""Check if multi-block theory splitting is causing issues."""

from src.config import get_config
from src.encoder.input_encoder import load_data_parallel
from src.ga.population import generate_course_group_aware_population_old
from src.workflows.context_builder import build_scheduling_context
from collections import defaultdict

config = get_config()
entities = load_data_parallel(config.data.root_directory)
context = build_scheduling_context(entities, config)

print(f"Total courses: {len(context.courses)}")
print(f"Total course-group pairs needed:")

course_group_pairs = []
for course_key, course in context.courses.items():
    for group_id in course.enrolled_group_ids:
        course_group_pairs.append((course_key, [group_id]))

print(f"  Total: {len(course_group_pairs)}")

# Count expected genes per course type
theory_quanta_dist = defaultdict(int)
practical_quanta_dist = defaultdict(int)

for course_key, course in context.courses.items():
    quanta = course.quanta_per_week
    if course.course_type == "theory":
        theory_quanta_dist[quanta] += 1
    else:
        practical_quanta_dist[quanta] += 1

print("\nTheory course quanta distribution:")
for quanta, count in sorted(theory_quanta_dist.items()):
    # Calculate expected genes per course
    num_blocks = (quanta + 1) // 2  # ceil(quanta / 2)
    print(f"  {quanta} quanta: {count} courses -> {num_blocks} genes each")

print("\nPractical course quanta distribution:")
for quanta, count in sorted(practical_quanta_dist.items()):
    print(f"  {quanta} quanta: {count} courses -> 1 gene each")

# Calculate total expected genes
total_expected_genes = 0
for course_key, course in context.courses.items():
    num_enrollments = len(course.enrolled_group_ids)
    if course.course_type == "theory":
        num_blocks = (course.quanta_per_week + 1) // 2
        total_expected_genes += num_enrollments * num_blocks
    else:
        total_expected_genes += num_enrollments

print(f"\nTotal expected genes (with splitting): {total_expected_genes}")
print(f"vs single-gene-per-course: {len(course_group_pairs)}")
