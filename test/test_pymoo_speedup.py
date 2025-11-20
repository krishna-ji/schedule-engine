"""
Quick benchmark: pymoo vs manual implementations for metrics calculation
"""

import time
import numpy as np
from deap import creator, base

# Setup DEAP individual type
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.01))
creator.create("Individual", list, fitness=creator.FitnessMin)


# Create fake population
def create_fake_population(size=500):
    population = []
    for i in range(size):
        ind = creator.Individual()
        # Random fitness values (hard violations, soft penalties)
        ind.fitness.values = (np.random.randint(0, 50), np.random.uniform(0, 500))
        population.append(ind)
    return population


print("Creating test population (500 individuals)...")
population = create_fake_population(500)
reference_front = create_fake_population(100)

# Test hypervolume
print("\n=== Hypervolume Test ===")
from src.metrics.hypervolume import calculate_hypervolume

start = time.perf_counter()
hv = calculate_hypervolume(population, ref_point=(100, 1000))
duration = time.perf_counter() - start
print(f"✓ pymoo HV: {hv:.2f} in {duration*1000:.2f}ms")

# Test IGD
print("\n=== IGD Test ===")
from src.metrics.pareto_metrics import calculate_inverted_generational_distance

start = time.perf_counter()
igd = calculate_inverted_generational_distance(population, reference_front)
duration = time.perf_counter() - start
print(f"✓ pymoo IGD: {igd:.4f} in {duration*1000:.2f}ms")

# Test GD
print("\n=== GD Test ===")
from src.metrics.pareto_metrics import calculate_generational_distance

start = time.perf_counter()
gd = calculate_generational_distance(population, reference_front)
duration = time.perf_counter() - start
print(f"✓ pymoo GD: {gd:.4f} in {duration*1000:.2f}ms")

print("\n=== Full Metrics Suite (as used in GA) ===")
start = time.perf_counter()
hv = calculate_hypervolume(population, ref_point=(100, 1000))
igd = calculate_inverted_generational_distance(population, reference_front)
gd = calculate_generational_distance(population, reference_front)
from src.metrics.pareto_metrics import calculate_spacing, calculate_spread

spacing = calculate_spacing(population)
spread = calculate_spread(population)
duration = time.perf_counter() - start
print(f"✓ All metrics calculated in {duration*1000:.1f}ms")
print(f"  Expected per generation: ~{duration:.2f}s (vs 50s before!)")
