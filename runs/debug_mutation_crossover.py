#!/usr/bin/env python3
"""Debug: trace what happens to best individual during one generation."""

import copy
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import BaselineExperiment

# Quick test setup
exp = BaselineExperiment(
    seed=42,
    pop_size=20,  # Small for debugging
    ngen=1,
    cxpb=0.9,
    mutpb=0.2,
    fitness_weights=(-1.0, -1.0),
    data_dir=PROJECT_ROOT / "data",
    output_dir=None,
    opening_time="10:00",
    closing_time="17:00",
    closed_days=["Saturday"],
    init_strategy="hybrid",
    log_interval=1,
    verbose=False,
)

# Initialize properly (mimicking run() method)
exp._init_seeds()
exp._load_data()
exp._create_evaluator()
exp._setup_population_factory()
exp._setup_toolbox()

# Create initial population
print("\n" + "=" * 60)
print("TRACING MUTATION AND CROSSOVER BEHAVIOR")
print("=" * 60)

pop = exp.create_initial_population()

# Sort by fitness for analysis
pop_sorted = sorted(pop, key=lambda x: (x.fitness.values[0], x.fitness.values[1]))

print("\nInitial population fitness (best 5):")
for i, ind in enumerate(pop_sorted[:5]):
    print(f"  {i}: Hard={ind.fitness.values[0]:.0f}, Soft={ind.fitness.values[1]:.0f}")

best = pop_sorted[0]
print(f"\nBest individual: Hard={best.fitness.values[0]:.0f}")

# Test mutation: What happens if we mutate the best individual?
print("\n" + "-" * 60)
print("TEST 1: Mutating best individual 10 times")
print("-" * 60)

for trial in range(10):
    test_ind = copy.deepcopy(best)
    orig_fitness = test_ind.fitness.values

    # Apply mutation
    exp.toolbox.mutate(test_ind)

    # Re-evaluate
    test_ind.fitness.values = exp.evaluate(test_ind)
    new_fitness = test_ind.fitness.values

    change = new_fitness[0] - orig_fitness[0]
    status = "BETTER" if change < 0 else ("SAME" if change == 0 else "WORSE")
    print(
        f"  Trial {trial}: {orig_fitness[0]:.0f} -> {new_fitness[0]:.0f} ({status}, delta={change:+.0f})"
    )

# Test crossover: What happens if we cross best with a random individual?
print("\n" + "-" * 60)
print("TEST 2: Crossing best with random individuals")
print("-" * 60)

for trial in range(10):
    # Pick a random partner
    partner = random.choice(pop)
    partner_fit = partner.fitness.values[0]

    ind1 = copy.deepcopy(best)
    ind2 = copy.deepcopy(partner)

    orig_best_fitness = ind1.fitness.values[0]

    # Apply crossover
    exp.toolbox.mate(ind1, ind2)

    # Re-evaluate
    ind1.fitness.values = exp.evaluate(ind1)
    ind2.fitness.values = exp.evaluate(ind2)

    new_fit1 = ind1.fitness.values[0]
    new_fit2 = ind2.fitness.values[0]

    # Check if we got anything better than original best
    best_offspring = min(new_fit1, new_fit2)
    change = best_offspring - orig_best_fitness
    status = "BETTER" if change < 0 else ("SAME" if change == 0 else "WORSE")

    print(
        f"  Trial {trial}: best({orig_best_fitness:.0f}) x partner({partner_fit:.0f}) -> offspring({new_fit1:.0f}, {new_fit2:.0f}) best={best_offspring:.0f} ({status})"
    )

# Summary
print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)
print("If crossover consistently makes offspring WORSE than the best parent,")
print("the best individual can never be improved through recombination.")
print("This would explain why min stays at 671 across all generations.")
