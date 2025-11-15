"""
Unit tests for diversity metric calculations in StateEncoder.

Tests phenotype_diversity and unique_fitness_ratio features.
"""

import pytest
import numpy as np
from deap import creator, base

from src.rl.gym_env.state_encoder import StateEncoder
from src.core.types import SessionGene


# Setup DEAP creator types (required for Individual)
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def create_test_individual(
    hard_violations: float, soft_violations: float, num_genes: int = 10
) -> creator.Individual:
    """Create a test individual with specified fitness."""
    genes = [
        SessionGene(
            course_id=f"COURSE{i}",
            course_type="LEC",
            group_ids=[f"GROUP{i}"],
            timeslot_index=i,
            room_id=f"ROOM{i % 3}",
        )
        for i in range(num_genes)
    ]
    individual = creator.Individual(genes)
    individual.fitness.values = (hard_violations, soft_violations)
    return individual


class TestPhenotypeDiversity:
    """Test phenotype diversity calculation."""

    def test_empty_population(self):
        """Empty population should return 0.0."""
        encoder = StateEncoder()
        diversity = encoder._calculate_phenotype_diversity([])
        assert diversity == 0.0

    def test_single_individual(self):
        """Single individual should return 0.0 (no comparison possible)."""
        encoder = StateEncoder()
        pop = [create_test_individual(10.0, 5.0)]
        diversity = encoder._calculate_phenotype_diversity(pop)
        assert diversity == 0.0

    def test_identical_fitness(self):
        """Population with identical fitness should have low diversity."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),
        ]
        diversity = encoder._calculate_phenotype_diversity(pop)
        assert diversity == 0.0, "Identical fitness should result in zero diversity"

    def test_diverse_fitness(self):
        """Population with diverse fitness values should have high diversity."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(0.0, 0.0),
            create_test_individual(50.0, 25.0),
            create_test_individual(100.0, 50.0),
        ]
        diversity = encoder._calculate_phenotype_diversity(pop)
        assert diversity > 0.0, "Diverse fitness should result in positive diversity"
        assert diversity <= 1.0, "Diversity should be normalized to [0, 1]"

    def test_two_clusters(self):
        """Population with two fitness clusters should have moderate diversity."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.0, 5.0),
            create_test_individual(10.5, 5.2),  # Close to first
            create_test_individual(50.0, 25.0),
            create_test_individual(50.5, 25.2),  # Close to third
        ]
        diversity = encoder._calculate_phenotype_diversity(pop)
        assert (
            0.0 < diversity <= 1.0
        ), "Clustered fitness should have moderate diversity"

    def test_large_population(self):
        """Test with larger population (50 individuals)."""
        encoder = StateEncoder()
        np.random.seed(42)
        pop = [
            create_test_individual(
                float(np.random.uniform(0, 100)), float(np.random.uniform(0, 50))
            )
            for _ in range(50)
        ]
        diversity = encoder._calculate_phenotype_diversity(pop)
        assert 0.0 <= diversity <= 1.0, "Diversity should be in [0, 1] range"


class TestUniqueFitnessRatio:
    """Test unique fitness ratio calculation."""

    def test_empty_population(self):
        """Empty population should return 0.0."""
        encoder = StateEncoder()
        ratio = encoder._calculate_unique_fitness_ratio([])
        assert ratio == 0.0

    def test_single_individual(self):
        """Single individual should return 1.0 (100% unique)."""
        encoder = StateEncoder()
        pop = [create_test_individual(10.0, 5.0)]
        ratio = encoder._calculate_unique_fitness_ratio(pop)
        assert ratio == 1.0, "Single individual should be 100% unique"

    def test_all_identical(self):
        """All identical fitness should return minimum ratio (1/pop_size)."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),
        ]
        ratio = encoder._calculate_unique_fitness_ratio(pop)
        assert ratio == 0.25, "4 identical individuals → 1 unique / 4 total = 0.25"

    def test_all_unique(self):
        """All unique fitness should return 1.0."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.0, 5.0),
            create_test_individual(20.0, 10.0),
            create_test_individual(30.0, 15.0),
            create_test_individual(40.0, 20.0),
        ]
        ratio = encoder._calculate_unique_fitness_ratio(pop)
        assert ratio == 1.0, "All unique fitness should result in 1.0 ratio"

    def test_partial_duplicates(self):
        """Some duplicates should return intermediate ratio."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.0, 5.0),
            create_test_individual(10.0, 5.0),  # Duplicate
            create_test_individual(20.0, 10.0),
            create_test_individual(30.0, 15.0),
        ]
        ratio = encoder._calculate_unique_fitness_ratio(pop)
        assert ratio == 0.75, "3 unique out of 4 total → 0.75"

    def test_floating_point_tolerance(self):
        """Nearly identical fitness (within rounding) should be treated as duplicate."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(10.00000, 5.00000),
            create_test_individual(10.00001, 5.00001),  # Very close (rounds to same)
            create_test_individual(20.0, 10.0),
        ]
        ratio = encoder._calculate_unique_fitness_ratio(pop)
        # With 4 decimal place rounding, these should be treated as duplicates
        assert ratio == pytest.approx(
            2 / 3, abs=0.01
        ), "Nearly identical should be treated as duplicates"

    def test_convergence_detection(self):
        """Test use case: detecting population convergence."""
        encoder = StateEncoder()

        # Diverse population (early in evolution)
        diverse_pop = [
            create_test_individual(float(i * 10), float(i * 5)) for i in range(10)
        ]
        diverse_ratio = encoder._calculate_unique_fitness_ratio(diverse_pop)

        # Converged population (late in evolution)
        converged_pop = [create_test_individual(10.0, 5.0) for _ in range(10)]
        converged_ratio = encoder._calculate_unique_fitness_ratio(converged_pop)

        assert (
            diverse_ratio > converged_ratio
        ), "Diverse population should have higher unique ratio"
        assert diverse_ratio == 1.0, "Fully diverse population should have ratio 1.0"
        assert (
            converged_ratio == 0.1
        ), "Fully converged population should have ratio 1/pop_size"


class TestStateEncoderIntegration:
    """Test integration of diversity metrics in full state encoding."""

    def test_encode_includes_new_metrics(self):
        """Verify encoded state includes phenotype_diversity and unique_fitness_ratio."""
        encoder = StateEncoder(max_generations=100, history_size=5)

        # Create diverse population
        pop = [
            create_test_individual(float(i * 10), float(i * 5), num_genes=20)
            for i in range(10)
        ]

        # Encode state
        obs = encoder.encode(
            pop, current_generation=50, generations_without_improvement=5
        )

        # Check observation dimension (17 base features + 5 history = 22)
        assert len(obs) == 22, f"Expected 22 features, got {len(obs)}"

        # Check all values are normalized to [0, 1] or [-1, 1]
        assert np.all(obs >= -1.0) and np.all(
            obs <= 1.0
        ), "All features should be normalized"

        # Indices 7 and 9 should contain phenotype_diversity and unique_fitness_ratio
        phenotype_idx = 7
        unique_ratio_idx = 9

        assert (
            0.0 <= obs[phenotype_idx] <= 1.0
        ), "Phenotype diversity should be in [0, 1]"
        assert (
            0.0 <= obs[unique_ratio_idx] <= 1.0
        ), "Unique fitness ratio should be in [0, 1]"

    def test_observation_dim_property(self):
        """Test observation_dim property returns correct dimension."""
        encoder = StateEncoder(max_generations=100, history_size=10)
        assert encoder.observation_dim == 27, "17 base features + 10 history = 27"

        encoder2 = StateEncoder(max_generations=200, history_size=5)
        assert encoder2.observation_dim == 22, "17 base features + 5 history = 22"

    def test_convergence_scenario(self):
        """Test encoding behavior during population convergence."""
        encoder = StateEncoder(max_generations=1000, history_size=5)

        # Early generation: diverse population
        diverse_pop = [
            create_test_individual(float(i * 10), float(i * 5), num_genes=20)
            for i in range(20)
        ]
        early_obs = encoder.encode(
            diverse_pop, current_generation=10, generations_without_improvement=0
        )

        # Late generation: converged population
        converged_pop = [
            create_test_individual(5.0, 2.5, num_genes=20) for _ in range(20)
        ]
        late_obs = encoder.encode(
            converged_pop, current_generation=900, generations_without_improvement=50
        )

        # Unique fitness ratio should be much lower in converged population
        unique_ratio_idx = 9
        assert (
            early_obs[unique_ratio_idx] > late_obs[unique_ratio_idx]
        ), "Converged population should have lower unique fitness ratio"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_fitness_values(self):
        """Test with very small fitness values (near zero)."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(0.001, 0.0001),
            create_test_individual(0.002, 0.0002),
        ]

        phenotype_div = encoder._calculate_phenotype_diversity(pop)
        unique_ratio = encoder._calculate_unique_fitness_ratio(pop)

        assert 0.0 <= phenotype_div <= 1.0
        assert 0.0 <= unique_ratio <= 1.0

    def test_very_large_fitness_values(self):
        """Test with very large fitness values."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(1000000.0, 500000.0),
            create_test_individual(2000000.0, 1000000.0),
        ]

        phenotype_div = encoder._calculate_phenotype_diversity(pop)
        unique_ratio = encoder._calculate_unique_fitness_ratio(pop)

        assert 0.0 <= phenotype_div <= 1.0
        assert 0.0 <= unique_ratio <= 1.0

    def test_negative_fitness_values(self):
        """Test with negative fitness values (though unusual in this domain)."""
        encoder = StateEncoder()
        pop = [
            create_test_individual(-10.0, -5.0),
            create_test_individual(-20.0, -10.0),
        ]

        phenotype_div = encoder._calculate_phenotype_diversity(pop)
        unique_ratio = encoder._calculate_unique_fitness_ratio(pop)

        assert 0.0 <= phenotype_div <= 1.0
        assert 0.0 <= unique_ratio <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
