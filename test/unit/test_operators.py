"""Unit tests for GA operators (crossover, mutation, repair)."""

import pytest
from unittest.mock import MagicMock
from copy import deepcopy

from src.ga.sessiongene import SessionGene
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual


class TestCrossover:
    """Test suite for crossover operators."""

    def test_crossover_preserves_all_genes(self):
        """Test that crossover doesn't lose or duplicate genes."""
        # Create two parent individuals
        parent1 = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R1", start_quanta=0, num_quanta=3
            ),
            SessionGene(
                "CS102", "theory", "I2", ["G2"], "R2", start_quanta=5, num_quanta=3
            ),
        ]
        parent2 = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R3", start_quanta=10, num_quanta=3
            ),
            SessionGene(
                "CS102", "theory", "I2", ["G2"], "R4", start_quanta=15, num_quanta=3
            ),
        ]

        context = MagicMock()

        # Perform crossover (crossover doesn't need context, it takes cx_prob)
        child1, child2 = crossover_course_group_aware(
            deepcopy(parent1), deepcopy(parent2), cx_prob=0.5
        )

        # Check children have same number of genes
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)

        # Check all course-group pairs are preserved
        p1_pairs = {(g.course_id, tuple(g.group_ids)) for g in parent1}
        c1_pairs = {(g.course_id, tuple(g.group_ids)) for g in child1}
        assert p1_pairs == c1_pairs

    def test_crossover_returns_tuple_of_two_individuals(self):
        """Test crossover returns exactly two offspring."""
        parent1 = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R1", start_quanta=0, num_quanta=3
            )
        ]
        parent2 = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R2", start_quanta=5, num_quanta=3
            )
        ]

        context = MagicMock()

        result = crossover_course_group_aware(
            deepcopy(parent1), deepcopy(parent2), cx_prob=0.5
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


class TestMutation:
    """Test suite for mutation operators."""

    def test_mutation_preserves_gene_count(self):
        """Test that mutation doesn't add or remove genes."""
        individual = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R1", start_quanta=0, num_quanta=3
            ),
            SessionGene(
                "CS102", "theory", "I2", ["G2"], "R2", start_quanta=5, num_quanta=3
            ),
        ]

        context = MagicMock()
        context.available_quanta = list(range(100))
        context.rooms = [
            MagicMock(room_id="R1"),
            MagicMock(room_id="R2"),
            MagicMock(room_id="R3"),
        ]

        original_len = len(individual)
        mutated = mutate_individual(deepcopy(individual), context, mut_prob=0.5)

        assert len(mutated[0]) == original_len

    def test_mutation_returns_tuple(self):
        """Test mutation returns tuple (required by DEAP)."""
        individual = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R1", start_quanta=0, num_quanta=3
            ),
        ]

        context = MagicMock()
        context.available_quanta = list(range(100))
        context.rooms = [MagicMock(room_id="R1")]

        result = mutate_individual(individual, context, mut_prob=0.2)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_mutation_with_zero_probability_unchanged(self):
        """Test mutation with mut_prob=0 returns unchanged individual."""
        individual = [
            SessionGene(
                "CS101", "theory", "I1", ["G1"], "R1", start_quanta=0, num_quanta=3
            ),
        ]

        context = MagicMock()
        context.available_quanta = list(range(100))
        context.rooms = [MagicMock(room_id="R1")]

        mutated = mutate_individual(deepcopy(individual), context, mut_prob=0.0)

        # With 0 probability, individual should be unchanged
        assert mutated[0][0].quanta == individual[0].quanta
        assert mutated[0][0].room_id == individual[0].room_id


class TestRepair:
    """Test suite for repair mechanisms."""

    def test_repair_reduces_violations(self):
        """Test that repair mechanism reduces constraint violations."""
        # This would require mocking the full repair system
        # Placeholder for now
        pytest.skip("Repair testing requires full context setup")

    def test_repair_preserves_gene_structure(self):
        """Test repair doesn't corrupt gene structure."""
        pytest.skip("Repair testing requires full context setup")
