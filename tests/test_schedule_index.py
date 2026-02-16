"""
Tests for ScheduleIndex caching system.

Tests verify:
- Correct conflict detection
- Caching behavior (maps built once, reused)
- Invalidation (maps rebuilt after modification)
- Performance characteristics
"""

from src.domain.gene import SessionGene
from src.ga.core.schedule_index import ScheduleIndex, create_schedule_index


def create_test_gene(
    course_id="CS101",
    course_type="LEC",
    group_ids=None,
    num_quanta=3,
    instructor_id="Prof_A",
    room_id="R101",
    start_quanta=0,
):
    """Helper to create test genes."""
    if group_ids is None:
        group_ids = ["G1"]

    return SessionGene(
        course_id=course_id,
        course_type=course_type,
        group_ids=group_ids,
        num_quanta=num_quanta,
        instructor_id=instructor_id,
        room_id=room_id,
        start_quanta=start_quanta,
    )


class TestScheduleIndexBasic:
    """Test basic ScheduleIndex functionality."""

    def test_create_index(self):
        """Test index creation."""
        genes = [create_test_gene()]
        index = ScheduleIndex.from_individual(genes)

        assert index is not None
        assert not index.is_valid()  # Not built yet

    def test_create_index_convenience(self):
        """Test convenience function."""
        genes = [create_test_gene()]
        index = create_schedule_index(genes)

        assert index is not None
        assert isinstance(index, ScheduleIndex)

    def test_empty_individual(self):
        """Test index with empty individual."""
        index = ScheduleIndex.from_individual([])

        conflicts = index.find_group_conflicts()
        assert conflicts == {}
        assert index.is_valid()  # Built (even if empty)


class TestGroupConflicts:
    """Test group overlap conflict detection."""

    def test_no_group_conflicts(self):
        """Test genes with no group overlaps."""
        genes = [
            create_test_gene(course_id="CS101", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS102", group_ids=["G1"], start_quanta=10),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()

        assert len(conflicts) == 0

    def test_simple_group_conflict(self):
        """Test two genes with group overlap."""
        genes = [
            create_test_gene(
                course_id="CS101", group_ids=["G1"], start_quanta=0, num_quanta=3
            ),
            create_test_gene(
                course_id="CS102", group_ids=["G1"], start_quanta=0, num_quanta=3
            ),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()

        assert len(conflicts) == 2
        assert 0 in conflicts
        assert 1 in conflicts
        assert 1 in conflicts[0]  # Gene 0 conflicts with gene 1
        assert 0 in conflicts[1]  # Gene 1 conflicts with gene 0

    def test_partial_overlap(self):
        """Test genes with partial time overlap."""
        genes = [
            create_test_gene(
                course_id="CS101", group_ids=["G1"], start_quanta=0, num_quanta=5
            ),
            create_test_gene(
                course_id="CS102", group_ids=["G1"], start_quanta=3, num_quanta=5
            ),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()

        assert len(conflicts) == 2  # Both genes conflicted

    def test_multiple_groups_same_gene(self):
        """Test gene with multiple groups."""
        genes = [
            create_test_gene(course_id="CS101", group_ids=["G1", "G2"], start_quanta=0),
            create_test_gene(course_id="CS102", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS103", group_ids=["G2"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()

        # Gene 0 conflicts with both 1 (G1 overlap) and 2 (G2 overlap)
        assert 0 in conflicts
        assert 1 in conflicts[0]
        assert 2 in conflicts[0]

    def test_three_way_conflict(self):
        """Test three genes all conflicting."""
        genes = [
            create_test_gene(course_id="CS101", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS102", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS103", group_ids=["G1"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_group_conflicts()

        assert len(conflicts) == 3
        assert len(conflicts[0]) == 2  # Gene 0 conflicts with 2 others
        assert len(conflicts[1]) == 2
        assert len(conflicts[2]) == 2


class TestRoomConflicts:
    """Test room exclusivity conflict detection."""

    def test_no_room_conflicts(self):
        """Test genes with no room conflicts."""
        genes = [
            create_test_gene(room_id="R101", start_quanta=0),
            create_test_gene(room_id="R101", start_quanta=10),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_room_conflicts()

        assert len(conflicts) == 0

    def test_simple_room_conflict(self):
        """Test two genes with room conflict."""
        genes = [
            create_test_gene(room_id="R101", start_quanta=0),
            create_test_gene(room_id="R101", start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_room_conflicts()

        assert len(conflicts) == 2
        assert 0 in conflicts
        assert 1 in conflicts

    def test_different_rooms_no_conflict(self):
        """Test genes in different rooms at same time."""
        genes = [
            create_test_gene(room_id="R101", start_quanta=0),
            create_test_gene(room_id="R102", start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_room_conflicts()

        assert len(conflicts) == 0


class TestInstructorConflicts:
    """Test instructor exclusivity conflict detection."""

    def test_no_instructor_conflicts(self):
        """Test genes with no instructor conflicts."""
        genes = [
            create_test_gene(instructor_id="Prof_A", start_quanta=0),
            create_test_gene(instructor_id="Prof_A", start_quanta=10),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_instructor_conflicts()

        assert len(conflicts) == 0

    def test_simple_instructor_conflict(self):
        """Test two genes with instructor conflict."""
        genes = [
            create_test_gene(instructor_id="Prof_A", start_quanta=0),
            create_test_gene(instructor_id="Prof_A", start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_instructor_conflicts()

        assert len(conflicts) == 2
        assert 0 in conflicts
        assert 1 in conflicts

    def test_different_instructors_no_conflict(self):
        """Test different instructors at same time."""
        genes = [
            create_test_gene(instructor_id="Prof_A", start_quanta=0),
            create_test_gene(instructor_id="Prof_B", start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        conflicts = index.find_instructor_conflicts()

        assert len(conflicts) == 0


class TestCaching:
    """Test caching behavior."""

    def test_maps_built_on_first_access(self):
        """Test that maps are built lazily."""
        genes = [create_test_gene()]
        index = ScheduleIndex.from_individual(genes)

        assert not index.is_valid()  # Not built yet

        index.find_group_conflicts()

        assert index.is_valid()  # Now built

    def test_maps_reused_on_second_access(self):
        """Test that maps are cached."""
        genes = [
            create_test_gene(course_id="CS101", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS102", group_ids=["G1"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)

        # First access builds
        conflicts1 = index.find_group_conflicts()
        assert index.is_valid()

        # Second access reuses (should be instant)
        conflicts2 = index.find_group_conflicts()

        assert conflicts1 == conflicts2
        assert index.is_valid()

    def test_all_conflict_types_use_same_cache(self):
        """Test that all conflict checks use same cached maps."""
        genes = [
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
        ]

        index = ScheduleIndex.from_individual(genes)

        # All these should build maps only once
        group_conflicts = index.find_group_conflicts()
        room_conflicts = index.find_room_conflicts()
        instructor_conflicts = index.find_instructor_conflicts()

        assert len(group_conflicts) == 2
        assert len(room_conflicts) == 2
        assert len(instructor_conflicts) == 2
        assert index.is_valid()


class TestInvalidation:
    """Test cache invalidation."""

    def test_invalidation_marks_stale(self):
        """Test that invalidate() marks cache as stale."""
        genes = [create_test_gene()]
        index = ScheduleIndex.from_individual(genes)

        index.find_group_conflicts()  # Build cache
        assert index.is_valid()

        index.invalidate()
        assert not index.is_valid()

    def test_rebuild_after_invalidation(self):
        """Test that maps are rebuilt after invalidation."""
        genes = [
            create_test_gene(course_id="CS101", group_ids=["G1"], start_quanta=0),
            create_test_gene(course_id="CS102", group_ids=["G1"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)

        # Initial conflicts
        conflicts_before = index.find_group_conflicts()
        assert len(conflicts_before) == 2

        # Modify gene (move to non-conflicting time)
        genes[1].start_quanta = 10
        index.invalidate()

        # Should rebuild and find no conflicts
        conflicts_after = index.find_group_conflicts()
        assert len(conflicts_after) == 0
        assert index.is_valid()

    def test_invalidation_affects_all_conflict_types(self):
        """Test that invalidation affects all cached maps."""
        genes = [
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
        ]

        index = ScheduleIndex.from_individual(genes)

        # Build all caches
        index.find_group_conflicts()
        index.find_room_conflicts()
        index.find_instructor_conflicts()

        # Modify and invalidate
        genes[1].start_quanta = 10
        index.invalidate()

        # All should show no conflicts after rebuild
        assert len(index.find_group_conflicts()) == 0
        assert len(index.find_room_conflicts()) == 0
        assert len(index.find_instructor_conflicts()) == 0


class TestUtilityMethods:
    """Test utility methods."""

    def test_find_all_conflicts(self):
        """Test find_all_conflicts() convenience method."""
        genes = [
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
        ]

        index = ScheduleIndex.from_individual(genes)
        all_conflicts = index.find_all_conflicts()

        assert "group" in all_conflicts
        assert "room" in all_conflicts
        assert "instructor" in all_conflicts
        assert len(all_conflicts["group"]) == 2
        assert len(all_conflicts["room"]) == 2
        assert len(all_conflicts["instructor"]) == 2

    def test_count_violations(self):
        """Test count_violations() method."""
        genes = [
            create_test_gene(group_ids=["G1"], start_quanta=0),
            create_test_gene(group_ids=["G1"], start_quanta=0),
            create_test_gene(group_ids=["G2"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        counts = index.count_violations()

        assert "group" in counts
        assert "room" in counts
        assert "instructor" in counts
        assert "total" in counts
        assert counts["group"] == 2  # Gene 0 and 1 conflict
        assert counts["total"] >= counts["group"]

    def test_get_all_violated_indices(self):
        """Test get_all_violated_indices() method."""
        genes = [
            create_test_gene(group_ids=["G1"], room_id="R101", start_quanta=0),
            create_test_gene(group_ids=["G1"], room_id="R101", start_quanta=0),
            create_test_gene(
                group_ids=["G2"], room_id="R102", start_quanta=10
            ),  # No conflict
        ]

        index = ScheduleIndex.from_individual(genes)
        violated = index.get_all_violated_indices()

        assert 0 in violated
        assert 1 in violated
        assert 2 not in violated

    def test_has_conflicts_true(self):
        """Test has_conflicts() returns True when conflicts exist."""
        genes = [
            create_test_gene(group_ids=["G1"], start_quanta=0),
            create_test_gene(group_ids=["G1"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)
        assert index.has_conflicts() is True

    def test_has_conflicts_false(self):
        """Test has_conflicts() returns False when no conflicts."""
        genes = [
            create_test_gene(group_ids=["G1"], start_quanta=0),
            create_test_gene(group_ids=["G1"], start_quanta=10),
        ]

        index = ScheduleIndex.from_individual(genes)
        assert index.has_conflicts() is False

    def test_get_gene_conflicts(self):
        """Test get_gene_conflicts() for specific gene."""
        genes = [
            create_test_gene(
                group_ids=["G1"], room_id="R101", instructor_id="Prof_A", start_quanta=0
            ),
            create_test_gene(
                group_ids=["G1"], room_id="R102", instructor_id="Prof_B", start_quanta=0
            ),
        ]

        index = ScheduleIndex.from_individual(genes)
        gene0_conflicts = index.get_gene_conflicts(0)

        assert "group" in gene0_conflicts
        assert "room" in gene0_conflicts
        assert "instructor" in gene0_conflicts
        assert 1 in gene0_conflicts["group"]  # Conflicts with gene 1 on group
        assert len(gene0_conflicts["room"]) == 0  # Different rooms
        assert len(gene0_conflicts["instructor"]) == 0  # Different instructors


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_large_individual(self):
        """Test with realistic-sized individual (200 genes)."""
        genes = [
            create_test_gene(
                course_id=f"CS{i}",
                group_ids=[f"G{i % 10}"],
                room_id=f"R{i % 20}",
                instructor_id=f"Prof_{i % 15}",
                start_quanta=i % 50,
            )
            for i in range(200)
        ]

        index = ScheduleIndex.from_individual(genes)

        # Should handle large individual efficiently
        conflicts = index.find_group_conflicts()
        assert isinstance(conflicts, dict)

        counts = index.count_violations()
        assert "total" in counts

    def test_multiple_invalidations(self):
        """Test multiple invalidation cycles."""
        genes = [
            create_test_gene(group_ids=["G1"], start_quanta=0),
            create_test_gene(group_ids=["G1"], start_quanta=0),
        ]

        index = ScheduleIndex.from_individual(genes)

        # Cycle 1
        conflicts1 = index.find_group_conflicts()
        assert len(conflicts1) == 2

        # Modify and invalidate
        genes[1].start_quanta = 10
        index.invalidate()

        # Cycle 2
        conflicts2 = index.find_group_conflicts()
        assert len(conflicts2) == 0

        # Modify and invalidate again
        genes[1].start_quanta = 0
        index.invalidate()

        # Cycle 3
        conflicts3 = index.find_group_conflicts()
        assert len(conflicts3) == 2

    def test_mixed_conflict_types(self):
        """Test individual with different conflict types."""
        genes = [
            create_test_gene(
                course_id="CS101",
                group_ids=["G1"],
                room_id="R101",
                instructor_id="Prof_A",
                start_quanta=0,
            ),
            create_test_gene(
                course_id="CS102",
                group_ids=["G1"],  # Group conflict with gene 0
                room_id="R102",
                instructor_id="Prof_A",  # Instructor conflict with gene 0
                start_quanta=0,
            ),
            create_test_gene(
                course_id="CS103",
                group_ids=["G2"],
                room_id="R101",  # Room conflict with gene 0
                instructor_id="Prof_B",
                start_quanta=0,
            ),
        ]

        index = ScheduleIndex.from_individual(genes)

        index.find_group_conflicts()
        index.find_room_conflicts()
        index.find_instructor_conflicts()

        # Gene 0 should have multiple conflict types
        gene0_conflicts = index.get_gene_conflicts(0)
        assert len(gene0_conflicts["group"]) > 0
        assert len(gene0_conflicts["room"]) > 0
        assert len(gene0_conflicts["instructor"]) > 0
