"""
Behavioral descriptors for archive-based diversity.

ENHANCEMENT #5: Extract behavioral features from schedules.
"""

import numpy as np
from numpy.typing import NDArray

from src.core.types import Individual
from src.entities.decoded_session import CourseSession


class BehavioralDescriptors:
    """
    Extracts behavioral features from schedules for diversity maintenance.

    Features (17 total):
    1-7. Temporal distribution (% sessions per day: Mon-Sun)
    8-10. Time slot distribution (morning/afternoon/evening)
    11-13. Room utilization (% capacity used per room type)
    14-15. Instructor load balance (std dev of sessions per instructor)
    16. Session compactness (avg gap between sessions)
    17. Cross-day spreading (sessions spread across days vs concentrated)

    These features capture schedule "behavior" independent of fitness,
    enabling discovery of diverse high-quality solutions.
    """

    def __init__(self):
        """Initialize behavioral descriptor extractor."""
        self.feature_names = [
            "monday_density",
            "tuesday_density",
            "wednesday_density",
            "thursday_density",
            "friday_density",
            "saturday_density",
            "sunday_density",
            "morning_density",
            "afternoon_density",
            "evening_density",
            "small_room_utilization",
            "medium_room_utilization",
            "large_room_utilization",
            "instructor_load_std",
            "student_load_std",
            "session_compactness",
            "cross_day_spreading",
        ]

    def extract(
        self, individual: Individual, sessions: list[CourseSession] = None
    ) -> NDArray[np.float64]:
        """
        Extract behavioral descriptor from individual.

        Args:
            individual: GA individual (chromosome)
            sessions: Decoded sessions (optional, for efficiency)

        Returns:
            17D behavioral descriptor vector
        """
        if sessions is None:
            # TODO: Decode individual to sessions
            # sessions = decode_individual(individual)
            pass

        features = np.zeros(17, dtype=np.float64)

        if sessions:
            # Extract temporal features
            features[0:7] = self._temporal_distribution(sessions)
            features[7:10] = self._time_slot_distribution(sessions)

            # Extract resource features
            features[10:13] = self._room_utilization(sessions)
            features[13:15] = self._load_balance(sessions)

            # Extract structural features
            features[15] = self._session_compactness(sessions)
            features[16] = self._cross_day_spreading(sessions)

        return features

    def _temporal_distribution(
        self, sessions: list[CourseSession]
    ) -> NDArray[np.float64]:
        """Percentage of sessions per day (Mon-Sun)."""
        day_counts = np.zeros(7)
        total_sessions = len(sessions)

        for session in sessions:
            # TODO: Extract day from session quanta
            # day_idx = get_day_index(session.session_quanta[0])
            # day_counts[day_idx] += 1
            pass

        return day_counts / max(total_sessions, 1)

    def _time_slot_distribution(
        self, sessions: list[CourseSession]
    ) -> NDArray[np.float64]:
        """Percentage of sessions in morning/afternoon/evening."""
        time_counts = np.zeros(3)  # [morning, afternoon, evening]
        total_sessions = len(sessions)

        for session in sessions:
            # TODO: Classify session start time
            # if is_morning(session): time_counts[0] += 1
            # elif is_afternoon(session): time_counts[1] += 1
            # else: time_counts[2] += 1
            pass

        return time_counts / max(total_sessions, 1)

    def _room_utilization(self, sessions: list[CourseSession]) -> NDArray[np.float64]:
        """Room utilization by type (small/medium/large)."""
        # TODO: Calculate room utilization by capacity
        return np.array([0.5, 0.5, 0.5])

    def _load_balance(self, sessions: list[CourseSession]) -> NDArray[np.float64]:
        """Standard deviation of load (instructor/student)."""
        # TODO: Calculate load distribution
        return np.array([0.0, 0.0])

    def _session_compactness(self, sessions: list[CourseSession]) -> float:
        """Average compactness of schedules."""
        # TODO: Calculate avg gap between sessions per group
        return 0.5

    def _cross_day_spreading(self, sessions: list[CourseSession]) -> float:
        """How spread out sessions are across days."""
        # TODO: Calculate Shannon entropy of day distribution
        return 0.5

    def distance(self, desc1: NDArray[np.float64], desc2: NDArray[np.float64]) -> float:
        """
        Calculate distance between two behavioral descriptors.

        Args:
            desc1: First descriptor
            desc2: Second descriptor

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(desc1 - desc2))
