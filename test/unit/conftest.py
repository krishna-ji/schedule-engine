"""Pytest configuration and shared fixtures for unit tests."""

import pytest
import sys
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_context():
    """
    Provide a minimal SchedulingContext for testing.

    Returns:
        Mock SchedulingContext with basic attributes
    """
    from unittest.mock import MagicMock

    context = MagicMock()
    context.available_quanta = list(range(100))
    context.courses = {}
    context.groups = []
    context.instructors = []
    context.rooms = []

    return context


@pytest.fixture
def sample_config():
    """
    Provide a sample Config object for testing.

    Returns:
        Loaded test configuration
    """
    from src.config import load_config

    try:
        config = load_config()
        return config
    except Exception:
        # If loading fails, return mock
        from unittest.mock import MagicMock

        return MagicMock()


@pytest.fixture
def sample_individual():
    """
    Provide a sample individual (list of SessionGenes) for testing.

    Returns:
        List of SessionGene objects
    """
    from src.ga.sessiongene import SessionGene

    return [
        SessionGene(
            course_id=("CS101", "Theory"),
            group_ids=["G1"],
            instructor_id="I1",
            room_id="R1",
            quanta=[0, 1, 2],
        ),
        SessionGene(
            course_id=("MATH101", "Theory"),
            group_ids=["G1"],
            instructor_id="I2",
            room_id="R2",
            quanta=[5, 6, 7],
        ),
    ]


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging

    # Clear all handlers
    logger = logging.getLogger("schedule_engine")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)  # Reduce noise during tests

    yield

    # Cleanup after test
    logger.handlers.clear()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
