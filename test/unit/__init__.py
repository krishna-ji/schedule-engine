"""
Unit Test Suite for Schedule Engine

This package contains unit tests for all major components of the
Schedule Engine project. Tests are organized by module:

- test_config_loader.py: Configuration loading and validation
- test_encoder.py: Input encoding and data loading
- test_constraints.py: Hard and soft constraint functions
- test_operators.py: GA operators (crossover, mutation, repair)
- test_utils.py: Utility functions and helpers

Running Tests:
    # Run all unit tests
    pytest test/unit/

    # Run with coverage
    pytest --cov=src --cov-report=html test/unit/

    # Run specific test file
    pytest test/unit/test_config_loader.py

    # Run specific test
    pytest test/unit/test_config_loader.py::TestConfigLoader::test_load_config_with_test_environment

    # Run tests matching pattern
    pytest -k "config" test/unit/

Fixtures:
    - sample_context: Provides minimal SchedulingContext
    - sample_config: Provides loaded test configuration
    - sample_individual: Provides sample GA individual

Configuration:
    See conftest.py for pytest configuration and fixtures.
"""

__all__ = []
