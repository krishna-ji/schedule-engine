"""Unit tests for configuration loading and validation."""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.config.loader import load_config
from src.config.models import Config


class TestConfigLoader:
    """Test suite for config loader functionality."""

    def test_load_config_with_test_environment(self):
        """Test loading test.yaml configuration."""
        config = load_config()
        assert config is not None
        assert isinstance(config, Config)
        assert config.ga.ngen > 0
        assert config.ga.pop_size > 0

    def test_load_config_with_explicit_path(self, tmp_path):
        """Test loading config from explicit file path."""
        # Create temporary config file
        config_data = {"ga": {"ngen": 100, "pop_size": 50, "cxpb": 0.8, "mutpb": 0.3}}
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load and verify
        config = load_config(str(config_file))
        assert config.ga.ngen == 100
        assert config.ga.pop_size == 50

    def test_load_config_with_missing_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_config_has_all_required_sections(self):
        """Test that loaded config has all required sections."""
        config = load_config()

        assert hasattr(config, "ga")
        assert hasattr(config, "parallel")
        assert hasattr(config, "repair")
        assert hasattr(config, "constraints")
        assert hasattr(config, "time")
        assert hasattr(config, "io")

    def test_config_ga_parameters_valid(self):
        """Test GA parameters are within valid ranges."""
        config = load_config()

        assert 0 < config.ga.cxpb <= 1.0
        assert 0 < config.ga.mutpb <= 1.0
        assert config.ga.ngen > 0
        assert config.ga.pop_size > 0

    def test_config_summary_returns_string(self):
        """Test that config.summary() returns formatted string."""
        config = load_config()
        summary = config.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "ga" in summary.lower() or "genetic" in summary.lower()


class TestConfigModels:
    """Test suite for Pydantic config models."""

    def test_ga_config_validation(self):
        """Test GAConfig validation."""
        from src.config.models import GAConfig

        # Valid config
        ga_config = GAConfig(ngen=100, pop_size=50, cxpb=0.8, mutpb=0.2)
        assert ga_config.ngen == 100

        # Invalid config (negative values)
        with pytest.raises(Exception):  # Pydantic ValidationError
            GAConfig(ngen=-10, pop_size=50, cxpb=0.8, mutpb=0.2)

    def test_parallel_config_defaults(self):
        """Test ParallelConfig has sensible defaults."""
        from src.config.models import ParallelConfig

        parallel_config = ParallelConfig()
        assert isinstance(parallel_config.use_multiprocessing, bool)
        assert parallel_config.num_workers is None or parallel_config.num_workers > 0
