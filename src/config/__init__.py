"""
Configuration module - Clean YAML-based configuration.

Usage:
    from src.config import config

    # Access config values
    ngen = config.ga.ngen
    pop_size = config.ga.pop_size
    fail_on_infeasibility = config.feasibility.fail_on_infeasibility
"""

from src.config.loader import load_config
from src.config.models import Config

# Global config object (set by main.py)
config: Config = None


def init_config(config_path: str = None, config_obj: Config = None) -> Config:
    """Initialize global config (called from main.py)"""
    global config
    if config_obj is not None:
        # Use provided config object (from runtime mode loading)
        config = config_obj
    else:
        # Load from path
        config = load_config(config_path)
    return config


def get_config() -> Config:
    """Get config, loading if necessary"""
    global config
    if config is None:
        config = load_config()
    return config
