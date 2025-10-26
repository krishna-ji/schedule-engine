"""
Configuration module - Clean YAML-based configuration.

Usage:
    from config import config

    # Access config values
    ngen = config.ga.ngen
    pop_size = config.ga.pop_size
    fail_on_infeasibility = config.feasibility.fail_on_infeasibility
"""

from config.loader import load_config
from config.models import Config

# Global config object (set by main.py)
config: Config = None


def init_config(config_path: str = None) -> Config:
    """Initialize global config (called from main.py)"""
    global config
    config = load_config(config_path)
    return config


def get_config() -> Config:
    """Get config, loading if necessary"""
    global config
    if config is None:
        config = load_config()
    return config
