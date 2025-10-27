"""
Configuration loader with environment detection.
Loads YAML configs with common base + environment-specific overrides.
"""

import os
import sys
from pathlib import Path
from src.config.models import Config
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML files with inheritance.

    Loading strategy:
    1. Load configs/common.yaml (base defaults)
    2. Merge with environment-specific config (test/dev/prod)
    3. Environment-specific values override common values

    Priority:
    1. Explicit config_path argument (standalone, no common merge)
    2. SCHEDULE_CONFIG environment variable (standalone, no common merge)
    3. configs/common.yaml + configs/{ENVIRONMENT}.yaml (merged)
    4. configs/common.yaml + configs/dev.yaml (default merged)
    5. Built-in defaults

    Args:
        config_path: Path to standalone config YAML file (no common merge)

    Returns:
        Config object
    """
    # Priority 1: Explicit path (standalone, no merge)
    if config_path:
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading standalone config: {config_path}")
        return Config.from_yaml(config_path)

    # Priority 2: Environment variable (standalone, no merge)
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            print(f"ERROR: Config file not found: {env_config}")
            sys.exit(1)
        print(f"Loading standalone config from SCHEDULE_CONFIG: {env_config}")
        return Config.from_yaml(env_config)

    # Priority 3+: Load common.yaml as base
    common_path = Path("configs/common.yaml")
    if not common_path.exists():
        print("WARNING: configs/common.yaml not found, using built-in defaults")
        common_data = {}
    else:
        with open(common_path, "r") as f:
            common_data = yaml.safe_load(f) or {}

    # Priority 3: Environment-specific config merged with common
    environment = os.getenv("ENVIRONMENT", "dev")
    env_path = Path(f"configs/{environment}.yaml")
    if env_path.exists():
        print(f"Loading {environment} config: common.yaml + {env_path.name}")
        with open(env_path, "r") as f:
            env_data = yaml.safe_load(f) or {}

        # Merge: common (base) + environment (override)
        merged_data = deep_merge(common_data, env_data)
        return Config(**merged_data)

    # Priority 4: Default dev config merged with common
    default_path = Path("configs/dev.yaml")
    if default_path.exists():
        print(f"Loading default config: common.yaml + dev.yaml")
        with open(default_path, "r") as f:
            dev_data = yaml.safe_load(f) or {}

        # Merge: common (base) + dev (override)
        merged_data = deep_merge(common_data, dev_data)
        return Config(**merged_data)

    # Priority 5: Built-in defaults only
    print("WARNING: No config files found, using built-in defaults")
    return Config()
