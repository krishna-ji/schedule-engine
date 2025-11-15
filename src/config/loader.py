"""
Configuration loader with base.yaml inheritance.
Loads configs with base.yaml + environment overrides.
"""

import os
import sys
import yaml
from pathlib import Path
from src.config.models import Config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    Override values take precedence over base values.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = None) -> Config:
    """
    Load configuration with base.yaml + environment overrides.

    Config structure:
    - base.yaml: All common settings
    - prod.yaml/test.yaml: Only environment-specific overrides

    Loading priority:
    1. Explicit config_path argument (--config flag) - loads with base.yaml merge
    2. SCHEDULE_CONFIG environment variable - loads with base.yaml merge
    3. configs/{ENVIRONMENT}.yaml (ENVIRONMENT env var) - loads with base.yaml merge
    4. configs/test.yaml (default) - loads with base.yaml merge
    5. Built-in defaults

    Args:
        config_path: Path to config YAML file

    Returns:
        Config object
    """
    base_path = Path("configs/base.yaml")

    # Load base config if it exists
    base_dict = {}
    if base_path.exists():
        with open(base_path) as f:
            base_dict = yaml.safe_load(f) or {}

    # Priority 1: Explicit path
    if config_path:
        if not Path(config_path).exists():
            print(f"[!ERR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        print(f"Loading config: {config_path} (merged with base.yaml)")
        return Config(**merged)

    # Priority 2: Environment variable
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            print(f"[!ERR] Config file not found: {env_config}")
            sys.exit(1)
        with open(env_config) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        print(
            f"Loading config from SCHEDULE_CONFIG: {env_config} (merged with base.yaml)"
        )
        return Config(**merged)

    # Priority 3: Environment-specific config
    environment = os.getenv("ENVIRONMENT", "test")
    env_path = Path(f"configs/{environment}.yaml")
    if env_path.exists():
        with open(env_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        print(f"Loading config: configs/{environment}.yaml (merged with base.yaml)")
        return Config(**merged)

    # Priority 4: Default test config
    default_path = Path("configs/test.yaml")
    if default_path.exists():
        with open(default_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        print("Loading config: configs/test.yaml (default, merged with base.yaml)")
        return Config(**merged)

    # Priority 5: Built-in defaults only
    print("[!WARN] No config files found, using built-in defaults")
    return Config()
