"""
Configuration loader with environment detection.
Loads YAML configs with fallback chain.
"""

import os
import sys
from pathlib import Path
from config.models import Config


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.

    Priority:
    1. Explicit config_path argument
    2. SCHEDULE_CONFIG environment variable
    3. configs/{ENVIRONMENT}.yaml
    4. configs/dev.yaml (default)
    5. Built-in defaults

    Args:
        config_path: Path to config YAML file

    Returns:
        Config object
    """
    # Priority 1: Explicit path
    if config_path:
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading config: {config_path}")
        return Config.from_yaml(config_path)

    # Priority 2: Environment variable
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            print(f"ERROR: Config file not found: {env_config}")
            sys.exit(1)
        print(f"Loading config from SCHEDULE_CONFIG: {env_config}")
        return Config.from_yaml(env_config)

    # Priority 3: Environment-specific config
    environment = os.getenv("ENVIRONMENT", "dev")
    env_path = Path(f"configs/{environment}.yaml")
    if env_path.exists():
        print(f"Loading {environment} config: {env_path}")
        return Config.from_yaml(str(env_path))

    # Priority 4: Default dev config
    default_path = Path("configs/dev.yaml")
    if default_path.exists():
        print(f"Loading default config: {default_path}")
        return Config.from_yaml(str(default_path))

    # Priority 5: Built-in defaults
    print("WARNING: No config file found, using built-in defaults")
    return Config()
