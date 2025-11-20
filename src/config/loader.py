"""
Configuration loader with base.yaml inheritance.
Loads configs with base.yaml + environment overrides + runtime mode overrides.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional
from src.config.models import Config
from src.config.runtime_mode import RuntimeMode


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


def load_config(
    config_path: str = None, runtime_mode: Optional[RuntimeMode] = None
) -> Config:
    """
    Load configuration with base.yaml + environment overrides + runtime mode overrides.

    Config structure:
    - base.yaml: All common settings
    - prod.yaml/test.yaml: Environment-specific overrides
    - configs/{category}/{mode}.yaml: Runtime mode overrides

    Loading priority:
    1. Runtime mode (--mode flag) - loads mode-specific config with base.yaml merge
    2. Explicit config_path argument (--config flag) - loads with base.yaml merge
    3. SCHEDULE_CONFIG environment variable - loads with base.yaml merge
    4. configs/{ENVIRONMENT}.yaml (ENVIRONMENT env var) - loads with base.yaml merge
    5. configs/test.yaml (default) - loads with base.yaml merge
    6. Built-in defaults

    Args:
        config_path: Path to config YAML file (overrides runtime_mode)
        runtime_mode: RuntimeMode enum for experiment (e.g., RuntimeMode.BASELINE)

    Returns:
        Config object

    Raises:
        ValueError: If runtime mode config violates mode constraints
    """
    base_path = Path("configs/base.yaml")

    # Load base config if it exists
    base_dict = {}
    if base_path.exists():
        with open(base_path) as f:
            base_dict = yaml.safe_load(f) or {}

    # Priority 1: Runtime mode
    if runtime_mode is not None:
        mode_path = runtime_mode.config_path
        if not mode_path.exists():
            print(f"[!ERR] Runtime mode config not found: {mode_path}")
            sys.exit(1)
        with open(mode_path) as f:
            mode_dict = yaml.safe_load(f) or {}
        
        # Merge Base + Mode
        merged = _deep_merge(base_dict, mode_dict)

        # Merge Environment Config (if exists) ON TOP
        # This allows prod.yaml to scale up population/generations while keeping mode constraints
        environment = os.getenv("ENVIRONMENT", "test")
        env_path = Path(f"configs/{environment}.yaml")
        if env_path.exists():
            with open(env_path) as f:
                env_dict = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, env_dict)
            if not os.environ.get("_GA_WORKER_PROCESS"):
                print(
                    f"Loading runtime mode: {runtime_mode.display_name} + {environment}.yaml"
                )
        else:
            if not os.environ.get("_GA_WORKER_PROCESS"):
                print(
                    f"Loading runtime mode: {runtime_mode.display_name} ({mode_path}, merged with base.yaml)"
                )

        # Validate config matches runtime mode constraints
        try:
            runtime_mode.validate_config(merged)
        except ValueError as e:
            print(f"[!ERR] Config validation failed for mode {runtime_mode.value}: {e}")
            sys.exit(1)

        return Config(**merged)

    # Priority 2: Explicit path
    if config_path:
        if not Path(config_path).exists():
            print(f"[!ERR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        if not os.environ.get("_GA_WORKER_PROCESS"):
            print(f"Loading config: {config_path} (merged with base.yaml)")
        return Config(**merged)

    # Priority 3: Environment variable
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            print(f"[!ERR] Config file not found: {env_config}")
            sys.exit(1)
        with open(env_config) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        if not os.environ.get("_GA_WORKER_PROCESS"):
            print(
                f"Loading config from SCHEDULE_CONFIG: {env_config} (merged with base.yaml)"
            )
        return Config(**merged)

    # Priority 4: Environment-specific config
    environment = os.getenv("ENVIRONMENT", "test")
    env_path = Path(f"configs/{environment}.yaml")
    if env_path.exists():
        with open(env_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        if not os.environ.get("_GA_WORKER_PROCESS"):
            print(f"Loading config: configs/{environment}.yaml (merged with base.yaml)")
        return Config(**merged)

    # Priority 5: Default test config
    default_path = Path("configs/test.yaml")
    if default_path.exists():
        with open(default_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        if not os.environ.get("_GA_WORKER_PROCESS"):
            print("Loading config: configs/test.yaml (default, merged with base.yaml)")
        return Config(**merged)

    # Priority 6: Built-in defaults only
    print("[!WARN] No config files found, using built-in defaults")
    return Config()
