"""
Configuration loader with base.yaml inheritance.
Loads configs with base.yaml + environment overrides + runtime mode overrides.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

from src.config.models import Config
from src.config.runtime_mode import RuntimeMode

logger = logging.getLogger("schedule_engine.config.loader")


def _log(message: str, level: str = "info") -> None:
    """Log configuration loader events and mirror to console for parent processes."""

    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)

    if not os.environ.get("_GA_WORKER_PROCESS"):
        print(message)


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
    config_path: str = None, runtime_mode: RuntimeMode | None = None
) -> Config:
    """
    Load configuration with layered merge strategy.

    Architecture:
    ============

    1. BASE LAYER (configs/base.yaml)
       - Common settings shared across all environments and modes
       - Default values for all configuration parameters

    2. MODE LAYER (configs/{category}/{mode}.yaml) [OPTIONAL]
       - Runtime mode specific settings (baseline, rl-guided, etc.)
       - Overrides base settings for that specific mode

    3. ENVIRONMENT LAYER (configs/{env}.yaml) [ALWAYS APPLIED]
       - Environment scaling (test=30 gens, prod=2000 gens)
       - Final overrides for deployment environment
       - CRITICAL: Always applied last regardless of priority path

    Merge Order: base.yaml → [mode.yaml] → env.yaml

    Loading Priority Paths:
    =======================

    Priority 1: Runtime Mode (--mode baseline)
        Flow: base.yaml → mode.yaml → env.yaml
        Use: Experiment-specific configuration with environment scaling

    Priority 2: Explicit Path (--config path/to/config.yaml)
        Flow: base.yaml → custom.yaml → env.yaml
        Use: Custom configurations with environment scaling

    Priority 3: SCHEDULE_CONFIG Environment Variable
        Flow: base.yaml → $SCHEDULE_CONFIG.yaml → env.yaml
        Use: CI/CD or automated workflows

    Priority 4: Default Environment (ENVIRONMENT=test)
        Flow: base.yaml → test.yaml
        Use: Quick testing without mode specification

    Environment Variable:
    ====================
    ENVIRONMENT: Controls which env.yaml to load (test/prod)
    - Must be set BEFORE calling load_config()
    - Default: "test" (safe for development)
    - Set in main.py from --env CLI argument

    Args:
        config_path: Path to config YAML file (Priority 2)
        runtime_mode: RuntimeMode enum for experiment (Priority 1)

    Returns:
        Config object with all layers merged

    Raises:
        ValueError: If runtime mode config violates mode constraints

    Example:
        # Via runtime mode (recommended)
        os.environ["ENVIRONMENT"] = "prod"
        config = load_config(runtime_mode=RuntimeMode.RL_GUIDED)

        # Via explicit path
        os.environ["ENVIRONMENT"] = "test"
        config = load_config("configs/custom.yaml")
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
            _log(f"[!ERR] Runtime mode config not found: {mode_path}", level="error")
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
            _log(
                f"Loading runtime mode: {runtime_mode.display_name} + {environment}.yaml"
            )
        else:
            _log(
                f"Loading runtime mode: {runtime_mode.display_name} ({mode_path}, merged with base.yaml)"
            )

        # Validate config matches runtime mode constraints
        try:
            runtime_mode.validate_config(merged)
        except ValueError as e:
            _log(
                f"[!ERR] Config validation failed for mode {runtime_mode.value}: {e}",
                level="error",
            )
            sys.exit(1)

        return Config(**merged)

    # Priority 2: Explicit path
    if config_path:
        if not Path(config_path).exists():
            _log(f"[!ERR] Config file not found: {config_path}", level="error")
            sys.exit(1)
        with open(config_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)

        # ALWAYS apply environment config on top (test/prod scaling)
        environment = os.getenv("ENVIRONMENT", "test")
        env_path = Path(f"configs/{environment}.yaml")
        if env_path.exists():
            with open(env_path) as f:
                env_dict = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, env_dict)
            _log(
                f"Loading config: {config_path} + {environment}.yaml (merged with base.yaml)"
            )
        else:
            _log(f"Loading config: {config_path} (merged with base.yaml)")
        return Config(**merged)

    # Priority 3: Environment variable
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            _log(f"[!ERR] Config file not found: {env_config}", level="error")
            sys.exit(1)
        with open(env_config) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        _log(
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
        _log(f"Loading config: configs/{environment}.yaml (merged with base.yaml)")
        return Config(**merged)

    # Priority 5: Default test config
    default_path = Path("configs/test.yaml")
    if default_path.exists():
        with open(default_path) as f:
            override_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(base_dict, override_dict)
        _log("Loading config: configs/test.yaml (default, merged with base.yaml)")
        return Config(**merged)

    # Priority 6: Built-in defaults only
    _log("[!WARN] No config files found, using built-in defaults", level="warning")
    return Config()
