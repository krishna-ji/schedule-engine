"""
Configuration loader with hierarchical config support.
Supports both legacy (single file) and new (multi-domain) structure.

Legacy structure (backward compatible):
- configs/base.yaml (all settings)
- configs/{env}.yaml (overrides)

New hierarchical structure:
- configs/common/{base,env}.yaml (shared settings)
- configs/ga/{base,env}.yaml (GA-specific)
- configs/rl/{base,env}.yaml (RL-specific)
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


def _load_yaml(path: Path) -> dict:
    """Load YAML file safely, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_hierarchical_config(environment: str) -> dict:
    """
    Load hierarchical configuration from configs/{common,ga,rl}/ structure.
    
    Args:
        environment: Environment name (prod, test, med)
    
    Returns:
        Merged configuration dictionary
    """
    # Load base configs for each domain
    common_base = _load_yaml(Path("configs/common/base.yaml"))
    ga_base = _load_yaml(Path("configs/ga/base.yaml"))
    rl_base = _load_yaml(Path("configs/rl/base.yaml"))
    
    # Load environment-specific overrides
    common_env = _load_yaml(Path(f"configs/common/{environment}.yaml"))
    ga_env = _load_yaml(Path(f"configs/ga/{environment}.yaml"))
    rl_env = _load_yaml(Path(f"configs/rl/{environment}.yaml"))
    
    # Merge in order: common -> ga -> rl, with env overrides
    result = {}
    result = _deep_merge(result, common_base)
    result = _deep_merge(result, common_env)
    result = _deep_merge(result, ga_base)
    result = _deep_merge(result, ga_env)
    result = _deep_merge(result, rl_base)
    result = _deep_merge(result, rl_env)
    
    return result


def _check_hierarchical_structure() -> bool:
    """Check if hierarchical config structure exists."""
    return (Path("configs/common").exists() and 
            Path("configs/ga").exists() and 
            Path("configs/rl").exists())


def load_config(config_path: str = None) -> Config:
    """
    Load configuration with support for hierarchical structure.

    Config structure (new hierarchical):
    - configs/common/{base,env}.yaml: Shared settings
    - configs/ga/{base,env}.yaml: GA-specific settings
    - configs/rl/{base,env}.yaml: RL-specific settings

    Legacy structure (backward compatible):
    - configs/base.yaml: All settings
    - configs/{env}.yaml: Environment overrides

    Loading priority:
    1. Explicit config_path argument (--config flag)
    2. SCHEDULE_CONFIG environment variable
    3. Hierarchical structure (if exists): configs/{common,ga,rl}/{env}.yaml
    4. Legacy structure: configs/{env}.yaml merged with base.yaml
    5. Built-in defaults

    Args:
        config_path: Path to config YAML file

    Returns:
        Config object
    """
    # Priority 1: Explicit path (legacy single-file mode)
    if config_path:
        if not Path(config_path).exists():
            print(f"[!ERR] Config file not found: {config_path}")
            sys.exit(1)
        
        # Load with legacy base.yaml merge if base exists
        base_path = Path("configs/base.yaml")
        if base_path.exists():
            base_dict = _load_yaml(base_path)
            override_dict = _load_yaml(Path(config_path))
            merged = _deep_merge(base_dict, override_dict)
            print(f"Loading config: {config_path} (merged with base.yaml)")
            return Config(**merged)
        else:
            # Direct load without base
            config_dict = _load_yaml(Path(config_path))
            print(f"Loading config: {config_path}")
            return Config(**config_dict)

    # Priority 2: Environment variable (legacy single-file mode)
    env_config = os.getenv("SCHEDULE_CONFIG")
    if env_config:
        if not Path(env_config).exists():
            print(f"[!ERR] Config file not found: {env_config}")
            sys.exit(1)
        
        base_path = Path("configs/base.yaml")
        if base_path.exists():
            base_dict = _load_yaml(base_path)
            override_dict = _load_yaml(Path(env_config))
            merged = _deep_merge(base_dict, override_dict)
            print(f"Loading config from SCHEDULE_CONFIG: {env_config} (merged with base.yaml)")
            return Config(**merged)
        else:
            config_dict = _load_yaml(Path(env_config))
            print(f"Loading config from SCHEDULE_CONFIG: {env_config}")
            return Config(**config_dict)

    # Determine environment
    environment = os.getenv("ENVIRONMENT", "test")
    
    # Priority 3: Hierarchical structure (new method)
    if _check_hierarchical_structure():
        merged = _load_hierarchical_config(environment)
        if merged:  # If we got any config from hierarchical structure
            print(f"Loading hierarchical config: configs/{{common,ga,rl}}/{environment}.yaml")
            return Config(**merged)

    # Priority 4: Legacy structure (backward compatibility)
    base_path = Path("configs/base.yaml")
    env_path = Path(f"configs/{environment}.yaml")
    
    if base_path.exists() or env_path.exists():
        base_dict = _load_yaml(base_path)
        override_dict = _load_yaml(env_path)
        merged = _deep_merge(base_dict, override_dict)
        print(f"Loading legacy config: configs/{environment}.yaml (merged with base.yaml)")
        return Config(**merged)

    # Priority 5: Built-in defaults only
    print("[!WARN] No config files found, using built-in defaults")
    return Config()
