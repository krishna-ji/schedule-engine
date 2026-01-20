#!/usr/bin/env python3
"""
Config Validation Script

Validates all experiment configs can be instantiated and have correct killswitches.
Run: uv run python scripts/utilities/validate_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.experiments import EXPERIMENT_REGISTRY


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{text}")
    print("=" * 60)


def print_status(passed: bool, message: str) -> None:
    """Print a status message."""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {status}: {message}")


def validate_experiment_config(name: str, experiment: dict) -> tuple[bool, list[str]]:
    """Validate a single experiment configuration."""
    errors = []

    try:
        # Test instantiation
        test_config = experiment["test_config"]
        prod_config = experiment["prod_config"]

        # Basic checks
        if not hasattr(test_config, "ngen"):
            errors.append("Missing 'ngen' attribute")
        if not hasattr(test_config, "pop_size"):
            errors.append("Missing 'pop_size' attribute")

        if not hasattr(prod_config, "ngen"):
            errors.append("Missing 'ngen' attribute in prod config")
        if not hasattr(prod_config, "pop_size"):
            errors.append("Missing 'pop_size' attribute in prod config")

        # Profile differentiation checks
        if test_config.ngen >= prod_config.ngen:
            errors.append(
                f"Test ngen ({test_config.ngen}) should be < prod ngen ({prod_config.ngen})"
            )
        if test_config.pop_size >= prod_config.pop_size:
            errors.append(
                f"Test pop_size ({test_config.pop_size}) should be < prod pop_size ({prod_config.pop_size})"
            )

        # Experiment-specific killswitch checks
        if name == "baseline":
            # Baseline should have everything disabled
            critical_flags = [
                ("repair_enabled", False),
                ("heuristics_master_enabled", False),
                ("rl_enabled", False),
                ("lns_enabled", False),
            ]
            for flag, expected in critical_flags:
                if hasattr(test_config, flag):
                    actual = getattr(test_config, flag)
                    if actual != expected:
                        errors.append(
                            f"Baseline {flag} should be {expected}, got {actual}"
                        )

        elif name == "memetic":
            # Memetic should have repair enabled, others disabled
            if (
                hasattr(test_config, "repair_enabled")
                and not test_config.repair_enabled
            ):
                errors.append("Memetic should have repair_enabled=True")
            if (
                hasattr(test_config, "heuristics_master_enabled")
                and test_config.heuristics_master_enabled
            ):
                errors.append("Memetic should have heuristics_master_enabled=False")

        elif name == "rl_guided":
            # RL should have RL enabled
            if hasattr(test_config, "rl_enabled") and not test_config.rl_enabled:
                errors.append("RL guided should have rl_enabled=True")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"Failed to validate: {e}"]


def main() -> None:
    """Run config validation for all experiments."""
    print_header(" Validating Experiment Configurations")

    all_passed = True
    total_experiments = len(EXPERIMENT_REGISTRY)

    for name, experiment in EXPERIMENT_REGISTRY.items():
        print(f"\nExperiment {experiment['id']}: {name}")
        print(f"  Description: {experiment['description']}")

        # Validate configs
        passed, errors = validate_experiment_config(name, experiment)

        if passed:
            print("  ✓ PASSED")

            # Show key settings
            test_cfg = experiment["test_config"]
            prod_cfg = experiment["prod_config"]
            print(f"    Test:  {test_cfg.ngen} gens, {test_cfg.pop_size} pop")
            print(f"    Prod:  {prod_cfg.ngen} gens, {prod_cfg.pop_size} pop")

            # Show key killswitches
            key_flags = [
                "repair_enabled",
                "heuristics_master_enabled",
                "rl_enabled",
                "lns_enabled",
            ]
            killswitches = []
            for flag in key_flags:
                if hasattr(test_cfg, flag):
                    value = getattr(test_cfg, flag)
                    status = "ON" if value else "OFF"
                    killswitches.append(f"{flag.replace('_enabled', '')}:{status}")

            if killswitches:
                print(f"    Flags: {' | '.join(killswitches)}")
        else:
            print("  ✗ FAILED")
            for error in errors:
                print(f"    - {error}")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"✓ ALL {total_experiments} EXPERIMENTS PASSED")
        print("\nAll experiment configs are properly structured and ready to use.")
    else:
        print("✗ VALIDATION FAILED")
        print("\nFix the errors above before running experiments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
