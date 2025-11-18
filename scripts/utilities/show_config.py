"""
Display current constraint configuration (both hard and soft).
Quick utility to see which constraints are enabled and their weights.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    get_config,
)  # get_config().hard_constraints, get_config().soft_constraints
from src.constraints.registry import (
    get_enabled_hard_constraints,
    get_enabled_soft_constraints,
)
from src.utils.console import write_header, write_separator, write_info


def main():
    write_header("CONSTRAINT CONFIGURATION (HARD + SOFT)")

    # Hard Constraints Section
    write_info("")
    write_separator()
    write_info("HARD CONSTRAINTS (Feasibility)".center(80))
    write_separator()

    hard_enabled_count = sum(
        1 for cfg in get_config().hard_constraints.values() if cfg["enabled"]
    )
    hard_total_count = len(get_config().hard_constraints)

    write_info(
        f"Status: {hard_enabled_count}/{hard_total_count} hard constraints enabled"
    )
    write_info("")

    # Show enabled hard constraints
    hard_enabled = get_enabled_hard_constraints()
    if hard_enabled:
        write_info("ENABLED HARD CONSTRAINTS:")
        write_separator("-")
        for name, info in hard_enabled.items():
            write_info(f"  {name:<45} weight = {info['weight']:.2f}")
        write_info("")

    # Show disabled hard constraints
    hard_disabled = [
        name
        for name, cfg in get_config().hard_constraints.items()
        if not cfg["enabled"]
    ]
    if hard_disabled:
        write_info("DISABLED HARD CONSTRAINTS:")
        write_separator("-")
        for name in hard_disabled:
            write_info(
                f"  {name:<45} (weight = {get_config().hard_constraints[name]['weight']:.2f})"
            )
        write_info("")

    # Show total hard weight
    total_hard_weight = sum(info["weight"] for info in hard_enabled.values())
    write_info(f"Total enabled hard weight: {total_hard_weight:.2f}")

    # Soft Constraints Section
    write_info("")
    write_separator()
    write_info("SOFT CONSTRAINTS (Quality)".center(80))
    write_separator()

    soft_enabled_count = sum(
        1 for cfg in get_config().soft_constraints.values() if cfg["enabled"]
    )
    soft_total_count = len(get_config().soft_constraints)

    write_info(
        f"Status: {soft_enabled_count}/{soft_total_count} soft constraints enabled"
    )
    write_info("")

    # Show enabled soft constraints
    soft_enabled = get_enabled_soft_constraints()
    if soft_enabled:
        write_info("ENABLED SOFT CONSTRAINTS:")
        write_separator("-")
        for name, info in soft_enabled.items():
            write_info(f"  {name:<45} weight = {info['weight']:.2f}")
        write_info("")

    # Show disabled soft constraints
    soft_disabled = [
        name
        for name, cfg in get_config().soft_constraints.items()
        if not cfg["enabled"]
    ]
    if soft_disabled:
        write_info("DISABLED SOFT CONSTRAINTS:")
        write_separator("-")
        for name in soft_disabled:
            write_info(
                f"  {name:<45} (weight = {get_config().soft_constraints[name]['weight']:.2f})"
            )
        write_info("")

    # Show total soft weight
    total_soft_weight = sum(info["weight"] for info in soft_enabled.values())
    write_info(f"Total enabled soft weight: {total_soft_weight:.2f}")

    # Summary
    write_info("")
    write_separator()
    write_info(
        f"SUMMARY: {hard_enabled_count + soft_enabled_count} total constraints enabled"
    )
    write_info(f"  - Hard: {hard_enabled_count}/{hard_total_count}")
    write_info(f"  - Soft: {soft_enabled_count}/{soft_total_count}")
    write_separator()
    write_info("")
    write_info("To modify: Edit config/constraints.py")


if __name__ == "__main__":
    main()
