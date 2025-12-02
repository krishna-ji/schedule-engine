#!/usr/bin/env python3
"""
Add # type: ignore[attr-defined] comments to .fitness attribute accesses.
DEAP adds .fitness at runtime, so these are legitimate type ignores.
"""

from pathlib import Path

# Files with .fitness attribute access
FILES_TO_FIX = [
    "src/rl/gym_env/schedule_env.py",
    "src/rl/rewards/base_reward.py",
    "src/rl/multi_agent/specialist_agents.py",
    "src/rl/multi_agent/rank_based_agents.py",
    "src/rl/local_search/solution_selector.py",
    "src/rl/local_search/memetic_policy.py",
    "src/ga/archive/novelty_archive.py",
    "src/ga/archive/map_elites.py",
]


def fix_fitness_access(file_path: Path) -> int:
    """Add type: ignore to .fitness accesses without it."""
    if not file_path.exists():
        print(f"Skip (not found): {file_path}")
        return 0

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    modified = 0

    for i, line in enumerate(lines):
        # Skip if already has type: ignore
        if "# type: ignore" in line:
            continue

        # Pattern: .fitness.something (values, valid, etc.)
        if ".fitness." in line:
            lines[i] = line.rstrip() + "  # type: ignore[attr-defined]\n"
            modified += 1

    if modified > 0:
        file_path.write_text("".join(lines), encoding="utf-8")
        print(f"✓ Fixed {modified} lines in {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

    return modified


def main():
    root = Path(__file__).parent.parent.parent
    total = 0

    for file_rel in FILES_TO_FIX:
        file_path = root / file_rel
        total += fix_fitness_access(file_path)

    print(f"\nTotal lines modified: {total}")


if __name__ == "__main__":
    main()
