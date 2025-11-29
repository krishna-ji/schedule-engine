#!/usr/bin/env python3
"""
Comprehensive fix for all .fitness type: ignore syntax errors.
This script fixes type: ignore comments that are placed inside expressions.
"""

import re
from pathlib import Path


def fix_fitness_syntax(file_path: Path) -> int:
    """Fix all .fitness type: ignore syntax issues in a file."""
    if not file_path.exists():
        return 0

    content = file_path.read_text(encoding="utf-8")
    original = content
    changes = 0

    # Pattern 1: np.array(ind.fitness.values  # type: ignore...) -> np.array(ind.fitness.values)  # type: ignore...
    pattern1 = r"np\.array\((\w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\)"
    if re.search(pattern1, content):
        content = re.sub(
            pattern1, r"np.array(\1)  # type: ignore[attr-defined]", content
        )
        changes += 1

    # Pattern 2: [ind.fitness.values  # type: ignore... for ind in ...] -> [ind.fitness.values for ind in ...]]  # type: ignore...
    # But ensure we don't double the closing bracket
    pattern2 = r"\[ind\.fitness\.values\s+# type: ignore\[attr-defined\]\s+for ind in ([^\]]+)\]"
    matches = list(re.finditer(pattern2, content))
    for match in matches:
        old_text = match.group(0)
        list_expr = match.group(1)
        # Only fix if it doesn't already have double brackets
        if content[match.end() : match.end() + 1] != "]":
            new_text = f"[ind.fitness.values for ind in {list_expr}]  # type: ignore[attr-defined]"
            content = content.replace(old_text, new_text, 1)
            changes += 1

    # Pattern 3: ind.fitness.values  # type: ignore...[0] -> ind.fitness.values[0]  # type: ignore...
    pattern3 = r"(\w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\[(\d+)\]"
    if re.search(pattern3, content):
        content = re.sub(pattern3, r"\1[\2]  # type: ignore[attr-defined]", content)
        changes += 1

    # Pattern 4: key=lambda x: x.fitness.values  # type: ignore...) -> key=lambda x: x.fitness.values)  # type: ignore...
    pattern4 = (
        r"(key=lambda \w+: \w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\)"
    )
    if re.search(pattern4, content):
        content = re.sub(pattern4, r"\1)  # type: ignore[attr-defined]", content)
        changes += 1

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        return changes
    return 0


# Process all RL files that might have .fitness access
files_to_check = [
    "src/rl/multi_agent/rank_based_agents.py",
    "src/rl/multi_agent/specialist_agents.py",
    "src/ga/archive/novelty_archive.py",
    "src/ga/archive/map_elites.py",
]

total = 0
for file_rel in files_to_check:
    file_path = Path(file_rel)
    changes = fix_fitness_syntax(file_path)
    if changes > 0:
        print(f"✓ Fixed {changes} patterns in {file_path}")
        total += changes

print(f"\nTotal changes: {total}")
