#!/usr/bin/env python3
"""Fix all malformed type: ignore comments."""

import re
from pathlib import Path


def fix_file(file_path: Path) -> int:
    """Fix syntax errors in a file."""
    if not file_path.exists():
        return 0

    content = file_path.read_text(encoding="utf-8")
    original = content

    # Pattern 1: .values  # type: ignore[attr-defined][0] -> .values[0]  # type: ignore[attr-defined]
    content = re.sub(
        r"\.fitness\.values\s+# type: ignore\[attr-defined\]\[(\d+)\]",
        r".fitness.values[\1]  # type: ignore[attr-defined]",
        content,
    )

    # Pattern 2: .values  # type: ignore[attr-defined][0], -> .values[0],  # type: ignore[attr-defined]
    content = re.sub(
        r"\.fitness\.values\s+# type: ignore\[attr-defined\]\[(\d+)\],",
        r".fitness.values[\1],  # type: ignore[attr-defined]",
        content,
    )

    # Pattern 3: .values  # type: ignore[attr-defined] for ind -> .values for ind]  # type: ignore[attr-defined]
    content = re.sub(
        r"\.fitness\.values\s+# type: ignore\[attr-defined\]\s+for ind in",
        r".fitness.values for ind in",
        content,
    )
    content = re.sub(
        r"(np\.array\(\[ind\.fitness\.values for ind in population\])\)",
        r"\1])  # type: ignore[attr-defined]",
        content,
    )

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"✓ Fixed {file_path}")
        return 1
    return 0


# Fix all files with syntax errors
files = [
    Path("src/rl/gym_env/schedule_env.py"),
    Path("src/rl/local_search/memetic_policy.py"),
]

total = sum(fix_file(f) for f in files)
print(f"\nTotal files fixed: {total}")
