#!/usr/bin/env python3
"""Fix syntax errors in schedule_env.py where type: ignore is placed incorrectly."""

from pathlib import Path

file_path = Path("src/rl/gym_env/schedule_env.py")
content = file_path.read_text(encoding="utf-8")

# Fix the malformed type ignore comments in lambda
# Pattern: ind.fitness.values  # type: ignore[attr-defined][0]
# Should be: ind.fitness.values[0])  # type: ignore[attr-defined]

content = content.replace(
    "ind.fitness.values  # type: ignore[attr-defined][0])",
    "ind.fitness.values[0])  # type: ignore[attr-defined]",
)

file_path.write_text(content, encoding="utf-8")
print(f"Fixed syntax errors in {file_path}")
