#!/usr/bin/env python3
"""Fix all type: ignore comments inside comprehensions/lambdas."""

import re
from pathlib import Path

files = [
    "src/rl/local_search/solution_selector.py",
    "src/rl/multi_agent/rank_based_agents.py",
    "src/ga/archive/novelty_archive.py",
]

for file_rel in files:
    file_path = Path(file_rel)
    if not file_path.exists():
        continue

    content = file_path.read_text(encoding="utf-8")
    original = content

    # Pattern: [ind.fitness.values  # type: ignore[attr-defined] for ind in ...]
    # To: [ind.fitness.values for ind in ...]]  # type: ignore[attr-defined]
    content = re.sub(
        r"\[ind\.fitness\.values\s+# type: ignore\[attr-defined\]\s+for ind in ([^\]]+)\]",
        r"[ind.fitness.values for ind in \1]]  # type: ignore[attr-defined]",
        content,
    )

    # Pattern: (ind.fitness.values  # type: ignore[attr-defined] for ...
    # To: (ind.fitness.values for ...)  # type: ignore[attr-defined]
    content = re.sub(
        r"\(ind\.fitness\.values\s+# type: ignore\[attr-defined\]\s+for",
        r"(ind.fitness.values for",
        content,
    )

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"✓ Fixed {file_path}")

print("Done!")
