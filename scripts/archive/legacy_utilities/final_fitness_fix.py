#!/usr/bin/env python3
"""Final comprehensive fix for ALL .fitness type: ignore syntax errors."""

import re
from pathlib import Path


def fix_all_patterns(content: str) -> str:
    """Apply all fix patterns."""

    # 1. Generator: min/max(ind.fitness.values  # type: ignore... for ind in pop)
    content = re.sub(
        r"(min|max)\((\w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\s+for",
        r"\1(\2 for",
        content,
    )

    # 2. List comp: [ind.fitness.values  # type: ignore... for ind in pop]
    content = re.sub(
        r"\[(\w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\s+for",
        r"[\1 for",
        content,
    )

    # 3. np.array(ind.fitness.values  # type: ignore...)
    content = re.sub(
        r"np\.array\((\w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\)",
        r"np.array(\1)  # type: ignore[attr-defined]",
        content,
    )

    # 4. lambda: x.fitness.values  # type: ignore...)
    content = re.sub(
        r"(lambda \w+: \w+\.fitness\.values)\s+# type: ignore\[attr-defined\]\)",
        r"\1)  # type: ignore[attr-defined]",
        content,
    )

    # 5. .fitness.values  # type: ignore...[0]
    content = re.sub(
        r"(\.fitness\.values)\s+# type: ignore\[attr-defined\]\[(\d+)\]",
        r"\1[\2]  # type: ignore[attr-defined]",
        content,
    )

    return content


# Process ALL files that might have fitness access
all_files = list(Path("src").rglob("*.py"))

fixed = []
for file_path in all_files:
    content = file_path.read_text(encoding="utf-8")
    original = content

    # Only process if it has .fitness
    if ".fitness" not in content:
        continue

    content = fix_all_patterns(content)

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        fixed.append(file_path)

print(f"Fixed {len(fixed)} files:")
for f in fixed:
    print(f"  ✓ {f}")
