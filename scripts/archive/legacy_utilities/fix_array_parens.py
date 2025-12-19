#!/usr/bin/env python3
from pathlib import Path

file_path = Path("src/rl/local_search/solution_selector.py")
content = file_path.read_text(encoding="utf-8")

# Fix np.array lines with missing closing paren
content = content.replace(
    "np.array(ind1.fitness.values  # type: ignore[attr-defined])",
    "np.array(ind1.fitness.values)  # type: ignore[attr-defined]",
)

content = content.replace(
    "np.array(ind2.fitness.values  # type: ignore[attr-defined])",
    "np.array(ind2.fitness.values)  # type: ignore[attr-defined]",
)

file_path.write_text(content, encoding="utf-8")
print(f"Fixed {file_path}")
