#!/usr/bin/env python3
from pathlib import Path

file_path = Path("src/rl/local_search/solution_selector.py")
content = file_path.read_text(encoding="utf-8")

# Fix double closing bracket
content = content.replace(
    "candidates]]  # type: ignore[attr-defined])",
    "candidates])  # type: ignore[attr-defined]",
)

file_path.write_text(content, encoding="utf-8")
print(f"Fixed {file_path}")
