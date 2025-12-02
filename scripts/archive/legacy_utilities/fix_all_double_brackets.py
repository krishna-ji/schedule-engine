#!/usr/bin/env python3
from pathlib import Path

file_path = Path("src/rl/local_search/solution_selector.py")
content = file_path.read_text(encoding="utf-8")

# Fix all double closing brackets created by previous regex
content = content.replace(
    "population]]  # type: ignore[attr-defined])",
    "population])  # type: ignore[attr-defined]",
)

file_path.write_text(content, encoding="utf-8")
print(f"Fixed all double brackets in {file_path}")
