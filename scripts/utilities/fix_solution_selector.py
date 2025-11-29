#!/usr/bin/env python3
"""Fix lambda type: ignore syntax in solution_selector.py"""

import re
from pathlib import Path

file_path = Path("src/rl/local_search/solution_selector.py")
content = file_path.read_text(encoding="utf-8")

# Fix: key=lambda ind: ind.fitness.values  # type: ignore[attr-defined])
# To: key=lambda ind: ind.fitness.values)  # type: ignore[attr-defined]

content = re.sub(
    r"(key=lambda ind: ind\.fitness\.values)\s+# type: ignore\[attr-defined\]\)",
    r"\1)  # type: ignore[attr-defined]",
    content,
)

file_path.write_text(content, encoding="utf-8")
print(f"Fixed {file_path}")
