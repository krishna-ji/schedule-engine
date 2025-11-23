#!/usr/bin/env python3
"""
Final migration script to replace all remaining .quanta attribute access
with the new SessionGene API (start_quanta + num_quanta).
"""

import re
from pathlib import Path

# Files that still need migration
FILES_TO_MIGRATE = [
    "src/ga/evaluator/gpu_batch_evaluator.py",
    "src/ga/operators/local_search.py",
    "src/ga/operators/mutation.py",
    "src/ga/operators/repair_selective.py",
    "src/ga/operators/repair.py",
    "src/ga/operators/violation_detector.py",
    "src/heuristics/diversity.py",
    "src/heuristics/improvement.py",
]


def main():
    base_path = Path(__file__).parent.parent

    for file_path_str in FILES_TO_MIGRATE:
        file_path = base_path / file_path_str

        if not file_path.exists():
            print(f"️  Skipping {file_path_str} (not found)")
            continue

        print(f"\n Processing {file_path_str}...")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern 1: gene.quanta when used to get list (should use get_quanta_list())
        # But be careful - we want to replace assignments differently

        # Pattern 2: if gene.quanta: → if gene.num_quanta > 0:
        content = re.sub(r"\bif\s+gene\.quanta\s*:", "if gene.num_quanta > 0:", content)
        content = re.sub(
            r"\bif\s+not\s+gene\.quanta\s*:", "if gene.num_quanta == 0:", content
        )

        # Pattern 3: len(set(gene.quanta)) → gene.num_quanta (contiguous means no duplicates)
        content = re.sub(r"\blen\(set\(gene\.quanta\)\)", "gene.num_quanta", content)

        # Pattern 4: for q in gene.quanta: → for q in range(gene.start_quanta, gene.end_quanta):
        content = re.sub(
            r"\bfor\s+(\w+)\s+in\s+gene\.quanta\s*:",
            r"for \1 in range(gene.start_quanta, gene.end_quanta):",
            content,
        )
        content = re.sub(
            r"\bfor\s+(\w+)\s+in\s+fixed\.quanta\s*:",
            r"for \1 in range(fixed.start_quanta, fixed.end_quanta):",
            content,
        )
        content = re.sub(
            r"\bfor\s+(\w+)\s+in\s+fixed_session\.quanta\s*:",
            r"for \1 in range(fixed_session.start_quanta, fixed_session.end_quanta):",
            content,
        )

        # Pattern 5: any(... for q in gene.quanta) → any(... for q in range(gene.start_quanta, gene.end_quanta))
        content = re.sub(
            r"\bfor\s+(\w+)\s+in\s+gene\.quanta\)",
            r"for \1 in range(gene.start_quanta, gene.end_quanta))",
            content,
        )

        # Pattern 6: return gene.quanta → return gene.get_quanta_list()
        content = re.sub(
            r"\breturn\s+gene\.quanta\b", "return gene.get_quanta_list()", content
        )

        # Pattern 7: gene.quanta = [...] → need to convert to start/num
        # This is more complex and needs manual review

        # Pattern 8: list(gene.quanta) → gene.get_quanta_list()
        content = re.sub(r"\blist\(gene\.quanta\)", "gene.get_quanta_list()", content)

        # Pattern 9: gene.quanta.copy() → gene.get_quanta_list()
        content = re.sub(r"\bgene\.quanta\.copy\(\)", "gene.get_quanta_list()", content)

        # Pattern 10: new_quanta == gene.quanta → comparison needs both start and num
        content = re.sub(
            r"\b(\w+)\s*==\s*gene\.quanta\b", r"(\1 == gene.get_quanta_list())", content
        )

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f" Updated {file_path_str}")
        else:
            print(f"️  No changes needed for {file_path_str}")

    print("\n" + "=" * 60)
    print("️  WARNING: Some patterns need manual review:")
    print("  - gene.quanta = ... (assignments need conversion)")
    print("  - Complex expressions involving gene.quanta")
    print("=" * 60)


if __name__ == "__main__":
    main()
