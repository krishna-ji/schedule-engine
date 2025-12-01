"""
Script to replace .quanta = ... assignments with new API.

Replaces patterns like:
    gene.quanta = new_quanta
With:
    from src.ga.quanta_converter import assign_quanta_to_gene
    assign_quanta_to_gene(gene, new_quanta)
"""

import re
from pathlib import Path


def replace_quanta_assignments(filepath: Path, dry_run: bool = True):
    """Replace .quanta = assignments in a file."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Pattern: gene.quanta = value
    pattern = r"(\s+)(\w+)\.quanta\s*=\s*(.+?)(?:\n|$)"

    def replacement(match):
        indent = match.group(1)
        var_name = match.group(2)
        value = match.group(3)

        # Add import at top of function if not present
        replacement_text = (
            f"{indent}# Convert quanta list to new API\n"
            f"{indent}from src.ga.quanta_converter import assign_quanta_to_gene\n"
            f"{indent}assign_quanta_to_gene({var_name}, {value})\n"
        )
        return replacement_text

    content = re.sub(pattern, replacement, content)

    if content != original:
        print(f"Modified: {filepath}")
        if not dry_run:
            filepath.write_text(content, encoding="utf-8")
        return True
    return False


if __name__ == "__main__":
    import sys

    dry_run = "--write" not in sys.argv
    print(f"Mode: {'DRY RUN' if dry_run else 'WRITE'}")

    files_to_process = [
        "src/ga/operators/repair.py",
        "src/ga/operators/repair_selective.py",
        "src/ga/operators/crossover.py",
        "src/ga/operators/constraint_guided_mutation.py",
        "src/heuristics/perturbation.py",
        "src/heuristics/utils.py",
        "src/decoder/individual_decoder.py",
    ]

    root = Path(".")
    modified = 0

    for file_path in files_to_process:
        p = root / file_path
        if p.exists() and replace_quanta_assignments(p, dry_run):
            modified += 1

    print(f"\nModified {modified} files")
