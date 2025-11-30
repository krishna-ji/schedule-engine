"""
Migration script to convert SessionGene API across the codebase.

This script performs automated replacements for common patterns:
1. SessionGene(..., quanta=[...]) → SessionGene(..., start_quanta=X, num_quanta=Y)
2. gene.quanta → gene.get_quanta_list() or range(gene.start_quanta, gene.end_quanta)
3. len(gene.quanta) → gene.num_quanta
4. gene.quanta[0] → gene.start_quanta

WARNING: This is a BREAKING CHANGE. Run tests after migration!
"""

import re
import sys
from pathlib import Path

# Migration patterns
PATTERNS = [
    # Pattern 1: len(gene.quanta) → gene.num_quanta
    (r"len\((\w+)\.quanta\)", r"\1.num_quanta"),
    # Pattern 2: gene.quanta[0] → gene.start_quanta
    (r"(\w+)\.quanta\[0\]", r"\1.start_quanta"),
    # Pattern 3: for q in gene.quanta → for q in range(gene.start_quanta, gene.end_quanta)
    (
        r"for (\w+) in (\w+)\.quanta:",
        r"for \1 in range(\2.start_quanta, \2.end_quanta):",
    ),
    # Pattern 4: for q in session.session_quanta → for q in range(session.start_quanta, session.end_quanta)
    # Note: session_quanta is in CourseSession (decoder output), not SessionGene
    # Keep this pattern for reference but apply carefully
    # Pattern 5: gene.quanta = [...] → gene.start_quanta = X; gene.num_quanta = Y
    # This is complex and needs manual review
    # Pattern 6: if gene.quanta: → if gene.num_quanta > 0:
    (r"if (\w+)\.quanta:", r"if \1.num_quanta > 0:"),
    # Pattern 7: max(gene.quanta) → gene.start_quanta + gene.num_quanta - 1
    (r"max\((\w+)\.quanta\)", r"(\1.start_quanta + \1.num_quanta - 1)"),
    # Pattern 8: min(gene.quanta) → gene.start_quanta
    (r"min\((\w+)\.quanta\)", r"\1.start_quanta"),
]


def migrate_file(filepath: Path, dry_run: bool = True) -> tuple[bool, list[str]]:
    """
    Migrate a single file to new SessionGene API.

    Args:
        filepath: Path to file to migrate
        dry_run: If True, only report changes without writing

    Returns:
        (changed, warnings) - Whether file was changed, and any warnings
    """
    try:
        content = filepath.read_text(encoding="utf-8")
        original_content = content
        warnings = []

        # Apply all patterns
        for pattern, replacement in PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                warnings.append(
                    f"Applied pattern: {pattern} → {replacement} ({len(matches)} matches)"
                )

        # Check for SessionGene instantiations (needs manual review)
        sessiongene_constructions = re.findall(r"SessionGene\([^)]+quanta\s*=", content)
        if sessiongene_constructions:
            warnings.append(
                f"️  MANUAL REVIEW NEEDED: Found {len(sessiongene_constructions)} SessionGene(..., quanta=...) constructions"
            )

        # Check for .quanta assignments (complex migration)
        quanta_assignments = re.findall(r"(\w+)\.quanta\s*=\s*(.+)", content)
        if quanta_assignments:
            warnings.append(
                f"️  MANUAL REVIEW NEEDED: Found {len(quanta_assignments)} .quanta = ... assignments"
            )

        changed = content != original_content

        if changed and not dry_run:
            filepath.write_text(content, encoding="utf-8")

        return changed, warnings

    except Exception as e:
        return False, [f"ERROR: {e}"]


def migrate_directory(directory: Path, pattern: str = "**/*.py", dry_run: bool = True):
    """
    Migrate all Python files in a directory.

    Args:
        directory: Root directory to scan
        pattern: Glob pattern for files
        dry_run: If True, only report changes without writing
    """
    files = list(directory.glob(pattern))
    print(f"Found {len(files)} files matching {pattern}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'WRITE MODE'}")
    print("=" * 80)

    changed_files = []
    all_warnings = []

    for filepath in files:
        if "__pycache__" in str(filepath) or "venv" in str(filepath):
            continue

        changed, warnings = migrate_file(filepath, dry_run=dry_run)

        if changed or warnings:
            print(f"\n {filepath.relative_to(directory)}")
            if changed:
                print("    Changed")
                changed_files.append(filepath)
            for warning in warnings:
                print(f"   {warning}")
            all_warnings.extend(warnings)

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Files changed: {len(changed_files)}")
    print(f"  Total warnings: {len(all_warnings)}")
    print(
        f"  Manual review items: {sum(1 for w in all_warnings if 'MANUAL REVIEW' in w)}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate SessionGene API")
    parser.add_argument("directory", type=Path, help="Directory to migrate")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write changes (default is dry-run)",
    )
    parser.add_argument("--pattern", default="**/*.py", help="File pattern to match")

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory {args.directory} does not exist")
        sys.exit(1)

    dry_run = not args.write
    migrate_directory(args.directory, args.pattern, dry_run=dry_run)
