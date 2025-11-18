"""
Script to remove redundant CSV exports from plot functions.

All CSV data is already in data/metrics.csv, so individual CSV files per plot
are redundant and waste disk space.
"""

import re
from pathlib import Path


def remove_csv_exports_from_file(filepath: Path):
    """Remove CSV export code from a plotting file."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Pattern 1: Remove CSV directory creation
    content = re.sub(
        r'\n\s+# Create CSVs subdirectory\n\s+csv_dir = os\.path\.join\(output_dir, "CSVs"\)\n\s+os\.makedirs\(csv_dir, exist_ok=True\)\n',
        "\n",
        content,
    )

    # Pattern 2: Remove individual CSV writes
    content = re.sub(
        r"\n\s+# Save .*? to CSV\n\s+csv_path = os\.path\.join\(csv_dir,.*?\n(?:\s+with open\(csv_path.*?\n(?:\s+.*?\n)*?\s+writer\.writerow\(.*?\)\n)+",
        "\n",
        content,
        flags=re.DOTALL,
    )

    # Pattern 3: Remove simpler CSV writes
    content = re.sub(
        r"\s+csv_path = os\.path\.join\(csv_dir,.*?\)\n\s+with open\(csv_path.*?\n(?:\s+.*?\n)+?\s+writer\.writerow\(row\)\n",
        "",
        content,
        flags=re.DOTALL,
    )

    # Update docstrings that mention CSVs/
    content = re.sub(r"- CSVs/(.*?)\.csv: .*?\n", "", content)

    # Add note about data/metrics.csv in docstrings
    content = re.sub(r"(    Saves:\n        - plots/)", r"\1", content)

    if content != original:
        print(f"✓ Modified: {filepath.name}")
        filepath.write_text(content, encoding="utf-8")
        return True
    else:
        print(f"  Skipped: {filepath.name} (no changes)")
        return False


def main():
    """Remove CSV exports from all plot files."""
    base_dir = Path(__file__).parent.parent / "src" / "exporter"

    plot_files = [
        "plot_detailed_constraints.py",
        "plot_hypervolume.py",
        "plot_metrics_comparison.py",
    ]

    modified = 0
    for filename in plot_files:
        filepath = base_dir / filename
        if filepath.exists():
            if remove_csv_exports_from_file(filepath):
                modified += 1
        else:
            print(f"✗ Not found: {filename}")

    print(f"\n{modified} files modified")


if __name__ == "__main__":
    main()
