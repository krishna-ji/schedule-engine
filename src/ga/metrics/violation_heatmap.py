"""
Constraint Violation Heatmap

Tracks which genes (course-group pairs) violate constraints most frequently.
Used to target repair efforts on "hot" genes that cause persistent violations.

Usage:
    from src.ga.metrics.violation_heatmap import ViolationHeatmap

    heatmap = ViolationHeatmap()

    # Record violations during fitness evaluation
    for gene in individual:
        if violates_availability(gene):
            heatmap.record_violation(gene, "availability")

    # Get hotspots for targeted repair
    hotspots = heatmap.get_hotspots(top_n=20)
    for gene_key, total_violations, breakdown in hotspots:
        print(f"{gene_key}: {total_violations} violations")
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.domain.gene import SessionGene


class ViolationHeatmap:
    """
    Tracks violation frequency per gene for targeted repair.

    Architecture:
    - Gene Key: (course_id, course_type, tuple(sorted(group_ids)))
    - Violation Types: availability, overlap, instructor_conflict, room_conflict,
                      qualification, room_type
    - Persistence: JSON file for cross-run analysis

    Attributes:
        violations: Dict mapping gene_key → violation_type → count
        generation_history: Track violations per generation
    """

    def __init__(self) -> None:
        """Initialize empty heatmap."""
        self.violations: dict[tuple, dict[str, int]] = defaultdict(
            lambda: {
                "availability": 0,
                "overlap": 0,
                "instructor_conflict": 0,
                "room_conflict": 0,
                "qualification": 0,
                "room_type": 0,
                "total": 0,
            }
        )
        self.generation_history: list[
            dict[str, Any]
        ] = []  # List of {gen: int, violations: dict}

    def record_violation(self, gene: SessionGene, violation_type: str) -> None:
        """
        Record that a gene violated a constraint.

        Args:
            gene: SessionGene that violated constraint
            violation_type: Type of violation (availability, overlap, etc.)
        """
        # Create gene key (unique identifier for course-group pair)
        key = self._make_gene_key(gene)

        # Increment violation count
        if violation_type in self.violations[key]:
            self.violations[key][violation_type] += 1
            self.violations[key]["total"] += 1

    def record_generation(self, gen: int) -> None:
        """
        Save current violation state for this generation.

        Args:
            gen: Generation number
        """
        # Deep copy current violations
        snapshot = {
            "generation": gen,
            "total_genes_violated": len(
                [v for v in self.violations.values() if v["total"] > 0]
            ),
            "total_violations": sum(v["total"] for v in self.violations.values()),
        }
        self.generation_history.append(snapshot)

    def get_hotspots(self, top_n: int = 20) -> list[tuple]:
        """
        Get genes with most frequent violations (hotspots).

        Args:
            top_n: Number of hotspots to return

        Returns:
            List of (gene_key, total_violations, violation_breakdown) tuples
            Sorted by total violations (descending)
        """
        scores = []
        for key, counts in self.violations.items():
            if counts["total"] > 0:
                scores.append((key, counts["total"], counts))

        # Sort by total violations (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def get_violations_by_type(self) -> dict[str, int]:
        """
        Get total violations grouped by type.

        Returns:
            Dict mapping violation_type → total_count
        """
        totals: dict[str, int] = defaultdict(int)
        for counts in self.violations.values():
            for vtype, count in counts.items():
                if vtype != "total":
                    totals[vtype] += count
        return dict(totals)

    def save_to_file(self, filepath: str) -> None:
        """
        Persist heatmap to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "violations": {str(k): v for k, v in self.violations.items()},
            "generation_history": self.generation_history,
            "summary": {
                "total_genes_tracked": len(self.violations),
                "genes_with_violations": len(
                    [v for v in self.violations.values() if v["total"] > 0]
                ),
                "total_violations": sum(v["total"] for v in self.violations.values()),
                "by_type": self.get_violations_by_type(),
            },
        }

        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        with filepath_obj.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ViolationHeatmap":
        """
        Load heatmap from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ViolationHeatmap instance
        """
        heatmap = cls()

        if not Path(filepath).exists():
            return heatmap

        with Path(filepath).open() as f:
            data = json.load(f)

        # Restore violations (convert string keys back to tuples)
        for key_str, counts in data.get("violations", {}).items():
            # Parse key string back to tuple
            key = eval(key_str)  # Safe here - we control the format
            heatmap.violations[key] = counts

        heatmap.generation_history = data.get("generation_history", [])

        return heatmap

    def merge(self, other: "ViolationHeatmap") -> None:
        """
        Merge another heatmap into this one (for multi-run analysis).

        Args:
            other: Another ViolationHeatmap to merge
        """
        for key, counts in other.violations.items():
            for vtype, count in counts.items():
                self.violations[key][vtype] += count

    def reset(self) -> None:
        """Clear all violation data."""
        self.violations.clear()
        self.generation_history.clear()

    def _make_gene_key(self, gene: SessionGene) -> tuple:
        """
        Create unique key for a gene.

        Args:
            gene: SessionGene

        Returns:
            Tuple of (course_id, course_type, sorted_group_ids_tuple)
        """
        return (gene.course_id, gene.course_type, tuple(sorted(gene.group_ids)))

    def print_summary(self, console: Any = None) -> None:
        """
        Print heatmap summary to console.

        Args:
            console: Rich Console instance (optional)
        """
        if console is None:
            from rich.console import Console

            console = Console()

        from rich.table import Table

        console.print("\n[bold cyan]Violation Heatmap Summary[/bold cyan]")

        # Overall stats
        total_genes = len(self.violations)
        genes_with_violations = len(
            [v for v in self.violations.values() if v["total"] > 0]
        )
        total_violations = sum(v["total"] for v in self.violations.values())

        console.print(f"  Total genes tracked: {total_genes}")
        console.print(f"  Genes with violations: {genes_with_violations}")
        console.print(f"  Total violations recorded: {total_violations}")

        # By type
        by_type = self.get_violations_by_type()
        if by_type:
            console.print("\n[bold]Violations by type:[/bold]")
            for vtype, count in sorted(
                by_type.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (
                    (count / total_violations * 100) if total_violations > 0 else 0
                )
                console.print(f"  {vtype:25s}: {count:5d} ({percentage:5.1f}%)")

        # Top hotspots
        hotspots = self.get_hotspots(top_n=10)
        if hotspots:
            console.print(
                "\n[bold yellow]Top 10 Hotspots (most violated genes):[/bold yellow]"
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", justify="right", width=5)
            table.add_column("Course", width=12)
            table.add_column("Type", width=10)
            table.add_column("Groups", width=15)
            table.add_column("Total", justify="right", width=8)
            table.add_column("Availability", justify="right", width=11)
            table.add_column("Overlap", justify="right", width=8)

            for rank, (key, total, breakdown) in enumerate(hotspots, 1):
                course_id, course_type, group_ids = key
                groups_str = ", ".join(group_ids[:2])  # Show first 2 groups
                if len(group_ids) > 2:
                    groups_str += f" +{len(group_ids) - 2}"

                table.add_row(
                    str(rank),
                    course_id[:12],
                    course_type[:10],
                    groups_str[:15],
                    str(total),
                    str(breakdown["availability"]),
                    str(breakdown["overlap"]),
                )

            console.print(table)


if __name__ == "__main__":
    """Quick test of violation heatmap."""
    from rich.console import Console

    console = Console()

    # Create test heatmap
    heatmap = ViolationHeatmap()

    # Simulate some violations
    from src.domain.gene import SessionGene

    gene1 = SessionGene("CS101", "theory", "INST1", ["GRP1"], "ROOM1", 0, 3)
    gene2 = SessionGene("CS102", "practical", "INST2", ["GRP2"], "ROOM2", 3, 2)

    # Record violations
    for _ in range(10):
        heatmap.record_violation(gene1, "availability")

    for _ in range(5):
        heatmap.record_violation(gene1, "overlap")

    for _ in range(3):
        heatmap.record_violation(gene2, "room_type")

    # Print summary
    heatmap.print_summary(console)

    # Test persistence
    test_file = "test_heatmap.json"
    heatmap.save_to_file(test_file)
    console.print(f"\n[green][!ok] Saved to {test_file}[/green]")

    # Load and verify
    loaded = ViolationHeatmap.load_from_file(test_file)
    console.print(f"[green][!ok] Loaded {len(loaded.violations)} gene records[/green]")

    Path(test_file).unlink()
    console.print("[dim]Cleaned up test file[/dim]\n")
