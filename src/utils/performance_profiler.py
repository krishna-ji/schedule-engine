"""Detailed performance profiling utilities for GA evolution."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import TypedDict

import psutil
from rich.console import Console
from rich.table import Table


class PhaseStats(TypedDict, total=False):
    """Aggregate metrics tracked per phase across generations."""

    total: float
    count: int
    min: float
    max: float
    avg: float


@dataclass
class PhaseProfile:
    """Profile data for a single phase of execution."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    thread_id: int | None = None
    worker_id: int | None = None
    items_processed: int = 0

    def format_duration(self) -> str:
        """Format duration in appropriate units."""
        if self.duration < 0.001:
            return f"{self.duration * 1000000:.0f}µs"
        elif self.duration < 1.0:
            return f"{self.duration * 1000:.0f}ms"
        else:
            return f"{self.duration:.2f}s"


@dataclass
class GenerationProfile:
    """Complete profile for one generation."""

    generation: int
    phases: dict[str, PhaseProfile] = field(default_factory=dict)
    total_duration: float = 0.0
    cpu_usage_peak: float = 0.0
    memory_usage_mb: float = 0.0

    def add_phase(self, phase: PhaseProfile) -> None:
        """Add a phase profile."""
        self.phases[phase.name] = phase
        self.total_duration += phase.duration

    def get_summary(self) -> str:
        """Get one-line summary."""
        parts = []
        for name, phase in self.phases.items():
            parts.append(f"{name}={phase.format_duration()}")
        return " | ".join(parts)


class PerformanceProfiler:
    """
    Tracks and displays detailed performance metrics for GA evolution.

    Features:
    - Phase-level timing (selection, crossover, mutation, evaluation, repair)
    - CPU and memory usage tracking
    - Multi-threading awareness (shows which core is doing what)
    - Real-time display with Rich console
    - Micro-breakdown per generation
    """

    def __init__(
        self,
        enabled: bool = True,
        console: Console | None = None,
        verbose: bool = False,
    ):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is enabled
            console: Rich console for output (optional)
            verbose: Show per-generation micro-breakdown (can interfere with progress bars)
        """
        self.enabled = enabled
        self.verbose = verbose
        self.console = console or Console()
        self.current_generation: int | None = None
        self.current_phase: PhaseProfile | None = None
        self.generation_profiles: list[GenerationProfile] = []
        self.process = psutil.Process(os.getpid())

        # Thread-local storage for worker tracking
        self._thread_local = threading.local()

    def start_generation(self, gen: int) -> None:
        """Start profiling a new generation."""
        if not self.enabled:
            return

        self.current_generation = gen
        self.generation_profiles.append(GenerationProfile(generation=gen))

    def end_generation(self) -> None:
        """End profiling current generation and display results."""
        if not self.enabled or self.current_generation is None:
            return

        profile = self.generation_profiles[-1]

        # Display micro-breakdown only if verbose mode enabled
        if self.verbose:
            self._display_generation_profile(profile)

        self.current_generation = None

    def start_phase(
        self, name: str, items_to_process: int = 0, worker_id: int | None = None
    ) -> None:
        """
        Start profiling a phase.

        Args:
            name: Phase name (e.g., "selection", "crossover", "evaluation")
            items_to_process: Number of items to process (for rate calculation)
            worker_id: Worker/core ID if using multiprocessing
        """
        if not self.enabled or self.current_generation is None:
            return

        self.current_phase = PhaseProfile(
            name=name,
            start_time=time.perf_counter(),
            thread_id=threading.get_ident(),
            worker_id=worker_id,
            items_processed=items_to_process,
        )

        # Capture initial CPU and memory
        try:
            self.current_phase.cpu_percent = self.process.cpu_percent(interval=None)
            self.current_phase.memory_mb = self.process.memory_info().rss / 1024 / 1024
        except:
            pass  # Ignore if process monitoring fails

    def end_phase(self) -> None:
        """End profiling current phase."""
        if not self.enabled or self.current_phase is None:
            return

        self.current_phase.end_time = time.perf_counter()
        self.current_phase.duration = (
            self.current_phase.end_time - self.current_phase.start_time
        )

        # Add to current generation profile
        if self.generation_profiles:
            self.generation_profiles[-1].add_phase(self.current_phase)

            # Update generation peak metrics
            profile = self.generation_profiles[-1]
            profile.cpu_usage_peak = max(
                profile.cpu_usage_peak, self.current_phase.cpu_percent
            )
            profile.memory_usage_mb = self.current_phase.memory_mb

        self.current_phase = None

    def _display_generation_profile(self, profile: GenerationProfile) -> None:
        """Display detailed profile for a generation."""
        if not self.console:
            return

        # Build performance breakdown string
        parts = []

        # Sort phases by duration (longest first)
        sorted_phases = sorted(
            profile.phases.values(), key=lambda p: p.duration, reverse=True
        )

        for phase in sorted_phases:
            # Format with color based on duration
            duration_str = phase.format_duration()

            # Add rate if items were processed
            if phase.items_processed > 0 and phase.duration > 0:
                rate = phase.items_processed / phase.duration
                if rate < 1:
                    rate_str = f"{1/rate:.1f}s/item"
                else:
                    rate_str = f"{rate:.0f}items/s"
                parts.append(f"{phase.name}={duration_str}({rate_str})")
            else:
                parts.append(f"{phase.name}={duration_str}")

        # Display as single line under main progress
        breakdown = " | ".join(parts)

        # Color code based on total time
        if profile.total_duration > 100:
            color = "red"
        elif profile.total_duration > 60:
            color = "yellow"
        else:
            color = "dim"

        self.console.print(f"[{color}]      {breakdown}[/{color}]")

    def get_statistics(self) -> dict[str, PhaseStats]:
        """
        Get aggregate statistics across all generations.

        Returns:
            Dictionary with timing statistics for each phase
        """
        stats: dict[str, PhaseStats] = {}

        for gen_profile in self.generation_profiles:
            for phase_name, phase in gen_profile.phases.items():
                phase_stats = stats.setdefault(
                    phase_name,
                    {
                        "total": 0.0,
                        "count": 0,
                        "min": float("inf"),
                        "max": 0.0,
                    },
                )
                phase_stats["total"] += phase.duration
                phase_stats["count"] += 1
                phase_stats["min"] = min(phase_stats["min"], phase.duration)
                phase_stats["max"] = max(phase_stats["max"], phase.duration)

        # Calculate averages
        for phase_stats in stats.values():
            if phase_stats["count"] > 0:
                phase_stats["avg"] = phase_stats["total"] / phase_stats["count"]
            else:
                phase_stats["avg"] = 0.0

        return stats

    def print_summary_table(self) -> None:
        """Print a summary table of all profiled phases."""
        if not self.enabled or not self.generation_profiles:
            return

        stats = self.get_statistics()

        table = Table(title="Performance Profile Summary")
        table.add_column("Phase", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Total Time", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Min Time", justify="right")
        table.add_column("Max Time", justify="right")
        table.add_column("% of Total", justify="right")

        # Calculate total time across all phases
        total_time = sum(s["total"] for s in stats.values())

        # Sort by total time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        for phase_name, phase_stats in sorted_stats:
            count = phase_stats["count"]
            total = phase_stats["total"]
            avg = phase_stats["avg"]
            min_time = phase_stats["min"]
            max_time = phase_stats["max"]
            percentage = (total / total_time * 100) if total_time > 0 else 0

            # Format times
            def fmt(t: float) -> str:
                if t < 0.001:
                    return f"{t * 1000000:.0f}µs"
                elif t < 1.0:
                    return f"{t * 1000:.0f}ms"
                else:
                    return f"{t:.2f}s"

            table.add_row(
                phase_name,
                str(count),
                fmt(total),
                fmt(avg),
                fmt(min_time),
                fmt(max_time),
                f"{percentage:.1f}%",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()


# Global profiler instance (can be accessed from anywhere)
_global_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled=False)
    return _global_profiler


def init_profiler(
    enabled: bool = True, console: Console | None = None, verbose: bool = False
) -> PerformanceProfiler:
    """
    Initialize global profiler.

    Args:
        enabled: Whether profiling is enabled
        console: Rich console for output
        verbose: Show per-generation micro-breakdown (disable during training to avoid progress bar interference)
    """
    global _global_profiler
    _global_profiler = PerformanceProfiler(
        enabled=enabled, console=console, verbose=verbose
    )
    return _global_profiler


def cleanup_profiler() -> None:
    """Cleanup and print summary."""
    global _global_profiler
    if _global_profiler and _global_profiler.enabled:
        _global_profiler.print_summary_table()
    _global_profiler = None
