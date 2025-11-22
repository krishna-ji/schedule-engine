"""Quick test of performance profiler"""

import time
from src.utils.performance_profiler import init_profiler, get_profiler
from src.utils.console_service import get_console

console = get_console()

# Initialize profiler
profiler = init_profiler(enabled=True, console=console)

console.print("\n[bold cyan]Testing Performance Profiler[/bold cyan]\n")

# Simulate 3 generations
for gen in range(3):
    profiler.start_generation(gen)

    # Simulate selection
    profiler.start_phase("selection", items_to_process=100)
    time.sleep(0.05)  # 50ms
    profiler.end_phase()

    # Simulate crossover
    profiler.start_phase("crossover", items_to_process=50)
    time.sleep(0.1)  # 100ms
    profiler.end_phase()

    # Simulate mutation
    profiler.start_phase("mutation", items_to_process=100)
    time.sleep(0.08)  # 80ms
    profiler.end_phase()

    # Simulate evaluation
    profiler.start_phase("evaluation", items_to_process=80)
    time.sleep(0.2)  # 200ms (longest phase)
    profiler.end_phase()

    # Simulate optional repair
    if gen % 2 == 0:
        profiler.start_phase("repair_memetic", items_to_process=10)
        time.sleep(0.15)  # 150ms
        profiler.end_phase()

    # Display generation breakdown
    console.print(f"[green][!ok] gen {gen+1}/3 : hc=3500, sc=4800.20, t=65.2s[/green]")
    profiler.end_generation()
    console.print()

# Show summary
from src.utils.performance_profiler import cleanup_profiler

cleanup_profiler()

console.print("\n[bold green] Profiler test complete![/bold green]\n")
