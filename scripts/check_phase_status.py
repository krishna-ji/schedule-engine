#!/usr/bin/env python3
"""
Check Phase 1 & 2 implementation status.

Shows what's complete, what's working, and what requires hardware/dependencies.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def check_code_implementation():
    """Check code implementation status."""
    console.print("\n[bold cyan]Code Implementation Status[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim")
    
    # Configuration
    from src.config.loader import _check_hierarchical_structure
    if _check_hierarchical_structure():
        table.add_row("Config System", "[green]✓ Done[/green]", "Hierarchical (common/ga/rl)")
    else:
        table.add_row("Config System", "[red]✗ Missing[/red]", "Hierarchical structure not found")
    
    # Heuristics
    try:
        import src.heuristics.construction
        import src.heuristics.perturbation
        import src.heuristics.improvement
        import src.heuristics.diversity
        import src.heuristics.meta
        from src.heuristics.registry import get_registry
        
        registry = get_registry()
        table.add_row("Heuristic Toolbox", "[green]✓ Done[/green]", f"{len(registry)} operators")
    except Exception as e:
        table.add_row("Heuristic Toolbox", "[red]✗ Error[/red]", str(e))
    
    # GA Core
    try:
        from src.core.ga_scheduler import GAScheduler
        from src.ga.population import generate_course_group_aware_population
        table.add_row("GA Core", "[green]✓ Done[/green]", "Scheduler + operators")
    except Exception as e:
        table.add_row("GA Core", "[red]✗ Error[/red]", str(e))
    
    # RL Code (check without importing heavy dependencies)
    rl_files = [
        "src/rl/gym_env/schedule_env.py",
        "src/rl/gym_env/state_encoder.py",
        "src/rl/gym_env/action_mapper.py",
        "src/rl/training/trainer.py",
        "src/rl/training/train_script.py",
        "src/rl/deployment/model_loader.py",
        "scripts/generate_validation_set.py",
        "scripts/select_best_checkpoint.py",
        "scripts/promote_model_to_prod.py",
    ]
    
    rl_exist = all((project_root / f).exists() for f in rl_files)
    if rl_exist:
        table.add_row("RL Code", "[green]✓ Done[/green]", "All modules present")
    else:
        missing = [f for f in rl_files if not (project_root / f).exists()]
        table.add_row("RL Code", "[yellow]⚠ Partial[/yellow]", f"{len(missing)} files missing")
    
    console.print(table)


def check_dependencies():
    """Check dependency status."""
    console.print("\n[bold cyan]Dependencies Status[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Package", style="cyan", width=25)
    table.add_column("Status", width=15)
    table.add_column("Required For", style="dim")
    
    deps = {
        'deap': ('GA core', True),
        'pydantic': ('Configuration', True),
        'rich': ('UI/logging', True),
        'torch': ('RL training', False),
        'stable_baselines3': ('RL agents', False),
        'gymnasium': ('RL environment', False),
        'tensorboard': ('RL monitoring', False),
    }
    
    for package, (purpose, critical) in deps.items():
        try:
            __import__(package)
            table.add_row(package, "[green]✓ Installed[/green]", purpose)
        except (ImportError, OSError, Exception) as e:
            # Catch all errors (ImportError, OSError for broken installs, etc.)
            if critical:
                table.add_row(package, "[red]✗ Missing[/red]", purpose)
            else:
                table.add_row(package, "[yellow]⚠ Not available[/yellow]", purpose)
    
    console.print(table)


def check_execution_readiness():
    """Check what can be executed now vs what needs hardware."""
    console.print("\n[bold cyan]Execution Readiness[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task", style="cyan", width=35)
    table.add_column("Status", width=15)
    table.add_column("Blockers", style="dim")
    
    # Check disk space
    import shutil
    disk = shutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    disk_ok = free_gb > 20
    
    # Check GPU
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except (ImportError, OSError, Exception):
        pass  # torch not available or broken
    
    # GA Execution
    try:
        from src.config import init_config
        init_config()
        table.add_row("GA Execution (test)", "[green]✓ Ready[/green]", "Can run now")
        table.add_row("GA Execution (prod)", "[green]✓ Ready[/green]", "Can run now (24-48h)")
    except Exception as e:
        table.add_row("GA Execution", "[red]✗ Blocked[/red]", str(e))
    
    # RL Training
    if gpu_available and disk_ok:
        table.add_row("RL Training", "[green]✓ Ready[/green]", "GPU + disk OK")
    else:
        blockers = []
        if not gpu_available:
            blockers.append("No GPU/CUDA")
        if not disk_ok:
            blockers.append(f"Low disk ({free_gb:.1f}GB free, need 20GB)")
        table.add_row("RL Training", "[yellow]⚠ Blocked[/yellow]", ", ".join(blockers))
    
    # Validation Set Generation
    try:
        from src.encoder import load_scheduling_data
        table.add_row("Validation Set Gen", "[green]✓ Ready[/green]", "Can run now")
    except Exception as e:
        table.add_row("Validation Set Gen", "[red]✗ Blocked[/red]", str(e))
    
    console.print(table)


def show_next_steps():
    """Show next steps based on current status."""
    console.print("\n[bold cyan]Next Steps[/bold cyan]\n")
    
    # Check if RL dependencies are available
    rl_available = True
    try:
        import torch
        import stable_baselines3
        import gymnasium
    except (ImportError, OSError, Exception):
        rl_available = False  # Dependencies not available or broken
    
    if rl_available:
        console.print("[green]✓ All dependencies available - Ready for Phase 2 execution[/green]\n")
        console.print("[bold]Phase 2 Execution Steps:[/bold]")
        console.print("1. [cyan]python scripts/generate_validation_set.py --stage all --num-problems 30[/cyan]")
        console.print("2. [cyan]python src/rl/training/train_script.py --timesteps 100000 --agent ppo[/cyan]")
        console.print("3. [cyan]python scripts/select_best_checkpoint.py --metric mean_reward --promote[/cyan]")
        console.print("4. [cyan]python scripts/promote_model_to_prod.py --checkpoint-id <id>[/cyan]")
        console.print("5. Edit [cyan]configs/rl/prod.yaml[/cyan] → set [cyan]rl.enabled: true[/cyan]")
        console.print("6. [cyan]uv run prod[/cyan] (with RL enabled)")
    else:
        console.print("[yellow]⚠ RL dependencies not available[/yellow]\n")
        console.print("[bold]What you can do now:[/bold]")
        console.print("1. [cyan]python scripts/validate_phase_integration.py[/cyan] - Validate integration")
        console.print("2. [cyan]python test/test_phase_integration.py[/cyan] - Run integration tests")
        console.print("3. [cyan]uv run test[/cyan] - Run GA smoke test (5-10 min)")
        console.print("4. [cyan]uv run prod[/cyan] - Run full GA (24-48h, RL disabled)")
        
        console.print("\n[bold]To enable Phase 2 execution:[/bold]")
        console.print("Install RL dependencies:")
        console.print("[cyan]pip install torch stable-baselines3 gymnasium tensorboard[/cyan]")
        console.print("Or:")
        console.print("[cyan]pip install -r requirements.txt[/cyan] (if available)")


def main():
    """Main function."""
    console.print(Panel.fit(
        "[bold cyan]Phase 1 & 2 Status Check[/bold cyan]\n"
        "Checking implementation, dependencies, and execution readiness",
        border_style="cyan"
    ))
    
    check_code_implementation()
    check_dependencies()
    check_execution_readiness()
    show_next_steps()
    
    console.print("\n" + "="*60)
    console.print("[bold green]Status check complete[/bold green]")
    console.print("="*60 + "\n")


if __name__ == "__main__":
    main()
