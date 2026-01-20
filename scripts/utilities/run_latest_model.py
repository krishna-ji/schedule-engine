"""Find and run the latest trained RL model."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

console = Console()


def find_latest_model(models_dir: Path = Path("models/rl_agents")) -> Path | None:
    """
    Find the most recently trained RL model.

    Args:
        models_dir: Directory containing trained models

    Returns:
        Path to latest .zip model file, or None if not found
    """
    model_files = list(models_dir.glob("*.zip"))

    if not model_files:
        return None

    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def main() -> int:
    """Find latest model and run RL inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RL inference with the latest trained model"
    )
    parser.add_argument(
        "--profile",
        choices=["test", "prod"],
        default="test",
        help="Profile to use (default: test)",
    )
    parser.add_argument("--test", action="store_const", const="test", dest="profile")
    parser.add_argument("--prod", action="store_const", const="prod", dest="profile")
    parser.add_argument("--name", help="Custom name for this run")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available models without running",
    )

    args = parser.parse_args()

    models_dir = project_root / "models" / "rl_agents"

    if not models_dir.exists():
        console.print(f"[red]Error:[/red] Models directory not found: {models_dir}")
        console.print("\n[yellow]Train a model first:[/yellow]")
        console.print("  uv run train-rl --prod")
        return 1

    # Find all models
    model_files = sorted(
        models_dir.glob("*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not model_files:
        console.print(f"[red]Error:[/red] No trained models found in {models_dir}")
        console.print("\n[yellow]Train a model first:[/yellow]")
        console.print("  uv run train-rl --prod")
        return 1

    # Display available models
    console.print("\n[cyan]Available Models:[/cyan]")
    for i, model in enumerate(model_files, 1):
        metadata_file = model.with_suffix(".json")
        size_mb = model.stat().st_size / (1024 * 1024)
        modified = model.stat().st_mtime

        import datetime

        mod_time = datetime.datetime.fromtimestamp(modified)
        time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")

        marker = "→ [bold green]LATEST[/bold green]" if i == 1 else "  "
        console.print(f"{marker} {i}. {model.name} ({size_mb:.1f} MB) - {time_str}")

        # Show metadata if available
        if metadata_file.exists():
            import json

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    agent_type = metadata.get("agent_type", "unknown")
                    timesteps = metadata.get(
                        "total_timesteps_trained", metadata.get("timesteps", "?")
                    )
                    console.print(
                        f"      [dim]{agent_type.upper()} | {timesteps:,} steps[/dim]"
                    )
            except Exception:
                pass

    latest_model = model_files[0]

    if args.list_only:
        console.print(f"\n[green]Latest model:[/green] {latest_model.name}")
        return 0

    # Run RL inference with latest model
    console.print(f"\n[bold green]Using latest model:[/bold green] {latest_model.name}")
    console.print(
        f"[cyan]Profile:[/cyan] {args.profile} ({'smoke test' if args.profile == 'test' else 'production'})"
    )

    console.print("\n[yellow]Starting RL-guided GA run...[/yellow]\n")

    # Use the launcher's RL experiment runner directly
    from scripts.launcher import run_experiment_universal

    try:
        return run_experiment_universal(
            experiment_id="rl-guided",
            profile=args.profile,
            name=args.name,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Run interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
