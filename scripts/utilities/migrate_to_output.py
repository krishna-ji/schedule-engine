#!/usr/bin/env python3
"""
Migration Script: Consolidate logs and models under output/

Moves existing logs/ and models/ directories into output/ structure
for better organization and easier cleanup.
"""

import shutil
from pathlib import Path
from rich.console import Console

console = Console()


def migrate_to_output_structure():
    """Migrate logs and models to consolidated output/ structure."""
    
    console.print("[cyan] Migrating to consolidated output/ structure...[/cyan]\n")
    
    # Create new directory structure
    output_dir = Path("output")
    new_logs_dir = output_dir / "logs"
    new_models_dir = output_dir / "models"
    new_analysis_dir = output_dir / "analysis"
    
    # Create directories
    new_logs_dir.mkdir(parents=True, exist_ok=True)
    new_models_dir.mkdir(parents=True, exist_ok=True)
    new_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"✓ Created: {new_logs_dir}")
    console.print(f"✓ Created: {new_models_dir}")
    console.print(f"✓ Created: {new_analysis_dir}")
    
    # Migrate logs/
    old_logs_dir = Path("logs")
    if old_logs_dir.exists():
        console.print(f"\n[yellow] Moving logs/ → output/logs/[/yellow]")
        
        for item in old_logs_dir.iterdir():
            dest = new_logs_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                console.print(f"  ✓ Copied: {item.name}/")
            else:
                shutil.copy2(item, dest)
                console.print(f"  ✓ Copied: {item.name}")
        
        # Remove old logs directory
        shutil.rmtree(old_logs_dir)
        console.print(f"  ️  Removed old: {old_logs_dir}")
    else:
        console.print("  ️  No logs/ directory found")
    
    # Migrate models/
    old_models_dir = Path("models")
    if old_models_dir.exists():
        console.print(f"\n[yellow] Moving models/ → output/models/[/yellow]")
        
        for item in old_models_dir.iterdir():
            dest = new_models_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                console.print(f"  ✓ Copied: {item.name}/")
            else:
                shutil.copy2(item, dest)
                console.print(f"  ✓ Copied: {item.name}")
        
        # Remove old models directory
        shutil.rmtree(old_models_dir)
        console.print(f"  ️  Removed old: {old_models_dir}")
    else:
        console.print("  ️  No models/ directory found")
    
    # Move old experiment folders to experiments/ subdirectory
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    console.print(f"\n[yellow] Organizing experiment folders...[/yellow]")
    
    moved_count = 0
    for item in output_dir.iterdir():
        # Move old evaluation_* folders to experiments/
        if item.is_dir() and item.name.startswith("evaluation_"):
            dest = experiments_dir / item.name
            shutil.move(str(item), str(dest))
            console.print(f"  ✓ Moved: {item.name} → experiments/")
            moved_count += 1
    
    if moved_count == 0:
        console.print("  ️  No old evaluation folders to move")
    
    console.print(f"\n[green] Migration complete![/green]")
    console.print("\n[bold green]New consolidated structure enforced:[/bold green]")
    console.print(" output/")
    console.print("  ├── experiments/    # GA experiment results (ORGANIZED)")
    console.print("  ├── logs/          # All logs (MOVED from logs/)")
    console.print("  ├── models/        # All trained models (MOVED from models/)")
    console.print("  └── analysis/      # Analysis results (analyze-results output)")
    console.print("\n[yellow]️  Analysis scripts now REQUIRE this structure[/yellow]")
    console.print("[yellow]   Old scattered structure is no longer supported[/yellow]")
    

def show_current_structure():
    """Show current directory structure."""
    console.print("[cyan] Current structure:[/cyan]\n")
    
    def show_tree(path: Path, prefix="", max_depth=2, current_depth=0):
        if current_depth > max_depth:
            return
        
        items = list(path.iterdir()) if path.exists() else []
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Show directories first
        for i, item in enumerate(dirs[:5]):  # Limit to first 5 items
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            connector = "└── " if is_last_dir else "├── "
            console.print(f"{prefix}{connector}{item.name}/")
            
            # Recurse into subdirectories
            if current_depth < max_depth:
                extension = "    " if is_last_dir else "│   "
                show_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        # Show some files
        for i, item in enumerate(files[:3]):  # Limit to first 3 files
            is_last = i == len(files) - 1
            connector = "└── " if is_last else "├── "
            console.print(f"{prefix}{connector}{item.name}")
        
        if len(files) > 3:
            console.print(f"{prefix}    ... ({len(files) - 3} more files)")
        if len(dirs) > 5:
            console.print(f"{prefix}    ... ({len(dirs) - 5} more directories)")
    
    # Show relevant directories
    for dir_name in ["output", "logs", "models"]:
        path = Path(dir_name)
        if path.exists():
            console.print(f" {dir_name}/")
            show_tree(path, "  ")
            console.print()


def main():
    """Main migration function for CLI entry point."""
    console.print("[bold cyan]Schedule Engine - Directory Migration[/bold cyan]\n")
    
    # Show current structure
    show_current_structure()
    
    # Ask for confirmation
    from rich.prompt import Confirm
    
    if Confirm.ask("\n[yellow]Proceed with migration?[/yellow]"):
        migrate_to_output_structure()
        console.print(f"\n[green] All experimental artifacts now consolidated under output/![/green]")
        console.print("[dim]Use 'uv run clean' to clear everything at once.[/dim]")
    else:
        console.print("[yellow]Migration cancelled.[/yellow]")


if __name__ == "__main__":
    main()