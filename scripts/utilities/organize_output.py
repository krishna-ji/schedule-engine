"""
Output Directory Organizer

Reorganizes flat evaluation_* directories into structured hierarchy:
  output/{category}/{mode}/evaluation_*/

Categories:
- baseline: Pure NSGA-II experiments
- nsga: NSGA-II variants (repairs, heuristics, full)
- rl: RL-guided experiments
- hybrid: Hybrid approaches (round-robin, archive)
- other: Unknown/unclassified runs
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import sys

def detect_category(dir_name: str, dir_path: Path) -> tuple:
    """
    Detect category and mode from directory name or contents.
    
    Returns:
        (category, mode_name) tuple
    """
    # Check manifest if exists
    manifest_path = dir_path / "experiment_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data = json.load(f)
                if data.get("runtime_mode"):
                    mode_value = data["runtime_mode"]
                    mode_number = mode_value.split("-")[0]
                    mode_name = mode_value.split("-", 1)[1]
                    
                    category_map = {
                        "1": "baseline", "2": "nsga", "3": "nsga", "4": "nsga",
                        "5": "rl", "6": "hybrid", "7": "rl", "8": "hybrid",
                        "9": "rl", "10": "rl"
                    }
                    return category_map.get(mode_number, "other"), mode_name
        except Exception as e:
            print(f"  [!] Error reading manifest: {e}")
    
    # Infer from config files in directory
    config_path = dir_path / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                content = f.read().lower()
                
                # Check for RL indicators
                if "rl:" in content and "enabled: true" in content:
                    return "rl", "rl-guided"
                
                # Check for round-robin
                if "round-robin" in content or "roundrobin" in content:
                    return "hybrid", "round-robin"
                
                # Check for heuristics
                if "heuristics:" in content and "enabled: true" in content:
                    return "nsga", "nsga-heuristics"
                
                # Check for repairs
                if "repair:" in content and "enabled: true" in content:
                    return "nsga", "nsga-repairs"
                
                # Baseline if minimal config
                if "repair:" in content and "enabled: false" in content:
                    return "baseline", "pure-nsga"
        except Exception as e:
            print(f"  [!] Error reading config: {e}")
    
    # Check directory name for hints
    name_lower = dir_name.lower()
    if "baseline" in name_lower or "pure" in name_lower:
        return "baseline", "pure-nsga"
    if "rl" in name_lower:
        return "rl", "rl-guided"
    if "roundrobin" in name_lower or "round-robin" in name_lower:
        return "hybrid", "round-robin"
    if "heuristic" in name_lower:
        return "nsga", "nsga-heuristics"
    
    # Default to other/unknown
    return "other", "unknown"


def organize_output_directory(dry_run: bool = True):
    """
    Reorganize output directory structure.
    
    Args:
        dry_run: If True, only print what would be done without moving files
    """
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("[!] output/ directory not found")
        return
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Analyzing output directory...\n")
    
    # Find all evaluation_* directories at root level
    flat_dirs = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("evaluation_")
    ]
    
    if not flat_dirs:
        print("[✓] No flat evaluation_* directories found - structure already organized!")
        return
    
    print(f"Found {len(flat_dirs)} flat directories to organize:\n")
    
    moves = []
    
    for old_path in flat_dirs:
        dir_name = old_path.name
        
        # Detect category and mode
        category, mode_name = detect_category(dir_name, old_path)
        
        # Build new path
        new_path = output_dir / category / mode_name / dir_name
        
        # Check if destination already exists
        if new_path.exists():
            print(f"[!] SKIP: {dir_name}")
            print(f"    → {new_path} (already exists)")
            print()
            continue
        
        moves.append((old_path, new_path, category, mode_name))
        
        print(f"[→] {dir_name}")
        print(f"    Category: {category}")
        print(f"    Mode: {mode_name}")
        print(f"    New path: {new_path}")
        print()
    
    if not moves:
        print("[✓] All directories already organized or skipped!")
        return
    
    print(f"\n{'Would move' if dry_run else 'Moving'} {len(moves)} directories...\n")
    
    if dry_run:
        print("[DRY RUN] Run with --execute flag to actually move files")
        return
    
    # Execute moves
    moved_count = 0
    error_count = 0
    
    for old_path, new_path, category, mode_name in moves:
        try:
            # Create parent directories
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move directory
            shutil.move(str(old_path), str(new_path))
            moved_count += 1
            print(f"[✓] Moved: {old_path.name} → {category}/{mode_name}/")
        except Exception as e:
            error_count += 1
            print(f"[✗] Error moving {old_path.name}: {e}")
    
    print(f"\n[✓] Complete!")
    print(f"    Moved: {moved_count}")
    print(f"    Errors: {error_count}")
    print(f"\nOrganized structure:")
    print("  output/")
    print("    baseline/")
    print("      pure-nsga/")
    print("        evaluation_*/")
    print("    nsga/")
    print("      nsga-repairs/")
    print("        evaluation_*/")
    print("      nsga-heuristics/")
    print("        evaluation_*/")
    print("    rl/")
    print("      rl-guided/")
    print("        evaluation_*/")
    print("    hybrid/")
    print("      round-robin/")
    print("        evaluation_*/")
    print("    other/")
    print("      unknown/")
    print("        evaluation_*/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize output directory structure")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default: dry run)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Output Directory Organizer")
    print("=" * 70)
    
    organize_output_directory(dry_run=not args.execute)
