#!/usr/bin/env python3
"""
Quick run script using UV (no activation needed!)
Cross-platform: Works on both Windows and Linux

Usage:
    python x              # Interactive menu
    python x dev          # Run directly with dev
    python x test         # Run directly with test
    python x prod         # Run directly with prod

On Linux/Mac, you can also make it executable:
    chmod +x x
    ./x
"""
import subprocess
import sys


def run_schedule_engine(env: str):
    """Run the schedule engine with the specified environment."""
    print(f"\n[yes!] Running with {env} environment...\n")
    cmd = ["uv", "run", "python", "main.py", "--env", env]

    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\n[err!] Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n[!bye] Execution interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except FileNotFoundError:
        print("\n[err!] Error: 'uv' not found. Please install uv first.")
        print("   Install: https://github.com/astral-sh/uv")
        sys.exit(1)


def show_menu():
    """Show interactive environment selection menu."""
    print("\n" + "=" * 50)
    print("  Schedule Engine - Environment Selection")
    print("=" * 50)
    print("\n  d) dev   - Development (100 generations)")
    print("  t) test  - Quick test (10 generations)")
    print("  p) prod  - Production (200+ generations)")
    print("\n" + "=" * 50)

    while True:
        try:
            choice = input("\nEnter choice (d/t/p) or 'q' to quit: ").strip().lower()

            if choice == "q":
                print("\n[!bye] Goodbye!")
                sys.exit(0)
            elif choice == "d":
                return "dev"
            elif choice == "t":
                return "test"
            elif choice == "p":
                return "prod"
            else:
                print("[err!] Invalid choice. Please enter d, t, p, or 'q'.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n[!bye] Goodbye!")
            sys.exit(0)


def main():
    """Main entry point."""
    # If environment provided as argument, use it directly
    if len(sys.argv) > 1:
        env = sys.argv[1].lower()
        if env in ["dev", "test", "prod"]:
            run_schedule_engine(env)
        else:
            print(f"[err!] Error: Invalid environment '{env}'")
            print("   Valid options: dev, test, prod")
            sys.exit(1)
    else:
        # Show interactive menu
        env = show_menu()
        run_schedule_engine(env)


if __name__ == "__main__":
    main()
