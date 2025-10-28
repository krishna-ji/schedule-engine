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
    print(f"\n🚀 Running with {env} environment...\n")
    cmd = ["uv", "run", "python", "main.py", "--env", env]

    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n👋 Execution interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except FileNotFoundError:
        print("\n❌ Error: 'uv' not found. Please install uv first.")
        print("   Install: https://github.com/astral-sh/uv")
        sys.exit(1)


def show_menu():
    """Show interactive environment selection menu."""
    print("\n" + "=" * 50)
    print("  Schedule Engine - Environment Selection")
    print("=" * 50)
    print("\n  1) dev   - Development (100 generations)")
    print("  2) test  - Quick test (10 generations)")
    print("  3) prod  - Production (200+ generations)")
    print("\n" + "=" * 50)

    while True:
        try:
            choice = input("\nEnter choice (1-3) or 'q' to quit: ").strip().lower()

            if choice == "q":
                print("\n👋 Goodbye!")
                sys.exit(0)
            elif choice == "1":
                return "dev"
            elif choice == "2":
                return "test"
            elif choice == "3":
                return "prod"
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Main entry point."""
    # If environment provided as argument, use it directly
    if len(sys.argv) > 1:
        env = sys.argv[1].lower()
        if env in ["dev", "test", "prod"]:
            run_schedule_engine(env)
        else:
            print(f"❌ Error: Invalid environment '{env}'")
            print("   Valid options: dev, test, prod")
            sys.exit(1)
    else:
        # Show interactive menu
        env = show_menu()
        run_schedule_engine(env)


if __name__ == "__main__":
    main()
