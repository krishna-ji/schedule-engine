#!/usr/bin/env python3
"""
Interactive git commit squashing utility.

Cross-platform Python replacement for commit_squash.ps1
"""

import subprocess
import sys


def run_git_command(cmd: list[str], check: bool = True) -> str | None:
    """Run git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + cmd, capture_output=True, text=True, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e.stderr}")
        return None


def get_current_branch() -> str:
    """Get current git branch name."""
    branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    return branch if branch else "unknown"


def get_commits(n: int) -> str | None:
    """Get last N commits."""
    return run_git_command(["log", "--oneline", f"-{n}"])


def squash_commits(n: int, message: str) -> bool:
    """Squash last N commits into one."""
    try:
        # Soft reset to N commits ago
        run_git_command(["reset", "--soft", f"HEAD~{n}"])

        # Create new commit with all changes
        run_git_command(["commit", "-m", message])

        return True
    except:
        # On error, try to restore original state
        print("\nError occurred. Attempting rollback...")
        run_git_command(["reset", "--hard", "ORIG_HEAD"], check=False)
        return False


def main():
    """Interactive commit squashing workflow."""

    print("GIT COMMIT SQUASHER")

    # Show current branch
    branch = get_current_branch()
    print(f"\nCurrent branch: {branch}")

    # Get number of commits to squash
    while True:
        try:
            n_input = input("\nHow many commits to squash? ")
            n = int(n_input)
            if n < 2:
                print("Must be 2 or more commits")
                continue
            break
        except ValueError:
            print("Invalid number")
            continue
        except KeyboardInterrupt:
            print("\n\nAborted")
            sys.exit(0)

    # Show commits that will be squashed
    print(f"\n{'=' * 60}")
    print(f"This will squash the last {n} commits into one:")

    commits = get_commits(n)
    if commits:
        print(commits)
    else:
        print("Error: Could not retrieve commits")
        sys.exit(1)

    # Confirm action
    print("\n" + "=" * 60)
    print("️  WARNING: This will rewrite git history!")

    try:
        confirm = input("\nContinue? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        print("\n\nAborted")
        sys.exit(0)

    if confirm != "y":
        print("Aborted")
        sys.exit(0)

    # Get new commit message
    try:
        message = input("\nEnter new commit message: ").strip()
    except KeyboardInterrupt:
        print("\n\nAborted")
        sys.exit(0)

    if not message:
        print("Error: Empty commit message")
        sys.exit(1)

    # Perform squash
    print(f"\nSquashing {n} commits...")
    if squash_commits(n, message):
        print("\n✓ Successfully squashed commits!")
        print("\nNew commit:")
        new_commit = get_commits(1)
        if new_commit:
            print(new_commit)

        print("\n" + "=" * 60)
        print("To push to remote (if previously pushed):")
        print("  git push --force-with-lease")

    else:
        print("\n✗ Failed to squash commits")
        sys.exit(1)


if __name__ == "__main__":
    main()
