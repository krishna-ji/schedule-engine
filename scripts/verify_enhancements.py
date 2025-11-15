#!/usr/bin/env python3
"""
Verification script for code quality enhancements.

Checks that all implemented features are working correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_files_exist():
    """Verify all new files were created."""
    print("🔍 Checking file creation...")

    required_files = [
        "src/__init__.py",
        "src/utils/console_helpers.py",
        "src/utils/logging_config.py",
        ".editorconfig",
        "CONTRIBUTING.md",
        "test/unit/__init__.py",
        "test/unit/conftest.py",
        "test/unit/test_config_loader.py",
        "test/unit/test_encoder.py",
        "test/unit/test_constraints.py",
        "test/unit/test_operators.py",
        "test/unit/test_utils.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING!")
            all_exist = False

    return all_exist


def check_imports():
    """Verify new modules can be imported."""
    print("\n🔍 Checking module imports...")

    imports_to_test = [
        ("src", "Package initialization"),
        ("src.utils.console_helpers", "Console helpers"),
        ("src.utils.logging_config", "Logging config"),
    ]

    all_imported = True
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description} ({module_name})")
        except ImportError as e:
            print(f"  ❌ {description} ({module_name}) - {e}")
            all_imported = False

    return all_imported


def check_console_helpers():
    """Verify console helpers work."""
    print("\n🔍 Testing console helpers...")

    try:
        from src.utils.console_helpers import (
            print_success,
            print_warning,
            print_error,
            print_info,
        )

        # These should not raise exceptions
        print_success("Test success message")
        print_warning("Test warning message")
        print_error("Test error message")
        print_info("Test info message")

        print("  ✅ Console helpers working")
        return True
    except Exception as e:
        print(f"  ❌ Console helpers failed: {e}")
        return False


def check_logging_config():
    """Verify logging configuration works."""
    print("\n🔍 Testing logging configuration...")

    try:
        from src.utils.logging_config import setup_logging, get_logger

        logger = setup_logging(level="WARNING")
        module_logger = get_logger("test_module")

        logger.info("Test log message")
        module_logger.debug("Test module message")

        print("  ✅ Logging configuration working")
        return True
    except Exception as e:
        print(f"  ❌ Logging config failed: {e}")
        return False


def check_bare_exceptions_fixed():
    """Verify bare exceptions were fixed."""
    print("\n🔍 Checking for bare exceptions...")

    files_to_check = [
        "src/ga/population.py",
        "src/exporter/plotpareto.py",
    ]

    bare_exceptions_found = False
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            # Check for "except:" without specific exception type
            if "\nexcept:\n" in content or "\n        except:\n" in content:
                print(f"  ❌ Bare exception found in {file_path}")
                bare_exceptions_found = True
            else:
                print(f"  ✅ {file_path} - No bare exceptions")
        else:
            print(f"  ⚠️  {file_path} - File not found")

    return not bare_exceptions_found


def check_package_metadata():
    """Verify package metadata."""
    print("\n🔍 Checking package metadata...")

    try:
        from src import __version__, __author__, __license__

        print(f"  ✅ Version: {__version__}")
        print(f"  ✅ Author: {__author__}")
        print(f"  ✅ License: {__license__}")
        return True
    except Exception as e:
        print(f"  ❌ Metadata check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Code Quality Enhancements - Verification Script")
    print("=" * 60)

    checks = [
        ("File Creation", check_files_exist),
        ("Module Imports", check_imports),
        ("Console Helpers", check_console_helpers),
        ("Logging Config", check_logging_config),
        ("Bare Exceptions Fixed", check_bare_exceptions_fixed),
        ("Package Metadata", check_package_metadata),
    ]

    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")

    print(f"\nTotal: {passed}/{total} passed")

    if failed == 0:
        print("\n🎉 All enhancements verified successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed} check(s) failed. Please review above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
