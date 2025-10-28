#!/bin/bash
# Quick run script using UV (no activation needed!)

# Prompt user to select environment
echo "Select environment to run:"
echo "  1) dev"
echo "  2) test"
echo "  3) prod"
echo -n "Enter choice (1-3): "
read choice

case $choice in
    1)
        echo "Running with dev environment..."
        uv run python main.py --env dev
        ;;
    2)
        echo "Running with test environment..."
        uv run python main.py --env test
        ;;
    3)
        echo "Running with prod environment..."
        uv run python main.py --env prod
        ;;
    *)
        echo "Invalid choice. Please enter 1, 2, or 3."
        exit 1
        ;;
esac
