#!/bin/bash
# Activate Python virtual environment and run main.py

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the virtual environment (same directory as script)
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate the virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "[OK] Activated virtual environment: $VENV_PATH"
    echo ""
    
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
            python main.py --env dev
            ;;
        2)
            echo "Running with test environment..."
            python main.py --env test
            ;;
        3)
            echo "Running with prod environment..."
            python main.py --env prod
            ;;
        *)
            echo "Invalid choice. Please enter 1, 2, or 3."
            exit 1
            ;;
    esac
else
    echo "[ERROR] Virtual environment not found at $VENV_PATH"
    echo "Please run setup-venv.sh first to create the virtual environment"
    exit 1
fi
