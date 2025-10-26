#!/bin/bash
# Setup Script for Schedule Engine Virtual Environment
# Creates a local .venv and installs all dependencies

echo "========================================"
echo "Schedule Engine - Virtual Environment Setup"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found in PATH"
    echo "Please install Python 3.8+ and add it to PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] Python found: $PYTHON_VERSION"

# Remove existing .venv if it exists
if [ -d ".venv" ]; then
    echo ""
    read -p "Existing .venv found. Delete and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing .venv..."
        rm -rf .venv
    else
        echo "Keeping existing .venv. Exiting..."
        exit 0
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

if [ ! -d ".venv" ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo "[OK] Virtual environment created at .venv/"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the schedule engine:"
echo "  python main.py --env test"
echo ""
