#!/bin/bash
# Setup Script for Schedule Engine with UV
# Creates a local .venv and installs all dependencies using UV
# UV is a blazingly fast Python package installer (10-100x faster than pip)

echo "========================================"
echo "Schedule Engine - UV Setup"
echo "========================================"
echo ""

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "[ERROR] UV not found in PATH"
    echo ""
    echo "Installing UV (standalone installer - no pip needed)..."
    echo ""
    
    # Install UV using standalone installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "[OK] UV installed successfully!"
        echo ""
        echo "Please restart your terminal and run this script again."
        exit 0
    else
        echo "[ERROR] Failed to install UV"
        echo "Please install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi

UV_VERSION=$(uv --version)
echo "[OK] UV found: $UV_VERSION"

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

# Create virtual environment with UV
echo ""
echo "Creating virtual environment with UV..."
uv venv .venv

if [ ! -d ".venv" ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo "[OK] Virtual environment created at .venv/"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with UV
echo ""
echo "Installing dependencies with UV (much faster than pip!)..."

# Check if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "Using pyproject.toml for installation..."
    uv sync
elif [ -f "requirements.txt" ]; then
    echo "Using requirements.txt for installation..."
    uv pip install -r requirements.txt
else
    echo "[ERROR] No pyproject.toml or requirements.txt found"
    exit 1
fi

# Verify installation
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Installed packages:"
uv pip list
echo ""
echo "To run the schedule engine (no activation needed!):"
echo "  uv run python main.py --env test"
echo "  uv run python main.py --env dev"
echo "  uv run python main.py --env prod"
echo ""
echo "Or activate manually:"
echo "  source .venv/bin/activate"
echo "  python main.py --env test"
echo ""
echo "To add a new package (auto-updates pyproject.toml!):"
echo "  uv add package-name"
echo ""
echo "To remove a package (auto-updates pyproject.toml!):"
echo "  uv remove package-name"
echo ""
echo "To sync dependencies after editing pyproject.toml:"
echo "  uv sync"
echo ""
