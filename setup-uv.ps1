# Setup Script for Schedule Engine with UV
# Creates a local .venv and installs all dependencies using UV
# UV is a blazingly fast Python package installer (10-100x faster than pip)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Schedule Engine - UV Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if UV is available
try {
    $uvVersion = uv --version 2>&1
    Write-Host "[OK] UV found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] UV not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing UV (standalone installer - no pip needed)..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Install UV using standalone installer
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        Write-Host ""
        Write-Host "[OK] UV installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Please restart your terminal and run this script again." -ForegroundColor Yellow
        exit 0
    } catch {
        Write-Host "[ERROR] Failed to install UV" -ForegroundColor Red
        Write-Host "Please install manually: https://github.com/astral-sh/uv" -ForegroundColor Yellow
        exit 1
    }
}

# Remove existing .venv if it exists
if (Test-Path ".venv") {
    Write-Host ""
    $response = Read-Host "Existing .venv found. Delete and recreate? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Removing existing .venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".venv"
    } else {
        Write-Host "Keeping existing .venv. Exiting..." -ForegroundColor Yellow
        exit 0
    }
}

# Create virtual environment with UV
Write-Host ""
Write-Host "Creating virtual environment with UV..." -ForegroundColor Cyan
uv venv .venv

if (-not (Test-Path ".venv")) {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Virtual environment created at .venv/" -ForegroundColor Green

# Install dependencies with UV
Write-Host ""
Write-Host "Installing dependencies with UV (much faster than pip!)..." -ForegroundColor Cyan

# Check if pyproject.toml exists
if (Test-Path "pyproject.toml") {
    Write-Host "Using pyproject.toml for installation..." -ForegroundColor Cyan
    uv sync
} elseif (Test-Path "requirements.txt") {
    Write-Host "Using requirements.txt for installation..." -ForegroundColor Cyan
    uv pip install -r requirements.txt
} else {
    Write-Host "[ERROR] No pyproject.toml or requirements.txt found" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
uv pip list
Write-Host ""
Write-Host "To run the schedule engine (no activation needed!):" -ForegroundColor Yellow
Write-Host "  uv run python main.py --env test" -ForegroundColor White
Write-Host "  uv run python main.py --env dev" -ForegroundColor White
Write-Host "  uv run python main.py --env prod" -ForegroundColor White
Write-Host ""
Write-Host "Or activate manually:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python main.py --env test" -ForegroundColor White
Write-Host ""
Write-Host "To add a new package (auto-updates pyproject.toml!):" -ForegroundColor Yellow
Write-Host "  uv add package-name" -ForegroundColor White
Write-Host ""
Write-Host "To remove a package (auto-updates pyproject.toml!):" -ForegroundColor Yellow
Write-Host "  uv remove package-name" -ForegroundColor White
Write-Host ""
Write-Host "To sync dependencies after editing pyproject.toml:" -ForegroundColor Yellow
Write-Host "  uv sync" -ForegroundColor White
Write-Host ""
