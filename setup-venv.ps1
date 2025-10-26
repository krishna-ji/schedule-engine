# Setup Script for Schedule Engine Virtual Environment
# Creates a local .venv and installs all dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Schedule Engine - Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to PATH" -ForegroundColor Yellow
    exit 1
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

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv

if (-not (Test-Path ".venv")) {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Virtual environment created at .venv/" -ForegroundColor Green

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

# Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
pip list
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run the schedule engine:" -ForegroundColor Yellow
Write-Host "  python main.py --env test" -ForegroundColor White
Write-Host ""
