# Installation Guide

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 20.04+)
- **Python**: 3.12 (exact version pinned)
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: NVIDIA GPU with CUDA 12.1 support (optional, for GPU acceleration)
- **Disk Space**: Minimum 5GB free space

### Required Software

1. **Python 3.12**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation: `python --version` (should show 3.12.x)

2. **UV Package Manager** (recommended)
   - Modern, fast Python package installer
   - Install via PowerShell (Windows):
     ```powershell
     irm https://astral.sh/uv/install.ps1 | iex
     ```
   - Install via curl (macOS/Linux):
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Verify installation: `uv --version`

3. **Git**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation: `git --version`

### Optional: NVIDIA GPU Setup

For GPU-accelerated constraint evaluation (10-50x speedup):

1. **NVIDIA Driver**
   - Download latest driver from [nvidia.com](https://www.nvidia.com/download/index.aspx)
   - Verify CUDA availability: `nvidia-smi`

2. **CUDA 12.1 Toolkit**
   - Bundled with PyTorch installation
   - No separate installation needed (handled by UV)

## Installation Steps

### 1. Clone Repository

```powershell
# Navigate to desired directory
cd C:\Users\<YourUsername>\Desktop

# Clone repository
git clone https://github.com/krishna-ji/schedule-engine.git
cd schedule-engine

# Switch to development branch (if applicable)
git checkout dev-krishna
```

### 2. Install Dependencies

```powershell
# Install all dependencies using UV (recommended)
uv sync --frozen

# Alternative: Using pip (if UV not available)
pip install -e .
```

**What gets installed:**
- **Core GA Libraries**: DEAP 1.4.1 (genetic algorithms)
- **RL Libraries**: Stable-Baselines3 2.3.2, Gymnasium 0.29.1
- **GPU Libraries**: PyTorch 2.4.1+cu121 (with CUDA 12.1)
- **Config/Validation**: Pydantic 2.10.3, PyYAML 6.0.2
- **Scientific Computing**: NumPy 1.26.4, SciPy 1.11.4, pandas 2.2.3
- **Visualization**: Matplotlib 3.9.4, Seaborn 0.13.2, Rich 13.9.4
- **Performance**: psutil 6.1.1 (system monitoring)

### 3. Verify Installation

```powershell
# Check Python version
python --version  # Should show 3.12.x

# Verify DEAP installation
python -c "import deap; print(f'DEAP {deap.__version__}')"

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify Stable-Baselines3
python -c "import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')"

# Run diagnostics script
uv run diagnose-system
```

Expected output:
```
✓ Python 3.12.x detected
✓ DEAP 1.4.1 installed
✓ PyTorch 2.4.1+cu121 installed
✓ CUDA available: True (GPU: NVIDIA GeForce RTX 3080)
✓ All dependencies installed successfully
```

### 4. Verify Data Files

```powershell
# Check data directory structure
ls data/

# Expected files:
# - Course.json
# - Groups.json
# - Instructors.json
# - Rooms.json
```

If data files are missing, contact project maintainer or check `data/archive/` for backups.

## Optional: Development Setup

For contributors and developers:

### 1. Install Development Dependencies

```powershell
# Install dev dependencies (includes pytest, black, ruff, mypy)
uv sync --frozen --group dev
```

### 2. Configure IDE

**VS Code (recommended):**
1. Install Python extension
2. Install Pylance extension
3. Set Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv/Scripts/python.exe`
4. Configure settings (`.vscode/settings.json`):
   ```json
   {
     "python.linting.enabled": true,
     "python.linting.ruffEnabled": true,
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "python.testing.pytestEnabled": true
   }
   ```

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Select existing virtual environment (`.venv`)
3. Configure external tools: Black (formatter), Ruff (linter)

### 3. Pre-commit Hooks (Optional)

```powershell
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Troubleshooting

### Issue: UV command not found

**Solution:**
- Ensure UV is in PATH
- Restart terminal/PowerShell after installation
- Try absolute path: `C:\Users\<YourUsername>\.cargo\bin\uv.exe`

### Issue: PyTorch CUDA not available

**Solution:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Check PyTorch installation:
   ```powershell
   python -c "import torch; print(torch.__version__)"
   ```
3. Reinstall with CUDA:
   ```powershell
   uv sync --frozen --reinstall-package torch
   ```

### Issue: Import errors

**Solution:**
- Ensure you're in project root: `cd schedule-engine`
- Activate virtual environment (if using venv):
  ```powershell
  .venv\Scripts\Activate.ps1  # Windows
  source .venv/bin/activate  # macOS/Linux
  ```
- Reinstall dependencies: `uv sync --frozen`

### Issue: Permission errors on Windows

**Solution:**
- Run PowerShell as Administrator
- Set execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## Next Steps

- [Setup Guide](02-setup.md) - Configure environment and data
- [First Run Guide](03-first-run.md) - Run your first experiment
- [UV Commands Reference](04-uv-commands.md) - All available commands

## Additional Resources

- [Official UV Documentation](https://github.com/astral-sh/uv)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
