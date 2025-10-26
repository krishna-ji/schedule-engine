# Virtual Environment Setup

This project uses a **local Python virtual environment** (`.venv`) instead of conda.

## Quick Start

### Windows (PowerShell)

```powershell
# Run the setup script
.\setup-venv.ps1

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Run the schedule engine
python main.py --env test
```

### Linux/Mac (Bash)

```bash
# Make the script executable
chmod +x setup-venv.sh

# Run the setup script
./setup-venv.sh

# Activate the environment
source .venv/bin/activate

# Run the schedule engine
python main.py --env test
```

## Manual Setup (if scripts don't work)

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Deactivating the Environment

When you're done working, deactivate the environment:

```bash
deactivate
```

## Requirements

- **Python**: 3.8+ (tested with 3.13)
- **No conda required!**

## Dependencies

All dependencies are listed in `requirements.txt`:

- **deap**: Genetic algorithm framework
- **pydantic**: Configuration validation
- **pyyaml**: YAML configuration files
- **rich**: Terminal UI and progress bars
- **matplotlib**: Data visualization
- **seaborn**: Statistical plotting
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **pandas**: Data manipulation

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error like "cannot be loaded because running scripts is disabled":

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup-venv.ps1
```

### Python Not Found

Make sure Python is in your PATH:

```bash
python --version
```

If not found, reinstall Python and check "Add to PATH" during installation.

### Module Not Found Errors

If you get import errors, reinstall dependencies:

```bash
pip install --force-reinstall -r requirements.txt
```

## Comparison with Conda

| Feature | Virtual Environment (.venv) | Conda |
|---------|----------------------------|-------|
| Size | ~50MB | ~3GB |
| Setup Time | ~30 seconds | ~5 minutes |
| Python-only | ✅ Yes | ❌ No (includes system libs) |
| Project-specific | ✅ Yes | ⚠️ Can be shared |
| Standard Python | ✅ Yes | ⚠️ Conda-specific |

## Why Not Conda?

This project **does not require conda** because:

1. All dependencies are pure Python packages
2. No system-level C libraries needed
3. Virtual environments are faster and lighter
4. Standard Python tooling (pip + venv)
5. Better for version control (.venv in .gitignore)

The `environment.yml` files are kept for historical reference only.
