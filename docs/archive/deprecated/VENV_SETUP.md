# Virtual Environment Setup

This project uses **UV** for blazingly fast dependency management (10-100x faster than pip).

##  Quick Start

### Windows (PowerShell)

```powershell
# Run the UV setup script (auto-installs UV if needed)
.\setup-uv.ps1

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Run the schedule engine
python main.py --env test
```

### Linux/Mac (Bash)

```bash
# Run the UV setup script (auto-installs UV if needed)
./setup-uv.sh

# Activate the environment
source .venv/bin/activate

# Run the schedule engine
python main.py --env test
```

**See `docs/UV_QUICKSTART.md` for more details.**

## Manual Setup

### Option 1: UV (Recommended - Fast!)

#### 1. Install UV (standalone, no pip needed)

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create Virtual Environment

```bash
uv venv .venv
```

#### 3. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

#### 4. Install Dependencies

```bash
# From pyproject.toml (recommended)
uv pip install -e .

# Or from requirements.txt
uv pip install -r requirements.txt
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

## Why UV?

| Feature | UV | pip | Conda |
|---------|----|----|-------|
| Size | ~50MB | ~50MB | ~3GB |
| Setup Time | **3-5 seconds** | 30-45 seconds | 5+ minutes |
| Installation Speed | **10-100x faster** | Baseline | Slower than pip |
| Dependency Resolution |  Excellent | ⚠️ Good |  Excellent |
| No pip needed |  Yes | N/A |  Yes |
| Production Ready |  Yes |  Yes |  Yes |

## UV vs Conda

This project uses **UV instead of Conda** because:

1.  **10-100x faster** installation
2.  **50x smaller** (50MB vs 3GB)
3.  **Pure Python** - no system libraries needed
4.  **Modern standard** - pyproject.toml (PEP 621)
5.  **Better for CI/CD** - faster builds
6.  **Standalone binary** - no pip dependency

UV is the modern choice for Python-only projects.
