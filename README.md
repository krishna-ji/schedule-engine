# Schedule Engine Project
[BEI-Major Project]
- Krishna Acharya
- Dinanath Padhya
- Bipul Dahal
- Claude काका
- Copilot मामा

## Quick Start

### Installation with UV ⚡ (10-100x faster than pip)

**Windows:**
```powershell
# Run the setup script (auto-installs UV if needed)
.\setup-uv.ps1

# Or manual setup:
# 1. Install UV (standalone installer - no pip needed)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create environment and install dependencies
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
```

**Linux/macOS:**
```bash
# Run the setup script (auto-installs UV if needed)
./setup-uv.sh

# Or manual setup:
# 1. Install UV (standalone installer - no pip needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Running the Engine

```bash
# Test run (fast, 10 generations)
python main.py --env test

# Development run (medium, 100 generations)
python main.py --env dev

# Production run (full quality, 200+ generations)
python main.py --env prod

# Custom configuration
python main.py --config path/to/custom.yaml
```


##  Documentation

See `docs/` folder for detailed documentation:

### 📊 Library Comparison & Analysis

**Wondering if Google OR-Tools or other libraries would be better?**

- **[Library Comparison Guide](docs/LIBRARY_COMPARISON.md)** - Comprehensive analysis comparing DEAP vs OR-Tools vs alternatives
- **[When to Use What](docs/WHEN_TO_USE_WHAT.md)** - Quick decision guide for choosing the right approach
- **[OR-Tools Proof-of-Concept](docs/ortools_poc.py)** - Demonstration comparing constraint programming vs evolutionary approach

**TL;DR:** The current DEAP-based implementation is well-suited for this problem. OR-Tools could be complementary but not a replacement. See guides for detailed analysis.

### 📚 Other Documentation


