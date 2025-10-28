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


