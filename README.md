# Schedule Engine Project
[BEI-Major Project]
- Krishna Acharya
- Dinanath Padhya
- Bipul Dahal
- Claude काका
- Copilot मामा

## Quick Start

### Installation with UV ⚡ (10-100x faster than pip)

**One-line setup (Windows/Linux/macOS):**
```bash
# Run the setup script (auto-installs UV if needed)
python setup-uv
```

**Manual setup:**
```bash
# 1. Install UV (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment and install dependencies
uv venv .venv
uv sync
```

### Running the Engine

**Quick run (interactive menu):**
```bash
python x
```

**Direct run (no menu):**
```bash
# Using the quick launcher
python x test    # Fast test (10 generations)
python x dev     # Development (100 generations)
python x prod    # Production (200+ generations)

# Or directly with main.py
python main.py --env test
python main.py --env dev
python main.py --env prod
python main.py --config path/to/custom.yaml
```

**Using UV (no activation needed):**
```bash
uv run python main.py --env dev
```


##  Documentation

See `docs/` folder for detailed documentation:

