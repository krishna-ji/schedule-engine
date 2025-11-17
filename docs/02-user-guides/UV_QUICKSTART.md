# UV Quick Start Guide

## 🚀 One-Command Setup

**Windows:**
```powershell
.\setup-uv.ps1
```

**Linux/macOS:**
```bash
./setup-uv.sh
```

That's it! The script will:
1. Auto-install UV if not found (no pip needed!)
2. Create virtual environment
3. Install all dependencies
4. Verify installation

## 📦 Manual Setup (3 Commands)

### Windows
```powershell
# 1. Install UV (standalone, no pip needed)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create venv & install
uv venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
uv pip install -e .
```

### Linux/macOS
```bash
# 1. Install UV (standalone, no pip needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv & install
uv venv .venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -e .
```

## ⚡ Speed Comparison

| Task | pip | UV | Speedup |
|------|-----|----|----|
| Install all deps | 45s | 5s | **9x faster** |
| Install one package | 8s | 1s | **8x faster** |
| Update all | 60s | 6s | **10x faster** |

## 🔧 Common Commands

```bash
# Install all dependencies
uv sync                          # From pyproject.toml (smart!)

# Add/remove packages (auto-updates pyproject.toml!)
uv add package-name              # Install AND update pyproject.toml ✨
uv remove package-name           # Remove AND update pyproject.toml ✨

# List packages
uv pip list

# Run without activation
uv run python main.py --env test  # No activation needed! ⚡

# Lock dependencies
uv lock                          # Create uv.lock with exact versions
```

## 📋 Daily Workflow

```bash
# No activation needed with native UV!
uv run python main.py --env test
uv run python main.py --env dev
uv run python main.py --env prod

# Or activate manually if you prefer
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/macOS
python main.py --env test
```

## 🆘 Troubleshooting

### "uv: command not found"
**Solution:** Restart your terminal after installing UV

### "Permission denied" (Windows)
**Solution:** Run PowerShell as Administrator

### Want to go back to pip?
**Solution:** Just use `.\setup-venv.ps1` instead

## 📚 More Info

- Full migration guide: `docs/UV_MIGRATION.md`
- Detailed setup: `docs/VENV_SETUP.md`
- Project docs: `docs/`

## Why UV?

✅ **10-100x faster** than pip  
✅ **No pip dependency** - standalone binary  
✅ **Better dependency resolution**  
✅ **Drop-in replacement** - same commands  
✅ **Production ready** - by Astral (Ruff creators)  

Ready to try? Just run:
```powershell
.\setup-uv.ps1  # Windows
./setup-uv.sh   # Linux/macOS
```
