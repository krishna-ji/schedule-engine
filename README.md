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

See `docs/` folder for detailed documentation.

---

## 🤔 "Should I Use OR-Tools Instead?"

**Quick Answer: NO! ✅ Your DEAP implementation is excellent!**

We've created a comprehensive analysis comparing DEAP vs Google OR-Tools vs other alternatives:

📍 **START HERE:** [Comparison Index](docs/COMPARISON_INDEX.md) - Navigation guide to all resources

### Quick Links by Time Available:

- ⚡ **2 minutes?** → [Quick Answer](docs/QUICK_ANSWER.md) - TL;DR: Keep DEAP!
- 📊 **5 minutes?** → [Visual Summary](docs/VISUAL_SUMMARY.md) - Charts & comparisons
- 🔍 **15 minutes?** → [When to Use What](docs/WHEN_TO_USE_WHAT.md) - Decision guide
- 📚 **30 minutes?** → [Library Comparison](docs/LIBRARY_COMPARISON.md) - Complete analysis (700+ lines)
- 💻 **Want code?** → [OR-Tools POC](docs/ortools_poc.py) - Run `python docs/ortools_poc.py`

### Key Findings:

**DEAP wins 6/9 categories:**
- ✅ Soft constraint handling (excellent vs poor)
- ✅ Multi-objective optimization (native vs none)
- ✅ Academic value (high vs low)
- ✅ Explainability (high vs low)
- ✅ Implementation status (complete vs not started)
- ✅ Thesis contribution (novel vs standard)

**Bottom Line:** Your DEAP implementation is the RIGHT choice for this problem. OR-Tools could be complementary (hybrid approach) but NOT a replacement. Don't waste time rewriting!

---

### 📚 Other Documentation

### 📚 Other Documentation


