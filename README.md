# 📅 Schedule Engine

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![UV](https://img.shields.io/badge/package%20manager-uv-orange)](https://github.com/astral-sh/uv)

**University Course Scheduling Engine using NSGA-II Genetic Algorithm**

A multi-objective optimization system for generating university timetables that balances hard constraints (must-satisfy) with soft preferences (nice-to-have). Built with Python, DEAP, and Rich terminal UI.

[BEI-Major Project] • Krishna Acharya • Dinanath Padhya • Bipul Dahal

---

## ✨ Features

- 🧬 **NSGA-II Multi-objective Optimization**: Pareto-optimal solutions for conflicting objectives
- ⚡ **Parallel Processing**: 3-6x speedup with multiprocessing support
- 🎯 **Constraint-Based**: Hard constraints (conflicts) + soft preferences (gaps, compactness)
- 🔧 **Intelligent Repair**: IGLS (Intensive Greedy Local Search) for constraint violation reduction
- 📊 **Rich Analytics**: Evolution plots, Pareto fronts, constraint heatmaps
- 📄 **PDF Calendar Export**: Color-coded timetables for easy visualization
- ✅ **Feasibility Analysis**: Pre-GA validation to detect impossible scheduling scenarios
- 🎨 **Terminal UI**: Real-time progress tracking with Rich formatting

---

## 🚀 Quick Start

### Installation with UV ⚡

**One-line setup (Windows/Linux/macOS):**
```bash
python setup-uv
```

**Manual setup:**
```bash
# Install UV package manager
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv .venv
uv sync
```

### Running the Engine

```bash
# Quick smoke test (30 generations, ~5-10 min)
uv run test

# Medium quality (400 generations, ~4-8 hours)
uv run notprod

# Best quality (2000 generations, ~24-48 hours)
uv run prod

# RL training (convenience):
# Start RL training using built-in train script with a profile; both flags and the shorthand positional profile work:
uv run train --profile prod
# or shorthand:
uv run train prod

# Custom configuration
python main.py --config path/to/custom.yaml
```

---

## 📁 Project Structure

```
schedule-engine/
├── main.py                 # CLI entry point
├── configs/                # YAML configurations
│   ├── base.yaml           # Shared settings
│   ├── test.yaml           # Smoke test (30 gen)
│   └── prod.yaml           # Best quality (2000 gen)
├── data/                   # Input JSON files
│   ├── Course.json
│   ├── Groups.json
│   ├── Instructors.json
│   └── Rooms.json
├── src/
│   ├── config/             # Configuration system
│   ├── core/               # GA scheduler
│   ├── ga/                 # Operators, population
│   ├── constraints/        # Hard & soft constraints
│   ├── encoder/            # JSON → entities
│   ├── decoder/            # Individual → schedule
│   ├── validation/         # Input & feasibility
│   ├── exporter/           # PDF, plots, reports
│   └── workflows/          # Orchestration
├── test/unit/              # Unit tests
└── output/                 # Generated schedules
```

---

## 🎯 Architecture

### Workflow Pipeline

```
Input JSON → Validation → Feasibility Check → GA Evolution → Decoding → Export
   ↓              ↓               ↓                  ↓            ↓         ↓
Courses      Consistency    Resource        NSGA-II with    Schedule   PDF +
Groups       Rules          Analysis         Repair          JSON       Plots
Instructors                                  Mechanisms
Rooms
```

### Key Components

**1. Genetic Algorithm (NSGA-II)**
- **Population**: 10-200 individuals (configurable)
- **Chromosome**: List of `SessionGene` (course + groups + instructor + room + time)
- **Fitness**: `(-hard_violations, -soft_penalty)` with weights `(-1.0, -0.01)`
- **Operators**: Course-group-aware crossover, constraint-guided mutation
- **Selection**: Non-dominated sorting with crowding distance

**2. Constraint System**
- **Hard**: No time conflicts (instructor/room/group), room capacity
- **Soft**: Minimize gaps, compact schedule, instructor/room preferences

**3. Repair Mechanisms**
- **IGLS**: Exhaustive search for better time slots
- **Stagnation Repair**: Triggered when evolution stagnates
- **Selective Repair**: Focuses on worst-violating individuals

**4. Time System**
- **Quantum-based**: Converts wall-clock time ↔ discrete quanta (default: 60 min)
- **Flexible**: Supports arbitrary session durations

---

## 📊 Output Structure

```
output/evaluation_<timestamp>/
├── run.log                    # Execution summary
├── feasibility.log            # Pre-GA analysis
├── violations.log             # Constraint violation report
├── schedule.json              # Best schedule (JSON)
├── calendar.pdf               # Color-coded timetable
├── data/
│   └── metrics.csv            # Per-generation metrics
└── plots/
    ├── hard_constraint_trend.pdf
    ├── soft_constraint_trend.pdf
    ├── diversity_trend.pdf
    ├── hypervolume.pdf
    └── constraints/           # Individual constraint plots
```

---

## 🛠️ Configuration

Configurations use YAML with inheritance:

- `base.yaml` - Common settings (shared across all environments)
- `test.yaml` - Overrides for quick testing
- `prod.yaml` - Best-quality production runs

**Key Settings:**

| Parameter | Test | Prod | Description |
|-----------|------|------|-------------|
| `ga.ngen` | 30 | 2000 | Generations |
| `ga.pop_size` | 10 | 200 | Population size |
| `parallel.use_multiprocessing` | false | true | Parallel eval |
| `repair.exhaustive.enabled` | false | true | IGLS repair |

See [`docs/CONFIG_QUICKSTART.md`](docs/CONFIG_QUICKSTART.md) for details.

---

## 📚 Documentation

### Quick References
- [`docs/QUICKREF.md`](docs/QUICKREF.md) - Command cheatsheet
- [`docs/CONFIG_QUICKSTART.md`](docs/CONFIG_QUICKSTART.md) - Configuration guide
- [`docs/UV_QUICKSTART.md`](docs/UV_QUICKSTART.md) - UV package manager

### Detailed Guides
- [`docs/PROD_RUN_GUIDE.md`](docs/PROD_RUN_GUIDE.md) - Production run workflow
- [`docs/METRICS_QUICKSTART.md`](docs/METRICS_QUICKSTART.md) - Metrics explanation
- [`docs/PARALLEL_QUICKSTART.md`](docs/PARALLEL_QUICKSTART.md) - Parallel processing
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - Contribution guidelines

### Architecture
- [`docs/for_report/`](docs/for_report/) - Thesis-ready documentation

---

## 🧪 Testing

```bash
# Run all unit tests
pytest test/unit/

# Run with coverage report
pytest --cov=src --cov-report=html test/unit/

# Run specific test file
pytest test/unit/test_config_loader.py

# Run tests matching pattern
pytest -k "constraint" test/unit/
```

---

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for:
- Development setup
- Code standards
- Testing requirements
- Commit guidelines
- Pull request process

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DEAP**: Distributed Evolutionary Algorithms in Python
- **Rich**: Beautiful terminal formatting
- **Pydantic**: Data validation and settings management

---

## 📬 Contact

**Authors**: Krishna Acharya, Dinanath Padhya, Bipul Dahal
**Project**: BEI Major Project (University Course Scheduling)

---

## 🎓 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
uv sync
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

**2. Multiprocessing Hangs**
```yaml
# Disable in config if issues persist
parallel:
  use_multiprocessing: false
```

**3. Memory Issues (Large Datasets)**
```yaml
# Reduce population size
ga:
  pop_size: 50  # Down from 200
```

**4. Slow Performance**
```yaml
# Enable parallelism
parallel:
  use_multiprocessing: true
  num_workers: null  # Auto-detect CPU cores
```

### Getting Help

1. Check [`docs/QUICKREF.md`](docs/QUICKREF.md)
2. Review configuration in `configs/`
3. Run with `--env test` first to verify setup
4. Check output logs in `output/evaluation_*/`

---

**Made with ❤️ for efficient university scheduling**
