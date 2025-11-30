# Schedule Engine

University course scheduling engine that combines NSGA-II genetic algorithms with reinforcement learning, CPU-parallelized fitness evaluation, and rich orchestration tooling. The system targets large university timetabling problems with hard/soft constraints and staged runtime modes for experiments and production runs.

## Highlights

- Progressive runtime modes (baseline NSGA-II through multi-agent RL) with killswitch-controlled features.
- CPU multiprocessing for parallel fitness evaluation, operators, and local search phases.
- GPU acceleration reserved for RL training and neural network inference (where it excels).
- YAML-driven configuration system with environment overrides (`test`, `prod`) and experiment manifests.
- Extensive CLI and UV script shortcuts for experiments, diagnostics, benchmarking, and RL training.
- Rich documentation in `docs/` covering algorithms, architecture, and thesis guides.

## Quick Start

```bash
uv sync --frozen          # Install dependencies
uv run launcher           # Interactive experiment launcher
uv run baseline --test    # Quick smoke test
```

## Experiment Framework

### **Unified Launcher & Experiment Management**

```bash
uv run launcher           # Interactive TUI menu for all experiments
uv run clean              # Clear all experimental artifacts
uv run migrate            # Organize output structure
uv run analyze-results    # Generate comparison analysis
```

### **Available Experiments**

| Command | Method | Description |
|---------|--------|-------------|
| `uv run baseline --test/prod` | **A1: Pure NSGA-II** | Minimal baseline (no repairs, no heuristics) |
| `uv run repairs --test/prod` | **B1: NSGA + Repairs** | NSGA-II with repair heuristics |
| `uv run heuristics --test/prod` | **B2: NSGA + Heuristics** | NSGA-II + 19 heuristic operators |
| `uv run full --test/prod` | **B3: Full GA** | Complete GA with local search |
| `uv run roundrobin --test/prod` | **C1: Round-Robin** | Fixed heuristic rotation |
| `uv run rl --test/prod` | **C2: RL-Guided** | Reinforcement learning selection |

**Profiles:** `--test` (30 gens, ~2 min) • `--prod` (2000 gens, ~3-5 hours)

### **Experiment Output Structure**

```
output/
├── experiments/{method}/{type}/evaluation_TIMESTAMP/  # Results per run
├── logs/           # Execution logs, TensorBoard data
├── models/         # Trained RL agents
└── analysis/       # Comparison plots, statistics
```

## Repository Layout

- `src/` – core engine modules (config, GA, constraints, RL, workflows, utils)
- `configs/` – base/test/prod plus per-mode YAML overrides
- `scripts/` – CLI utilities, diagnostics, benchmarking, training helpers
- `test/` – unit and integration tests
- `docs/` – user guides, architecture notes, experiment playbooks, thesis material
- `output/` – experiment logs, evaluation artifacts, manifests

## Documentation

Start with `docs/00-INDEX.md` for navigation, or see:

- `docs/02-user-guides/runtime-modes.md` – runtime mode reference
- `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md` – GA/RL enhancements
- `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md` – thesis experiment walkthroughs

When contributing, follow the coding standards in `.github/copilot-instructions.md`, run formatting (`black`, `ruff`), and execute targeted tests before opening a PR.
