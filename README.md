# Schedule Engine

University course scheduling engine that combines NSGA-II genetic algorithms with reinforcement learning, GPU-accelerated fitness evaluation, and rich orchestration tooling. The system targets large university timetabling problems with hard/soft constraints and staged runtime modes for experiments and production runs.

## Highlights

- Progressive runtime modes (baseline NSGA-II through multi-agent RL) with killswitch-controlled features.
- GPU batch evaluator, parallel operators, and local search phases for 10-50x throughput gains.
- YAML-driven configuration system with environment overrides (`test`, `prod`) and experiment manifests.
- Extensive CLI and UV script shortcuts for experiments, diagnostics, benchmarking, and RL training.
- Rich documentation in `docs/` covering algorithms, architecture, and thesis guides.

## Quick Start

```bash
uv sync --frozen          # install dependencies
uv run test               # smoke test (30 generations)
uv run prod               # full production run (2000 generations)
```

Common runtime shortcuts:

```bash
uv run launcher           # interactive menu
uv run baseline           # Mode 1: pure NSGA-II
uv run repairs            # Mode 2: NSGA-II + repairs
uv run heuristics         # Mode 3: + heuristics
uv run full               # Mode 4: full GA stack
uv run rl                 # Mode 5: RL-guided selection
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
