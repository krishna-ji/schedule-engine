---
applyTo: "{src/rl/**/*.py,scripts/**/rl*.py,scripts/promote_model_to_prod.py,test/rl/**/*.py}"
---

# Reinforcement Learning Modules Instructions

## Overview
The RL stack (environment, training, deployment, promotion) extends the GA scheduler with a PPO/DQN-powered hyper-heuristic. Code quality and reproducibility are critical because these modules gate production scheduling.

## General Principles
- **Determinism first**: Seed Stable-Baselines3 algorithms via config; expose seed parameters in CLIs.
- **Config-driven**: New knobs belong in `configs/base.yaml` (and `prod.yaml` overrides). Never hardcode hyperparameters in code.
- **Docstring everything**: Each public class/function in `src/rl/**` requires a concise docstring describing inputs, outputs, and side effects.
- **Manifest integrity**: Any code writing checkpoints must update `models/rl_agents/manifest.json` via `CheckpointManager`.
- **Timeout budgets**: Respect targets (<100 ms load, <10 ms inference). Include lightweight benchmarks when touching deployment modules.

## Environment & State Encoder (`src/rl/gym_env/`)
- Keep observation layout synchronized with `StateEncoder.observation_dim`; update associated tests in `test/rl/test_diversity_metrics.py` when features change.
- Avoid importing heavy GA modules at import time; lazy-load inside functions to keep env initialization fast.
- When adding features, include normalization logic and document new indices in the encoder docstring.

## Training (`src/rl/training/`)
- CLIs (`train_script.py`, `generate_validation_set.py`, `select_best_checkpoint.py`) must support UV execution (`uv run train`, `uv run python ...`).
- Training profiles in `configs/training/` (test, med, prod) control curriculum stages and hyperparameters.
- Curriculum changes require YAML updates plus doc updates in `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`.
- Callbacks should log to TensorBoard using the provided writer; do not create new logging sinks unless coordinated.
- Always gate long-running operations behind `if __name__ == "__main__":` to keep modules import-safe.
- GPU acceleration is enabled by default (`device: cuda` in configs/base.yaml); see `docs/04-algorithms/nvidia-gpu/` for details.

## Deployment & Promotion (`src/rl/deployment/`, `scripts/promote_model_to_prod.py`)
- Use `ModelLoader` for all SB3 model loading; no direct `PPO.load()` calls elsewhere.
- Promotion flow: `select_best_checkpoint.py --promote` → `promote_model_to_prod.py --checkpoint-id <id>` → update `configs/prod.yaml`. Keep this contract documented.
- `ModelRegistry` writes configs atomically (temp file + replace). Preserve this behavior when editing.
- Add regression tests in `test/rl/test_registry.py` or new files when touching registry/promotion logic.

## GA Integration Touchpoints
- `_init_rl()` and `_apply_rl_operators()` live in `src/core/ga_scheduler.py`. Any RL change impacting those methods must be coordinated with GA maintainers and validated with `uv run prod` (RL on/off).
- When extending hybrid controller modes or fallback strategies, update `configs/base.yaml` docs and onboarding guide.

## Documentation & Reporting
- Record significant RL changes in `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`.
- Document technical Q&A in `docs/08-qna/technical-questions.md` for discussion history.
- For new utilities, add usage examples (CLI snippets) similar to existing documentation sections.
- Performance improvements should reference `docs/05-performance/` documentation.
- GPU-related changes should coordinate with `docs/04-algorithms/nvidia-gpu/` guides.

## Testing Checklist
- `pytest test/rl/test_diversity_metrics.py` for encoder/state changes.
- `pytest test/rl/test_registry.py` for promotion/registry edits.
- High-level smoke: `uv run python src/rl/training/train_script.py --timesteps 1000 --agent ppo` (documented in onboarding guide).