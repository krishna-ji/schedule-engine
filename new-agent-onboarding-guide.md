# Schedule Engine – New Agent Onboarding Guide

_Last updated: 2025-11-16_

Welcome! This guide condenses everything a fresh agent needs to ramp up quickly: what has been built, what remains, and how to execute day-to-day workflows.

## 1. Project Snapshot
- **Problem**: University course scheduling via NSGA-II + constraint repair.
- **Language / Tools**: Python 3.11+, DEAP, Stable-Baselines3, Pydantic, Rich UI, UV package manager.
- **Entry Points**:
  - GA runs: `uv run test` (smoke) / `uv run prod` (full) / `python main.py --env <env>`
  - RL training: `uv run python src/rl/training/train_script.py --timesteps <N> --agent ppo`
- **Configs**: `configs/base.yaml` (shared), `configs/{test,prod}.yaml` (overrides). Always load via `from src.config import get_config`.

## 2. Phase Timeline & Status
| Phase | Date | Status | Highlights |
| --- | --- | --- | --- |
| **Phase 1.5 – Heuristic Toolbox** | 2025-11-15 | ✅ Complete | 19 operators, decorator registry, documented in `docs/PHASE_1.5_SUMMARY.md`.
| **Phase 2.1 – Gym Environment** | 2025-11-15 | ✅ Complete | Gymnasium env + state encoder + reward, see `docs/PHASE_2.1_SUMMARY.md`.
| **Phase 2.2-2.4 – RL Integration** | 2025-11-15 | ✅ Code complete | Training pipeline, checkpoints, deployment, GA integration (`docs/code/PHASE_2_RL_COMPLETE.md`).
| **Phase 3 – Advanced RL / Evaluation** | Planned | 🚧 Not started | Multi-agent, transfer learning, evaluation suite (`Todo.md`).

The remaining work for Phase 2 is operational: run the RL training curriculum, validate/promo best checkpoint, and benchmark GA with RL enabled.

## 3. Current Priority Backlog (Nov 2025)
1. **Run PPO curriculum training** (100K–300K timesteps) with TensorBoard logging.
2. **Generate validation sets** via `scripts/generate_validation_set.py --stage all`.
3. **Select & promote best checkpoint** using `scripts/select_best_checkpoint.py --metric mean_reward --promote` followed by `scripts/promote_model_to_prod.py --checkpoint-id <id>`.
4. **Enable RL in `configs/prod.yaml`** (`rl.enabled: true`, `mode: inference|hybrid`) and execute `uv run prod`.
5. **Baseline comparison**: repeat `uv run prod` with RL disabled; capture hard/soft violations, convergence time, run duration.
6. **Documentation updates**: append run results to `docs/code/PHASE_2_RL_COMPLETE.md` and this guide.

Track these in `Todo.md` (master plan) and the VS Code Tasks view.

## 4. Quick Runbooks
### 4.1 Environment Setup
```pwsh
uv sync
uv run python -m pip install -e .
```

### 4.2 RL Training (Smoke vs. Deep Run)
```pwsh
# Smoke (≈15 min)
uv run python src/rl/training/train_script.py --timesteps 100000 --agent ppo --save-path models/rl_agents/ppo_smoke

# Full (≈60 min)
uv run python src/rl/training/train_script.py --timesteps 300000 --agent ppo --save-path models/rl_agents/ppo_full
```
Launch TensorBoard in another terminal if desired:
```pwsh
uv run python -m tensorboard --logdir logs/tensorboard
```

### 4.3 Validation & Promotion
```pwsh
uv run python scripts/generate_validation_set.py --stage all --num-problems 10
uv run python scripts/select_best_checkpoint.py --metric mean_reward --promote
uv run python scripts/promote_model_to_prod.py --checkpoint-id <checkpoint_id>
```

### 4.4 Production GA with RL
1. Edit `configs/prod.yaml`:
   ```yaml
   rl:
     enabled: true
     mode: inference
     agent:
       model_path: models/rl_agents/<promoted>.zip
   ```
2. Run:
   ```pwsh
   uv run prod
   ```
3. Record metrics (hard/soft, time per generation, generation where feasibility achieved) in `output/` and docs.

### 4.5 Regression/Baseline Run
Toggle `rl.enabled: false` and rerun `uv run prod` with the same RNG seed for apples-to-apples comparison.

## 5. Directory Cheatsheet
- `src/core/ga_scheduler.py`: NSGA-II core + RL hooks (`_init_rl`, `_apply_rl_operators`).
- `src/rl/gym_env/`: State encoder, reward, observation logic.
- `src/rl/training/`: Trainer, curriculum, callbacks, checkpointing.
- `src/rl/deployment/`: Model loader, inference, registry.
- `scripts/`: Validation/promotion utilities (`generate_validation_set`, `select_best_checkpoint`, `promote_model_to_prod`).
- `test/rl/`: Regression tests for encoder/registry.
- `docs/`: Phase summaries (`docs/PHASE_1.5_SUMMARY.md`, `docs/PHASE_2.1_SUMMARY.md`, `docs/code/PHASE_2_RL_COMPLETE.md`).

## 6. Coding Standards & Instructions
- Follow `.github/copilot-instructions.md` for global policy.
- Path-specific rules live in `.github/instructions/*.instructions.md` (notably `ga-core.instructions.md` and `rl.instructions.md`).
- Use docstrings + type hints; keep edits ASCII unless explicitly justified.
- Document minor fixes in `docs/code/ENHANCE.md`; major design work goes into `docs/for_report/`.

## 7. Validation Checklist Before Commit
- `pytest test/rl/test_diversity_metrics.py` after touching state encoder or observation logic.
- `pytest test/rl/test_registry.py` for promotion/registry edits.
- `uv run python src/rl/training/train_script.py --timesteps 1000 --agent ppo` as a smoke test for training changes.
- `uv run prod` (with RL toggled on/off) when modifying GA scheduler or RL hooks.

## 8. Communication & Reporting
- Log meaningful experiment outputs in `output/evaluation_*` with timestamped folders.
- Summaries belong in `docs/code/PHASE_2_RL_COMPLETE.md` (Phase 2) or upcoming docs for Phase 3.
- Update this onboarding guide when processes or priorities shift.

## 9. Future Directions (Phase 3+)
- **Advanced RL**: Multi-agent, transfer learning, online adaptation.
- **Evaluation Suite**: Baseline strategies, statistical tests, visualization pipeline.
- **Documentation**: RL Architecture/Training/Integration guides + quick start (see `Todo.md`).

Stay aligned with the backlog in `Todo.md`, and keep `.github/copilot-instructions.md` + this document in sync. Happy scheduling! 🚀
