# Library Cheatsheet

Quick reference for the core third-party libraries used throughout Schedule Engine.

## DEAP (Genetic Algorithms)

| Aspect | Usage |
| --- | --- |
| Install | Included via UV (see `pyproject.toml`) |
| Key APIs | `deap.base.Toolbox`, `deap.tools.selNSGA2`, `deap.creator` |
| Where Used | `src/core/ga_scheduler.py`, `src/ga/population.py` |
| Patterns | Register evaluation/crossover/mutation functions in toolbox, rely on DEAP's `HallOfFame` for elitism |

Example:
```python
from deap import base, tools

toolbox = base.Toolbox()
toolbox.register("mate", crossover_course_group_aware)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selNSGA2)
```

## Stable-Baselines3 (RL Agents)

| Aspect | Usage |
| --- | --- |
| Algorithms | `PPO`, `DQN` |
| Where Used | `src/rl/agents/ppo_agent.py`, `src/rl/training/trainer.py` |
| Config | `rl.training.*` section in YAML |
| Notes | Wrap SB3 models to abstract serialization, device handling, and inference timeouts |

Example policy loading:
```python
from stable_baselines3 import PPO
model = PPO.load("models/rl_agents/ppo_prod.zip", device="cuda")
action, _ = model.predict(state, deterministic=True)
```

## PyTorch (GPU & RL Backend)

| Aspect | Usage |
| --- | --- |
| Version | 2.4.1+cu121 |
| GA Role | GPU batch constraint evaluation |
| RL Role | SB3 backend, custom reward ops |
| Files | `src/ga/evaluator/gpu_batch_evaluator.py`, `src/rl/*` |

Best practices:
- Call `torch.set_num_threads(1)` in RL training to avoid oversubscribing CPU.
- Use `torch.no_grad()` during inference to reduce memory.

## Pydantic (Configuration Models)

| Aspect | Usage |
| --- | --- |
| Version | 2.10.3 |
| Purpose | Strongly typed config schemas (`src/config/models/*.py`) |
| Features | Custom validators, default factories, type coercion |

Example model:
```python
from pydantic import BaseModel, Field

class GAConfig(BaseModel):
    pop_size: int = Field(200, ge=10)
    ngen: int = Field(2000, ge=1)
```

## Rich (Console UI)

| Aspect | Usage |
| --- | --- |
| Purpose | Colorful CLI output, progress bars, tables |
| Files | `src/utils/console_service.py`, `src/workflows/standard_run.py` |
| Tips | Use `console = get_console()` once per module; avoid instantiating Rich objects repeatedly inside loops |

Example progress bar:
```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Evolution", total=config.ga.ngen)
    for gen in range(config.ga.ngen):
        run_generation()
        progress.update(task, advance=1)
```

## NumPy & SciPy

- NumPy powers deterministic shuffling, statistics, and RL feature calculations.
- SciPy (if enabled) assists with advanced statistics in diagnostics scripts.

## Gymnasium

- Provides the RL environment interface (`gym.Env`).
- Environment ID: `ScheduleEnv-v0` (registered in `src/rl/gym_env/__init__.py`).

## Testing Utilities

| Library | Role |
| --- | --- |
| `pytest` | Core test runner |
| `hypothesis` (optional) | Property-based tests for constraints |
| `pytest-benchmark` | Used in benchmarking scripts for heuristic speed |

Keep this cheat sheet nearby when upgrading dependencies or onboarding new contributors.
