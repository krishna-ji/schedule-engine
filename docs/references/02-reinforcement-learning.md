# Reinforcement Learning Reference

This page documents how RL augments the GA by selecting heuristic operators adaptively.

## Architecture

```mermaid
flowchart LR
    GA[GA Scheduler] --> State[State Encoder]
    State --> Agent[RL Agent (PPO/DQN)]
    Agent --> Action[Action Mapper]
    Action --> Heuristic[Heuristic Registry]
    GA --> Reward[Reward Calculator]
    Reward --> Agent
```

| Component | File(s) | Notes |
| --- | --- | --- |
| State Encoder | `src/rl/gym_env/state_encoder.py` | Builds 25D vector (normalized) |
| Reward Calculator | `src/rl/gym_env/reward_calculator.py` | Combines fitness delta, diversity delta, time penalty |
| Action Mapper | `src/rl/gym_env/action_mapper.py` | Maps discrete action → heuristic callable |
| Agents | `src/rl/agents/ppo_agent.py`, `dqn_agent.py`, `random_agent.py` | Wrap Stable-Baselines3 |
| Training | `src/rl/training/trainer.py`, `curriculum.py` | Handles curriculum + callbacks |
| Deployment | `src/rl/deployment/inference.py` | Fast inference + safety checks |

## Observation Vector (25 Dimensions)

| # | Feature | Description |
| --- | --- | --- |
| 1 | `gen_progress` | `current_gen / total_gen` |
| 2 | `best_hard_ratio` | `best_hard / max_hard` |
| 3 | `best_soft_ratio` | `best_soft / max_soft` |
| 4 | `avg_hard_ratio` | Population avg |
| 5 | `avg_soft_ratio` | Population avg |
| 6-13 | `hard_histogram` | Per-constraint normalized counts |
| 14-17 | `soft_histogram` | Soft constraint penalties |
| 18 | `diversity_score` | Normalized genotype diversity |
| 19 | `stagnation_ratio` | `gens_since_improvement / max` |
| 20 | `repair_success_rate` | Rolling average |
| 21 | `heuristic_success_rate` | Last 20 applications |
| 22 | `rl_action_entropy` | Tracking exploration |
| 23 | `cx_rate` | Actual crossover probability realized |
| 24 | `mut_rate` | Actual mutation probability realized |
| 25 | `wall_clock_norm` | Normalized runtime per generation |

## Action Space

- 19 actions map to registered heuristics (IDs defined in `ACTION_MAP`).
- Action 19 reserved for "adjust probabilities" (scales `cxpb`/`mutpb`).
- Config: `rl.actions.allow_probability_scaling` toggles this behavior.

## Reward Function

```python
reward = (
    weights.fitness * clamp(delta_fitness, -1.0, 1.0)
    + weights.diversity * clamp(delta_diversity, -0.5, 0.5)
    - weights.time * normalized_wall_clock
)
```

Default weights (from `configs/base.yaml`):
- `fitness = 0.7`
- `diversity = 0.2`
- `time = 0.1`

Reward is computed every generation; PPO uses GAE(λ=0.95).

## Training Defaults

| Parameter | Value |
| --- | --- |
| Algorithm | PPO (SB3) |
| Policy | MLP (3 layers × 256 units) |
| Learning Rate | 3e-4 |
| Batch Size | 4096 |
| n_steps | 2048 |
| Entropy Coeff | 0.01 |
| Clip Range | 0.2 |
| VecEnvs | 16 (requires 16 CPU cores) |

See `configs/training/ppo_default.yaml` for overrides.

## Safety Mechanisms

- **Timeout:** inference limited to 20ms; fallback triggers round-robin heuristic.
- **Checksum:** model manifest stores SHA256; loader verifies before use.
- **Versioning:** `models/rl_agents/manifest.json` links model to git commit + config hash.
- **Disable switch:** `rl.enabled = false` bypasses RL entirely for regression testing.

## Testing

- `test/rl/test_state_encoder.py` – ensures deterministic observation output.
- `test/rl/test_action_mapper.py` – validates ID ↔ heuristic mapping.
- `test/rl/test_reward_calculator.py` – checks reward scaling.
- `test/rl/test_rl_integration.py` – end-to-end step with mocked GA.

## References

- See [How-To: Training RL Agents](../how-to/03-training-rl-agents.md).
- Research background in `docs/research-papers/00-paper-index.md` (PPO, curriculum learning).
