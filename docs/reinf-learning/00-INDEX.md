# Reinforcement Learning Documentation

Complete guide to the RL-powered hyper-heuristic system for course timetabling optimization.

## Quick Links

### Getting Started
- [01-overview.md](01-overview.md) - Architecture and concepts
- [02-quickstart.md](02-quickstart.md) - Training your first RL agent
- [03-profiles.md](03-profiles.md) - Training profiles (test/med/prod)

### Core Components
- [04-environment.md](04-environment.md) - Gymnasium environment (`ScheduleEnv`)
- [05-state-representation.md](05-state-representation.md) - State vector design
- [06-action-space.md](06-action-space.md) - Heuristic operator toolbox
- [07-reward-function.md](07-reward-function.md) - Reward calculation and shaping

### Training System
- [08-trainer.md](08-trainer.md) - `RLTrainer` class and workflow
- [09-agents.md](09-agents.md) - PPO and DQN implementation
- [10-curriculum.md](10-curriculum.md) - Curriculum learning system
- [11-callbacks.md](11-callbacks.md) - Training callbacks and monitoring

### Advanced Features
- [12-multi-agent.md](12-multi-agent.md) - Multi-agent RL systems
- [13-adaptive-rewards.md](13-adaptive-rewards.md) - Dynamic reward shaping
- [14-specialist-agents.md](14-specialist-agents.md) - Constraint-specific experts

### Configuration & Deployment
- [15-configuration.md](15-configuration.md) - Config files and parameters
- [16-checkpoints.md](16-checkpoints.md) - Model saving and selection
- [17-deployment.md](17-deployment.md) - Production deployment guide

### Analysis & Debugging
- [18-tensorboard.md](18-tensorboard.md) - Monitoring with TensorBoard
- [19-visualization.md](19-visualization.md) - Training visualizations
- [20-troubleshooting.md](20-troubleshooting.md) - Common issues and fixes

## File Structure

```
src/rl/
├── __init__.py              # Public API
├── environment.py           # Gymnasium environment
├── state.py                 # State representation
├── reward.py                # Reward calculation
├── actions.py               # Action/heuristic definitions
├── hyper_heuristic_loop.py  # Main RL optimization loop
│
├── training/
│   ├── trainer.py           # RLTrainer class
│   ├── train_script.py      # CLI training entry point
│   ├── config_loader.py     # Training config management
│   ├── curriculum.py        # Curriculum learning
│   ├── callbacks.py         # Custom SB3 callbacks
│   └── checkpoints.py       # Checkpoint utilities
│
├── agents/
│   ├── __init__.py          # Agent factory functions
│   ├── ppo_agent.py         # PPO configuration
│   └── dqn_agent.py         # DQN configuration
│
├── policies/
│   ├── probability_policy.py # Adaptive probability selection
│   └── credit_assignment.py  # Credit assignment strategies
│
├── rewards/
│   ├── base_reward.py       # Base reward calculator
│   └── adaptive_reward.py   # Adaptive reward shaping
│
├── multi_agent/
│   ├── specialist_agents.py # Constraint-specific agents
│   ├── rank_based_agents.py # Rank-based coordination
│   └── agent_coordinator.py # Multi-agent orchestration
│
└── visualization/
    ├── training_plots.py    # Training progress plots
    └── heatmaps.py          # Action selection heatmaps
```

## Training Commands

```bash
# Quick smoke test (2-3 min)
uv run train-rl --test

# Medium training (30-45 min)
uv run train-rl --med

# Production training (1-2 hours)
uv run train-rl --prod

# With curriculum learning
uv run train-rl --prod --curriculum

# Interactive launcher
uv run launcher  # Select option 4-6 for RL
```

## Key Concepts

### Hyper-Heuristic Approach
Instead of hand-coding operator selection logic, we train an RL agent to learn **which operator to apply when** based on current state.

### Gymnasium Integration
Standard OpenAI Gym/Gymnasium interface for clean RL agent integration with Stable-Baselines3.

### Multi-Objective Optimization
RL agent learns to balance hard constraints (feasibility) and soft penalties (quality) in NSGA-II framework.

### Curriculum Learning
Progressive difficulty stages (easy → medium → hard) for stable training convergence.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│              RL Training Loop (PPO/DQN)             │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│          ScheduleEnv (Gymnasium Interface)          │
│  ┌───────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │   State   │  │  Action  │  │     Reward      │  │
│  │  (5D vec) │  │ (19 ops) │  │  (improvement)  │  │
│  └───────────┘  └──────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│            GA Scheduler (NSGA-II Core)              │
│  - Population management                            │
│  - Fitness evaluation                               │
│  - Metrics tracking (every 10 gens)                 │
└─────────────────────────────────────────────────────┘
```

## Performance

| Profile | Timesteps | Envs | Time | Model Quality |
|---------|-----------|------|------|---------------|
| Test    | 500       | 1    | ~3m  | Verification  |
| Med     | 100K      | 4    | ~45m | Experiments   |
| Prod    | 300K      | 8    | ~2h  | **Thesis**    |

## Next Steps

1. **Read [01-overview.md](01-overview.md)** - Understand the system architecture
2. **Try [02-quickstart.md](02-quickstart.md)** - Run your first training
3. **Explore [04-environment.md](04-environment.md)** - Deep dive into core components
4. **Configure [15-configuration.md](15-configuration.md)** - Customize for your needs

## Support

- **Issues**: See [20-troubleshooting.md](20-troubleshooting.md)
- **CLI Reference**: `../CLI_REFERENCE.md`
- **Instructions**: `../.github/instructions/rl.instructions.md`
