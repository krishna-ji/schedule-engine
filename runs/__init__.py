"""
Schedule Engine - Standalone Experiment Runners

Migrated from notebooks/ to standalone Python scripts with:
- Full logging to dedicated log files
- Automatic figure exports
- JSON results export
- Timestamped output directories

GA Experiments (progressive complexity):
    python runs/ga_01_baseline.py             # Pure NSGA-II (no repair)
    python runs/ga_02_memetic.py              # + Local Search
    python runs/ga_03_repair_sequential.py    # + Round-robin repairs
    python runs/ga_04_repair_bandit.py        # + UCB/epsilon-greedy
    python runs/ga_05_repair_qlearning.py     # + Q-learning selection

RL Training:
    python runs/rl_01_train_ppo.py            # Train PPO agent
    python runs/rl_02_train_dqn.py            # Train DQN agent
    python runs/rl_03_train_curriculum.py     # Curriculum learning
    python runs/rl_04_train_specialist.py     # Specialist agents

RL Analysis:
    python runs/rl_05_compare_rewards.py      # Reward function comparison
    python runs/rl_06_adaptive_params.py      # Learn GA parameters
    python runs/rl_07_ablation.py             # Method comparison
    python runs/rl_08_hyperparam_sweep.py     # LR sensitivity
    python runs/rl_09_multi_agent.py          # Agent dynamics
    python runs/rl_10_verify.py               # Component check
"""
