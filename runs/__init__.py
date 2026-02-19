"""
Schedule Engine — Experiment Runners

All experiments use the ``src.experiments`` runner classes.
Each script declares configuration at the top, creates an experiment
instance, and calls ``exp.run()``.

GA Experiments (pymoo NSGA-II, progressive complexity):
    python runs/ga_01_baseline.py           # Pure NSGA-II
    python runs/ga_02_memetic.py            # + Elite bitset repair
    python runs/ga_03_aggressive.py         # 2x offspring, full-pop repair
    python runs/ga_04_adaptive.py           # Stagnation-aware escalation
    python runs/ga_05_cp_hybrid.py          # + CP-SAT deep polish

RL Training:
    python runs/rl_01_train_ppo.py          # PPO baseline
    python runs/rl_02_train_dqn.py          # DQN baseline
    python runs/rl_03_train_curriculum.py   # Curriculum learning
    python runs/rl_04_train_specialist.py   # Specialist agents

RL Analysis:
    python runs/rl_05_compare_rewards.py    # Reward function comparison
    python runs/rl_06_adaptive_params.py    # Fixed vs adaptive params
    python runs/rl_07_ablation.py           # Method comparison
    python runs/rl_08_hyperparam_sweep.py   # LR sensitivity
    python runs/rl_09_multi_agent.py        # Agent coordination
    python runs/rl_10_verify.py             # Component check

Utilities:
    python runs/pre_scheduling_audit.py     # Data validation audit
"""
