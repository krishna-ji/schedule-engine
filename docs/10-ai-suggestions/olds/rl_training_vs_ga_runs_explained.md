# RL Training vs GA Runs: Complete Guide

**Your Confusion Clarified**: Running GA 30 times ≠ 30 RL models ≠ RL training

This guide explains the difference between:
1. **RL Training** (learning phase, happens once)
2. **GA Execution** (production usage, happens many times)
3. **Curriculum Learning** (smart training strategy)

---

## Part 1: The Fundamental Difference

### Scenario A: Traditional GA (Current System, No RL)

```
Run 1: GA with random heuristic selection → Schedule with fitness -150
Run 2: GA with random heuristic selection → Schedule with fitness -145
Run 3: GA with random heuristic selection → Schedule with fitness -160
...
Run 30: GA with random heuristic selection → Schedule with fitness -155

Result: 30 different schedules, NO learning across runs
Each run is independent, no improvement in heuristic selection
```

### Scenario B: RL Training Phase (One-Time, Learning)

```
RL Training (Single Session, 30-60 minutes):
┌────────────────────────────────────────────────────────┐
│ Episode 1: Run mini-GA (100 gens)                      │
│   Agent picks random actions, gets rewards             │
│   Store experiences, update neural network             │
│                                                         │
│ Episode 2: Run mini-GA (100 gens)                      │
│   Agent slightly smarter, better actions               │
│   Store experiences, update neural network             │
│                                                         │
│ Episode 3-500: Keep learning...                        │
│   Agent gets progressively smarter                     │
│                                                         │
│ Result: ONE trained model (schedule_ppo_final.zip)     │
└────────────────────────────────────────────────────────┘

NOT 30 models - just ONE smart model!
```

### Scenario C: GA with Trained RL Model (Production)

```
Run 1: GA with RL agent → Schedule with fitness -120 ✓ (better!)
Run 2: GA with RL agent → Schedule with fitness -118 ✓
Run 3: GA with RL agent → Schedule with fitness -122 ✓
...
Run 30: GA with RL agent → Schedule with fitness -115 ✓

Result: 30 different schedules, ALL using SAME trained RL model
Each run uses smart heuristic selection (no more learning)
```

---

## Part 2: What is RL Training Really?

### The Training Process (Simplified)

```python
# RL Training = Agent learns by running GA MANY times internally

def rl_training():
    """
    ONE training session creates ONE model.
    Inside this session, agent runs mini-GA hundreds of times.
    """
    agent = PPO("MlpPolicy", env)  # Random brain
    
    # Training loop (runs mini-GA 500+ times internally)
    for episode in range(500):  # 500 episodes, not 500 models!
        
        # Each episode = one mini-GA run (100 generations)
        obs = env.reset()  # Start fresh GA population
        episode_reward = 0
        
        for step in range(100):  # 100 GA generations
            # Agent picks heuristic
            action = agent.predict(obs)
            
            # Apply heuristic to population
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        # After episode: Update agent's brain
        agent.learn_from_episode()
        
        print(f"Episode {episode}: Reward = {episode_reward}")
    
    # After ALL episodes: Save ONE trained model
    agent.save("models/rl_agents/schedule_ppo_final.zip")
    
    return agent  # ONE model, trained on 500 mini-GA runs
```

**Key insight**: 
- **500 episodes** during training = agent practices on 500 mini-GA runs
- **Result**: 1 model (not 500 models!)
- **Model contains**: Learned policy for heuristic selection

---

## Part 3: Training vs Production Usage

### Training Phase (One-Time, Offline)

```
Purpose: Teach agent how to select heuristics
Duration: 30-60 minutes
Happens: Before deployment
Result: ONE trained model file

┌─────────────────────────────────────────────────────┐
│ RL Training Session                                 │
│ ───────────────────────────────────────────────     │
│                                                      │
│ Agent runs mini-GA 500 times internally:            │
│                                                      │
│ Episode 1:   Random actions → Fitness -180          │
│ Episode 2:   Random actions → Fitness -175          │
│ Episode 10:  Learning... → Fitness -165             │
│ Episode 50:  Getting better → Fitness -145          │
│ Episode 100: Smarter → Fitness -135                 │
│ Episode 200: Even smarter → Fitness -125            │
│ Episode 500: Expert level → Fitness -118            │
│                                                      │
│ Save trained model: schedule_ppo_final.zip          │
└─────────────────────────────────────────────────────┘

Output: 1 model file (10-20 MB)
        Contains: Neural network weights encoding learned policy
```

### Production Phase (Many Times, Fast)

```
Purpose: Use trained agent to solve scheduling problems
Duration: 2-48 hours per GA run
Happens: After deployment
Result: 30 schedules (using SAME model)

┌─────────────────────────────────────────────────────┐
│ Production: Run GA 30 times with trained RL agent  │
│ ───────────────────────────────────────────────────  │
│                                                      │
│ Load model ONCE: schedule_ppo_final.zip             │
│                                                      │
│ Run 1:  GA (2000 gens) with RL → Fitness -120      │
│         Model picks smart heuristics throughout     │
│                                                      │
│ Run 2:  GA (2000 gens) with RL → Fitness -118      │
│         SAME model, different random seed           │
│                                                      │
│ Run 3:  GA (2000 gens) with RL → Fitness -122      │
│         SAME model, different initial population    │
│                                                      │
│ ... (Runs 4-30 use SAME trained model)             │
│                                                      │
│ Run 30: GA (2000 gens) with RL → Fitness -115      │
│         SAME model, consistently good               │
└─────────────────────────────────────────────────────┘

Output: 30 schedules (30 JSON files)
        All use SAME trained model for heuristic selection
```

---

## Part 4: Curriculum Learning Explained

### What is Curriculum Learning?

**Analogy**: Learning math
-  Bad: Start with calculus (too hard, give up)
-  Good: Start with addition → multiplication → algebra → calculus

**In RL**: Train on easy problems first, gradually increase difficulty

### Why Use Curriculum Learning?

```
Without Curriculum (Flat Training):
┌────────────────────────────────────────────────────┐
│ Episode 1-500: Train on hard problems (40 courses) │
│                                                     │
│ Result: Agent struggles, learns slowly             │
│         Final performance: -135 fitness            │
│         Takes 60 minutes                           │
└────────────────────────────────────────────────────┘

With Curriculum (Progressive Training):
┌────────────────────────────────────────────────────┐
│ Stage 1 (Episodes 1-100): Easy (10 courses)        │
│   Agent learns basic patterns quickly              │
│                                                     │
│ Stage 2 (Episodes 101-200): Medium (20 courses)    │
│   Agent applies learned patterns to harder problems│
│                                                     │
│ Stage 3 (Episodes 201-500): Hard (40 courses)      │
│   Agent masters complex scheduling                 │
│                                                     │
│ Result: Agent learns faster, better generalization │
│         Final performance: -120 fitness (better!)  │
│         Takes 60 minutes (same time)               │
└────────────────────────────────────────────────────┘
```

### Curriculum Learning Implementation

#### Stage 1: Easy Problems (10 courses)

```python
# Easy stage: Fewer courses, fewer constraints
easy_context = filter_context(
    num_courses=10,      # Small problem
    num_rooms=5,
    num_instructors=8
)

# Train agent on easy problems
agent.learn(
    env=ScheduleEnv(easy_context),
    total_timesteps=20000  # 20K steps
)

# Agent learns: "Basic heuristics work, understand environment"
```

#### Stage 2: Medium Problems (20 courses)

```python
# Medium stage: More courses, more constraints
medium_context = filter_context(
    num_courses=20,      # Moderate problem
    num_rooms=10,
    num_instructors=15
)

# Continue training on medium problems (same agent!)
agent.learn(
    env=ScheduleEnv(medium_context),
    total_timesteps=30000  # Additional 30K steps
)

# Agent learns: "Apply basics to harder problems, refine strategy"
```

#### Stage 3: Hard Problems (40 courses)

```python
# Hard stage: Full complexity
hard_context = load_full_context("data/")  # All courses

# Continue training on hard problems (same agent!)
agent.learn(
    env=ScheduleEnv(hard_context),
    total_timesteps=50000  # Additional 50K steps
)

# Agent learns: "Master complex scheduling, expert level"

# Save final model (trained across all stages)
agent.save("models/rl_agents/schedule_ppo_curriculum.zip")
```

### Curriculum Parameters to Vary

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| **Courses** | 10 | 20 | 40 (full dataset) |
| **Rooms** | 5 | 10 | 20 (full dataset) |
| **Instructors** | 8 | 15 | 30 (full dataset) |
| **GA Generations** | 50 | 100 | 200 (per episode) |
| **Constraint Density** | Loose | Moderate | Tight (full constraints) |
| **Training Steps** | 20K | 30K | 50K |

---

## Part 5: Complete Training Architecture

### Full Training Pipeline

```
Training Phase (Happens ONCE):
════════════════════════════════════════════════════════

Step 1: Curriculum Stage 1 (Easy)
┌──────────────────────────────────────────────────────┐
│ Create easy environment (10 courses)                 │
│ Agent runs 100 mini-GA episodes                      │
│ Total: 20,000 timesteps                              │
│ Duration: ~10 minutes                                │
│ Checkpoint: models/stage_easy.zip                    │
└──────────────────────────────────────────────────────┘
                        ↓
Step 2: Curriculum Stage 2 (Medium)
┌──────────────────────────────────────────────────────┐
│ Create medium environment (20 courses)               │
│ Load checkpoint from Stage 1                         │
│ Agent runs 150 mini-GA episodes                      │
│ Total: 30,000 timesteps                              │
│ Duration: ~15 minutes                                │
│ Checkpoint: models/stage_medium.zip                  │
└──────────────────────────────────────────────────────┘
                        ↓
Step 3: Curriculum Stage 3 (Hard)
┌──────────────────────────────────────────────────────┐
│ Create hard environment (40 courses, full dataset)   │
│ Load checkpoint from Stage 2                         │
│ Agent runs 250 mini-GA episodes                      │
│ Total: 50,000 timesteps                              │
│ Duration: ~25 minutes                                │
│ Final model: models/schedule_ppo_final.zip           │
└──────────────────────────────────────────────────────┘
                        ↓
        TRAINING COMPLETE (Total: 50 min)
                        ↓
┌──────────────────────────────────────────────────────┐
│ Result: ONE trained model                            │
│ File: models/rl_agents/schedule_ppo_final.zip        │
│ Size: 15 MB                                          │
│ Contains: Neural network trained on 500 mini-GAs     │
└──────────────────────────────────────────────────────┘
```

### Production Phase (Happens MANY times)

```
Production Usage (30 Independent GA Runs):
════════════════════════════════════════════════════════

Load trained model ONCE:
┌──────────────────────────────────────────────────────┐
│ model = PPO.load("models/schedule_ppo_final.zip")   │
│ Duration: 50ms                                       │
└──────────────────────────────────────────────────────┘
                        ↓
Run GA 30 times with trained model:
┌──────────────────────────────────────────────────────┐
│ Run 1:  GA(2000 gens) → Fitness -120  (24 hours)    │
│ Run 2:  GA(2000 gens) → Fitness -118  (24 hours)    │
│ Run 3:  GA(2000 gens) → Fitness -122  (24 hours)    │
│ ...                                                   │
│ Run 30: GA(2000 gens) → Fitness -115  (24 hours)    │
│                                                       │
│ All runs use SAME model for heuristic selection     │
│ Model picks smart actions consistently              │
└──────────────────────────────────────────────────────┘
                        ↓
Result: 30 schedules, statistical analysis
        Mean fitness: -119.5 (± 3.2)
        All use SAME trained RL model
```

---

## Part 6: Key Concepts Clarified

### Concept 1: Episodes vs Models

```python
# WRONG UNDERSTANDING:
"500 training episodes = 500 models"  

# CORRECT UNDERSTANDING:
"500 training episodes = 1 model trained 500 times"  ✓
```

**Analogy**: 
- 500 episodes = 500 practice sessions
- 1 model = 1 brain that learns from all 500 sessions

### Concept 2: Training vs Testing

| Phase | Purpose | Duration | Output | Model State |
|-------|---------|----------|--------|-------------|
| **Training** | Learn policy | 30-60 min | 1 model file | Weights updating |
| **Testing** | Evaluate policy | 5-10 min | Performance metrics | Weights frozen |
| **Production** | Use policy | 24 hours/run | 1 schedule | Weights frozen |

### Concept 3: When Model Changes

```
Model Changes (Weights Update):
✓ During RL training (every 2048 steps)
✓ During fine-tuning (if you retrain later)

Model Doesn't Change (Weights Frozen):
✓ During testing/evaluation
✓ During production GA runs
✓ When running GA 30 times for statistical analysis
```

---

## Part 7: Practical Implementation Guide

### Training Script with Curriculum

```python
"""
Complete RL training with curriculum learning.
File: src/rl/training/train_with_curriculum.py
"""
from pathlib import Path
from stable_baselines3 import PPO
from rich.console import Console

from src.rl.gym_env import ScheduleEnv
from src.workflows.standard_run import load_input_data
from src.core.types import create_initial_population

console = Console()


def filter_courses_by_count(context, num_courses: int):
    """Filter context to include only N courses."""
    filtered_courses = context.courses[:num_courses]
    # Filter related instructors and groups
    return create_filtered_context(filtered_courses, context)


def train_curriculum():
    """Train agent using curriculum learning."""
    
    console.print("[bold] RL Training with Curriculum Learning[/bold]\n")
    
    # Load full dataset
    qts, full_context = load_input_data("data")
    
    # Stage 1: Easy (10 courses)
    console.print("[bold cyan]Stage 1: Easy (10 courses)[/bold cyan]")
    easy_context = filter_courses_by_count(full_context, 10)
    easy_pop = create_initial_population(easy_context, pop_size=30)
    easy_env = ScheduleEnv(
        initial_population=easy_pop,
        context=easy_context,
        max_generations=50  # Shorter episodes
    )
    
    # Create agent
    agent = PPO("MlpPolicy", easy_env, verbose=1)
    
    # Train on easy problems
    console.print("Training on easy problems (20K steps)...")
    agent.learn(total_timesteps=20000)
    agent.save("models/rl_agents/checkpoints/stage_easy.zip")
    console.print("✓ Stage 1 complete\n")
    
    # Stage 2: Medium (20 courses)
    console.print("[bold yellow]Stage 2: Medium (20 courses)[/bold yellow]")
    medium_context = filter_courses_by_count(full_context, 20)
    medium_pop = create_initial_population(medium_context, pop_size=40)
    medium_env = ScheduleEnv(
        initial_population=medium_pop,
        context=medium_context,
        max_generations=100
    )
    
    # Continue training (same agent!)
    agent.set_env(medium_env)  # Update environment, keep weights
    console.print("Training on medium problems (30K steps)...")
    agent.learn(total_timesteps=30000)
    agent.save("models/rl_agents/checkpoints/stage_medium.zip")
    console.print("✓ Stage 2 complete\n")
    
    # Stage 3: Hard (40 courses - full dataset)
    console.print("[bold red]Stage 3: Hard (Full dataset)[/bold red]")
    hard_pop = create_initial_population(full_context, pop_size=50)
    hard_env = ScheduleEnv(
        initial_population=hard_pop,
        context=full_context,
        max_generations=200
    )
    
    # Final training stage
    agent.set_env(hard_env)
    console.print("Training on hard problems (50K steps)...")
    agent.learn(total_timesteps=50000)
    
    # Save final model
    final_path = "models/rl_agents/schedule_ppo_curriculum.zip"
    agent.save(final_path)
    console.print(f"\n [bold green]Training complete![/bold green]")
    console.print(f"Model saved: {final_path}")
    
    return agent


if __name__ == "__main__":
    train_curriculum()
```

### Production Usage (30 GA Runs)

```python
"""
Run GA 30 times with trained RL model.
File: src/evaluation/run_30_ga_trials.py
"""
from stable_baselines3 import PPO
from src.core.ga_scheduler import GAScheduler
from src.workflows.standard_run import load_input_data

def run_30_trials():
    """Run GA 30 times using SAME trained RL model."""
    
    # Load trained model ONCE
    trained_model = PPO.load("models/rl_agents/schedule_ppo_final.zip")
    
    # Load problem data
    qts, context = load_input_data("data")
    
    results = []
    
    for trial in range(30):
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/30")
        print(f"{'='*60}")
        
        # Create GA scheduler with RL enabled
        scheduler = GAScheduler(
            context=context,
            use_rl=True,
            rl_model=trained_model  # SAME model for all trials
        )
        
        # Run GA (2000 generations)
        best_solution = scheduler.run()
        
        # Record results
        results.append({
            'trial': trial + 1,
            'fitness': best_solution.fitness.values[0],
            'hard_violations': best_solution.fitness.values[0],  # Assuming multi-objective
            'convergence_gen': scheduler.convergence_generation
        })
        
        print(f"Fitness: {best_solution.fitness.values[0]}")
    
    # Statistical summary
    import numpy as np
    fitnesses = [r['fitness'] for r in results]
    print(f"\n{'='*60}")
    print(f"SUMMARY (30 trials with SAME RL model)")
    print(f"{'='*60}")
    print(f"Mean fitness: {np.mean(fitnesses):.2f}")
    print(f"Std dev: {np.std(fitnesses):.2f}")
    print(f"Best: {np.min(fitnesses):.2f}")
    print(f"Worst: {np.max(fitnesses):.2f}")
    
    return results


if __name__ == "__main__":
    run_30_trials()
```

---

## Part 8: Summary & Answers to Your Questions

### Q1: "If I run GA 30 times, do I have 30 models?"

**Answer**:  NO - You have 1 model, used 30 times

```
Training (once):    → 1 model created
Production (30x):   → 1 model used 30 times → 30 schedules
```

### Q2: "How does the RL model learn?"

**Answer**: By running mini-GA hundreds of times DURING TRAINING

```
Training session:
- Agent runs mini-GA 500 times internally
- Each mini-GA run = 1 episode = 100 generations
- Agent tries different heuristics, sees rewards
- Neural network weights update after each episode
- Result: ONE smart model
```

### Q3: "What is curriculum learning?"

**Answer**: Progressive difficulty training (easy → medium → hard)

```
Stage 1: Train on 10 courses  (easy)    → Agent learns basics
Stage 2: Train on 20 courses  (medium)  → Agent improves
Stage 3: Train on 40 courses  (hard)    → Agent masters it

Result: ONE model, better generalization than flat training
```

### Q4: "What to vary in curriculum?"

**Answer**: Multiple options:

| Parameter | Easy Stage | Medium Stage | Hard Stage |
|-----------|-----------|--------------|-----------|
| **Courses** | 10 | 20 | 40 |
| **Generations/episode** | 50 | 100 | 200 |
| **Population size** | 30 | 40 | 50 |
| **Constraint tightness** | Loose | Moderate | Full |
| **Training steps** | 20K | 30K | 50K |

**Recommended**: Start with varying **number of courses** only (simplest)

### Q5: "What is this aspect called?"

**Answer**: Several terms apply:

- **RL Training** = Learning phase (agent practices)
- **Curriculum Learning** = Progressive difficulty training strategy
- **Transfer Learning** = Knowledge transfers from easy → hard
- **Episode** = One mini-GA run during training
- **Model** = Trained neural network (policy)
- **Inference** = Using trained model in production

---

## Part 9: Quick Reference

### Training Lifecycle

```
┌────────────────────────────────────────────────────┐
│ TRAINING PHASE (Once, 30-60 min)                  │
├────────────────────────────────────────────────────┤
│ Input:  Environment + untrained agent             │
│ Process: Agent runs 500 mini-GAs internally       │
│ Output: 1 trained model (schedule_ppo_final.zip)  │
└────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────┐
│ PRODUCTION PHASE (Many times, 24h each)           │
├────────────────────────────────────────────────────┤
│ Input:  1 trained model (loaded once)             │
│ Process: Run GA 30 times using model              │
│ Output: 30 schedules (30 JSON files)              │
└────────────────────────────────────────────────────┘
```

### File Structure After Training

```
models/rl_agents/
├── schedule_ppo_final.zip          # Main trained model (USE THIS)
├── schedule_dqn_final.zip          # Alternative algorithm
└── checkpoints/                    # Intermediate saves
    ├── stage_easy.zip              # Checkpoint after easy training
    ├── stage_medium.zip            # Checkpoint after medium training
    └── stage_hard.zip              # Checkpoint after hard training

output/
├── ga_run_01_result.json          # GA run 1 using trained model
├── ga_run_02_result.json          # GA run 2 using trained model
├── ...
└── ga_run_30_result.json          # GA run 30 using trained model
```

### Commands Summary

```bash
# Train RL model with curriculum (DO ONCE)
uv run python src/rl/training/train_with_curriculum.py
# Output: models/rl_agents/schedule_ppo_curriculum.zip

# Run GA 30 times with trained model (DO MANY TIMES)
uv run python src/evaluation/run_30_ga_trials.py
# Output: 30 schedule files, all using SAME model

# View training progress
tensorboard --logdir logs/tensorboard
```

---

## Conclusion

**Key Takeaways**:

1.  **Training produces 1 model** (not 30 models)
2.  **Model learns by running mini-GA 500+ times internally**
3.  **Production uses that 1 model for all 30 GA runs**
4.  **Curriculum learning = easy→medium→hard progressive training**
5.  **Vary course count in curriculum** (10→20→40 courses)
6.  **Training = learn phase** | **Production = use phase**

**Next step**: Implement training script and run your first RL training session! 



#  2nd prmpt answer:
Optimizing tool selection...## Production-Ready Model Creation

### What is an Episode?

**1 Episode = 1 Complete GA Run (Start → Finish)**

```
Episode Timeline:
├── Initialize population (generation 0)
├── Generation 1: RL agent makes decisions → population evolves
├── Generation 2: RL agent makes decisions → population evolves
├── ...
├── Generation 200: Final population
└── Episode Reward: Based on final schedule quality
```

**During each episode:**
- RL agent experiences a full scheduling problem
- Makes hundreds of decisions (mutation selection, crossover, repair strategies)
- Gets feedback on whether those decisions led to good/bad schedules
- This experience is stored and used to update the neural network

### Making a Production-Ready Model

**Training Process (One-Time, ~30-60 minutes):**

```python
# Step 1: Train with Curriculum Learning (3 stages)
# ================================================

# Stage 1: Easy Problems (10 courses)
for episode in range(200):  # 200 episodes
    # Run 1 complete GA with 10 courses
    # RL agent learns basic scheduling patterns
    # Takes ~0.1-0.2 seconds per episode
    pass
# → Checkpoint: model_stage1.zip (basic knowledge)

# Stage 2: Medium Problems (20 courses)
for episode in range(300):  # 300 episodes
    # Run 1 complete GA with 20 courses
    # RL agent learns complex constraint handling
    # Takes ~0.2-0.4 seconds per episode
    pass
# → Checkpoint: model_stage2.zip (intermediate knowledge)

# Stage 3: Hard Problems (40 courses - your real data)
for episode in range(500):  # 500 episodes
    # Run 1 complete GA with 40 courses
    # RL agent masters real-world complexity
    # Takes ~0.4-0.8 seconds per episode
    pass
# → Final Model: ppo_scheduler_final.zip (production-ready)

# Total: 1000 episodes = 1000 GA runs during training
# Time: 200×0.15s + 300×0.3s + 500×0.6s = 30s + 90s + 300s = 420s ≈ 7 minutes
```

**What Makes It Production-Ready?**

After 1000 training episodes, the model has:
1. **Experienced 1000 different scheduling scenarios**
2. **Learned which decisions lead to good schedules**
3. **Neural network weights optimized** through gradient descent
4. **Validation performance above threshold** (e.g., better than random baseline)

### Production Model Structure

```
ppo_scheduler_final.zip (10-20 MB)
├── policy.pth                  # Neural network weights (main file)
├── policy.optimizer.pth        # Training state (not used in production)
├── pytorch_variables.pth       # Normalization stats
└── metadata.json              # Hyperparameters, architecture info
```

### Using Production Model (Many Times)

```python
# Load model ONCE (100ms)
model = PPO.load("ppo_scheduler_final.zip")

# Use it 30 times for 30 different schedules
for trial in range(30):
    result = run_ga_with_rl_model(
        model=model,              # Same model, frozen weights
        generations=2000,         # Full production GA
        data=your_courses         # Your actual data
    )
    # Each run: 5-10 minutes
    # Model just helps make better decisions
    # Weights DON'T change during these runs
```

### Key Differences

| Aspect | Training | Production |
|--------|----------|------------|
| **Purpose** | Teach the model | Use the model |
| **Episodes/Runs** | 1000 episodes | 30 GA runs |
| **GA Length** | Short (100-200 gens) | Full (2000 gens) |
| **Time per Run** | 0.1-0.8 seconds | 5-10 minutes |
| **Total Time** | 7-60 minutes (one-time) | 2.5-5 hours (reusable) |
| **Model Changes** | Weights update constantly | Weights frozen |
| **Output** | 1 trained model file | 30 different schedules |

### Episode Breakdown Example

```
Training Episode #427 (one of 1000):
─────────────────────────────────────
Problem: Schedule 40 courses
Duration: 0.6 seconds
GA Generations: 200

Generation 0: [Agent: "Use uniform_crossover"] → Reward: -5.2
Generation 1: [Agent: "Use swap_mutation"] → Reward: -4.8
Generation 2: [Agent: "Use greedy_repair"] → Reward: -4.1
...
Generation 200: Final fitness = -2.3

Episode Reward: -2.3 (good! agent learns this sequence worked)
Network Update: Gradient descent updates weights based on this experience
```

### Summary

**To make a production-ready model:**
1. **Train for 1000 episodes** (1000 mini-GA runs, ~7-60 min total)
2. **Use curriculum learning** (easy→medium→hard)
3. **Validate performance** (better than baselines)
4. **Save final model** (ppo_scheduler_final.zip)

**That ONE model is then used:**
- 30 times to generate 30 schedules (production)
- 100 times for research experiments
- Any number of times without retraining

**Episode = 1 learning experience** where the agent runs a complete GA and learns from the outcome.