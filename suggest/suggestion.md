# Hyper-Heuristic Timetabling: Comprehensive Technical Blueprint

**Project:** HYPERHEURISTIC OPTIMIZATION OF UNIVERSITY COURSE TIMETABLING PROBLEM USING RL AND GENETIC ALGORITHMS
**Version:** 3.0 - Technical Implementation Guide
**Date:** 2025-11-14

---

## Executive Summary

Your project has reached a critical juncture. The current NSGA-II implementation is mature and effective, but the **failed CP-SAT experiment** (19.7M constraints, intractable) has revealed a crucial insight: **no single, rigid algorithm can optimally solve this problem at scale**. A more powerful, flexible heuristic approach is needed for high-intensity repair.

**The Path Forward:** Evolve from a pure metaheuristic to a **Reinforcement Learning-based Hyper-Heuristic** that intelligently orchestrates multiple optimization techniques. This approach is:
- **Academically Novel:** Perfectly aligned with your thesis title
- **Technically Sound:** Leverages your existing GA and IGLS while adding adaptive intelligence
- **State-of-the-Art:** Represents current best practices in combinatorial optimization

---

## 1. Current State Analysis

### What You Have (Strengths)
✅ **Robust GA Implementation:** NSGA-II with sophisticated operators
✅ **Modular Architecture:** Clean separation of concerns (`src/constraints`, `src/ga`, `src/core`)
✅ **Repair Heuristics:** IGLS system with exhaustive search and stagnation handling
✅ **Comprehensive Evaluation:** Multi-objective fitness with hard/soft constraints
✅ **Flexible Configuration:** YAML-based parameter management

### What You Learned (Critical Insight)
❌ **Pure CP-SAT Fails at Scale:** 239 courses → 19.7M constraints (115x expected)
✅ **CP-SAT Works on Small Problems:** The key to hybrid success
✅ **Low Resource Utilization (15.7%):** Problem is vast but loosely constrained → ideal for heuristics

### What's Missing (The Gap)
🔲 **Adaptive Strategy Selection:** Currently, operators are applied with fixed probabilities
🔲 **Problem-Aware Optimization:** The system doesn't "learn" which techniques work when
🔲 **Hybrid Approach:** The powerful IGLS solver exists in isolation, not integrated as a high-intensity repair operator.

---

## 2. Proposed Architecture: RL-Powered Hyper-Heuristic

### The Core Concept: Three Layers

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: RL Agent (The "Brain")                        │
│  • Observes: Current schedule state                     │
│  • Decides: Which heuristic to apply next               │
│  • Learns: Optimal strategy through experience          │
└─────────────────────────────────────────────────────────┘
                           ↓ selects action
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: Heuristic Toolbox (The "Operators")          │
│  • Low-Intensity: mutate_time, mutate_room              │
│  • Medium-Intensity: crossover, LNS_random              │
│  • High-Intensity: LNS_conflicted                       │
│  • Surgical: LNS_IGLS_Repair (uses IGLS on subproblems)│
└─────────────────────────────────────────────────────────┘
                           ↓ modifies
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Timetabling Environment (The "Problem")       │
│  • Current Schedule: Individual (chromosome)            │
│  • Fitness Function: Hard/soft constraint evaluation    │
│  • State Representation: Feature vector for RL          │
└─────────────────────────────────────────────────────────┘
```

### Key Innovation: The RL Agent as "Conductor"

The RL agent doesn't solve the timetabling problem directly. Instead, it learns **which heuristic to apply when**, based on the current state of the schedule.

**Example Learned Behaviors:**
- *"When hard violations are high → use expensive LNS_IGLS_Repair"*
- *"When solution is good but stagnant → use LNS_random for exploration"*
- *"When near optimal → use cheap mutations for fine-tuning"*

---

## 3. Technical Specification

### 3.1 State Representation (What the RL Agent "Sees")

The state must be a **fixed-size numerical vector** that captures the schedule's current condition.

**Proposed State Vector (5 dimensions):**

```python
S = [
    norm_hard_violations,      # [0.0-1.0] Hard violations / max_possible_conflicts
    norm_soft_violations,      # [0.0-1.0] Soft violations / theoretical_max
    fitness_delta,             # [-inf, +inf] Change from previous iteration
    norm_stagnation,           # [0.0-1.0] Iterations_since_improvement / 100
    progress                   # [0.0-1.0] Current_iteration / max_iterations
]
```

**Rationale:**
- **Normalized values** ensure stability for neural network training
- **Stagnation tracking** allows the agent to detect when bold moves are needed
- **Progress indicator** enables different strategies for early vs. late optimization

### 3.2 Action Space (What the RL Agent Can Do)

Each action is a heuristic function that modifies the current schedule.

| Action ID | Function | Intensity | Computational Cost | Use Case |
|-----------|----------|-----------|-------------------|----------|
| 0 | `mutate_session_time()` | Low | ~1ms | Quick exploration, fine-tuning |
| 1 | `mutate_session_room()` | Low | ~1ms | Room constraint adjustments |
| 2 | `crossover_one_point()` | Medium | ~5ms | Combine good features from archive |
| 3 | `LNS_destroy_random_10pct()` | Medium | ~50ms | Escape local optima |
| 4 | `LNS_destroy_conflicted()` | High | ~100ms | Target hard violations |
| 5 | `LNS_IGLS_Repair()` | Very High | ~500ms-10s | **The surgical tool** - IGLS on subproblem |

**Action 5 Detail: LNS_IGLS_Repair**

This is the most powerful operator, leveraging your existing IGLS implementation in a more focused way:

1. **Destroy:** Extract only sessions involved in hard conflicts (e.g., 5-15 sessions).
2. **Repair:** Use your existing IGLS algorithm to intensively re-optimize this small subset within the context of the larger, fixed schedule.
3. **Reintegrate:** Merge the repaired sessions back into the main schedule.

*Why this works:* Instead of a rigid, external solver, you use your proven, flexible IGLS heuristic on a concentrated area, allowing for powerful, targeted improvements without the overhead and brittleness of a CP model.

### 3.3 Reward Function (What the RL Agent Optimizes)

A carefully shaped reward that balances improvement, cost, and strategic goals.

```python
def calculate_reward(old_fitness, new_fitness, action_cost, hard_fixed):
    # Component 1: Fitness improvement (primary driver)
    improvement = (old_fitness[0] + 0.01 * old_fitness[1]) - \
                  (new_fitness[0] + 0.01 * new_fitness[1])

    # Component 2: Cost penalty (teach efficiency)
    cost_penalty = action_cost * 0.1

    # Component 3: Hard constraint bonus (strategic incentive)
    hard_bonus = 10.0 if hard_fixed else 0.0

    return improvement - cost_penalty + hard_bonus
```

**Design Rationale:**
- **Positive reward** for improving fitness (moving toward better solutions)
- **Small penalty** for expensive actions (prevents overuse of LNS_IGLS_Repair)
- **Large bonus** for fixing hard constraints (prioritizes feasibility)

---

## 4. Detailed Algorithms

### Algorithm 1: Main RL-Driven Optimization Loop

```python
def RL_HyperHeuristic_Solve(
    initial_schedule: Individual,
    max_iterations: int = 2000
) -> Individual:
    """
    Main optimization loop with RL-based heuristic selection.

    Returns:
        Best schedule found during the optimization process.
    """
    # === INITIALIZATION ===
    env = TimetablingEnvironment(initial_schedule)
    rl_agent = DQNAgent(
        state_size=5,
        action_size=6,
        learning_rate=1e-4
    )
    archive = SolutionArchive(max_size=10)  # For crossover operations
    archive.add(initial_schedule)

    best_solution = initial_schedule
    best_fitness = env.evaluate()

    # === OPTIMIZATION LOOP ===
    for iteration in range(1, max_iterations + 1):
        # 1. Observe current state
        state = env.get_state_vector(iteration, max_iterations)

        # 2. RL agent selects action (epsilon-greedy)
        action_id = rl_agent.choose_action(state)
        heuristic = HEURISTIC_TOOLBOX[action_id]

        # 3. Apply heuristic to modify schedule
        old_fitness = env.get_fitness()

        if heuristic.needs_partner:  # e.g., crossover
            partner = archive.get_random()
            new_schedule = heuristic.apply(env.current, partner)
        else:
            new_schedule = heuristic.apply(env.current)

        env.update(new_schedule)
        new_fitness = env.get_fitness()

        # 4. Calculate reward
        hard_fixed = (old_fitness[0] > new_fitness[0])
        reward = calculate_reward(
            old_fitness,
            new_fitness,
            heuristic.cost,
            hard_fixed
        )

        # 5. RL agent learns from experience
        next_state = env.get_state_vector(iteration + 1, max_iterations)
        rl_agent.store_experience(state, action_id, reward, next_state)

        if iteration % 10 == 0:  # Train every 10 steps
            rl_agent.train_on_batch(batch_size=32)

        # 6. Update best solution and archive
        if new_fitness < best_fitness:
            best_solution = new_schedule
            best_fitness = new_fitness
            archive.add(new_schedule)

        # 7. Logging (every 50 iterations)
        if iteration % 50 == 0:
            log_progress(iteration, best_fitness, rl_agent.epsilon)

    return best_solution
```

### Algorithm 2: LNS with IGLS Repair (The "Surgical Tool")

```python
def LNS_IGLS_Repair(schedule: Individual) -> Individual:
    """
    Large Neighborhood Search with IGLS for repair.

    Key Insight: A powerful, focused heuristic (IGLS) is better for
    repairing small, complex subproblems than a rigid global solver.

    Steps:
        1. Identify sessions in hard conflicts.
        2. Remove them to create a partial schedule.
        3. Run a full IGLS optimization on just those removed sessions
           within the context of the fixed partial schedule.
        4. Reintegrate the repaired sessions.
    """
    # === DESTROY PHASE ===
    conflicted_sessions = find_hard_conflict_sessions(schedule)

    if not conflicted_sessions:
        return schedule  # Nothing to fix

    # Limit the scope to prevent excessive runtime
    if len(conflicted_sessions) > 20:
        conflicted_sessions = sorted(
            conflicted_sessions, key=lambda s: s.get_conflict_severity(), reverse=True
        )[:20]

    # Create partial schedule by removing conflicted sessions
    partial_schedule = schedule.remove(conflicted_sessions)

    # === REPAIR PHASE (IGLS) ===
    # Initialize a new IGLS run focused only on the subproblem.
    # The IGLS solver needs to be adapted to handle a fixed background schedule.
    igls_solver = IGLSSubproblemSolver(
        partial_schedule,
        conflicted_sessions,
        max_iterations=100, # Focused, intense search
        max_stagnation=30
    )
    
    repaired_schedule = igls_solver.solve()

    # === REINTEGRATION PHASE ===
    # The IGLS solver should return the best complete schedule it found.
    # If it failed to improve, we can just return the original.
    if repaired_schedule.fitness < schedule.fitness:
        return repaired_schedule
    else:
        return schedule
```

### Algorithm 3: DQN Agent (Neural Network Implementation)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    """Deep Q-Network agent for heuristic selection."""

    def __init__(self, state_size=5, action_size=6, learning_rate=1e-4):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor

        # Neural network (Q-function approximator)
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)  # Output: Q-value for each action
        )

        # Target network (for stable training)
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Experience replay buffer
        self.memory = deque(maxlen=10000)

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def store_experience(self, state, action, reward, next_state):
        """Add experience to replay buffer."""
        self.memory.append((state, action, reward, next_state))

    def train_on_batch(self, batch_size=32):
        """Train the network on a random batch from memory."""
        if len(self.memory) < batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Target Q-values (using target network)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q

        # Compute loss and update
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())
```

---

## 5. Phased Implementation Plan

### Phase 1: LNS-IGLS Hybrid (4-6 weeks)

**Goal:** Prove that integrating IGLS as a targeted repair tool improves the baseline GA.

**Tasks:**
1. ✅ Implement `find_hard_conflict_sessions()` to identify problematic sessions.
2. ✅ Adapt IGLS to work on subproblems (`IGLSSubproblemSolver`).
3. ✅ Implement `LNS_IGLS_Repair()` (Algorithm 2).
4. ✅ Integrate into GA: Call `LNS_IGLS_Repair` every 50 generations or when stagnant.
5. ✅ Benchmark: GA Baseline vs. GA+LNS/IGLS.

**Success Metric:** GA+LNS/IGLS finds feasible solutions faster or achieves lower hard violations.

**Deliverable:** Report showing comparative performance on your dataset.

### Phase 2: RL Environment (3-4 weeks)

**Goal:** Build the hyper-heuristic framework with a learning agent.

**Tasks:**
1. ✅ Create `TimetablingEnv` class (wraps schedule, exposes state/reward)
2. ✅ Implement state vector calculation
3. ✅ Implement reward function
4. ✅ Build heuristic toolbox (6 actions from Section 3.2)
5. ✅ Implement simple baseline: Random heuristic selector

**Success Metric:** Environment can execute any heuristic and return valid state/reward.

**Deliverable:** Working environment that can be used by any policy (random, RL, or human-designed).

### Phase 3: DQN Agent (4-6 weeks)

**Goal:** Train a Deep RL agent to learn optimal heuristic selection.

**Tasks:**
1. ✅ Implement `DQNAgent` class (Algorithm 3)
2. ✅ Integrate with main loop (Algorithm 1)
3. ✅ Train on your dataset
4. ✅ Analyze learned policy: Which actions does it prefer in which states?
5. ✅ Benchmark: DQN vs. Random vs. GA+LNS/CP

**Success Metric:** DQN agent outperforms random selection and achieves competitive or superior results to GA+LNS/CP.

**Deliverable:**
- Trained DQN model
- Plots showing convergence and action selection over time
- Analysis report on learned strategy

### Phase 4: Evaluation & Thesis Writing (6-8 weeks)

**Goal:** Comprehensive evaluation and documentation.

**Tasks:**
1. ✅ Test on multiple problem instances (vary size, constraint density)
2. ✅ Statistical analysis of results
3. ✅ Generate visualizations (convergence curves, action heatmaps, etc.)
4. ✅ Write thesis chapters
5. ✅ Prepare publication draft

**Success Metric:** Clear demonstration that RL-based hyper-heuristic is adaptive and effective.

---

## 6. Technical Challenges & Solutions

### Challenge 1: IGLS Subproblem Modeling

**Problem:** Adapting the existing IGLS to work on a subproblem (a few movable sessions against a large fixed background) requires careful implementation.

**Solution:**
- Create a new `IGLSSubproblemSolver` class.
- The solver's evaluation function must check for conflicts against both the other movable sessions *and* the fixed sessions in the partial schedule.
- The search neighborhood should only consider moves for the small set of repairable sessions.
- Time-limit the IGLS run (e.g., 10-15 seconds max) to ensure it acts as a fast "surgical" tool.

### Challenge 2: Reward Shaping

**Problem:** A poorly designed reward function can lead to pathological behavior (e.g., agent only uses cheap mutations).

**Solution:**
- Start with basic `improvement_only` reward
- Log all agent actions during training
- If behavior is suboptimal, incrementally add:
  - Cost penalties (to discourage expensive actions unless necessary)
  - Hard constraint bonuses (to prioritize feasibility)
  - Exploration bonuses (to encourage trying diverse actions)
- Use reward visualization to understand agent incentives

### Challenge 3: Training Instability

**Problem:** Deep RL can be unstable, especially early in training.

**Solution:**
- Use experience replay (implemented in DQNAgent)
- Use target network (updated every 100 steps)
- Use Huber loss (more stable than MSE)
- Start with high epsilon (1.0) and decay slowly
- Monitor loss curves - if they diverge, reduce learning rate

### Challenge 4: Computational Cost

**Problem:** Training RL agents requires many evaluations.

**Solution:**
- Use the `test` configuration for initial experiments (30 gens, fast)
- Parallelize fitness evaluations (already supported in your codebase)
- Train on smaller problem instances first
- Use early stopping: If agent performance plateaus, end training

---

## 7. Expected Outcomes & Academic Contribution

### What You Will Achieve

1. **A State-of-the-Art Solver:** A hybrid system that combines:
   - Global search (GA)
   - Local exact solving (CP-SAT)
   - Adaptive intelligence (RL)

2. **Novel Research Contribution:** Demonstrating that an RL agent can learn to select heuristics more effectively than fixed strategies.

3. **Publishable Results:** Comparative analysis showing:
   - Baseline GA performance
   - Improvement from LNS-CP integration
   - Further improvement from RL-based selection
   - Adaptiveness across different problem instances

### Why This Is Better Than Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Pure GA** | Simple, proven | Gets stuck in local optima | Good baseline, not competitive |
| **Pure CP-SAT** | Optimal (if it finishes) | **Intractable** (you proved this) | ❌ Infeasible at scale |
| **SA/TS** | Simple to add | Single-objective, parameter-sensitive | Good additions to toolbox |
| **ACO** | Interesting | Complex adaptation to timetabling | Lower priority |
| **RL Hyper-Heuristic** | **Adaptive, learns**, combines all | Complex implementation | ✅ **Best long-term approach** |

**Recommendation:** Don't abandon RL for SA/TS/ACO. Instead, **add SA/TS as additional actions** in your RL agent's toolbox. The RL agent can then learn when to use each technique.

---

## 8. Success Criteria & Milestones

### Minimum Viable Project (Pass Threshold)
- ✅ Working GA+LNS/CP hybrid
- ✅ Basic RL environment
- ✅ Simple RL agent (even if not optimal)
- ✅ Comparative evaluation showing some improvement

### Strong Project (High Pass)
- ✅ DQN agent that learns non-trivial policy
- ✅ Demonstration of adaptive behavior
- ✅ Comprehensive benchmarking
- ✅ Analysis of learned strategy

### Exceptional Project (Top Grade / Publication)
- ✅ DQN agent outperforms all baselines
- ✅ Transfer learning: Agent trained on one instance works on others
- ✅ Ablation studies showing contribution of each component
- ✅ Publication-ready paper with rigorous experimental design

---

### Immediate Next Steps (This Week)

1. **Decision Point:** Commit to the RL hyper-heuristic approach with IGLS repair.
2. **Setup:** Ensure `pytorch` is installed.
3. **Phase 1 Start:** Implement `find_hard_conflict_sessions()`.
4. **Prototype:** Create a minimal `LNS_IGLS_Repair` that works on 5 sessions.
5. **Test:** Verify the subproblem IGLS can solve a small, targeted problem quickly.

**First Concrete Task:**

```python
# File: src/ga/lns_igls_repair.py

def find_hard_conflict_sessions(individual: Individual) -> List[SessionGene]:
    """
    Identify sessions involved in hard constraint violations.

    Returns:
        List of SessionGene objects that are in conflict.
    """
    from src.ga.evaluation import evaluate_individual

    # Get all hard constraint violations
    violations = evaluate_individual(individual, return_details=True)

    conflicted_session_ids = set()
    for violation in violations.hard:
        # Each violation object should indicate which sessions are involved
        conflicted_session_ids.update(violation.session_ids)

    # Return the actual SessionGene objects
    conflicted_sessions = [
        gene for gene in individual
        if gene.session_id in conflicted_session_ids
    ]

    return conflicted_sessions
```

**Test it immediately** with a known-bad individual to verify it correctly identifies conflicted sessions.

---

## 10. Conclusion

You are at an exciting point in your research. Your CP-SAT failure is not a setback—it's a valuable finding that justifies the need for a more sophisticated approach. The RL-based hyper-heuristic framework will:

1. **Leverage your existing work** (the GA and IGLS are not wasted; they are the foundation)
2. **Integrate the IGLS insight** (use it surgically on small subproblems)
3. **Add adaptive intelligence** (RL learns the optimal strategy)
4. **Create a publishable contribution** (novel, rigorous, and effective)

**Your project is ready for this evolution.** The architecture is modular, the baseline is strong, and the path forward is clear.

**Start with Phase 1. Build incrementally. Benchmark continuously.**

You will create something genuinely novel and powerful.
