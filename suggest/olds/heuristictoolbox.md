Searched codebase for "I also need heuristic toolbox right?: before RL or after RL? implementing it?", no results

**Before RL** — you need the heuristic toolbox first! Here's why:

## The Dependency Chain

```
Heuristic Toolbox (Phase 1.5)
    ↓
RL Agent observes/learns from heuristics
    ↓  
RL Integration (Phase 2)
```

---

## Why Heuristic Toolbox Comes First

### 1. **RL Needs Baseline Behaviors to Learn From**
- RL agents learn by exploring and comparing actions
- Heuristics provide **expert demonstrations** (imitation learning)
- Without heuristics, RL starts from random policy (very slow convergence)

### 2. **Hybrid RL-Heuristic Architecture**
Your best approach is:
- **Heuristics** = Fast, domain-specific rules (always available)
- **RL** = Meta-controller that decides **when/which** heuristic to apply
- **Fallback** = If RL fails, heuristics ensure system still works

### 3. **What You Already Have vs Need**

**Already Implemented** ✅:
- LNS repair (heuristic destroy-repair)
- IGLS local search (greedy swaps)
- Adaptive mutation/crossover
- Conflict detection heuristics

**Missing Heuristics** ❌:
- Construction heuristics (build schedule from scratch intelligently)
- Perturbation heuristics (smart schedule shaking)
- Diversity maintenance (prevent premature convergence)
- Constraint-specific repair moves (room swap, time shift, instructor reassignment)

---

## Recommended Heuristic Toolbox Structure

```python
src/heuristics/
├── __init__.py
├── construction.py       # Build schedules greedily
│   ├── largest_degree_first()
│   ├── most_constrained_first()
│   └── earliest_deadline_first()
│
├── perturbation.py       # Shake current solution
│   ├── random_swap()
│   ├── temporal_shift()
│   ├── room_shuffle()
│   └── instructor_reassign()
│
├── improvement.py        # Local search moves
│   ├── kempe_chain()     # Graph coloring move
│   ├── ejection_chain()  # Advanced local search
│   └── variable_depth_search()
│
├── diversity.py          # Population diversity
│   ├── distance_preserving_crossover()
│   ├── crowding_mutation()
│   └── niching_selection()
│
└── meta.py              # High-level strategies
    ├── variable_neighborhood_descent()
    ├── iterated_local_search()
    └── adaptive_large_neighborhood()
```

---

## RL Integration Points (After Heuristics)

Once heuristics are ready, RL can:

### **Option 1: Heuristic Selection RL**
```python
# RL agent picks which heuristic to apply
state = get_state(population, generation)
action = rl_agent.select_action(state)  # action ∈ {h1, h2, h3, ...}

if action == "lns_repair":
    apply_lns_repair()
elif action == "kempe_chain":
    apply_kempe_chain()
elif action == "time_shift":
    apply_temporal_shift()
# ... etc
```

### **Option 2: Adaptive Weight RL**
```python
# RL adjusts heuristic weights dynamically
weights = rl_agent.get_weights(state)  # e.g., [0.3, 0.5, 0.2]
selected_heuristic = random.choices(heuristics, weights=weights)[0]
apply_heuristic(selected_heuristic)
```

### **Option 3: Hierarchical RL**
```python
# High-level RL picks strategy, low-level heuristics execute
strategy = high_level_rl.decide(state)  # "intensify" or "diversify"

if strategy == "intensify":
    # Use exploitation heuristics
    apply_local_search()
elif strategy == "diversify":
    # Use exploration heuristics
    apply_perturbation()
```

---

## My Recommendation: Implementation Order

### **Phase 1.5: Heuristic Toolbox** (do this now)
1. **Construction heuristics** (2-3 variants)
   - Most-constrained-first (prioritize hard sessions)
   - Largest-degree-first (schedule busy instructors first)
2. **Perturbation heuristics** (3-4 variants)
   - Temporal shift (move session to different time)
   - Room swap (exchange rooms between sessions)
   - Instructor reassign (if qualified)
3. **Meta-heuristic wrappers**
   - Variable neighborhood descent
   - Iterated local search with heuristics

### **Phase 2: RL Integration** (after heuristics)
1. **Gym environment** wrapping GA + heuristics
2. **RL agent** (PPO/DQN) learning heuristic selection
3. **Training loop** with reward shaping
4. **Evaluation** comparing RL vs random vs fixed heuristics

---

## Should I Build the Heuristic Toolbox Now?

**Yes! I recommend:**

1. **Start with 3 core heuristics** (quick wins):
   - `temporal_shift(session, delta)` - move session by ±delta quanta
   - `room_swap(session1, session2)` - exchange rooms if both suitable
   - `kempe_chain(session)` - graph coloring move for conflict resolution

2. **Add heuristic manager**:
   ```python
   class HeuristicManager:
       def __init__(self):
           self.heuristics = {
               "temporal_shift": temporal_shift,
               "room_swap": room_swap,
               "kempe_chain": kempe_chain,
           }
       
       def apply(self, name, individual, **kwargs):
           return self.heuristics[name](individual, **kwargs)
   ```

3. **Integrate into GA scheduler**:
   - Add heuristic application after crossover/mutation
   - Log which heuristics improve fitness
   - Collect statistics for RL training later

**Want me to implement the heuristic toolbox now?** I can:
- Create the module structure
- Implement 3-5 core heuristics
- Add the manager/registry system
- Integrate into GA scheduler with logging
- Document each heuristic's use case

This will take ~30 minutes and sets you up perfectly for RL integration after. Should I proceed?