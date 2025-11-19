# Pre-Phase 2 Checklist - RL Implementation Readiness

**Date:** November 15, 2025  
**Status:** Verification in progress

##  Code Organization

### Current Structure
```
src/
├── ga/operators/          # ✓ Well-organized, keep as-is
├── lns/                   # ✓ Clean, CP-SAT removed
├── rl/                    # ✓ Placeholders created
├── core/                  # ✓ GA scheduler updated
├── config/                # ✓ Config system ready
└── entities/              # ✓ Domain models stable
```

### Recommendation: **NO REORGANIZATION NEEDED**
-  Keep `ga/operators/` as low-level primitives
-  Create `rl/actions/` as high-level wrappers
-  Maintain clear layer separation

---

##  Code Quality Checks

### 1. No CP-SAT Remnants
- [ ] Run: `grep -r "ortools\|cp_model\|cp_sat" src/`
- [ ] Expected: No matches
- [ ] Status: ___________

### 2. Import Consistency
- [ ] All LNS imports use `lns_igls_repair` (not `lns_cp_repair`)
- [ ] No `repair_with_cp_sat` imports
- [ ] Status: ___________

### 3. Configuration Validity
- [ ] `configs/base.yaml` has only IGLS params (no `repair_strategy`, `cp_time_limit`)
- [ ] `configs/prod.yaml` references only IGLS
- [ ] Status: ___________

### 4. Function Signatures
- [ ] `lns_igls_repair()` exists and has correct signature
- [ ] No `lns_repair()` with `repair_strategy` parameter
- [ ] Status: ___________

---

##  Dependency Check

### Python Dependencies (pyproject.toml)
- [x]  `torch` installed (for Phase 3 DQN)
- [x]  `ortools` REMOVED
- [ ] Verify: `uv pip list | grep torch`
- [ ] Verify: `uv pip list | grep ortools` (should be empty)

### Missing Dependencies?
- [ ] Any imports failing?
- [ ] Any module not found errors?

---

##  Architecture Decisions

### Decision 1: Heuristic Action Design
**Question:** Separate `ga/operators/` from `rl/actions/`?

** DECISION: YES - Use Wrapper Pattern**

**Rationale:**
- GA operators are low-level, reusable building blocks
- RL actions are high-level, with cost tracking and metadata
- Clean separation enables independent testing and evolution

**Implementation Plan:**
```python
# src/rl/actions/base.py
class HeuristicAction:
    """Base class for all RL-selectable actions."""
    
    name: str
    intensity: str  # "low", "medium", "high", "very_high"
    avg_cost_ms: float
    
    def apply(self, individual, context) -> Individual:
        """Apply heuristic to individual."""
        pass
    
    def get_cost(self) -> float:
        """Return computational cost (for reward penalty)."""
        return self.avg_cost_ms

# src/rl/actions/mutation_actions.py
class MutateTimeAction(HeuristicAction):
    name = "mutate_time"
    intensity = "low"
    avg_cost_ms = 1.0
    
    def apply(self, individual, context):
        # Wraps ga.operators.mutation.mutate_individual()
        from src.ga.operators.mutation import mutate_individual
        return mutate_individual(individual, context, mut_prob=1.0)
```

### Decision 2: State Representation
**Question:** What features should the RL agent observe?

** DECISION: 5D State Vector (as per design doc)**

1. `norm_hard_violations` - [0.0, 1.0]
2. `norm_soft_violations` - [0.0, 1.0]
3. `fitness_delta` - [-inf, +inf]
4. `norm_stagnation` - [0.0, 1.0]
5. `progress` - [0.0, 1.0]

**Implementation:** Create `StateCalculator` in `src/rl/state.py`

### Decision 3: Reward Function
**Question:** How to balance fitness improvement vs. action cost?

** DECISION: Multi-component reward**

```python
reward = (
    fitness_improvement       # Primary: encourage progress
    - (action_cost * 0.1)    # Penalty: discourage expensive actions
    + (hard_fixed_bonus * 10) # Bonus: prioritize feasibility
)
```

**Implementation:** Create `calculate_reward()` in `src/rl/reward.py`

---

##  Testing Strategy

### Unit Tests to Create
- [ ] `test/rl/test_state.py` - State calculation and normalization
- [ ] `test/rl/test_actions.py` - Each action wrapper
- [ ] `test/rl/test_reward.py` - Reward calculation edge cases
- [ ] `test/rl/test_environment.py` - Environment transitions

### Integration Tests
- [ ] RandomAgent can run full optimization
- [ ] State/action/reward flow works end-to-end
- [ ] Archive management (for crossover)

---

##  Phase 2 Implementation Order

### Week 1: Core Infrastructure
1. **Day 1-2:** Implement `HeuristicAction` base class
2. **Day 3-4:** Wrap 3 basic actions (mutate_time, mutate_room, crossover)
3. **Day 5:** Implement `StateCalculator`

### Week 2: Environment & Baseline
4. **Day 6-7:** Implement `TimetablingEnvironment`
5. **Day 8-9:** Implement `RandomAgent`
6. **Day 10:** Implement main RL loop (with random agent)

### Week 3: Experiments & Refinement
7. **Day 11-12:** Run baseline experiments (Random vs. GA-only)
8. **Day 13-14:** Add remaining actions (LNS-IGLS, destroy variants)
9. **Day 15:** Documentation and metrics

---

##  Pre-Implementation Tasks

### Before Writing Any Code:

1. **Verify Clean State**
   ```bash
   # No CP-SAT code
   grep -r "ortools\|cp_model" src/
   
   # All imports valid
   python -c "from src.lns import lns_igls_repair; print('OK')"
   
   # Config loads
   python -c "from src.config import get_config; print(get_config().lns.enabled)"
   ```

2. **Review Design Documents**
   - [ ] Read `suggest/suggestion.md` - Algorithm 1 & 2
   - [ ] Read `suggest/future_direction.md` - Architecture
   - [ ] Read `Todo.md` - Phase 2 tasks

3. **Understand Dependencies**
   - [ ] Review `src/ga/operators/` - What exists?
   - [ ] Review `src/lns/` - How does LNS-IGLS work?
   - [ ] Review `src/core/ga_scheduler.py` - How does GA loop work?

4. **Set Up Development Environment**
   - [ ] Ensure PyTorch works: `python -c "import torch; print(torch.__version__)"`
   - [ ] Ensure all tests pass: `pytest test/` (if any exist)
   - [ ] Code formatter configured: `black`, `ruff`

---

##  GO/NO-GO Decision

### GREEN LIGHT Criteria (all must be ):
- [ ] No CP-SAT code remains
- [ ] All imports resolve correctly
- [ ] Configuration is valid
- [ ] PyTorch is installed
- [ ] Architecture decisions documented
- [ ] Clear implementation plan

### Current Status: ** PENDING VERIFICATION**

Once all items are , you're ready to start Phase 2!

---

##  Notes & Questions

### Open Questions:
1. Should we implement all 6 actions immediately, or start with 3-4?
   - **Recommendation:** Start with 3-4, add others after baseline works

2. Should we add Simulated Annealing / Tabu Search actions now?
   - **Recommendation:** NO - wait for Phase 5 (after DQN proves value)

3. What's the priority: Random baseline or full action set?
   - **Recommendation:** Random baseline first (prove framework works)

### Design Refinements:
- Consider adding `action_history` to state (last N actions taken)
- Consider adding `time_budget_remaining` to state
- Consider adaptive epsilon decay based on fitness improvement

---

##  Final Checklist Summary

**Code:**
- [ ] No CP-SAT remnants
- [ ] All imports valid
- [ ] Configuration clean

**Architecture:**
- [x] Decision: Keep `ga/operators/` separate
- [x] Decision: Create `rl/actions/` wrappers
- [x] Decision: 5D state vector
- [x] Decision: Multi-component reward

**Infrastructure:**
- [x] `src/rl/` directories created
- [x] Placeholder files with TODOs
- [ ] PyTorch verified
- [ ] Tests outlined

**Documentation:**
- [x] Design documents reviewed
- [x] Implementation plan created
- [x] Phase 2 tasks clear

**Status:**  **READY PENDING FINAL VERIFICATION**

Run the verification commands above, then proceed to Phase 2 implementation!
