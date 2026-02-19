# DEAP → pymoo Migration Analysis

## Executive Summary

**Recommendation**: Proceed with migration to pymoo with 2×events encoding (1098 variables)

**Key Benefits**:

- Access to 40+ multi-objective algorithms (NSGA-II, NSGA-III, etc.)
- Better operator library with constraint handling
- Performance gains for large populations
- Built-in Pareto front analysis
- Modern active development vs DEAP stagnation

## 1. Current Implementation Analysis (Verified)

### Environment Status

```
Python 3.12.12 (venv)
DEAP 1.4.1
pymoo 0.6.1.3 (ALREADY INSTALLED)
numpy 1.26.4, scipy 1.11.4
```

### Problem Scale (Verified)

```
Events per individual: 549
Constraint system: 8 hard + 6 soft constraints
Current evaluation time: ~7.6ms per individual
Course-group combinations: 159 schedulable courses
Total domain elements: 549×(rooms+instructors+time_slots)
```

### Decision Variables (Verified from SessionGene)

```python
# Current DEAP representation per event:
- course_id (string, never mutated)
- group_ids (list, never mutated) 
- instructor_id (int, mutable)
- room_id (int, mutable)
- start_quanta (int, mutable)

# Effectively: [instructor, room, start_time] × 549 events
```

### Domain Constraints (Verified with Real Data)

```
ALLOWED ROOMS per event:
  Min: 1, Max: 75, Average: 32.7
  39% of events have ≤5 suitable rooms

ALLOWED INSTRUCTORS per event:  
  Min: 1, Max: 12, Average: 3.5
  50% of events have ≤2 qualified instructors

ALLOWED START TIMES per event:
  Min: 33, Max: 42, Average: 40.9
  Time availability very high (98%+)
```

### Current Constraint Analysis (Verified)

```python
# Hard constraints (exclusivity-based):
- StudentGroupExclusivity: No group in 2+ places simultaneously  
- InstructorExclusivity: No instructor in 2+ places simultaneously
- RoomExclusivity: No room hosting 2+ sessions simultaneously
- InstructorQualifications: instructor_id ∈ course.qualified_instructor_ids
- RoomSuitability: Uses find_suitable_rooms_for_course() logic
- InstructorAvailability: instructor available at start_quanta
- GroupAvailability: group available at start_quanta
- CompleteCoverage: All enrollments scheduled

# Soft constraints (optimization preferences):
- Compactness: Minimize gaps in student schedules  
- BreakFrequency: Adequate breaks between sessions
- Continuity: Back-to-back sessions when beneficial
- Plus 3 others with weight-based scoring
```

## 2. Migration Architecture

### Encoding Strategy: 2×E (Recommended)

```python
# Chromosome: [I₁, R₁, I₂, R₂, ..., I₅₄₉, R₅₄₉]
# Where: Iᵢ = instructor index, Rᵢ = room index
# Time slots computed deterministically to reduce variables

class ScheduleProblem(Problem):
    def __init__(self, events_data):
        super().__init__(
            n_var=1098,  # 2 * 549 events
            n_obj=2,     # hard + soft violations  
            n_constr=8,  # hard constraints as constraints
            xl=0,        # Lower bounds per event
            xu=max_idx   # Upper bounds per event  
        )
        self.events = events_data
        self.evaluator = FastEvaluator()
```

### Foundation Components (Implemented)

#### 1. Precomputed Domains (`build_events.py`)

```python
# Exports for each event:
{
    "event_id": 123,
    "course_id": "CS101", 
    "group_ids": ["BCS1A", "BCS1B"],
    "allowed_rooms": [1, 3, 5, 7, 12],      # Valid room indices
    "allowed_instructors": [45, 67, 89],     # Valid instructor indices  
    "allowed_starts": [0,1,2,...,38,39],    # Valid time slot indices
    "duration": 3                           # quanta duration
}
```

#### 2. Fast Numeric Evaluator (`fast_evaluator.py`)

```python
def fast_conflict_evaluator(instructor_assignments, room_assignments, 
                          start_times, events_data):
    """
    Returns: (room_conflicts, instructor_conflicts, group_conflicts, soft_penalty)
    Uses occupancy maps for O(1) conflict detection
    """
```

**Performance**: Currently 20.35ms (needs optimization, target: <7ms)

## 3. Migration Implementation Plan

### Phase 1: Problem Class Implementation

```python
from pymoo.core.problem import Problem
import numpy as np
import pickle

class SchedulingProblem(Problem):
    def __init__(self):
        # Load precomputed domain data
        with open('events_with_domains.pkl', 'rb') as f:
            self.events = pickle.load(f)
        
        # Build variable bounds
        xl, xu = [], []
        for event in self.events:
            xl.extend([0, 0])  # instructor, room
            xu.extend([len(event['allowed_instructors'])-1, 
                      len(event['allowed_rooms'])-1])
        
        super().__init__(
            n_var=len(xl),
            n_obj=2,  # hard_violations, soft_violations
            n_constr=8,  # Each hard constraint type
            xl=np.array(xl),
            xu=np.array(xu)
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        # Convert chromosome to assignments
        n_solutions, n_vars = X.shape
        objectives = []
        constraints = []
        
        for i in range(n_solutions):
            chromosome = X[i]
            # Decode chromosome to instructor/room assignments
            instructor_assignments = chromosome[::2]  # Every 2nd starting at 0
            room_assignments = chromosome[1::2]       # Every 2nd starting at 1
            
            # Compute time slots (deterministic or heuristic)
            start_times = self._compute_start_times(instructor_assignments, room_assignments)
            
            # Evaluate using fast evaluator
            room_conf, inst_conf, group_conf, soft_penalty = self.evaluator.evaluate(
                instructor_assignments, room_assignments, start_times
            )
            
            hard_total = room_conf + inst_conf + group_conf
            objectives.append([hard_total, soft_penalty])
            
            # Individual constraint values for constraint handling
            constraints.append([room_conf, inst_conf, group_conf, ...])
        
        out["F"] = np.array(objectives)
        out["G"] = np.array(constraints)
```

### Phase 2: Algorithm Selection

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling

# Multi-objective optimization
algorithm = NSGA2(
    pop_size=100,
    sampling=IntegerRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=20, prob=1.0/n_var),  
    eliminate_duplicates=True
)

from pymoo.optimize import minimize

res = minimize(
    SchedulingProblem(),
    algorithm,  
    ('n_gen', 200),
    verbose=True
)
```

### Phase 3: Performance Optimization

1. **Fast Evaluator Optimization** (Priority)
   - Current: 20.35ms (too slow)
   - Target: <7ms (match DEAP performance)
   - Optimizations: Cython, vectorization, memory layout

2. **Constraint-Aware Operators**
   - Custom crossover respecting domain constraints
   - Intelligent mutation within allowed domains
   - Repair operators for constraint violations

3. **Time Slot Algorithm**
   - Deterministic scheduling (earliest-fit)
   - Heuristic placement (best-fit)
   - Hybrid approaches

## 4. Expected Benefits

### Algorithm Variety (Major Advantage)

```python
# Multi-objective algorithms available:
- NSGA-II, NSGA-III (non-dominated sorting)
- MOEA/D (decomposition-based) 
- SMS-EMOA (hypervolume-based)
- Plus 35+ other algorithms

# Constraint handling methods:
- Death penalty (current approach)
- Constraint violation as objective
- Repair operators
- Adaptive penalty methods
```

### Performance Improvements

- Better scalability for large populations (1000+ individuals)
- Vectorized operations across population
- NumPy-optimized constraint evaluation
- Parallel evaluation support

### Analysis Capabilities

```python
# Pareto front analysis
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

# Multi-criteria decision making
from pymoo.mcdm.pseudo_weights import PseudoWeights

# Automated hyperparameter tuning
from pymoo.util.grid_search import GridSearch
```

## 5. Migration Risks & Mitigations

### Risk 1: Performance Regression

- **Mitigation**: Optimize fast evaluator before full migration
- **Fallback**: Keep DEAP version until performance matches

### Risk 2: Constraint Handling Differences  

- **Mitigation**: Extensive validation with known test cases
- **Testing**: Compare results on identical problem instances

### Risk 3: Integration Complexity

- **Mitigation**: Incremental migration with interface compatibility
- **Strategy**: Wrap pymoo in DEAP-compatible interface initially

## 6. Implementation Timeline

### Week 1: Foundation

- [x] Verify domain constraints with real data
- [x] Implement event builder with precomputed domains
- [x] Create fast numeric evaluator
- [ ] Optimize evaluator performance (<7ms target)

### Week 2: Core Migration

- [ ] Implement SchedulingProblem class
- [ ] Test 2×E encoding with simple algorithms
- [ ] Validate constraint evaluation accuracy
- [ ] Benchmark against DEAP baseline

### Week 3: Advanced Features  

- [ ] Implement constraint-aware operators
- [ ] Test multi-objective algorithms
- [ ] Performance optimization
- [ ] Integration with existing workflows

### Week 4: Validation & Deployment

- [ ] Comprehensive testing on real instances
- [ ] Documentation and training
- [ ] Gradual rollout with fallback plan

## 7. Code Foundation Status

### Completed Components

- **Domain Analysis**: Real constraint data verified
- **Event Builder**: `build_events.py` exports 549 events with domains
- **Fast Evaluator**: `fast_evaluator.py` provides numeric evaluation  
- **Data Exports**: JSON (595KB) and pickle (117KB) files ready

### File Outputs Generated

```
events_with_domains.json  # Human-readable event definitions
events_with_domains.pkl   # Fast loading for Python
```

### Next Implementation Priority

1. **Optimize fast_evaluator.py** (currently 20ms, needs to be <7ms)
2. **Create SchedulingProblem class** using foundation components
3. **Test basic pymoo algorithms** with the 2×E encoding

---

**Migration Assessment: RECOMMENDED**

The verified analysis shows pymoo migration is highly beneficial for this 549-event scheduling problem. Foundation components are implemented and working. The main remaining work is performance optimization and algorithm integration.
