# Codebase Refactoring Plan

**Date**: November 17, 2025  
**Version**: 1.0  
**Status**: Action Plan

---

## Executive Summary

This document outlines necessary refactorings to improve code quality, maintainability, and performance of the schedule-engine codebase. Refactorings are prioritized by impact and risk.

---

## Priority Levels

- **P0 (Critical)**: Must do, affects correctness or security
- **P1 (High)**: Should do soon, affects quality significantly  
- **P2 (Medium)**: Nice to have, improves maintainability
- **P3 (Low)**: Future enhancement, minimal impact

---

## Refactoring Categories

1. [Code Quality & Structure](#1-code-quality--structure)
2. [Performance Optimizations](#2-performance-optimizations)
3. [Error Handling & Robustness](#3-error-handling--robustness)
4. [Testing Infrastructure](#4-testing-infrastructure)
5. [Documentation & Type Hints](#5-documentation--type-hints)

---

## 1. Code Quality & Structure

### R1.1: Standardize Heuristic Return Types [P0]

**Problem**: Heuristics have inconsistent return types (some return `Individual`, others return tuples).

**Current State**:
```python
# Some heuristics
def temporal_shift(...) -> Individual:
    return modified_individual

# Others
def kempe_chain(...) -> Tuple[Individual, Dict]:
    return modified_individual, metadata
```

**Refactored**:
```python
# ALL heuristics should return Individual only
def temporal_shift(...) -> Individual:
    return modified_individual

def kempe_chain(...) -> Individual:
    # Log metadata instead of returning
    logger.debug(f"Kempe chain applied: {metadata}")
    return modified_individual
```

**Files to Modify**:
- `src/heuristics/improvement.py` (kempe_chain, conflict_repair)
- `src/heuristics/perturbation.py` (ejection_chain)
- `src/heuristics/meta.py` (adaptive_intensity)

**Estimated Effort**: 2-4 hours

---

### R1.2: Consolidate State Encoder Feature Extraction [P1]

**Problem**: Feature extraction logic scattered across multiple methods.

**Current State**:
```python
class StateEncoder:
    def encode(self, ...):
        # Mix of inline calculations
        best_fitness = min(...)
        avg_fitness = mean(...)
        # ... 20+ lines ...
```

**Refactored**:
```python
class StateEncoder:
    def encode(self, population, generation, stagnation, max_generations):
        """Main encoding method."""
        features = {
            **self._extract_fitness_features(population),
            **self._extract_diversity_features(population),
            **self._extract_progress_features(generation, stagnation, max_generations),
            **self._extract_constraint_features(population),
            **self._extract_history_features()
        }
        return self._normalize_features(features)
    
    def _extract_fitness_features(self, population) -> Dict[str, float]:
        """Extract fitness-related features."""
        fitness_values = [ind.fitness.values for ind in population]
        return {
            'best_fitness': min(fitness_values),
            'avg_fitness': mean(fitness_values),
            'worst_fitness': max(fitness_values),
            'fitness_std': std(fitness_values),
            'fitness_range': max(fitness_values) - min(fitness_values)
        }
    
    def _extract_diversity_features(self, population) -> Dict[str, float]:
        """Extract diversity metrics."""
        return {
            'population_diversity': self._compute_hamming_diversity(population),
            'genotype_diversity': self._compute_gene_diversity(population),
            'phenotype_diversity': self._compute_fitness_diversity(population),
            'unique_fitness_ratio': self._compute_unique_ratio(population)
        }
    
    # ... similar for other feature groups
```

**Benefits**:
- **Maintainability**: Easy to add/remove features
- **Testability**: Can test feature groups independently
- **Clarity**: Clear separation of concerns

**Files to Modify**:
- `src/rl/gym_env/state_encoder.py`

**Estimated Effort**: 4-6 hours

---

### R1.3: Extract Configuration Validation [P1]

**Problem**: Config validation logic spread across multiple files.

**Current State**:
```python
# In ga_scheduler.py
if not rl_config.enabled:
    return False

# In model_loader.py
if not Path(model_path).exists():
    raise FileNotFoundError(...)

# In inference.py
if timeout_ms < 1.0:
    raise ValueError(...)
```

**Refactored**:
```python
# src/config/validators.py
class RLConfigValidator:
    """Validate RL configuration."""
    
    @staticmethod
    def validate(config: RLConfig) -> Tuple[bool, List[str]]:
        """Validate RL config, return (is_valid, errors)."""
        errors = []
        
        # Check if enabled
        if not config.enabled:
            return True, []  # Valid to be disabled
        
        # Check model path exists
        if not Path(config.agent.model_path).exists():
            errors.append(f"Model not found: {config.agent.model_path}")
        
        # Check timeout range
        if not (1.0 <= config.inference.timeout_ms <= 1000.0):
            errors.append(f"Invalid timeout: {config.inference.timeout_ms}ms (must be 1-1000)")
        
        # Check mode compatibility
        if config.mode not in ['inference', 'hybrid', 'training']:
            errors.append(f"Invalid mode: {config.mode}")
        
        # Check agent type
        if config.agent.type not in ['ppo', 'dqn']:
            errors.append(f"Invalid agent type: {config.agent.type}")
        
        is_valid = len(errors) == 0
        return is_valid, errors

# Usage in ga_scheduler.py
validator = RLConfigValidator()
is_valid, errors = validator.validate(get_config().rl)
if not is_valid:
    for error in errors:
        console.print(f"[red]✗ {error}[/red]")
    return False
```

**Files to Create/Modify**:
- Create: `src/config/validators.py`
- Modify: `src/core/ga_scheduler.py`, `src/rl/deployment/model_loader.py`

**Estimated Effort**: 3-5 hours

---

### R1.4: Unified Logging Strategy [P2]

**Problem**: Mix of `print()`, `console.print()`, and `logger` calls.

**Refactored**:
```python
# Establish clear guidelines:
# - User-facing messages: console.print() (Rich)
# - Debug/trace logs: logger.debug()
# - Error logs: logger.error()
# - Performance metrics: logger.info()
# - Never use print() in production code

# Example refactoring
# BEFORE
print(f"Loading model from {path}")  # ❌

# AFTER
console.print(f"[cyan]Loading model from {path}...[/cyan]")  # User message
logger.info(f"Model load initiated: {path}")  # Log for debugging
```

**Files to Audit**:
- All `src/rl/` files
- All `src/heuristics/` files
- `src/core/ga_scheduler.py`

**Estimated Effort**: 6-8 hours (audit + refactor)

---

## 2. Performance Optimizations

### R2.1: Cache State Encoder Computations [P1]

**Problem**: Diversity metrics recomputed every generation.

**Current**:
```python
def encode(self, population, generation, ...):
    diversity = compute_diversity(population)  # Expensive!
    # ...
```

**Optimized**:
```python
class StateEncoder:
    def __init__(self, ...):
        self._cache = {}
        self._cache_gen = -1
    
    def encode(self, population, generation, ...):
        # Check cache
        cache_key = (generation, id(population))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute and cache
        state = self._compute_state(population, generation, ...)
        self._cache[cache_key] = state
        
        # Evict old cache entries
        if len(self._cache) > 10:
            oldest_key = min(self._cache.keys(), key=lambda k: k[0])
            del self._cache[oldest_key]
        
        return state
```

**Expected Speedup**: 2-3x for state encoding (5-10ms → 2-3ms)

**Files to Modify**:
- `src/rl/gym_env/state_encoder.py`

**Estimated Effort**: 2-3 hours

---

### R2.2: Lazy Model Loading [P1]

**Problem**: Model loaded synchronously, blocking GA start.

**Current**:
```python
def _init_rl(self):
    # ... setup ...
    model, metadata = loader.load_model(model_path, agent_type)  # Blocks!
    # ...
```

**Optimized**:
```python
def __init__(self, config, context):
    # ... existing init ...
    
    # Start async RL initialization
    if get_config().rl.enabled:
        self._rl_init_future = self._async_init_rl()

def _async_init_rl(self):
    """Initialize RL in background thread."""
    return ThreadPoolExecutor(max_workers=1).submit(self._init_rl_impl)

def run(self):
    # Wait for RL init if needed
    if hasattr(self, '_rl_init_future'):
        self._rl_init_future.result(timeout=30.0)
    
    # ... continue GA ...
```

**Expected Speedup**: Eliminate 50-100ms startup delay

**Files to Modify**:
- `src/core/ga_scheduler.py`

**Estimated Effort**: 3-4 hours

---

### R2.3: Batch Fitness Evaluation in RL [P2]

**Problem**: Evaluating modified individuals one-by-one.

**Current**:
```python
for ind in modified_individuals:
    fitness = evaluate(ind, context)
    ind.fitness.values = fitness
```

**Optimized**:
```python
# Batch evaluation if toolbox supports it
if hasattr(self.toolbox, 'map') and len(modified_individuals) > 1:
    fitness_values = list(self.toolbox.map(
        self.toolbox.evaluate,
        modified_individuals
    ))
    for ind, fit in zip(modified_individuals, fitness_values):
        ind.fitness.values = fit
else:
    # Fallback to sequential
    for ind in modified_individuals:
        fitness = evaluate(ind, context)
        ind.fitness.values = fitness
```

**Expected Speedup**: 1.5-2x for multi-individual operations

**Files to Modify**:
- `src/core/ga_scheduler.py::_apply_rl_operators()`

**Estimated Effort**: 2-3 hours

---

## 3. Error Handling & Robustness

### R3.1: Wrap Heuristic Execution with Error Handling [P0]

**Problem**: Heuristic failures can crash GA.

**Current**:
```python
def apply_action(self, action, individual, context):
    result = action_info.function(individual, context)  # May raise!
    return result, True
```

**Hardened**:
```python
def apply_action(self, action, individual, context):
    action_info = self.actions[action]
    
    # No-op case
    if action_info.function is None:
        return [individual], True
    
    try:
        # Clone to avoid mutation
        individual_copy = copy.deepcopy(individual)
        
        # Apply heuristic with timeout
        with timeout(seconds=5):
            result = action_info.function(individual_copy, context)
        
        # Validate result
        if not self._validate_result(result):
            logger.warning(f"Heuristic {action_info.name} returned invalid result")
            return [individual], False
        
        return [result], True
    
    except TimeoutError:
        logger.error(f"Heuristic {action_info.name} timed out")
        return [individual], False
    
    except Exception as e:
        logger.error(f"Heuristic {action_info.name} failed: {e}", exc_info=True)
        return [individual], False

def _validate_result(self, result) -> bool:
    """Validate heuristic result."""
    if result is None:
        return False
    if not isinstance(result, list):
        return False
    if len(result) == 0:
        return False
    if not all(hasattr(gene, 'course_id') for gene in result):
        return False
    return True
```

**Files to Modify**:
- `src/rl/gym_env/action_mapper.py`

**Estimated Effort**: 4-6 hours

---

### R3.2: Add RL Component Cleanup [P1]

**Problem**: RL resources not released after GA run.

**Implementation**:
```python
def run(self):
    try:
        # ... GA evolution ...
    finally:
        # Always cleanup
        self._cleanup_rl()

def _cleanup_rl(self):
    """Release RL resources."""
    if not hasattr(self, 'rl_controller'):
        return
    
    try:
        # Clear caches
        if hasattr(self.rl_controller, 'inference_engine'):
            self.rl_controller.inference_engine.clear_cache()
        
        # Release model
        if hasattr(self, 'rl_action_mapper'):
            del self.rl_action_mapper
        
        if hasattr(self, 'rl_state_encoder'):
            del self.rl_state_encoder
        
        if hasattr(self, 'rl_controller'):
            del self.rl_controller
        
        logger.info("RL components cleaned up")
    
    except Exception as e:
        logger.warning(f"Error during RL cleanup: {e}")
```

**Files to Modify**:
- `src/core/ga_scheduler.py`

**Estimated Effort**: 2-3 hours

---

### R3.3: Safe YAML Loading [P0]

**Problem**: Using unsafe `yaml.load()`.

**Current**:
```python
config = yaml.load(f)  # ❌ Unsafe
```

**Fixed**:
```python
config = yaml.safe_load(f)  # ✅ Safe
```

**Files to Audit**:
- `src/rl/deployment/registry.py`
- `src/config/loader.py`
- All files using `yaml.load()`

**Estimated Effort**: 1 hour

---

## 4. Testing Infrastructure

### R4.1: Add Integration Tests for RL-GA [P0]

**Implementation**:
```python
# test/integration/test_rl_ga_integration.py
import pytest
from src.core.ga_scheduler import GAScheduler
from src.config import init_config

def test_rl_ga_full_pipeline():
    """Test complete RL-GA integration."""
    # Setup
    init_config(env='test', overrides={'rl.enabled': True})
    context = create_test_context()
    
    # Create scheduler with RL
    scheduler = GAScheduler(
        config=GAConfig(generations=10, pop_size=4),
        context=context
    )
    
    # Run GA
    scheduler.run()
    
    # Verify RL was used
    assert scheduler.rl_enabled
    assert scheduler.rl_controller is not None
    
    # Verify RL actions were applied
    stats = scheduler.rl_controller.get_usage_stats()
    assert stats['rl_actions'] > 0
    
    # Verify solution quality
    best = scheduler.get_best_solution()
    assert best.fitness.values[0] <= 20  # Reasonable for test

def test_rl_fallback_on_error():
    """Test fallback when RL fails."""
    # Force RL error by corrupting model
    init_config(env='test', overrides={
        'rl.enabled': True,
        'rl.agent.model_path': 'invalid/path.zip'
    })
    
    context = create_test_context()
    scheduler = GAScheduler(...)
    
    # Should fall back gracefully
    scheduler.run()
    
    # GA should still complete
    assert scheduler.best_individual is not None

def test_rl_disabled():
    """Test GA works without RL."""
    init_config(env='test', overrides={'rl.enabled': False})
    
    scheduler = GAScheduler(...)
    scheduler.run()
    
    # RL should not be initialized
    assert not scheduler.rl_enabled
    assert not hasattr(scheduler, 'rl_controller')
```

**Files to Create**:
- `test/integration/test_rl_ga_integration.py`
- `test/integration/test_rl_deployment.py`
- `test/integration/test_rl_training.py`

**Estimated Effort**: 12-16 hours

---

### R4.2: Increase Unit Test Coverage [P1]

**Current Coverage**: ~60% for RL modules

**Target Coverage**: 80%+

**Priority Areas**:
1. `ActionMapper` - test all edge cases
2. `StateEncoder` - test normalization, caching
3. `RLInference` - test timeout, errors
4. `HybridController` - test mode switching, fallback
5. `Heuristics` - test return types, error handling

**Estimated Effort**: 20-30 hours

---

## 5. Documentation & Type Hints

### R5.1: Add Missing Docstrings [P2]

**Audit Results**: ~15% of public methods lack docstrings

**Priority Files**:
- `src/rl/gym_env/schedule_env.py::step()`
- `src/rl/hybrid/hybrid_controller.py::_apply_fallback()`
- `src/heuristics/improvement.py` (several functions)

**Standard Format** (Google Style):
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """One-line summary.
    
    Longer description if needed. Explain what the function does,
    when to use it, and any important caveats.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails
    
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
        expected_output
    """
```

**Estimated Effort**: 10-15 hours

---

### R5.2: Add Type Hints Throughout [P2]

**Current Status**: ~70% of functions have type hints

**Target**: 95%+

**Priority Areas**:
- All RL modules
- All heuristic functions
- All public APIs

**Example**:
```python
# BEFORE
def evaluate(individual, context):
    ...

# AFTER
def evaluate(
    individual: Individual,
    context: SchedulingContext
) -> Tuple[float, float]:
    ...
```

**Tool**: Use `mypy` to find missing type hints
```bash
mypy src/ --strict --show-error-codes
```

**Estimated Effort**: 15-20 hours

---

## Implementation Priority

### Phase 1: Critical Fixes (Week 1-2)
1. R3.3: Safe YAML loading [P0]
2. R1.1: Standardize heuristic return types [P0]
3. R3.1: Wrap heuristic execution [P0]
4. R4.1: Add integration tests [P0]

### Phase 2: Quality Improvements (Week 3-4)
5. R1.2: Consolidate state encoder [P1]
6. R1.3: Extract config validation [P1]
7. R2.1: Cache state computations [P1]
8. R2.2: Lazy model loading [P1]
9. R3.2: RL component cleanup [P1]

### Phase 3: Enhancements (Week 5-6)
10. R1.4: Unified logging [P2]
11. R2.3: Batch fitness evaluation [P2]
12. R4.2: Increase test coverage [P1-P2]
13. R5.1: Add docstrings [P2]
14. R5.2: Add type hints [P2]

---

## Success Criteria

### Code Quality Metrics
- [ ] No `print()` statements in production code
- [ ] All heuristics return consistent types
- [ ] All YAML loading uses `safe_load()`
- [ ] 95%+ functions have type hints
- [ ] 95%+ public methods have docstrings

### Testing Metrics
- [ ] Integration test suite exists (10+ tests)
- [ ] Unit test coverage >80%
- [ ] All critical paths tested
- [ ] CI/CD pipeline passes

### Performance Metrics
- [ ] State encoding <5ms (cached)
- [ ] Model loading <100ms (first time)
- [ ] Heuristic execution <50ms (99th percentile)
- [ ] Memory leaks eliminated

---

## Risk Assessment

| Refactoring | Risk Level | Mitigation |
|-------------|------------|------------|
| R1.1: Return types | Low | Comprehensive tests before/after |
| R2.2: Lazy loading | Medium | Thorough testing of edge cases |
| R3.1: Error handling | Low | Wrap carefully, preserve behavior |
| R4.1: Integration tests | Low | Start with simple cases |

---

**Document Status**: ✅ Complete - Ready for implementation planning
