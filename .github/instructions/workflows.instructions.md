---
applyTo: "src/workflows/**/*.py"
---

# Workflow Orchestration Instructions

## Overview
High-level workflow orchestration that coordinates all components. Main workflow in `src/workflows/standard_run.py`, reporting in `src/workflows/reporting.py`, experiment tracking in `src/workflows/experiment_manager.py`.

## Standard Workflow Pipeline

### run_standard_workflow() - Main Entry Point
```
1. Initialize RNG and output directory
2. Load input data from JSON files
3. Validate input data (optional)
4. Run feasibility checks (optional)
5. Setup and run GA scheduler
6. Decode best solution
7. Generate reports and exports
```

### Function Signature
```python
def run_standard_workflow(
    pop_size: int,
    generations: int,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.2,
    data_dir: str = "data",
    output_dir: Optional[str] = None,
    seed: int = 69,
    validate: bool = True,
    config: Config = None  # NEW: Pass config object
) -> Dict:
    """
    Returns:
        Dict with keys:
            - best_individual: Best GA solution
            - decoded_schedule: List[CourseSession]
            - metrics: GAMetrics object
            - output_path: str (output directory)
    """
```

## Workflow Stages

### Stage 1: Initialization
```python
# Set random seed
random.seed(seed)

# Create output directory
if output_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"evaluation_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
```

### Stage 2: Data Loading
```python
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.encoder.input_encoder import load_courses, load_groups, ...

qts = QuantumTimeSystem()
courses = load_courses(os.path.join(data_dir, "Course.json"))
groups = load_groups(os.path.join(data_dir, "Groups.json"), qts)
# ... load instructors, rooms ...

context = SchedulingContext(
    courses=courses,
    groups=groups,
    instructors=instructors,
    rooms=rooms,
    available_quanta=qts.get_all_operating_quanta(),
    qts=qts
)
```

### Stage 3: Validation (Optional)
```python
if validate:
    from src.validation import validate_input
    is_valid, warnings = validate_input(context)
    if not is_valid:
        console.print("[yellow]Validation warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")
```

### Stage 4: Feasibility Check (Optional)
```python
from src.validation.feasibility_checker import run_feasibility_checks

if config.feasibility.enable_checks:
    results = run_feasibility_checks(context, config)
    if results.has_critical_failures and config.feasibility.fail_on_infeasibility:
        raise RuntimeError("Problem is infeasible, cannot proceed")
```

### Stage 5: GA Execution
```python
from src.core.ga_scheduler import GAScheduler, GAConfig

ga_config = GAConfig(
    pop_size=pop_size,
    generations=generations,
    crossover_prob=crossover_prob,
    mutation_prob=mutation_prob,
    repair_config=config.repair.dict()
)

scheduler = GAScheduler(
    ga_config, context, hard_names, soft_names,
    pool=pool, logger=logger, seed=seed
)
scheduler.setup_toolbox()
scheduler.initialize_population()
scheduler.evolve()
```

### Stage 6: Decoding
```python
from src.decoder.individual_decoder import decode_individual

best_individual = scheduler.get_best_solution()
decoded_schedule = decode_individual(best_individual, context)
```

### Stage 7: Reporting
```python
from src.workflows.reporting import generate_reports

generate_reports(
    decoded_schedule=decoded_schedule,
    metrics=scheduler.metrics,
    output_dir=output_dir,
    context=context
)
```

## Rules

### Error Handling
- Wrap each stage in try-except
- Log errors to console and logger
- Continue with partial results if possible
- Return None or raise exception for critical failures

### Progress Reporting
- Use Rich console for user-facing messages
- Show spinners for long operations
- Display summary after each major stage
- Save detailed logs to `logger.txt`

### Multiprocessing
- Create worker pool AFTER data loading (pass data_dir to workers)
- Close pool in finally block
- Handle Windows spawn method (serialize context)
- Suppress worker output via environment variable

### Configuration Passthrough
- Accept `config` parameter (Config object)
- Pass to all subcomponents that need it
- Don't mix config object with individual parameters
- Use config for all optional features (validation, feasibility, repair)

## Adding New Workflow Stages

### Step 1: Implement Stage Function
```python
# In src/workflows/standard_run.py or new file
def run_my_stage(context, config, output_dir):
    """New workflow stage."""
    console.print("[bold]Running My Stage...[/bold]")
    # Implementation
    console.print("   [cyan]✓ My stage complete[/cyan]")
    return result
```

### Step 2: Insert in Pipeline
```python
def run_standard_workflow(...):
    # ... existing stages ...
    
    # Stage N: My Stage
    if config.my_stage.enabled:
        my_result = run_my_stage(context, config, output_dir)
    
    # ... continue workflow ...
```

### Step 3: Add Config Settings
```python
# In config/models.py
class MyStageConfig(BaseModel):
    enabled: bool = True
    # ... other settings ...

class Config(BaseModel):
    # ... existing ...
    my_stage: MyStageConfig = Field(default_factory=MyStageConfig)
```

## Workflow Variants

### Quick Test Run
```python
result = run_standard_workflow(
    pop_size=4,
    generations=10,
    validate=False,  # Skip validation for speed
    config=test_config
)
```

### Production Run
```python
result = run_standard_workflow(
    pop_size=100,
    generations=500,
    validate=True,
    config=prod_config
)
```

### Custom Workflow
```python
# Load data manually
qts, context = load_input_data(data_dir)

# Custom GA setup
ga_config = GAConfig(...)
scheduler = GAScheduler(ga_config, context, ...)

# Custom evolution loop
for gen in range(custom_generations):
    # ... custom logic ...

# Export manually
generate_reports(...)
```

## Performance Monitoring
- Track time for each stage
- Log memory usage if available
- Report GA convergence speed
- Save timing data to logger

## Experiment Management

### ExperimentManager
Track experiments systematically with `src/workflows/experiment_manager.py`:

```python
from src.workflows.experiment_manager import ExperimentManager
from src.config.runtime_mode import RuntimeMode

# Initialize manager
manager = ExperimentManager(base_dir="output")

# Register experiment run
run = manager.register_run(
    experiment_name="prod-baseline-r01",
    runtime_mode=RuntimeMode.BASELINE,
    config_snapshot=config.dict(),
    tags=["production", "ablation", "baseline"]
)

# Run experiment (manager creates organized output directory)
result = run_standard_workflow(
    output_dir=run.output_dir,
    config=config
)

# Update with results
manager.update_run(
    run_id=run.run_id,
    final_fitness=(hard_violations, soft_penalty),
    duration_seconds=elapsed_time
)
```

### Experiment Comparison

```python
# Compare runtime modes
comparison = manager.compare_modes(
    experiment_name="prod-baseline-r01",
    modes=[RuntimeMode.BASELINE, RuntimeMode.FULL, RuntimeMode.RL_GUIDED]
)

# Export to CSV
manager.export_comparison_csv("output/comparison.csv")

# CLI usage
python main.py --compare  # View all experiments
```

### Manifest Structure

Experiments tracked in `output/manifest.json`:
```json
{
  "experiments": [
    {
      "run_id": "run_20251118_143022",
      "experiment_name": "prod-baseline-r01",
      "runtime_mode": "baseline",
      "timestamp": "2025-11-18T14:30:22",
      "output_dir": "output/evaluation_20251118_143022_baseline",
      "config_snapshot": {...},
      "final_fitness": [0, 123.45],
      "duration_seconds": 3600.5,
      "tags": ["production", "ablation"],
      "notes": "Baseline run for comparison"
    }
  ]
}
```

### Best Practices

- **Use ExperimentManager for all production runs**: Ensures structured tracking
- **Tag experiments meaningfully**: Use tags like `["ablation", "production", "gpu"]`
- **Include notes**: Document experiment purpose and observations
- **Compare systematically**: Run all modes with same data for fair comparison
- **Archive manifest**: Commit `manifest.json` to git for reproducibility

## Never Do
-  Skip stage ordering (must follow: load → validate → feasibility → GA → decode → report)
-  Create output directory after GA runs (need it for logging)
-  Forget to close multiprocessing pool
-  Modify context between stages (should be immutable)
-  Hardcode paths (use parameters)
-  Skip error handling (catch and log all exceptions)
-  Run production experiments without ExperimentManager (lose tracking)
-  Forget to register experiment runs (breaks reproducibility)
-  Skip runtime mode validation (always validate killswitches)
