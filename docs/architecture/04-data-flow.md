# Data Flow Architecture

## Complete System Data Flow

```mermaid
flowchart TD
    Start([User Input]) --> CLI[CLI Parser<br/>main.py]
    CLI --> LoadConfig[Load Configuration<br/>configs/*.yaml]
    LoadConfig --> LoadData[Load JSON Data<br/>data/]
    
    LoadData --> Course[Course.json<br/>150 courses]
    LoadData --> Groups[Groups.json<br/>30 groups]
    LoadData --> Instructors[Instructors.json<br/>50 instructors]
    LoadData --> Rooms[Rooms.json<br/>40 rooms]
    
    Course --> Encoder[Input Encoder<br/>encoder/]
    Groups --> Encoder
    Instructors --> Encoder
    Rooms --> Encoder
    
    Encoder --> Entities[Domain Entities<br/>Course, Group, Instructor, Room]
    Entities --> TimeSystem[Quantum Time System<br/>Discretization]
    TimeSystem --> Context[Scheduling Context<br/>Complete problem data]
    
    Context --> Validation[Input Validation<br/>validation/input_validator.py]
    Validation --> Feasibility[Feasibility Checks<br/>validation/feasibility_checker.py]
    
    Feasibility -->|Pass| InitPop[Initialize Population<br/>ga/population.py]
    Feasibility -->|Fail| Report1[Feasibility Report<br/>Warning/Error]
    
    InitPop --> Pop[Population<br/>N individuals]
    
    Pop --> EvalLoop{Evolution Loop<br/>ngen generations}
    
    EvalLoop --> Evaluate[Fitness Evaluation<br/>ga/evaluator/]
    Evaluate -->|GPU Available| GPU[GPU Batch Evaluator<br/>10-50x speedup]
    Evaluate -->|CPU Only| CPU[CPU Sequential Evaluator]
    
    GPU --> Constraints[Constraint Evaluation<br/>constraints/]
    CPU --> Constraints
    
    Constraints --> Hard[Hard Constraints<br/>14 types]
    Constraints --> Soft[Soft Constraints<br/>4 types]
    
    Hard --> Fitness[Fitness Values<br/>-hard, -soft]
    Soft --> Fitness
    
    Fitness --> Selection[NSGA-II Selection<br/>Pareto ranking]
    Selection --> Elite[Elite Preservation<br/>10% best]
    Selection --> Breeding[Breeding Pool<br/>90%]
    
    Breeding --> Operators{Apply Operators}
    
    Operators --> Crossover[Crossover<br/>cxpb=0.75]
    Operators --> Mutation[Mutation<br/>mutpb=0.25]
    Operators -->|Heuristics Enabled| HeurChoice{Heuristic Selection}
    
    HeurChoice -->|RL Enabled| RLAgent[RL Agent<br/>PPO/DQN]
    HeurChoice -->|RL Disabled| Random[Random/Round-Robin]
    
    RLAgent --> StateEnc[State Encoder<br/>25D observation]
    StateEnc --> RLModel[RL Model<br/>Neural network]
    RLModel --> Action[Action Mapper<br/>20 actions]
    Action --> SelectHeur[Selected Heuristic<br/>1 of 19 operators]
    
    Random --> SelectHeur
    SelectHeur --> ApplyHeur[Apply Heuristic<br/>heuristics/]
    
    Crossover --> Offspring[Offspring<br/>New generation]
    Mutation --> Offspring
    ApplyHeur --> Offspring
    
    Offspring --> Stagnation{Stagnation<br/>Detected?}
    Stagnation -->|Yes| Repair[IGLS Repair<br/>repair_igls.py]
    Stagnation -->|No| NextGen[Next Generation]
    Repair --> NextGen
    
    NextGen --> Elite
    Elite --> NewPop[Updated Population]
    NewPop --> Termination{Termination<br/>Criteria?}
    
    Termination -->|No| EvalLoop
    Termination -->|Yes| BestInd[Best Individual<br/>Chromosome]
    
    BestInd --> Decoder[Result Decoder<br/>decoder/]
    Decoder --> Sessions[Course Sessions<br/>450 sessions]
    
    Sessions --> Exporter[Report Exporter<br/>exporter/]
    Exporter --> JSON[schedule.json<br/>Detailed data]
    Exporter --> PDF[calendar.pdf<br/>Visual timetable]
    Exporter --> Plots[plots/<br/>Evolution curves]
    Exporter --> Report2[report.txt<br/>Summary]
    
    JSON --> Output([Output Directory<br/>output/])
    PDF --> Output
    Plots --> Output
    Report2 --> Output
    Report1 --> Output
    
    style CLI fill:#e1f5ff
    style Evaluate fill:#ffe1f5
    style GPU fill:#fff5e1
    style RLAgent fill:#e1ffe1
    style Decoder fill:#ffe1e1
```

## Message Passing Patterns

### 1. Configuration Propagation

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant ConfigLoader
    participant BaseYAML
    participant EnvYAML
    participant ModeYAML
    participant Pydantic
    participant Components
    
    User->>CLI: --env prod --mode full
    CLI->>ConfigLoader: load_config(mode, env)
    ConfigLoader->>BaseYAML: Load base.yaml
    BaseYAML-->>ConfigLoader: Base settings
    ConfigLoader->>EnvYAML: Load prod.yaml
    EnvYAML-->>ConfigLoader: Environment overrides
    ConfigLoader->>ModeYAML: Load 4-nsga-full.yaml
    ModeYAML-->>ConfigLoader: Mode-specific settings
    ConfigLoader->>ConfigLoader: Deep merge (base + env + mode)
    ConfigLoader->>Pydantic: Validate merged config
    Pydantic-->>ConfigLoader: Validated config object
    ConfigLoader-->>CLI: Final config
    CLI->>Components: Pass config to all modules
    Components->>Components: Access via get_config()
```

### 2. Data Encoding Pipeline

```mermaid
sequenceDiagram
    participant JSON as JSON Files
    participant Encoder as Input Encoder
    participant Linker as Cross-Referencer
    participant TimeSystem as Time System
    participant Entities as Domain Entities
    participant Context as Scheduling Context
    
    JSON->>Encoder: Load Course.json
    Encoder-->>Entities: Course objects
    
    JSON->>Encoder: Load Groups.json
    Encoder-->>Entities: Group objects
    
    JSON->>Encoder: Load Instructors.json
    Encoder-->>Entities: Instructor objects
    
    JSON->>Encoder: Load Rooms.json
    Encoder-->>Entities: Room objects
    
    Entities->>Linker: Cross-reference entities
    Linker->>Linker: Link courses  groups
    Linker->>Linker: Link courses  instructors
    Linker-->>Entities: Updated relationships
    
    Entities->>TimeSystem: Initialize time system
    TimeSystem->>TimeSystem: Create quantum mapping
    TimeSystem-->>Entities: Time quantum lookup
    
    Entities->>Context: Build scheduling context
    Context-->>Context: SchedulingContext object
    Note over Context: Contains all problem data<br/>Passed to GA scheduler
```

### 3. Fitness Evaluation Flow

```mermaid
sequenceDiagram
    participant GA as GA Scheduler
    participant Evaluator as Fitness Evaluator
    participant GPU as GPU Batch Evaluator
    participant Constraints as Constraint Functions
    participant Individual as Individual
    
    GA->>Evaluator: Evaluate population (N individuals)
    Evaluator->>Evaluator: Check GPU availability
    
    alt GPU Available
        Evaluator->>GPU: Batch evaluate (N individuals)
        GPU->>GPU: Prepare batch tensors
        GPU->>Constraints: Evaluate all constraints (GPU)
        Constraints-->>GPU: Violation tensors
        GPU->>GPU: Aggregate violations
        GPU-->>Evaluator: Fitness values (batch)
    else CPU Only
        loop For each individual
            Evaluator->>Constraints: Evaluate constraints (CPU)
            Constraints-->>Evaluator: Violation counts
            Evaluator->>Evaluator: Calculate fitness
        end
    end
    
    Evaluator->>Individual: Set fitness attribute
    Individual-->>Evaluator: Updated individuals
    Evaluator-->>GA: Population with fitness
```

### 4. RL Agent Interaction

```mermaid
sequenceDiagram
    participant GA as GA Scheduler
    participant RLController as RL Controller
    participant StateEnc as State Encoder
    participant Agent as RL Agent (PPO/DQN)
    participant ActionMap as Action Mapper
    participant Heuristic as Heuristic Operator
    
    GA->>RLController: Request heuristic action
    RLController->>StateEnc: Encode current state
    StateEnc->>StateEnc: Extract population metrics
    StateEnc->>StateEnc: Extract constraint breakdown
    StateEnc->>StateEnc: Extract historical context
    StateEnc-->>RLController: 25D state vector
    
    RLController->>Agent: Predict action (state)
    Agent->>Agent: Forward pass (neural network)
    Agent-->>RLController: Action ID (0-19)
    
    RLController->>ActionMap: Map action to heuristic
    ActionMap->>ActionMap: Lookup heuristic in registry
    ActionMap-->>RLController: Heuristic function
    
    RLController->>Heuristic: Apply heuristic (individual)
    Heuristic->>Heuristic: Execute operator logic
    Heuristic-->>RLController: Modified individual
    
    RLController->>RLController: Calculate reward
    RLController->>Agent: Store experience (s, a, r, s')
    RLController-->>GA: Modified individual
```

### 5. Repair System Trigger

```mermaid
sequenceDiagram
    participant GA as GA Scheduler
    participant Monitor as Stagnation Monitor
    participant Repair as IGLS Repair System
    participant SubProblem as Subproblem Solver
    participant Individual as Best Individual
    
    loop Every generation
        GA->>Monitor: Update best fitness
        Monitor->>Monitor: Check for improvement
        
        alt No improvement for N generations
            Monitor-->>GA: Stagnation detected!
            GA->>Repair: Trigger repair (best individual)
            Repair->>Individual: Extract violated sessions
            Individual-->>Repair: Conflicted sessions (subset)
            
            Repair->>SubProblem: Solve subproblem (IGLS)
            SubProblem->>SubProblem: Iterative greedy local search
            SubProblem->>SubProblem: Try moves (swap/shift)
            SubProblem->>SubProblem: Accept improving moves
            SubProblem-->>Repair: Repaired subset
            
            Repair->>Repair: Reintegrate into schedule
            Repair->>Repair: Validate repair
            
            alt Repair successful
                Repair-->>GA: Repaired individual (better)
                GA->>Monitor: Reset stagnation counter
            else Repair failed
                Repair-->>GA: Original individual (unchanged)
                GA->>Monitor: Increment stagnation counter
            end
        else Improvement detected
            Monitor->>Monitor: Reset stagnation counter
        end
    end
```

### 6. Result Export Pipeline

```mermaid
sequenceDiagram
    participant GA as GA Scheduler
    participant Decoder as Result Decoder
    participant SessionBuilder as Session Builder
    participant Exporter as Report Exporter
    participant JSONWriter as JSON Writer
    participant PDFGenerator as PDF Generator
    participant PlotGenerator as Plot Generator
    participant FileSystem as Output Directory
    
    GA->>Decoder: Decode best individual
    Decoder->>SessionBuilder: Build course sessions
    
    loop For each session gene
        SessionBuilder->>SessionBuilder: Extract course
        SessionBuilder->>SessionBuilder: Extract time slot
        SessionBuilder->>SessionBuilder: Extract room
        SessionBuilder->>SessionBuilder: Extract instructor
        SessionBuilder->>SessionBuilder: Create CourseSession
    end
    
    SessionBuilder-->>Decoder: List of CourseSession objects
    Decoder-->>GA: Decoded schedule (450 sessions)
    
    GA->>Exporter: Generate reports
    
    par JSON Export
        Exporter->>JSONWriter: Serialize schedule
        JSONWriter->>JSONWriter: Convert to JSON
        JSONWriter->>FileSystem: Write schedule.json
    and PDF Export
        Exporter->>PDFGenerator: Generate calendar
        PDFGenerator->>PDFGenerator: Create timetable grid
        PDFGenerator->>PDFGenerator: Add sessions (color-coded)
        PDFGenerator->>FileSystem: Write calendar.pdf
    and Plot Export
        Exporter->>PlotGenerator: Generate evolution plots
        PlotGenerator->>PlotGenerator: Fitness evolution curve
        PlotGenerator->>PlotGenerator: Diversity evolution curve
        PlotGenerator->>PlotGenerator: Pareto front scatter
        PlotGenerator->>FileSystem: Write plots/*.png
    end
    
    Exporter->>FileSystem: Write report.txt (summary)
    FileSystem-->>Exporter: Output directory path
    Exporter-->>GA: Export complete
```

## Component Communication

### Global State Management

**Config Access Pattern:**
```python
# Singleton pattern via module-level cache
from src.config import get_config

config = get_config()  # Returns cached config
print(config.ga.ngen)  # Access nested settings
```

**Scheduling Context:**
```python
# Passed explicitly through function calls
context = SchedulingContext(
    courses=courses,
    groups=groups,
    instructors=instructors,
    rooms=rooms,
    time_system=time_system
)

# Passed to GA scheduler
scheduler = GAScheduler(context, config)
```

### Event-Driven Communication

**Callback System (RL Training):**
```python
# Custom callbacks for training events
callbacks = [
    PeriodicEvaluationCallback(eval_freq=5000),
    EarlyStoppingCallback(patience=5),
    CheckpointCallback(save_freq=10000)
]

trainer.train(callbacks=callbacks)
```

**Progress Tracking (Rich):**
```python
# Live progress updates
with Progress() as progress:
    task = progress.add_task("Evolution", total=ngen)
    for gen in range(ngen):
        # ... evolution logic ...
        progress.update(task, advance=1)
```

### Parallel Communication

**Multiprocessing (CPU):**
```python
# Worker pool for parallel evaluation
with ThreadPoolExecutor(max_workers=num_cores) as executor:
    futures = [executor.submit(evaluate, ind) for ind in population]
    results = [f.result() for f in futures]
```

**GPU Batch Communication:**
```python
# Single GPU call for batch
fitness_batch = gpu_evaluator.evaluate_batch(
    population,
    batch_size=100
)
```

## Data Structures

### Core Data Types

**Individual (Chromosome):**
```python
# DEAP Individual (list of SessionGene)
Individual = list[SessionGene]

# SessionGene structure
@dataclass
class SessionGene:
    course_id: str
    session_index: int  # Which session of the course
    time_quantum_start: int  # Discretized time
    duration_quanta: int  # Duration in quanta
    room_id: str
    instructor_id: str
    group_ids: list[str]
```

**Fitness:**
```python
# Two-objective minimization
fitness = (
    -hard_violations,  # First objective (minimize)
    -soft_penalty      # Second objective (minimize)
)

# Pareto dominance: fitness1 dominates fitness2 if:
# - fitness1 is no worse in all objectives
# - fitness1 is better in at least one objective
```

**Scheduling Context:**
```python
@dataclass
class SchedulingContext:
    courses: list[Course]
    groups: list[Group]
    instructors: list[Instructor]
    rooms: list[Room]
    time_system: QuantumTimeSystem
    config: Config
```

### Message Formats

**RL State (25D vector):**
```python
state = [
    gen / max_gen,                    # Progress
    best_hard / total_hard,           # Hard violation ratio
    best_soft / max_soft,             # Soft penalty ratio
    avg_hard / total_hard,            # Population avg hard
    avg_soft / max_soft,              # Population avg soft
    diversity,                        # Genotypic diversity
    stagnation / max_stagnation,      # Stagnation ratio
    # ... 18 more features
]
```

**RL Action (discrete):**
```python
action_id = 0..19  # Maps to heuristic operator

# Action mapping
ACTION_MAP = {
    0: "greedy_construction",
    1: "smart_construction",
    2: "swap_rooms_all",
    # ... 17 more operators
}
```

**Constraint Violation Report:**
```python
{
    "hard_constraints": {
        "student_group_exclusivity": 15,  # 15 violations
        "instructor_exclusivity": 20,
        "room_exclusivity": 15,
        # ... other constraints
    },
    "soft_constraints": {
        "avoid_early_sessions": 12,
        "avoid_late_sessions": 8,
        # ... other constraints
    },
    "total_hard": 50,
    "total_soft": 28.7
}
```

## See Also

- [High-Level Architecture](01-high-level-architecture.md) - System overview
- [Components](03-components/) - Module details
- [Code Structure](../code/01-code-structure.md) - File organization
