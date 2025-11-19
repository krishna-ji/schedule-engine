# High-Level Architecture

## System Overview

Schedule Engine is a modular constraint-based optimization system that combines genetic algorithms (NSGA-II), reinforcement learning, and local search to solve university course timetabling problems.

```mermaid
graph TB
    A[CLI Entry Point<br/>main.py] --> B[Workflow Orchestrator<br/>workflows/]
    B --> C[Data Encoder<br/>encoder/]
    B --> D[Validation<br/>validation/]
    B --> E[GA Scheduler<br/>core/ga_scheduler.py]
    B --> F[Result Decoder<br/>decoder/]
    B --> G[Report Exporter<br/>exporter/]
    
    C --> H[Domain Entities<br/>entities/]
    E --> I[GA Operators<br/>ga/operators/]
    E --> J[Fitness Evaluation<br/>ga/evaluator/]
    E --> K[Heuristic Toolbox<br/>heuristics/]
    E --> L[RL Integration<br/>rl/]
    E --> M[Repair System<br/>ga/operators/repair_igls.py]
    
    J --> N[Constraints<br/>constraints/]
    J --> O[GPU Acceleration<br/>ga/evaluator/gpu_batch_evaluator.py]
    
    L --> P[Gym Environment<br/>rl/gym_env/]
    L --> Q[RL Agents<br/>rl/agents/]
    L --> R[Training<br/>rl/training/]
    L --> S[Deployment<br/>rl/deployment/]
    
    style A fill:#e1f5ff
    style E fill:#ffe1e1
    style J fill:#ffe1f5
    style L fill:#e1ffe1
    style O fill:#fff5e1
```

## Core Components

### 1. Entry Layer (`main.py`)
**Purpose:** CLI interface and workflow coordination

**Responsibilities:**
- Parse command-line arguments
- Initialize configuration
- Route to appropriate workflow
- Handle experiment management

**Key Files:**
- `main.py` - Entry point with environment/mode routing
- `src/workflows/standard_run.py` - Standard workflow orchestration

### 2. Data Layer
**Purpose:** Data loading, validation, and domain modeling

```mermaid
graph LR
    A[JSON Files<br/>data/] --> B[Input Encoder<br/>encoder/]
    B --> C[Domain Entities<br/>entities/]
    C --> D[Validation<br/>validation/]
    D --> E[Scheduling Context<br/>core/types.py]
    
    style A fill:#e1f5ff
    style C fill:#ffe1e1
    style E fill:#ffe1f5
```

**Components:**
- **Encoder** (`src/encoder/`) - JSON → Python objects
- **Entities** (`src/entities/`) - Domain models (Course, Instructor, Room, Group)
- **Validation** (`src/validation/`) - Input integrity + feasibility checks
- **Time System** (`src/encoder/quantum_time_system.py`) - Time discretization

### 3. Optimization Core
**Purpose:** Genetic algorithm execution and population management

```mermaid
graph TB
    A[GA Scheduler<br/>ga_scheduler.py] --> B[Population Init<br/>ga/population.py]
    A --> C[GA Operators<br/>ga/operators/]
    A --> D[Selection<br/>NSGA-II]
    A --> E[Fitness Evaluation<br/>ga/evaluator/]
    
    C --> F[Crossover<br/>crossover.py]
    C --> G[Mutation<br/>mutation.py]
    C --> H[Repair<br/>repair_igls.py]
    
    E --> I[Constraint Evaluation<br/>constraints/]
    E --> J[GPU Batch Evaluator<br/>gpu_batch_evaluator.py]
    
    style A fill:#ffe1e1
    style E fill:#ffe1f5
    style I fill:#e1ffe1
    style J fill:#fff5e1
```

**Components:**
- **GA Scheduler** (`src/core/ga_scheduler.py`) - Main evolution loop
- **Population** (`src/ga/population.py`) - Individual generation
- **Operators** (`src/ga/operators/`) - Crossover, mutation, repair
- **Evaluator** (`src/ga/evaluator/`) - Fitness calculation (CPU/GPU)

### 4. Constraint System
**Purpose:** Evaluate hard/soft constraint violations

**Architecture:**
- **Hard Constraints** - Must be satisfied (0 violations required)
- **Soft Constraints** - Preferences (minimize violations)
- **Weighted Aggregation** - Configurable penalty weights

**Evaluation Modes:**
- **CPU Sequential** - Standard evaluation
- **CPU Parallel** - Multiprocessing (3-5x speedup)
- **GPU Batch** - CUDA acceleration (10-50x speedup)

**Key Files:**
- `src/constraints/hard_*.py` - Hard constraint functions
- `src/constraints/soft_*.py` - Soft constraint functions
- `src/ga/evaluator/gpu_batch_evaluator.py` - GPU implementation

### 5. Heuristic System
**Purpose:** 19 specialized repair/improvement operators

```mermaid
graph TB
    A[Heuristic Registry<br/>heuristics/registry.py] --> B[Construction Operators]
    A --> C[Perturbation Operators]
    A --> D[Repair Operators]
    A --> E[Optimization Operators]
    A --> F[Diversity Operators]
    
    B --> G[Room Assignment<br/>Greedy/Smart]
    C --> H[Swap/Shift<br/>Time/Room/Instructor]
    D --> I[Violated Session Repair<br/>Targeted Fixes]
    E --> J[Block Consolidation<br/>Fragment Reduction]
    F --> K[Diversity Injection<br/>Novelty Search]
    
    style A fill:#e1f5ff
    style D fill:#ffe1e1
    style K fill:#ffe1f5
```

**Categories:**
1. **Construction** (2 ops) - Build schedules from scratch
2. **Perturbation** (9 ops) - Modify existing schedules
3. **Repair** (3 ops) - Fix specific violations
4. **Optimization** (3 ops) - Improve quality
5. **Diversity** (2 ops) - Maintain population variety

**Execution:**
- **Sequential** - One heuristic at a time
- **Parallel** - Multiple heuristics concurrently (3-5x speedup)
- **RL-Guided** - Adaptive selection via trained agent

### 6. RL Integration
**Purpose:** Learn optimal heuristic selection strategies

```mermaid
graph TB
    A[RL Agent<br/>PPO/DQN] --> B[State Encoder<br/>25D observation]
    A --> C[Action Mapper<br/>20 discrete actions]
    A --> D[Reward Calculator<br/>Multi-component]
    
    B --> E[Population Metrics<br/>Fitness, Diversity]
    B --> F[Constraint Breakdown<br/>Per-type violations]
    B --> G[Historical Context<br/>Last 5 generations]
    
    C --> H[Heuristic Selection<br/>19 operators]
    C --> I[Probability Control<br/>cxpb/mutpb adjust]
    
    D --> J[Fitness Improvement<br/>Delta reward]
    D --> K[Diversity Bonus<br/>Exploration reward]
    D --> L[Time Penalty<br/>Computational cost]
    
    style A fill:#e1ffe1
    style B fill:#ffe1e1
    style C fill:#ffe1f5
    style D fill:#fff5e1
```

**Components:**
- **Gym Environment** (`src/rl/gym_env/`) - OpenAI Gym interface
- **Agents** (`src/rl/agents/`) - PPO/DQN implementations
- **Training** (`src/rl/training/`) - Curriculum learning + checkpoints
- **Deployment** (`src/rl/deployment/`) - Model loading + inference

### 7. Configuration System
**Purpose:** Hierarchical YAML-based configuration

```mermaid
graph TB
    A[base.yaml<br/>Common Settings] --> B[Environment Configs<br/>test/prod]
    B --> C[Runtime Mode Configs<br/>10 modes]
    C --> D[Config Loader<br/>Deep Merge]
    D --> E[Pydantic Validation<br/>Type Checking]
    E --> F[Global Config Object<br/>get_config()]
    
    style A fill:#e1f5ff
    style C fill:#ffe1e1
    style E fill:#ffe1f5
```

**Hierarchy:**
1. `configs/base.yaml` - Base settings (shared)
2. `configs/{env}.yaml` - Environment overrides (test/prod)
3. `configs/{category}/{mode}.yaml` - Runtime modes (10 modes)

**Loading:** Deep merge with Pydantic validation

### 8. Output Layer
**Purpose:** Result decoding and report generation

**Components:**
- **Decoder** (`src/decoder/`) - Individual → CourseSession mapping
- **Exporter** (`src/exporter/`) - JSON/PDF/Plot generation
- **Metrics** (`src/metrics/`) - Diversity, convergence, performance

## Data Flow

### End-to-End Pipeline

```mermaid
sequenceDiagram
    participant CLI as CLI (main.py)
    participant Config as Config Loader
    participant Encoder as Data Encoder
    participant Validator as Validator
    participant GA as GA Scheduler
    participant Evaluator as Fitness Evaluator
    participant Decoder as Result Decoder
    participant Exporter as Report Exporter
    
    CLI->>Config: Load config (env + mode)
    Config-->>CLI: Config object
    
    CLI->>Encoder: Load data (JSON files)
    Encoder-->>CLI: Scheduling context
    
    CLI->>Validator: Validate input
    Validator-->>CLI: Validation report
    
    CLI->>Validator: Feasibility checks
    Validator-->>CLI: Feasibility report
    
    CLI->>GA: Initialize population
    GA-->>GA: Generate individuals
    
    loop Evolution (ngen generations)
        GA->>Evaluator: Evaluate population
        Evaluator-->>GA: Fitness values
        
        alt RL Enabled
            GA->>RL: Get heuristic action
            RL-->>GA: Selected operator
        end
        
        GA->>GA: Apply operators (crossover/mutation)
        GA->>GA: NSGA-II selection
        
        alt Stagnation Detected
            GA->>Repair: Trigger IGLS repair
            Repair-->>GA: Repaired individual
        end
    end
    
    GA-->>CLI: Best individual
    
    CLI->>Decoder: Decode schedule
    Decoder-->>CLI: Course sessions
    
    CLI->>Exporter: Generate reports
    Exporter-->>CLI: Output files
    
    CLI->>CLI: Display results
```

### GA Evolution Cycle

```mermaid
graph TB
    A[Population<br/>N individuals] --> B{Evaluate Fitness<br/>GPU/CPU}
    B --> C[NSGA-II Selection<br/>Pareto ranking]
    C --> D{Apply Operators}
    
    D --> E[Crossover<br/>cxpb probability]
    D --> F[Mutation<br/>mutpb probability]
    D --> G{Heuristics Enabled?}
    
    G -->|Yes| H{RL Enabled?}
    G -->|No| I[Standard Operators]
    
    H -->|Yes| J[RL Agent Selects<br/>Operator]
    H -->|No| K[Random/Round-Robin<br/>Selection]
    
    J --> L[Apply Selected<br/>Heuristic]
    K --> L
    I --> M[Next Generation]
    L --> M
    
    M --> N{Stagnation?}
    N -->|Yes| O[IGLS Repair<br/>Best Individual]
    N -->|No| P[Continue]
    
    O --> P
    P --> Q{Termination?}
    Q -->|No| A
    Q -->|Yes| R[Return Best]
    
    style B fill:#ffe1f5
    style G fill:#e1ffe1
    style H fill:#e1ffe1
    style J fill:#e1ffe1
    style O fill:#ffe1e1
```

## Design Principles

### 1. Modularity
- **Clear separation of concerns** - Each module has single responsibility
- **Loose coupling** - Modules communicate via interfaces
- **High cohesion** - Related functionality grouped together

### 2. Extensibility
- **Plugin architecture** - New heuristics register in registry
- **Config-driven** - Behavior controlled via YAML
- **Killswitches** - Enable/disable features without code changes

### 3. Performance
- **GPU acceleration** - 10-50x speedup for constraint evaluation
- **Parallel execution** - Multiprocessing for operators
- **Lazy loading** - Models loaded on demand
- **Caching** - Reuse computed results

### 4. Reliability
- **Input validation** - Pydantic type checking
- **Feasibility checks** - Pre-flight constraint analysis
- **Error handling** - Graceful degradation (GPU → CPU fallback)
- **Testing** - >80% code coverage target

### 5. Reproducibility
- **Deterministic execution** - Seed control
- **Experiment tracking** - Manifest-based logging
- **Version control** - Config + code + data versioning
- **Documentation** - Comprehensive guides

## Technology Stack

### Core Libraries

**Genetic Algorithms:**
- **DEAP 1.4.1** - GA framework (toolbox, operators, selection)
- Custom NSGA-II implementation with multi-objective optimization

**Reinforcement Learning:**
- **Stable-Baselines3 2.3.2** - PPO/DQN agents
- **Gymnasium 0.29.1** - OpenAI Gym environment
- **PyTorch 2.4.1+cu121** - Neural network backend

**GPU Acceleration:**
- **PyTorch CUDA 12.1** - GPU batch evaluation
- **NumPy 1.26.4** - Scientific computing
- 10-50x speedup on NVIDIA GPUs

**Configuration & Validation:**
- **Pydantic 2.10.3** - Config validation + type checking
- **PyYAML 6.0.2** - YAML parsing
- Hierarchical config with deep merge

**Visualization & UI:**
- **Rich 13.9.4** - Terminal UI (progress bars, tables)
- **Matplotlib 3.9.4** - Plotting (evolution curves, Pareto fronts)
- **Seaborn 0.13.2** - Statistical plots

### Development Tools

- **Python 3.12** - Language (pinned exact version)
- **UV** - Package manager (modern, fast)
- **pytest** - Testing framework
- **black** - Code formatter (88 line length)
- **ruff** - Linter (fast, comprehensive)
- **mypy** - Type checker (static analysis)

## Performance Characteristics

### Execution Time

| Configuration | Time (without GPU) | Time (with GPU) | Speedup |
|---------------|-------------------|-----------------|---------|
| Test (30 gens) | 5-10 min | 1-2 min | 3-5x |
| Prod (2000 gens) | 24-48 hours | 1-2.5 hours | 13-34x |

### Speedup Breakdown

1. **GPU Batch Evaluation** - 10-50x (constraint evaluation)
2. **Parallel Operators** - 3-5x (crossover/mutation)
3. **Parallel Feasibility** - 3-5x (validation)
4. **Combined** - 13-34x (end-to-end)

### Scalability

- **Population size** - Linear scaling (O(n))
- **Constraint evaluation** - Linear with individuals (O(n))
- **Heuristic execution** - Parallel (independent operations)
- **GPU memory** - Batch size limited by VRAM (auto-tune)

## System Requirements

### Minimum

- **CPU:** 4 cores
- **RAM:** 8 GB
- **Disk:** 5 GB
- **GPU:** None (CPU fallback)

### Recommended

- **CPU:** 8+ cores (Intel i7/AMD Ryzen 7)
- **RAM:** 16 GB
- **Disk:** 10 GB SSD
- **GPU:** NVIDIA RTX 3060+ (8GB VRAM, CUDA 12.1)

### Optimal (Thesis Experiments)

- **CPU:** 16+ cores (Intel i9/AMD Ryzen 9)
- **RAM:** 32 GB
- **Disk:** 20 GB NVMe SSD
- **GPU:** NVIDIA RTX 3080+ (10GB VRAM, CUDA 12.1)

## Deployment Modes

### 1. Development
- Test environment (`--env test`)
- Quick iterations (30 gens)
- Debug logging enabled

### 2. Production
- Prod environment (`--env prod`)
- Full runs (2000 gens)
- Result logging only

### 3. Research
- Custom configs
- Long runs (5000+ gens)
- Extensive metrics collection

## See Also

- [Design Principles](02-design-principles.md) - Detailed design rationale
- [Components](03-components/) - Module-level documentation
- [Data Flow](04-data-flow.md) - Detailed sequence diagrams
- [Code Organization](../code/01-code-structure.md) - File layout guide
