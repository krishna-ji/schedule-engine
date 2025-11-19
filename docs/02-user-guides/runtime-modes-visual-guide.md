# Runtime Mode Architecture - Visual Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Schedule Engine CLI                      │
│                                                              │
│  python main.py --mode baseline --env prod --experiment X   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Mode Selector                     │
│                                                              │
│  RuntimeMode.from_string("baseline")                        │
│  → RuntimeMode.BASELINE                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Config Loader                            │
│                                                              │
│  1. Load configs/base.yaml (common settings)                │
│  2. Load configs/baseline/1-pure-nsga.yaml (overrides)      │
│  3. Deep merge (base + overrides)                           │
│  4. Validate killswitches (repair=false, rl=false)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Experiment Manager                          │
│                                                              │
│  1. Create output dir: output/baseline/pure-nsga/eval_...   │
│  2. Register run in manifest.json                           │
│  3. Track experiment metadata                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GA Scheduler (NSGA-II)                     │
│                                                              │
│  - Population initialization (random/smart/hybrid)          │
│  - Evolution loop (crossover + mutation)                    │
│  - Optional: Repairs (IGLS)                                 │
│  - Optional: Heuristics (Phase 1.5)                         │
│  - Optional: Local search (LNS-IGLS)                        │
│  - Optional: RL guidance (PPO agent)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Results & Reports                         │
│                                                              │
│  - best_schedule.json                                       │
│  - schedule_report.pdf                                      │
│  - evolution_plots.png                                      │
│  - metrics.json                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Experiment Manager Update                     │
│                                                              │
│  - Update manifest.json with results                        │
│  - Record duration, fitness, hypervolume                    │
│  - Enable comparison analysis                               │
└─────────────────────────────────────────────────────────────┘
```

---

##  Mode Progression Flow

```
┌──────────────┐
│   BASELINE   │  Mode 1: Pure NSGA-II
│   (Start)    │  - Random initialization
└──────┬───────┘  - Basic crossover/mutation
       │          - NO enhancements
       │
       ▼
┌──────────────┐
│   + REPAIRS  │  Mode 2: NSGA-II + Repairs
│              │  - IGLS repair system
└──────┬───────┘  - Hybrid initialization
       │          - Stagnation repair
       │
       ▼
┌──────────────┐
│ + HEURISTICS │  Mode 3: NSGA-II + Repairs + Heuristics
│              │  - 19 heuristic operators
└──────┬───────┘  - Construction/perturbation/improvement
       │          - Diversity/meta-heuristics
       │
       ▼
┌──────────────┐
│ + LOCAL SRCH │  Mode 4: NSGA-II + Full
│   (Best GA)  │  - Memetic local search (LNS-IGLS)
└──────┬───────┘  - All enhancements enabled
       │          - Adaptive probabilities
       │
       ├────────────┐
       │            │
       ▼            ▼
┌─────────┐  ┌─────────────┐
│   + RL  │  │ ROUND-ROBIN │  Mode 5: RL-Guided
│         │  │             │  - RL agent controls heuristic selection
└─────────┘  └─────────────┘  Mode 6: Round-Robin
                              - Fixed rotation (RL baseline)
```

---

##  Config Inheritance

```
base.yaml (Common Settings)
    │
    ├─► baseline/1-pure-nsga.yaml
    │   └─► ALL FEATURES = OFF
    │
    ├─► nsga/2-nsga-repairs.yaml
    │   └─► repair.enabled = true
    │
    ├─► nsga/3-nsga-heuristics.yaml
    │   └─► repair.enabled = true
    │       heuristics.*.enabled = true
    │
    ├─► nsga/4-nsga-full.yaml
    │   └─► repair.enabled = true
    │       heuristics.*.enabled = true
    │       repair.memetic_mode = true
    │       lns.enabled = true
    │       enhancements.master_enabled = true
    │
    ├─► rl/5-rl-guided.yaml
    │   └─► (all from mode 4)
    │       rl.enabled = true
    │       rl.mode = inference
    │
    └─► hybrid/6-roundrobin.yaml
        └─► (all from mode 4)
            ga.use_adaptive_probabilities = false
```

---

## 🗂️ Output Organization Tree

```
output/
├── experiment_manifest.json        Master tracking database
│
├── baseline/
│   └── pure-nsga/
│       ├── evaluation_20251118_140530_exp1/
│       │   ├── best_schedule.json
│       │   ├── schedule_report.pdf
│       │   ├── evolution_plots.png
│       │   └── metrics.json
│       └── evaluation_20251118_153200_exp2/
│
├── nsga/
│   ├── nsga-repairs/
│   │   └── evaluation_20251118_141000_exp1/
│   ├── nsga-heuristics/
│   │   └── evaluation_20251118_142000_exp1/
│   └── nsga-full/
│       └── evaluation_20251118_143000_exp1/
│
├── rl/
│   └── rl-guided/
│       └── evaluation_20251118_144000_exp1/
│
└── hybrid/
    └── roundrobin/
        └── evaluation_20251118_145000_exp1/
```

---

##  Decision Flow

```
User runs: python main.py --mode baseline --env prod

    ┌─────────────────┐
    │  Parse CLI Args │
    └────────┬────────┘
             │
    ┌────────▼────────────────────┐
    │  mode = "baseline"          │  ◄── User input
    │  env = "prod"                │
    │  experiment = None           │
    └────────┬────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  RuntimeMode.from_string("baseline")      │  ◄── Parse mode
    │  → RuntimeMode.BASELINE                   │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Get config path:                         │  ◄── Resolve path
    │  configs/baseline/1-pure-nsga.yaml        │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Load & merge:                            │  ◄── Load config
    │  1. base.yaml                             │
    │  2. 1-pure-nsga.yaml                      │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Validate killswitches:                   │  ◄── Validate
    │  - repair.enabled = false               │
    │  - rl.enabled = false                   │
    │  - enhancements.master_enabled = false  │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Create output dir:                       │  ◄── Organize
    │  output/baseline/pure-nsga/eval_...       │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Register in manifest.json                │  ◄── Track
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Run GA with config:                      │  ◄── Execute
    │  - pop_size = 200                         │
    │  - ngen = 2000                            │
    │  - repairs = OFF                          │
    │  - heuristics = OFF                       │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │  Export results & update manifest         │  ◄── Report
    └───────────────────────────────────────────┘
```

---

##  Killswitch Validation

```
┌─────────────────────────────────────────────────────────┐
│              RuntimeMode.validate_config()               │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   BASELINE    │  │   NSGA_FULL   │  │   RL_GUIDED   │
├───────────────┤  ├───────────────┤  ├───────────────┤
│ Must have:    │  │ Must have:    │  │ Must have:    │
│               │  │               │  │               │
│ repair =    │  │ repair =    │  │ rl =        │
│ rl =        │  │ enhancements  │  │ mode =        │
│ enhancements  │  │   =         │  │  inference    │
│   =         │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────────────────────────────────────────┐
│   Valid      Invalid (raises ValueError)      │
└───────────────────────────────────────────────────┘
```

---

##  Experiment Lifecycle

```
┌────────────────────────────────────────────────────────┐
│                    1. PLANNING                          │
│  - Select runtime mode (baseline/full/rl/etc.)         │
│  - Choose environment (test/prod)                      │
│  - Name experiment (optional)                          │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│                    2. SETUP                             │
│  - Load config (base.yaml + mode override)             │
│  - Validate killswitches                               │
│  - Create output directory                             │
│  - Register run in manifest.json                       │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│                 3. EXECUTION                            │
│  - Initialize population                               │
│  - Evolution loop (crossover, mutation, selection)     │
│  - Apply features (repairs, heuristics, LS, RL)        │
│  - Track metrics (fitness, hypervolume, time)          │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│                  4. REPORTING                           │
│  - Export best schedule (JSON, PDF)                    │
│  - Generate plots (evolution, Pareto)                  │
│  - Save metrics (metrics.json)                         │
│  - Update manifest.json with results                   │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│                  5. ANALYSIS                            │
│  - Compare modes (--compare flag)                      │
│  - Export CSV for stats analysis                       │
│  - Visualize performance profiles                      │
│  - Select best mode for production                     │
└────────────────────────────────────────────────────────┘
```

---

##  Color-Coded Mode Map

```
 Mode 1: Baseline (Pure NSGA-II)
   └─► Research baseline
       NO enhancements
       
 Mode 2: NSGA-II + Repairs
   └─► + IGLS repair system
       Test repair effectiveness
       
 Mode 3: NSGA-II + Repairs + Heuristics
   └─► + 19 heuristic operators
       Test heuristic toolbox
       
 Mode 4: NSGA-II + Full
   └─► + Local search + all enhancements
       Best non-RL configuration
       
 Mode 5: RL-Guided
   └─► + RL agent (PPO) controls selection
       AI-driven optimization
       
 Mode 6: Round-Robin
   └─► + Fixed heuristic rotation
       Deterministic baseline for RL
```

---

##  Comparison Matrix

```
┌──────────────┬─────────┬─────────┬────────────┬──────┬────┬────────────┐
│   Feature    │ Baseline│ Repairs │ Heuristics │ Full │ RL │ RoundRobin │
├──────────────┼─────────┼─────────┼────────────┼──────┼────┼────────────┤
│ Repairs      │       │       │          │    │  │          │
│ Heuristics   │       │       │          │    │  │          │
│ Memetic LS   │       │       │          │    │  │          │
│ LNS-IGLS     │       │       │          │    │  │          │
│ Enhancements │       │       │          │    │  │          │
│ Adaptive     │       │       │          │    │  │          │
│ RL Agent     │       │       │          │    │  │          │
├──────────────┼─────────┼─────────┼────────────┼──────┼────┼────────────┤
│ Complexity   │  Simple │   Low   │   Medium   │ High │Max │    High    │
│ Runtime      │   Fast  │  Medium │   Medium   │ Slow │Med │    Slow    │
│ Quality      │   Low   │  Medium │   High     │Best* │?   │    High    │
└──────────────┴─────────┴─────────┴────────────┴──────┴────┴────────────┘

* Best non-RL configuration
```

---

This visual guide complements the detailed documentation in:
- `docs/02-user-guides/runtime-modes.md`
- `docs/06-development/implementation-notes/RUNTIME_MODES_IMPLEMENTATION.md`
- `RUNTIME_MODES_SUMMARY.md`
