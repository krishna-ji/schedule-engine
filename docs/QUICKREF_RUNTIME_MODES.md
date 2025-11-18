# Runtime Modes - Quick Reference

## 🚀 Quick Start

```bash
# List all modes
python main.py --list-modes

# Run with specific mode
python main.py --mode baseline --env prod

# UV shortcuts
uv run baseline     # Pure NSGA-II
uv run repairs      # + Repairs
uv run heuristics   # + Heuristics
uv run full         # + Local Search (best non-RL)
uv run rl           # RL-guided (requires trained model)
uv run roundrobin   # Round-robin heuristics

# Compare all runs
python main.py --compare
```

---

## 📊 Modes at a Glance

| # | Mode | Features | Use Case | Command |
|---|------|----------|----------|---------|
| 1 | **Baseline** | Pure NSGA-II | Research baseline | `uv run baseline` |
| 2 | **Repairs** | + IGLS repairs | Test repair effectiveness | `uv run repairs` |
| 3 | **Heuristics** | + 19 operators | Test heuristic toolbox | `uv run heuristics` |
| 4 | **Full** | + Local search | Best non-RL GA | `uv run full` |
| 5 | **RL-Guided** | + RL agent (PPO) | RL-guided search | `uv run rl` |
| 6 | **Round-Robin** | + Fixed rotation | RL baseline comparison | `uv run roundrobin` |

---

## 🎯 Common Workflows

### Feature Ablation Study
```bash
uv run baseline     # 0 features
uv run repairs      # +1 feature
uv run heuristics   # +2 features
uv run full         # +3 features
python main.py --compare
```

### RL Evaluation
```bash
python main.py --mode rl-guided --env prod --experiment "rl-test-1"
python main.py --mode roundrobin --env prod --experiment "rr-test-1"
python main.py --mode nsga-full --env prod --experiment "full-test-1"
python main.py --compare
```

### Production Deployment
```bash
# Test all modes
for mode in baseline repairs heuristics full; do
  python main.py --mode $mode --env prod --experiment "prod-$mode"
done

# Select best mode from comparison
python main.py --compare

# Deploy
uv run full  # Or whichever performed best
```

---

## 📁 Output Structure

```
output/
├── experiment_manifest.json   # All runs tracked here
├── baseline/
│   └── pure-nsga/
│       └── evaluation_20251118_140530_exp1/
├── nsga/
│   ├── nsga-repairs/
│   ├── nsga-heuristics/
│   └── nsga-full/
├── rl/
│   └── rl-guided/
└── hybrid/
    └── roundrobin/
```

---

## 🔧 Configuration Files

```
configs/
├── base.yaml                    # Common settings
├── baseline/1-pure-nsga.yaml   # Mode 1
├── nsga/
│   ├── 2-nsga-repairs.yaml     # Mode 2
│   ├── 3-nsga-heuristics.yaml  # Mode 3
│   └── 4-nsga-full.yaml        # Mode 4
├── rl/5-rl-guided.yaml         # Mode 5
└── hybrid/6-roundrobin.yaml    # Mode 6
```

---

## 🎛️ Killswitch Matrix

| Feature | 1 | 2 | 3 | 4 | 5 | 6 |
|---------|---|---|---|---|---|---|
| Repairs | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Heuristics | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Memetic LS | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| LNS-IGLS | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Enhancements | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Adaptive Prob | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| RL Agent | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

---

## 💡 Tips

**Smoke Testing:** Use `--env test` for quick validation (30 gens, ~5 min)
```bash
python main.py --mode full --env test
```

**Experiment Names:** Use descriptive names for easy tracking
```bash
python main.py --mode baseline --experiment "paper-baseline-v1"
```

**GPU Acceleration:** RL mode automatically uses CUDA if available
```bash
# Check GPU usage
nvidia-smi
uv run rl  # Will use GPU for inference
```

**Export Results:** Generate CSV for statistical analysis
```python
from src.workflows.experiment_manager import ExperimentManager
manager = ExperimentManager()
manager.export_comparison_csv("output/results.csv")
```

---

## 📚 Documentation

- **Full Guide:** `docs/02-user-guides/runtime-modes.md` (2500+ lines)
- **Implementation:** `docs/06-development/implementation-notes/RUNTIME_MODES_IMPLEMENTATION.md`
- **Architecture:** `docs/03-architecture/` (system design)
- **Algorithms:** `docs/04-algorithms/` (GA, RL, heuristics)

---

## 🐛 Troubleshooting

**Q: "Config validation failed"**  
A: Config violates mode constraints. Check killswitches in config file match expected values.

**Q: "Missing RL model"**  
A: Train RL agent first: `uv run train` or edit config to use fallback mode.

**Q: "Output directory exists"**  
A: Use unique experiment names or timestamps (automatic with `--experiment` flag).

---

## 🚦 Status Indicators

- 🔴 **Baseline** - Pure GA, no enhancements
- 🟡 **Repairs** - Basic repair system
- 🟢 **Heuristics** - Full heuristic toolbox
- 🔵 **Full** - Best non-RL configuration
- 🤖 **RL-Guided** - AI-driven heuristic selection
- 🔄 **Round-Robin** - Fixed heuristic rotation

---

For detailed information, see `docs/02-user-guides/runtime-modes.md`
