# Output Structure Consolidation

## 📁 **Recommended Structure: Everything Under `output/`**

### ✅ **Benefits:**
1. **Single Clean Command**: `uv run clean` clears all experimental artifacts
2. **Easy Backup/Archive**: One folder contains entire project state
3. **Thesis Organization**: All experimental artifacts in one logical place
4. **No File Hunting**: No more searching across `logs/`, `models/`, `output/`
5. **Automated Analysis**: Scripts can find everything in one tree structure

### 🏗️ **New Consolidated Structure:**
```
output/
├── experiments/           # Organized GA experiment results
│   ├── baseline/          # Pure NSGA-II experiments
│   │   └── pure-nsga/
│   ├── repairs/           # NSGA-II + repairs experiments  
│   ├── heuristics/        # NSGA-II + heuristics experiments
│   └── full/              # Full GA experiments
├── logs/                  # All logs consolidated here
│   ├── nsga/              # GA execution logs (moved from logs/nsga/)
│   ├── tensorboard/       # TensorBoard logs (moved from logs/tensorboard/)
│   └── training/          # RL training logs (moved from logs/training/)
├── models/                # All trained models
│   └── rl_agents/         # RL model checkpoints (moved from models/rl_agents/)
└── analysis/              # Analysis results
    ├── statistical_analysis.json
    ├── convergence_comparison.png
    └── runtime_vs_quality.png
```

### 🔧 **Migration Process:**

#### 1. **Automatic Migration Script:**
```bash
uv run migrate  # Run consolidation script
```
- Moves `logs/` → `output/logs/`  
- Moves `models/` → `output/models/`
- Organizes old `evaluation_*` folders → `output/experiments/`
- Creates `output/analysis/` for results

#### 2. **Updated Configuration:**
All paths in `configs/base.yaml` now point to consolidated locations:
- `io.logs_dir: output/logs`
- `io.models_dir: output/models`  
- `rl.training.tensorboard_log: output/logs/tensorboard`
- `rl.training.save_dir: output/models/rl_agents`
- `rl.evaluation.metrics_dir: output/analysis/rl_metrics`

#### 3. **Updated Analysis Scripts:**
- `analyze-results` now searches both old and new structures
- Saves all analysis outputs to `output/analysis/`
- Backwards compatible with existing experiment folders

### 🧹 **Simplified Cleanup:**

#### Before (Scattered):
```bash
# Had to clean multiple directories
rm -rf output/
rm -rf logs/  
rm -rf models/
```

#### After (Consolidated):
```bash
uv run clean  # Cleans everything at once
```

### 📊 **Thesis Benefits:**

1. **Easy Archival**: `zip -r thesis_experiments.zip output/`
2. **Simple Backup**: One folder to sync/backup for all experimental data
3. **Clear Organization**: Logical separation (experiments/logs/models/analysis)
4. **Automated Analysis**: All scripts know where to find data
5. **Clean Workspace**: Only `output/` folder contains experimental artifacts

### 🎯 **Comparison:**

| Aspect | **Old Structure** | **New Structure** |
|--------|------------------|-------------------|
| **Cleanup** | Manual deletion of 3+ folders | `uv run clean` |
| **Backup** | Multiple scattered directories | Single `output/` folder |
| **Analysis** | Search across multiple locations | Unified search tree |
| **Organization** | Mixed with source code | Isolated experimental area |
| **Thesis Work** | Complex file management | Simple single-folder archival |

### ✅ **Implementation Status:**

- ✅ **Config Updated**: All paths point to consolidated structure
- ✅ **Migration Script**: `uv run migrate` available
- ✅ **Analysis Updated**: Uses new organized structure only
- ✅ **Clean Command**: Works with consolidated structure  
- ✅ **No Backward Compatibility**: Clean new architecture only

**Requirement**: **Must use consolidated structure** - run `uv run migrate` to organize existing experiments into new structure.