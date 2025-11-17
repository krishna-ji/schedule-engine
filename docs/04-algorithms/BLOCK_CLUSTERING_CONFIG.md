# Block Clustering Configuration Guide

## Overview

Block clustering penalties enforce pedagogically sound session scheduling by penalizing undesirable time block distributions. The system is **course-type aware**, applying different rules to theory and practical courses.

## Configuration Parameters

All parameters are configured in `configs/{test,dev,prod}.yaml` under the `time:` section:

```yaml
time:
  # Block size preferences (applies to theory courses)
  preferred_block_size_min: 2             # Minimum preferred block size
  preferred_block_size_max: 3             # Maximum preferred block size
  
  # Theory course penalties
  theory_isolated_penalty: 2              # Penalty for isolated sessions (after excused)
  theory_oversized_penalty_per_quantum: 1  # Penalty per quantum for blocks > max
  theory_max_excused_isolated: 1          # Number of isolated sessions excused per day
  
  # Practical course penalties
  practical_fragmentation_penalty: 20     # Penalty per split in practical sessions
```

## Theory Courses

### Rules

1. **Preferred Blocks**: 2-3 consecutive quanta (no penalty)
2. **Isolated Sessions**: Single-quantum blocks penalized after first excused session
3. **Oversized Blocks**: Blocks > 3 quanta incur penalty per excess quantum

### Examples

| Distribution | Penalty Calculation | Total |
|-------------|---------------------|-------|
| [3, 3] | 0 (ideal) | 0 |
| [2, 2, 2] | 0 (acceptable) | 0 |
| [1, 2, 3] | 0 (first isolated excused) | 0 |
| [1, 1, 4] | 2 (2nd isolated) + 1 (oversized by 1) | 3 |
| [6] | 3 (oversized by 3) | 3 |
| [1, 1, 1, 3] | 2 + 2 (2nd and 3rd isolated) | 4 |

### Tuning

- **Increase `theory_isolated_penalty`**: Stricter enforcement against fragmentation
- **Increase `theory_oversized_penalty_per_quantum`**: Discourage long continuous sessions
- **Decrease `theory_max_excused_isolated`**: More aggressive clustering (set to 0 to penalize all isolated)

## Practical Courses

### Rules

1. **Single Block Required**: All quanta must be consecutive (no splits allowed)
2. **Heavy Fragmentation Penalty**: Each split incurs configurable penalty

### Examples

| Distribution | Penalty Calculation | Total |
|-------------|---------------------|-------|
| [3] | 0 (ideal - single block) | 0 |
| [6] | 0 (single block, any size OK) | 0 |
| [2, 1] | 20 × 1 (one split) | 20 |
| [1, 1, 1] | 20 × 2 (two splits) | 40 |
| [2, 2, 2] | 20 × 2 (two splits) | 40 |

### Tuning

- **Increase `practical_fragmentation_penalty`**: Stronger enforcement of single-block rule
- Practical courses should typically have higher penalties than theory to reflect pedagogical importance

## Environment Presets

### Test Environment (fast convergence)
```yaml
theory_isolated_penalty: 2
theory_oversized_penalty_per_quantum: 1
theory_max_excused_isolated: 1
practical_fragmentation_penalty: 20
```

### Dev Environment (balanced)
```yaml
theory_isolated_penalty: 2
theory_oversized_penalty_per_quantum: 1
theory_max_excused_isolated: 1
practical_fragmentation_penalty: 20
```

### Production Environment (strict quality)
```yaml
theory_isolated_penalty: 3              # Higher penalty
theory_oversized_penalty_per_quantum: 2  # Higher penalty
theory_max_excused_isolated: 1
practical_fragmentation_penalty: 50     # Much higher penalty
```

## Repair Heuristics

The `repair_session_clustering` heuristic automatically attempts to fix clustering violations:

### Theory Repair Strategies
1. **Complete Rebuild**: For genes with 4+ quanta and poor clustering, rebuild distribution from scratch
2. **Block Splitting**: Split oversized blocks into 2-3 quantum chunks
3. **Local Rearrangement**: Move isolated quanta to adjacent positions

### Practical Repair Strategy
- **Single Block Consolidation**: Searches for consecutive time window to accommodate all quanta
- Prioritizes same-day consolidation
- Iterates through all days to find suitable blocks

## Troubleshooting

### Problem: Too many isolated theory sessions

**Solutions:**
1. Increase `theory_isolated_penalty` (e.g., 2 → 4)
2. Decrease `theory_max_excused_isolated` (1 → 0)
3. Enable more aggressive repair: `repair.max_iterations: 5+`

### Problem: Practical courses still fragmented

**Solutions:**
1. Increase `practical_fragmentation_penalty` (e.g., 20 → 50 or 100)
2. Increase constraint weight in hard_constraints section:
   ```yaml
   hard_constraints:
     session_block_clustering_penalty:
       weight: 4.0  # or higher
   ```
3. Enable memetic repair mode: `repair.memetic_mode: true`

### Problem: Oversized theory blocks (4+ quanta)

**Solutions:**
1. Increase `theory_oversized_penalty_per_quantum` (1 → 2 or 3)
2. Verify `preferred_block_size_max: 3` is set correctly

## Best Practices

1. **Start with defaults**: Use test/dev presets before customizing
2. **Incremental tuning**: Change one parameter at a time
3. **Monitor convergence**: Check if penalties reduce over generations
4. **Balance penalties**: Theory penalties should be lower than practical (pedagogical flexibility vs requirement)
5. **Consider course load**: Higher `quanta_per_week` courses may need more tolerance

## Related Configuration

- **Constraint Weight**: Set in `hard_constraints.session_block_clustering_penalty.weight`
- **Repair Iterations**: Set in `repair.max_iterations`
- **Memetic Mode**: Set in `repair.memetic_mode` (enables every-generation repair)

## Testing

To verify configuration changes:

```bash
# Run test suite
python test/test_block_clustering_course_type.py

# Quick schedule run
python main.py --env test

# Full quality run
python main.py --env prod
```

Check output for clustering violations in `violation_report.txt` and constraint evolution in `logger_constraints.csv`.
