<!-- Suggested thesis placement: Chapter 3 - Constraint Design, Section 3.5 - Session Clustering Constraints -->

## Course-Type-Aware Block Clustering Penalty

### Problem Statement

The original block clustering constraint applied uniform penalty rules to all course sessions regardless of their pedagogical nature. However, theory courses and practical courses have fundamentally different scheduling requirements that necessitate differentiated treatment.

Theory courses benefit from moderate fragmentation (2-3 hour blocks spread across the week) to maintain student attention and allow time for homework between sessions. Practical courses, conversely, require continuous time blocks to complete laboratory experiments, project work, or hands-on activities without interruption.

### Solution Design

The enhanced block clustering penalty implements course-type-aware evaluation logic that distinguishes between theory and practical sessions:

#### Theory Course Rules

1. **Oversized Block Penalty**: Blocks exceeding 3 consecutive quanta incur a penalty of 1 per excess quantum. This discourages marathon sessions while allowing reasonable flexibility.

2. **Isolated Session Handling**: Single-quantum blocks (isolated sessions) are penalized with a weight of 2, but the first isolated session per course per day is excused. This policy recognizes that one isolated session may be unavoidable due to scheduling constraints, but multiple isolated sessions indicate poor clustering.

3. **Preferred Block Sizes**: Blocks of 2-3 consecutive quanta incur no penalty, representing the pedagogically optimal session length.

#### Practical Course Rules

Practical courses must be scheduled as a single coalesced block without fragmentation. Any split across non-consecutive time slots incurs a heavy penalty of 20 per fragmentation instance. This ensures that laboratory sessions, workshops, or project-based activities can be completed in one continuous period.

### Implementation

The constraint function analyzes sessions grouped by `(course_id, course_type, day)` tuples to detect consecutive quantum blocks. For each course-day combination:

1. Quanta are sorted and partitioned into consecutive blocks
2. Course type determines which penalty logic applies
3. Penalties accumulate based on block size distribution

#### Configuration System

All penalty parameters are externalized to YAML configuration files, enabling environment-specific tuning without code modification:

```yaml
time:
  # Block size preferences
  preferred_block_size_min: 2
  preferred_block_size_max: 3
  
  # Theory course penalties
  theory_isolated_penalty: 2              # Penalty for isolated sessions (after excused ones)
  theory_oversized_penalty_per_quantum: 1  # Penalty per quantum for blocks > max
  theory_max_excused_isolated: 1          # Number of isolated sessions excused per day
  
  # Practical course penalties
  practical_fragmentation_penalty: 20     # Penalty per split in practical sessions
```

Environment-specific values:
- **Test/Dev**: Balanced penalties (isolated=2, fragmentation=20) for faster convergence
- **Production**: Stricter penalties (isolated=3, fragmentation=50) for higher quality

#### Repair Heuristics Integration

The repair system includes course-type-aware strategies in `repair_session_clustering()`:

**Theory Course Repair:**
1. For genes with 4+ quanta and poor clustering: Complete rebuild into optimal 2-3 block distribution
2. For oversized blocks (>3): Split into smaller preferred blocks
3. For isolated quanta: Local rearrangement to adjacent positions

**Practical Course Repair:**
- `_rebuild_practical_single_block()`: Searches for a single consecutive time window that can accommodate all required quanta
- Prioritizes same-day consolidation to minimize disruption
- Iterates through all available days to find suitable consecutive blocks

The repair system uses `_calculate_gene_clustering_penalty_typed()` which applies course-type-specific penalty calculation matching the constraint evaluation logic.

### Pedagogical Rationale

This design reflects established best practices in university scheduling:

- **Theory sessions** benefit from spaced repetition and distributed practice, with 2-3 hour blocks allowing adequate coverage without cognitive overload
- **Practical sessions** require uninterrupted time for setup, experimentation, and cleanup, with fragmentation disrupting workflow and reducing educational effectiveness
- **Single isolated theory sessions** (one per day) may be necessary for schedule feasibility and do not significantly harm learning outcomes, but multiple isolated sessions indicate suboptimal clustering

### Impact on Schedule Quality

The course-type-aware clustering penalty guides the genetic algorithm toward schedules that respect pedagogical requirements while maintaining scheduling flexibility. Theory courses achieve better time distribution without excessive fragmentation, while practical courses maintain session integrity. The asymmetric penalty structure (heavy for practical fragmentation, moderate for theory violations) reflects the relative importance of these constraints in educational delivery.
