# Archive-Based Diversity Maintenance

**Enhancement**: #5 - Novelty Search and Quality-Diversity  
**Difficulty**: Medium  
**Impact**: Medium  
**Priority**: 6

---

## Problem Statement

### Current Diversity Maintenance

```python
# NSGA-II: Crowding distance for diversity
def select_next_generation(population):
    # 1. Non-dominated sorting (Pareto fronts)
    fronts = fast_non_dominated_sort(population)
    
    # 2. Crowding distance within each front
    for front in fronts:
        assign_crowding_distance(front)
    
    # 3. Select based on: Pareto rank > crowding distance
    return select_by_crowding(fronts, pop_size)
```

**Problem**: Crowding distance only maintains **fitness diversity**, not **behavioral diversity**.

### What's Missing: Behavioral Diversity

**Example**:
```
Solution A: (hard=0, soft=100)  Schedule pattern: [Mon-heavy, Tue-light, ...]
Solution B: (hard=0, soft=100)  Schedule pattern: [distributed evenly]

NSGA-II: Both have same fitness → might drop one
Reality: Different structures → both valuable for search
```

**Impact**: 
- Premature convergence to similar solution structures
- Loss of exploration in behavioral space
- Miss alternative high-quality regions

---

## Solution: Archive-Based Diversity

Maintain an **archive** of behaviorally diverse solutions alongside fitness-based population.

### Key Concepts

#### 1. Behavioral Characterization

Define features that describe solution **structure**, not just fitness.

**Examples for scheduling**:
```python
def behavioral_descriptor(individual, context):
    """
    Extract behavioral features (independent of fitness).
    """
    decoded = decode_individual(individual, context)
    
    features = [
        # Temporal distribution
        sessions_per_day[0],  # Monday
        sessions_per_day[1],  # Tuesday
        # ... (7 features)
        
        # Room utilization
        avg_sessions_per_room,
        room_utilization_std,
        # ... (3 features)
        
        # Instructor load
        avg_sessions_per_instructor,
        instructor_load_std,
        max_instructor_load,
        # ... (4 features)
        
        # Temporal patterns
        avg_session_start_time,
        session_time_spread,
        # ... (3 features)
    ]
    
    return np.array(features)  # 17-dimensional descriptor
```

#### 2. Novelty Metric

Measure how **different** a solution is from archived solutions.

**Sparseness calculation**:
$$\text{novelty}(x) = \frac{1}{k} \sum_{i=1}^{k} \text{dist}(x, x_i)$$

Where:
- $x$ = behavioral descriptor of new solution
- $x_i$ = $k$ nearest neighbors in archive
- $k$ = typically 15 (parameter)

```python
def compute_novelty(descriptor, archive, k=15):
    """
    Compute novelty as average distance to k-nearest neighbors.
    """
    if len(archive) < k:
        return float('inf')  # Very novel if archive is small
    
    distances = [
        np.linalg.norm(descriptor - archived_descriptor)
        for archived_descriptor in archive
    ]
    
    # k-nearest neighbors
    distances.sort()
    knn_distances = distances[:k]
    
    novelty = np.mean(knn_distances)
    return novelty
```

---

## Architecture 1: Novelty Search

**Objective**: Maximize novelty instead of fitness.

### Algorithm

```python
class NoveltySearch:
    def __init__(self, archive_size=100, novelty_threshold=0.5):
        self.archive = []  # List of (individual, descriptor, fitness)
        self.archive_size = archive_size
        self.novelty_threshold = novelty_threshold
    
    def evolve_one_generation(self, population):
        """
        Evolve population using novelty as selection criterion.
        """
        # 1. Compute novelty for each individual
        novelties = []
        for ind in population:
            descriptor = behavioral_descriptor(ind, self.context)
            novelty = compute_novelty(descriptor, 
                                     [a[1] for a in self.archive])
            novelties.append(novelty)
        
        # 2. Select by novelty (not fitness!)
        selected = tournament_selection_by_novelty(population, novelties)
        
        # 3. Apply operators
        offspring = crossover_and_mutate(selected)
        
        # 4. Update archive with novel solutions
        for ind, novelty in zip(population, novelties):
            if novelty > self.novelty_threshold:
                self.add_to_archive(ind)
        
        return offspring
    
    def add_to_archive(self, individual):
        """
        Add individual to archive if novel enough.
        """
        descriptor = behavioral_descriptor(individual, self.context)
        fitness = individual.fitness.values
        
        self.archive.append((individual, descriptor, fitness))
        
        # Maintain archive size
        if len(self.archive) > self.archive_size:
            # Remove least novel (closest to others)
            self.archive = self._prune_archive()
    
    def _prune_archive(self):
        """
        Keep most novel solutions in archive.
        """
        # Compute novelty of each archived solution
        novelties = []
        for i, (ind, desc, fit) in enumerate(self.archive):
            other_descriptors = [self.archive[j][1] 
                               for j in range(len(self.archive)) if j != i]
            novelty = compute_novelty(desc, other_descriptors)
            novelties.append((novelty, i))
        
        # Keep top archive_size most novel
        novelties.sort(reverse=True)
        indices_to_keep = [idx for _, idx in novelties[:self.archive_size]]
        
        return [self.archive[i] for i in indices_to_keep]
```

### Pros and Cons

**Pros**:
- Explores solution space thoroughly
- Discovers unexpected high-quality regions
- Maintains behavioral diversity automatically

**Cons**:
- **Ignores fitness completely** (might generate many infeasible solutions)
- Requires good behavioral descriptor design
- Slower convergence to high-fitness regions

---

## Architecture 2: Quality-Diversity (MAP-Elites)

**Objective**: Fill a map of behavioral niches with highest-fitness solutions.

### Concept: Feature Map

Divide behavioral space into cells, keep best solution per cell.

```
Behavioral Space Discretization:

     Sessions per day (avg)
         ^
   High  │ [Cell] [Cell] [Cell]
         │ [Cell] [Best] [Cell]
         │        ★ fitness=10
   Low   │ [Cell] [Cell] [Empty]
         └──────────────────────> Room utilization
           Low          High
```

Each cell stores the **best-fitness** solution with that behavioral pattern.

### Algorithm: MAP-Elites

```python
class MAPElites:
    def __init__(self, feature_ranges, num_bins=10):
        """
        Args:
            feature_ranges: List of (min, max) for each behavioral feature
            num_bins: Number of bins per dimension
        """
        self.feature_ranges = feature_ranges
        self.num_bins = num_bins
        self.num_features = len(feature_ranges)
        
        # Initialize empty map
        shape = tuple([num_bins] * self.num_features)
        self.map = np.empty(shape, dtype=object)  # Stores individuals
        self.fitness_map = np.full(shape, -np.inf)  # Stores fitness values
    
    def evolve(self, num_iterations, batch_size=100):
        """
        Main MAP-Elites loop.
        """
        # 1. Initialize with random solutions
        for _ in range(batch_size):
            ind = generate_random_individual()
            self.try_add(ind)
        
        # 2. Iterative improvement
        for iteration in range(num_iterations):
            # Sample random cells
            batch = []
            for _ in range(batch_size):
                ind = self.sample_from_map()
                
                if ind is not None:
                    # Mutate
                    offspring = mutate(ind)
                    batch.append(offspring)
            
            # Try to add offspring to map
            for ind in batch:
                self.try_add(ind)
            
            # Report progress
            if iteration % 100 == 0:
                coverage = self.get_coverage()
                print(f"Iteration {iteration}: {coverage*100:.1f}% coverage")
    
    def try_add(self, individual):
        """
        Try to add individual to map.
        Replaces existing if better fitness in same cell.
        """
        # Compute behavioral descriptor
        descriptor = behavioral_descriptor(individual, self.context)
        
        # Get cell index
        cell_index = self.get_cell_index(descriptor)
        
        # Get fitness
        fitness = individual.fitness.values[0]  # Assuming single objective for simplicity
        
        # Add if cell empty or better than existing
        if self.fitness_map[cell_index] < fitness:
            self.map[cell_index] = individual
            self.fitness_map[cell_index] = fitness
            return True
        
        return False
    
    def get_cell_index(self, descriptor):
        """
        Discretize continuous descriptor into cell index.
        """
        indices = []
        for i, (value, (min_val, max_val)) in enumerate(zip(descriptor, self.feature_ranges)):
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            
            # Discretize
            bin_index = int(normalized * (self.num_bins - 1))
            indices.append(bin_index)
        
        return tuple(indices)
    
    def sample_from_map(self):
        """
        Sample a random non-empty cell.
        """
        filled_cells = np.argwhere(self.map != None)
        if len(filled_cells) == 0:
            return None
        
        idx = filled_cells[np.random.randint(len(filled_cells))]
        return self.map[tuple(idx)]
    
    def get_coverage(self):
        """
        Fraction of cells that contain solutions.
        """
        filled = np.sum(self.map != None)
        total = np.prod(self.map.shape)
        return filled / total
    
    def get_best_solutions(self, n=10):
        """
        Return top-n solutions across all cells.
        """
        solutions = []
        for idx in np.ndindex(self.map.shape):
            if self.map[idx] is not None:
                ind = self.map[idx]
                fitness = self.fitness_map[idx]
                solutions.append((fitness, ind))
        
        solutions.sort(reverse=True, key=lambda x: x[0])
        return [ind for _, ind in solutions[:n]]
```

### Pros and Cons

**Pros**:
- **Balances quality and diversity** (best of both worlds)
- Produces a diverse portfolio of high-quality solutions
- Interpretable (can visualize feature map)
- Works well with constrained problems

**Cons**:
- Discretization can be tricky (how many bins?)
- High-dimensional behavioral spaces are challenging
- Computational overhead of maintaining map

---

## Hybrid: NSGA-II + Archive

Combine Pareto-based selection with behavioral archive.

### Algorithm

```python
class NSGA2WithArchive:
    def __init__(self, pop_size=100, archive_size=50):
        self.pop_size = pop_size
        self.archive = []  # Behavioral archive
        self.archive_size = archive_size
    
    def evolve_one_generation(self, population):
        """
        NSGA-II evolution with archive injection.
        """
        # 1. Standard NSGA-II selection
        offspring = nsga2_selection_and_variation(population)
        
        # 2. Evaluate novelty of offspring
        for ind in offspring:
            descriptor = behavioral_descriptor(ind, self.context)
            novelty = compute_novelty(descriptor, 
                                     [a[0] for a in self.archive])
            ind.novelty = novelty
        
        # 3. Update archive with novel solutions
        for ind in offspring:
            if ind.novelty > self.novelty_threshold:
                self.add_to_archive(ind)
        
        # 4. Combine: 90% NSGA-II selected + 10% archive sampled
        num_from_nsga = int(0.9 * self.pop_size)
        num_from_archive = self.pop_size - num_from_nsga
        
        next_gen = nsga2_select(population + offspring, num_from_nsga)
        
        if len(self.archive) > 0:
            archive_samples = random.sample(self.archive, 
                                           min(num_from_archive, len(self.archive)))
            next_gen.extend([a[1] for a in archive_samples])  # a[1] is individual
        
        return next_gen
    
    def add_to_archive(self, individual):
        """
        Add to archive if novel and feasible.
        """
        # Only archive feasible solutions
        if individual.fitness.values[0] > 0:
            return
        
        descriptor = behavioral_descriptor(individual, self.context)
        self.archive.append((descriptor, individual))
        
        # Prune if too large
        if len(self.archive) > self.archive_size:
            self.archive = self._prune_by_novelty()
```

**Benefits**:
- Maintains NSGA-II's fitness-driven search
- Adds behavioral exploration via archive
- Archive acts as "diversity reservoir"

---

## RL Integration with Archive

Use archive to enhance RL training.

### Reward Shaping with Novelty

```python
class NoveltyAwareReward:
    def __init__(self, archive, novelty_weight=0.2):
        self.archive = archive
        self.novelty_weight = novelty_weight
    
    def calculate(self, individual, fitness_improvement):
        """
        Reward = fitness improvement + novelty bonus
        """
        descriptor = behavioral_descriptor(individual, self.context)
        novelty = compute_novelty(descriptor, self.archive)
        
        # Normalize novelty
        novelty_normalized = novelty / (self.archive_max_dist + 1e-6)
        
        reward = fitness_improvement + self.novelty_weight * novelty_normalized
        return reward
```

### Archive as Experience Replay

```python
class ArchiveReplayBuffer:
    def __init__(self, archive):
        self.archive = archive  # Behavioral archive
        self.replay_buffer = []  # RL transitions
    
    def add_transition(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
    
    def sample_batch(self, batch_size):
        """
        Sample transitions with diversity.
        """
        # 50% recent transitions
        recent_batch = random.sample(self.replay_buffer[-1000:], batch_size // 2)
        
        # 50% from archive-related transitions
        archive_batch = []
        for _ in range(batch_size // 2):
            # Sample archived solution
            archived_ind = random.choice(self.archive)
            
            # Find transitions involving similar individuals
            similar_transitions = [
                t for t in self.replay_buffer
                if behavioral_similarity(t[0], archived_ind) > 0.8
            ]
            
            if similar_transitions:
                archive_batch.append(random.choice(similar_transitions))
        
        return recent_batch + archive_batch
```

---

## Implementation Roadmap

### Phase 1: Behavioral Descriptors (1 week)
- [ ] Define behavioral features for schedules
- [ ] Implement `behavioral_descriptor()` function
- [ ] Test: Do descriptors capture meaningful differences?
- [ ] Visualize: Plot solutions in behavioral space (PCA/t-SNE)

### Phase 2: Novelty Search (2 weeks)
- [ ] Implement `NoveltySearch` class
- [ ] Add novelty calculation functions
- [ ] Test: Can it find diverse feasible solutions?
- [ ] Compare: Novelty-only vs fitness-only vs hybrid

### Phase 3: MAP-Elites (3 weeks)
- [ ] Implement `MAPElites` class
- [ ] Design feature map (2-3 dimensions initially)
- [ ] Visualize feature map during evolution
- [ ] Benchmark: Coverage vs quality trade-off

### Phase 4: NSGA-II + Archive Hybrid (2 weeks)
- [ ] Integrate archive into GA scheduler
- [ ] Tune: Novelty threshold, archive size, injection rate
- [ ] Evaluate: Diversity metrics improvement
- [ ] Production: Deploy best configuration

---

## Expected Benefits

### 1. Better Exploration
- **Current**: Converges to single Pareto front region
- **Expected**: Discovers multiple high-quality regions
- **Impact**: 30% more diverse solution set

### 2. Escape Local Optima
- **Current**: Stagnates when population converges
- **Expected**: Archive injection provides fresh genetic material
- **Impact**: 20% better final fitness on hard problems

### 3. Solution Portfolio
- **Current**: Returns single best solution
- **Expected**: Returns diverse set of high-quality alternatives
- **Impact**: Better for decision makers (multiple options)

---

## Evaluation Metrics

### 1. Behavioral Diversity
```python
def evaluate_behavioral_diversity(population):
    """
    Measure average pairwise distance in behavioral space.
    """
    descriptors = [behavioral_descriptor(ind) for ind in population]
    
    distances = []
    for i in range(len(descriptors)):
        for j in range(i+1, len(descriptors)):
            dist = np.linalg.norm(descriptors[i] - descriptors[j])
            distances.append(dist)
    
    return np.mean(distances)
```

### 2. Feature Map Coverage (MAP-Elites)
```python
def evaluate_map_coverage(map_elites):
    """
    Fraction of cells filled + average fitness per cell.
    """
    coverage = map_elites.get_coverage()
    avg_fitness = np.mean(map_elites.fitness_map[map_elites.fitness_map > -np.inf])
    
    return coverage, avg_fitness
```

### 3. Archive Quality
```python
def evaluate_archive_quality(archive):
    """
    How many archived solutions are Pareto-optimal?
    """
    archive_solutions = [a[1] for a in archive]
    pareto_optimal = extract_pareto_front(archive_solutions)
    
    return len(pareto_optimal) / len(archive)
```

---

## Configuration

```yaml
# configs/base.yaml
diversity:
  archive:
    enabled: true
    type: "hybrid"  # Options: "novelty", "map_elites", "hybrid"
    
    novelty_search:
      archive_size: 100
      novelty_threshold: 0.5
      k_nearest: 15
      
    map_elites:
      num_bins: 10
      feature_dimensions: 3  # Sessions/day, room utilization, instructor load
      
    hybrid:
      injection_rate: 0.1  # 10% archive, 90% NSGA-II
      novelty_threshold: 0.5
      archive_size: 50
  
  behavioral_descriptor:
    features:
      - "sessions_per_day_distribution"
      - "room_utilization"
      - "instructor_load_balance"
      - "temporal_clustering"
    normalize: true
```

---

## Related Work

### Papers
1. **"Novelty Search"** (Lehman & Stanley, 2011)
   - Original novelty search algorithm
   - Applications to robotics and games

2. **"MAP-Elites"** (Mouret & Clune, 2015)
   - Quality-diversity algorithm
   - Illuminating solution space

3. **"Surprise Search"** (Gravina et al., 2016)
   - Behavioral diversity for procedural content generation

---

## Summary

**Problem**: Fitness-only selection leads to premature convergence

**Solution**: Maintain behavioral archive with novelty search or MAP-Elites

**Recommended Start**: NSGA-II + Archive hybrid (practical, proven)

**Expected Impact**: 30% more diverse solutions, 20% better on hard problems

**Next Steps**: Define behavioral descriptors → Implement hybrid archive → Benchmark
