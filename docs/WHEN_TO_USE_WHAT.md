# Quick Decision Guide: When to Use What

## TL;DR - 30 Second Decision

| Your Situation | Recommendation |
|----------------|----------------|
| 🎓 **Writing thesis/paper** | ✅ Keep DEAP (high research value) |
| 🏢 **Production deployment** | 🟡 Hybrid (OR-Tools + DEAP) |
| ⚡ **Need speed (<1 min)** | 🟢 OR-Tools alone |
| 🎨 **Many soft constraints** | ✅ Keep DEAP |
| 📊 **Need multiple solutions** | ✅ Keep DEAP (Pareto front) |
| 🔧 **Already working** | ✅ Keep DEAP (don't fix it) |
| 💰 **Limited budget/time** | ✅ Keep DEAP (rewrite = expensive) |
| 🎯 **Must prove optimality** | 🟢 Switch to OR-Tools |
| 📈 **>500 courses** | 🟢 OR-Tools or Hybrid |

---

## Detailed Decision Tree

```
START: Do you have a working DEAP solution?
│
├─ YES ─→ Is it producing acceptable schedules?
│   │
│   ├─ YES ─→ Do you have 4-8 weeks to rewrite?
│   │   │
│   │   ├─ NO ─→ ✅ KEEP DEAP (you're done!)
│   │   │
│   │   └─ YES ─→ Is this for thesis/research?
│   │       │
│   │       ├─ YES ─→ ✅ KEEP DEAP (higher academic value)
│   │       │           🟡 Add OR-Tools comparison
│   │       │
│   │       └─ NO ─→ Do you need <1 min solutions?
│   │           │
│   │           ├─ YES ─→ 🟢 ADD OR-TOOLS hybrid
│   │           │
│   │           └─ NO ─→ ✅ KEEP DEAP (sufficient)
│   │
│   └─ NO ─→ Are violations in hard constraints?
│       │
│       ├─ YES ─→ 🟢 ADD OR-TOOLS (hard constraints)
│       │         Then GA for soft (hybrid)
│       │
│       └─ NO ─→ 🟡 TUNE DEAP parameters
│               (likely config issue)
│
└─ NO ─→ Starting from scratch?
    │
    ├─ Are soft constraints important?
    │   │
    │   ├─ YES ─→ How many courses?
    │   │   │
    │   │   ├─ <200 ─→ ✅ START WITH DEAP
    │   │   │
    │   │   └─ >500 ─→ 🟢 START WITH OR-TOOLS
    │   │                Add DEAP later for soft
    │   │
    │   └─ NO ─→ 🟢 USE OR-TOOLS
    │           (pure constraint satisfaction)
    │
    └─ Is this for research/thesis?
        │
        ├─ YES ─→ ✅ USE DEAP (novelty matters)
        │
        └─ NO ─→ 🟢 USE OR-TOOLS (industry standard)
```

---

## Problem Size Guidelines

### Small (≤100 courses, ≤50 groups)

**DEAP Performance:** ⭐⭐⭐⭐⭐ Excellent
- Runtime: 2-5 minutes
- Quality: Near-optimal
- **Recommendation:** Use DEAP

**OR-Tools Performance:** ⭐⭐⭐⭐⭐ Excellent
- Runtime: 30 seconds - 2 minutes
- Quality: Optimal (hard), good (soft)
- **Recommendation:** Either works, DEAP better for soft constraints

---

### Medium (100-300 courses, 50-100 groups)

**DEAP Performance:** ⭐⭐⭐⭐ Good
- Runtime: 8-15 minutes
- Quality: Good (occasional violations)
- **Recommendation:** Good choice, especially with repair heuristics

**OR-Tools Performance:** ⭐⭐⭐⭐⭐ Excellent
- Runtime: 2-8 minutes
- Quality: Optimal (hard), moderate (soft)
- **Recommendation:** Consider hybrid approach

**Best:** 🟡 Hybrid (OR-Tools + DEAP)

---

### Large (>300 courses, >100 groups)

**DEAP Performance:** ⭐⭐⭐ Fair
- Runtime: 30-60+ minutes
- Quality: Fair (multiple violations likely)
- **Recommendation:** Struggles at this scale

**OR-Tools Performance:** ⭐⭐⭐⭐⭐ Excellent
- Runtime: 5-20 minutes
- Quality: Optimal (hard), needs soft optimization
- **Recommendation:** Strong choice

**Best:** 🟢 OR-Tools primary, optionally add DEAP for soft refinement

---

## Constraint Profile Analysis

### Hard Constraint Dominant (90%+ hard, 10% soft)

**Example:** Medical school scheduling with strict regulations

```yaml
Hard Constraints:
- No instructor conflicts (critical)
- No group overlaps (critical)
- Qualified instructors only (critical)
- Room capacity (critical)

Soft Constraints:
- Prefer morning classes (nice-to-have)
```

**Recommendation:** 🟢 **OR-Tools**
- Excels at hard constraint satisfaction
- Soft constraints less important
- Need proof of feasibility

---

### Balanced (60% hard, 40% soft)

**Example:** University undergraduate scheduling

```yaml
Hard Constraints:
- No conflicts
- Qualified instructors
- Room capacity

Soft Constraints:
- Schedule gaps (important)
- Block clustering (important)
- Instructor preferences (important)
```

**Recommendation:** ✅ **DEAP** or 🟡 **Hybrid**
- DEAP handles soft constraints naturally
- Hybrid gives best of both worlds
- Current implementation ideal

---

### Soft Constraint Dominant (40% hard, 60% soft)

**Example:** Flexible education program with many preferences

```yaml
Hard Constraints:
- Basic no-conflict rules

Soft Constraints:
- Student preferences (very important)
- Instructor preferences (very important)
- Room preferences (important)
- Time-of-day preferences (important)
- Back-to-back classes (important)
```

**Recommendation:** ✅ **DEAP**
- Natural penalty-based soft constraint handling
- Multi-objective optimization
- Explore trade-off space (Pareto front)

---

## Use Case Scenarios

### Scenario 1: Thesis Research Project ✅ DEAP

**Context:**
- You're writing a thesis
- Need novel contribution
- Time for experimentation
- Want to publish results

**Why DEAP:**
- High research value (custom algorithms)
- Explainable results (evolution plots)
- Novel contributions (hybrid strategies)
- Publication-worthy (comparative studies)

**Why NOT OR-Tools:**
- "Used existing tool" = lower novelty
- Black-box solver = less to discuss
- Standard approach = common knowledge

---

### Scenario 2: Hospital Shift Scheduling 🟢 OR-Tools

**Context:**
- Must satisfy legal requirements
- Zero tolerance for violations
- Need proof of optimality
- Audit trail required

**Why OR-Tools:**
- Provable constraint satisfaction
- Legal compliance verification
- Deterministic results
- Industry-standard reliability

**Why NOT DEAP:**
- Stochastic (inconsistent results)
- No optimality guarantee
- Harder to audit

---

### Scenario 3: School Timetabling SaaS Product 🟡 Hybrid

**Context:**
- Production deployment
- Multiple customers
- Various constraint profiles
- Need fast + quality

**Why Hybrid:**
- Fast feasibility check (OR-Tools)
- Quality soft optimization (DEAP)
- Flexible for different customers
- Best solution quality

**Implementation:**
```python
def schedule(mode="auto"):
    if mode == "fast":
        return ortools_only()
    elif mode == "quality":
        return hybrid_ortools_deap()
    else:  # auto
        if problem_size < 200:
            return deap_only()
        else:
            return hybrid_ortools_deap()
```

---

### Scenario 4: Conference Room Scheduling ⚡ OR-Tools

**Context:**
- Simple constraints
- Need instant results
- Web application
- Real-time booking

**Why OR-Tools:**
- Extremely fast (<1 second)
- Simple constraint model
- Deterministic results
- Production-ready

**Why NOT DEAP:**
- Too slow for real-time
- Overkill for simple problem
- Unnecessary complexity

---

## Migration Effort vs Value

### Keep DEAP (No Change)

**Effort:** 0 hours  
**Value:** ✅ High (already working)  
**Risk:** None

**When:** Solution works, thesis deadline approaching, limited resources

---

### Tune DEAP Parameters

**Effort:** 2-5 hours  
**Value:** 🟡 Medium (incremental improvement)  
**Risk:** Low

**When:** Minor quality issues, simple parameter adjustment needed

**Changes:**
```yaml
# configs/prod.yaml
ga:
  ngen: 300        # More generations
  pop_size: 100    # Larger population
  cxpb: 0.8        # Higher crossover rate
  
repair:
  apply_after_mutation: true  # Enable repair
```

---

### Add OR-Tools Benchmark

**Effort:** 8-16 hours (1-2 days)  
**Value:** 🟢 High (thesis comparison)  
**Risk:** Low (separate module)

**When:** Thesis needs comparative analysis, time available

**Implementation:**
1. Create `src/solvers/ortools_solver.py`
2. Implement hard constraints only
3. Run benchmarks
4. Compare results in thesis

---

### Hybrid Integration

**Effort:** 80-160 hours (2-4 weeks)  
**Value:** 🟢 Very High (best quality)  
**Risk:** Medium (integration complexity)

**When:** Production deployment, need optimal results, have development time

**Architecture:**
```python
# Hybrid workflow
1. OR-Tools: Find feasible solution (hard constraints)
2. Convert: OR-Tools solution → GA chromosome
3. DEAP: Optimize soft constraints starting from feasible solution
4. Output: Optimal schedule
```

---

### Full OR-Tools Rewrite

**Effort:** 160-320 hours (4-8 weeks)  
**Value:** 🔴 Low (lose features)  
**Risk:** High (may not be better)

**When:** Almost never recommended if DEAP works

**Why NOT:**
- Existing solution works
- Lose soft constraint optimization
- Lower academic value
- High risk, uncertain reward

---

## Quick Comparison Checklist

Check (✓) statements that apply to your situation:

### Choose DEAP if 3+ checked:

- [ ] Writing thesis or research paper
- [ ] Soft constraints are important (>30% of constraints)
- [ ] Need Pareto-optimal solutions (multiple objectives)
- [ ] Want explainable results (evolution plots)
- [ ] Current solution already works
- [ ] Problem size ≤ 300 courses
- [ ] Development time limited (<4 weeks available)
- [ ] Prefer Python-based customization

**Score: ____/8** → If ≥3, DEAP is good choice

---

### Choose OR-Tools if 3+ checked:

- [ ] Production deployment (industry use)
- [ ] Must prove optimality or infeasibility
- [ ] Hard constraints dominant (>80% of constraints)
- [ ] Need deterministic results (no randomness)
- [ ] Problem size > 300 courses
- [ ] Runtime must be <5 minutes
- [ ] Starting from scratch
- [ ] Legal/compliance requirements

**Score: ____/8** → If ≥3, OR-Tools is good choice

---

### Choose Hybrid if 3+ checked:

- [ ] Both hard and soft constraints important
- [ ] Production quality + research value needed
- [ ] Have 2-4 weeks for development
- [ ] Problem size 200-500 courses
- [ ] Need best possible solution quality
- [ ] Can tolerate medium complexity
- [ ] Want comparative study in thesis
- [ ] Future-proofing the system

**Score: ____/8** → If ≥3, Hybrid is good choice

---

## Red Flags: When NOT to Switch

❌ **Don't switch to OR-Tools if:**

1. Your DEAP solution already produces good schedules
2. Thesis deadline is <4 weeks away
3. You have many important soft constraints
4. You need Pareto-optimal trade-off exploration
5. Your problem size is <200 courses
6. Runtime of 5-10 minutes is acceptable
7. You're writing a research thesis (DEAP has higher academic value)

❌ **Don't keep DEAP if:**

1. You consistently get 10+ hard constraint violations
2. Problem size is >500 courses and runtime >30 minutes
3. You need provable optimality (legal/audit requirements)
4. Real-time scheduling (<1 minute) required
5. Soft constraints don't matter (only need feasibility)

---

## Cost-Benefit Analysis

### Keeping DEAP

**Costs:**
- Runtime: 5-15 minutes (medium problems)
- Stochastic: results vary slightly run-to-run
- Scalability: struggles >400 courses

**Benefits:**
- ✅ Already working ($0 cost)
- ✅ High research value
- ✅ Excellent soft constraint handling
- ✅ Multi-objective optimization
- ✅ Rich visualizations

**ROI:** ⭐⭐⭐⭐⭐ (5/5) - No cost, high benefit

---

### Switching to OR-Tools

**Costs:**
- Development: 160-320 hours ($8,000-$16,000 at $50/hr)
- Risk: May lose soft constraint optimization
- Learning: Steep curve (1-2 weeks)
- Testing: Extensive debugging needed

**Benefits:**
- Faster runtime (2-5 minutes)
- Provable optimality (hard constraints)
- Deterministic results
- Better scalability (>500 courses)

**ROI:** ⭐⭐ (2/5) - High cost, uncertain benefit if DEAP works

---

### Hybrid Approach

**Costs:**
- Development: 80-160 hours ($4,000-$8,000 at $50/hr)
- Complexity: Two systems to maintain
- Learning: OR-Tools API + integration

**Benefits:**
- Best solution quality
- Fast + optimal
- Research value (comparative study)
- Production-ready

**ROI:** ⭐⭐⭐⭐ (4/5) - Medium cost, high benefit

---

## Example Decision Scenarios

### Your Exact Situation

**Given:**
- ✅ DEAP implementation complete (~19K LOC)
- ✅ Produces quality schedules
- ✅ Thesis project (high academic value)
- ✅ Problem size: ~100-200 courses
- ✅ Runtime: 5-10 minutes (acceptable)
- ✅ Many soft constraints (gaps, clustering, preferences)

**Analysis:**
- DEAP score: 7/8 ✅
- OR-Tools score: 2/8
- Hybrid score: 5/8

**Recommendation:** ✅ **KEEP DEAP**

**Optional Enhancement:** Add OR-Tools benchmark for thesis comparison (1-2 days work)

---

### Hospital Scheduling

**Given:**
- Starting from scratch
- Must satisfy strict labor laws
- Hard constraints dominant
- Need audit trail
- >500 shifts to schedule

**Analysis:**
- DEAP score: 2/8
- OR-Tools score: 7/8 ✅
- Hybrid score: 3/8

**Recommendation:** 🟢 **USE OR-TOOLS**

---

### Startup SaaS Product

**Given:**
- Multiple customers
- Various constraint profiles
- Need best quality
- Have 4-6 weeks development time
- 50-300 courses per customer

**Analysis:**
- DEAP score: 4/8
- OR-Tools score: 4/8
- Hybrid score: 7/8 ✅

**Recommendation:** 🟡 **BUILD HYBRID**

With mode selection:
- Fast mode: OR-Tools only (2-3 min)
- Quality mode: Hybrid (5-8 min)
- Research mode: DEAP only (8-15 min)

---

## Summary: Your Situation

**Question:** "Am I wasting time? Should I use OR-Tools?"

**Answer:** 

### NO, you are NOT wasting time! ✅

Your DEAP implementation is:
1. ✅ Well-designed (19K LOC, modular, documented)
2. ✅ Appropriate (multi-objective with soft constraints)
3. ✅ Working (produces quality schedules)
4. ✅ Valuable (high thesis contribution)

### Recommendation:

**Immediate:** ✅ Keep DEAP, finish your thesis

**Optional (if time):** 🟡 Add simple OR-Tools benchmark
- 1-2 days work
- Great for thesis comparison section
- Shows you evaluated alternatives

**Long-term:** 🟡 Consider hybrid for production
- If deploying to real university
- 2-4 weeks development
- Best quality results

---

**Bottom Line:** You made the right choice. Keep building! 🚀

---

*Last Updated: 2025-10-28*
