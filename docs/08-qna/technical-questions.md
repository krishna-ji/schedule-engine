# Technical Questions & Answers

**Your active Q&A workspace for technical questions about the schedule-engine project**

---

## How to Use This Document

1. **Ask questions** - Technical, implementation, algorithmic questions
2. **AI will document** - Conversations recorded here
3. **Search later** - Reference past Q&A when similar issues arise
4. **Build knowledge** - Accumulate project-specific solutions

---

## Q&A Sessions

### [Session 1: November 17, 2025] - GPU Acceleration Implementation

**Q: Can we use NVIDIA GPU for both RL training and constraint checking?**

**A:** 
- **RL Training:** ✅ YES - GPU provides 3-5× speedup
  - Implementation: Changed `device: auto` → `device: cuda` in configs/base.yaml
  - Framework: Stable-Baselines3 with PyTorch backend already GPU-ready
  - Expected: 22.5 hours → 6 hours for full curriculum training
  - Hardware: 8GB GPU sufficient (only needs 200-800 MB VRAM)

- **Constraint Checking:** ❌ NO - GPU would be 2.4× slower
  - Reason: Small, irregular, dictionary-based operations
  - Memory transfer overhead: 50ms vs 40ms CPU computation
  - Constraint checking is only 4% of RL training time
  - Recommendation: Keep constraints on CPU

**Documentation:** See [05-performance/nvidia-gpu/](../05-performance/nvidia-gpu/)

---

### [Session 2: November 17, 2025] - Documentation Reorganization

**Q: How can I categorize docs/ folder? Too many loose files, hard to find things.**

**A:** Implemented 10-category structure:
1. **01-getting-started/** - User onboarding
2. **02-user-guides/** - Daily usage guides
3. **03-architecture/** - System design
4. **04-algorithms/** - Technical deep dives
5. **05-performance/** - Optimization & analysis
6. **06-development/** - Changelogs & dev notes
7. **07-thesis-report/** - Academic content
8. **08-qna/** - This document (Q&A workspace)
9. **09-future-plans/** - Roadmap & ideas
10. **10-ai-suggestions/** - AI-generated content
11. **archive/** - Historical reference

**Benefits:**
- 30+ loose files → 2 files + 10 organized categories
- Clear purpose for each folder
- Easy navigation with README.md files
- Nothing lost (all archived, not deleted)

**Documentation:** See [REORGANIZATION_SUMMARY.md](../REORGANIZATION_SUMMARY.md)

---

## Template for New Q&A

```markdown
### [Session X: YYYY-MM-DD] - Topic Title

**Q: Your question here**

**A:** Detailed answer with:
- Key points
- Implementation details
- Code examples if relevant
- References to documentation

**Documentation:** Links to related docs
**Status:** Open/Resolved/Implemented
```

---

## Categories of Questions

### Technical Implementation
- How to implement specific features
- Code architecture questions
- Integration challenges
- Performance optimization

### Algorithms & Mathematics
- Algorithm complexity analysis
- Mathematical formulations
- Optimization strategies
- Constraint satisfaction approaches

### Architecture & Design
- System design decisions
- Component interactions
- Data flow questions
- Integration patterns

### Debugging & Troubleshooting
- Error investigation
- Performance issues
- Configuration problems
- Unexpected behavior

---

## Quick Links

- [Index](../INDEX.md)
- [Architecture](../03-architecture/)
- [Algorithms](../04-algorithms/)
- [Performance](../05-performance/)
- [Development](../06-development/)

---

**Ask anything!** AI agents will document conversations here for future reference.
