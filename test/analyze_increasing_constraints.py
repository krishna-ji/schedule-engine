"""
Analyze why hard constraints are INCREASING during evolution.
This should NEVER happen - hard constraints should only decrease!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Read the CSV
df = pd.read_csv("output/evaluation_20251024_141624/CSVs/hard_constraints_all.csv")

print("=" * 70)
print("HARD CONSTRAINT EVOLUTION ANALYSIS")
print("=" * 70)

print("\n1. COMPARISON: Gen 0 vs Gen 100")
print("-" * 70)
gen0 = df.iloc[0]
gen100 = df.iloc[-1]

constraints = [
    "no_group_overlap",
    "no_instructor_conflict",
    "room_type_mismatch",
    "availability_violations",
]

for c in constraints:
    change = gen100[c] - gen0[c]
    pct = (change / gen0[c] * 100) if gen0[c] > 0 else 0
    status = "❌ WORSE" if change > 0 else "✅ BETTER" if change < 0 else "➖ SAME"
    print(
        f"{c:30s}: {gen0[c]:6.0f} → {gen100[c]:6.0f} ({change:+6.0f}, {pct:+6.1f}%) {status}"
    )

print("\n2. MAXIMUM VALUES (Worst Point)")
print("-" * 70)
for c in constraints:
    max_gen = df[c].idxmax()
    max_val = df[c].max()
    print(f"{c:30s}: Gen {max_gen:3d} = {max_val:.0f}")

print("\n3. CONSTRAINT TRENDS")
print("-" * 70)
for c in constraints:
    increasing_gens = 0
    for i in range(1, len(df)):
        if df[c].iloc[i] > df[c].iloc[i - 1]:
            increasing_gens += 1
    print(
        f"{c:30s}: Increased in {increasing_gens}/{len(df)-1} generations ({increasing_gens/(len(df)-1)*100:.1f}%)"
    )

print("\n4. KEY PROBLEM AREAS")
print("-" * 70)

# Find where constraints jumped significantly
print("\nLargest INCREASES per constraint:")
for c in constraints:
    diffs = df[c].diff()
    max_increase_idx = diffs.idxmax()
    max_increase = diffs.max()
    if max_increase > 0:
        print(
            f"{c:30s}: Gen {max_increase_idx-1}→{max_increase_idx}: +{max_increase:.0f}"
        )

print("\n" + "=" * 70)
print("🔍 ROOT CAUSE ANALYSIS")
print("=" * 70)

# Identify the culprit
no_group = gen100["no_group_overlap"] - gen0["no_group_overlap"]
no_instructor = gen100["no_instructor_conflict"] - gen0["no_instructor_conflict"]
room_type = gen100["room_type_mismatch"] - gen0["room_type_mismatch"]
avail = gen100["availability_violations"] - gen0["availability_violations"]

print("\nProblem Summary:")
if no_group < 0:
    print(f"  ✅ Group overlaps: DECREASING ({no_group:.0f})")
else:
    print(f"  ❌ Group overlaps: INCREASING (+{no_group:.0f}) - CRITICAL!")

if no_instructor > 0:
    print(f"  ❌ Instructor conflicts: INCREASING (+{no_instructor:.0f}) - CRITICAL!")
else:
    print(f"  ✅ Instructor conflicts: DECREASING ({no_instructor:.0f})")

if room_type < 0:
    print(f"  ✅ Room type mismatches: DECREASING ({room_type:.0f})")
else:
    print(f"  ❌ Room type mismatches: INCREASING (+{room_type:.0f})")

if avail < 0:
    print(f"  ✅ Availability violations: DECREASING ({avail:.0f})")
else:
    print(f"  ❌ Availability violations: INCREASING (+{avail:.0f}) - CRITICAL!")

print("\n" + "=" * 70)
print("💡 LIKELY ROOT CAUSES")
print("=" * 70)

issues = []

if no_group < 0 and no_instructor > 0 and avail < 0:
    issues.append(
        "TRADE-OFF PROBLEM: Reducing group overlaps & availability, but INCREASING instructor conflicts"
    )
    issues.append(
        "→ Repair/mutation may be moving sessions to fix one constraint but breaking instructor constraint"
    )

if no_instructor > 0:
    issues.append(
        "INSTRUCTOR CONFLICT INCREASING: Genetic operators not preserving instructor uniqueness"
    )
    issues.append(
        "→ Check: crossover/mutation may assign same instructor to overlapping time slots"
    )

if room_type < 0 and (no_group < 0 or avail < 0):
    issues.append("Good: Room types improving while other constraints improve")

if no_group < 0:
    issues.append("Group overlaps decreasing - selection is working")

for i, issue in enumerate(issues, 1):
    print(f"\n{i}. {issue}")

print("\n" + "=" * 70)
