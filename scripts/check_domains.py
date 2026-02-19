"""Quick check of instructor/room domain sizes for CP model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import collections

from src.io.data_store import DataStore

store = DataStore.from_json("data")
ctx = store.to_context()

# Check qualified instructor counts per course
qi_counts = []
for key, course in ctx.courses.items():
    qi = list(course.qualified_instructor_ids)
    qi_counts.append(len(qi))

cc = collections.Counter(qi_counts)
print("Qualified instructors per course:")
for k in sorted(cc):
    print(f"  {k} qualified: {cc[k]} courses")
print(
    f"Courses with <3 qualified: {sum(v for k, v in cc.items() if k < 3)}/{len(ctx.courses)}"
)
print(f"Total instructors: {len(ctx.instructors)}, Total rooms: {len(ctx.rooms)}")

# Room domain sizes
from src.utils.room_compatibility import (
    is_room_suitable_for_course,
    is_room_type_compatible,
)

room_counts = []
for key, course in ctx.courses.items():
    req = str(getattr(course, "required_room_features", "lecture")).lower().strip()
    lab = getattr(course, "specific_lab_features", None)
    suitable = 0
    type_compat = 0
    for room in ctx.rooms.values():
        rt = str(getattr(room, "room_features", "lecture")).lower().strip()
        rf = getattr(room, "specific_features", None)
        if is_room_suitable_for_course(req, rt, lab, rf):
            suitable += 1
        elif is_room_type_compatible(req, rt):
            type_compat += 1
    room_counts.append((key, suitable, suitable + type_compat))

print("\nRoom domains per course (suitable + type_compat):")
rc_sizes = [s + t for _, s, t in room_counts]
print(
    f"  Min={min(rc_sizes)}, Max={max(rc_sizes)}, Avg={sum(rc_sizes) / len(rc_sizes):.1f}"
)
for key, s, t in sorted(room_counts, key=lambda x: -x[2])[:10]:
    print(f"  {key}: {s} suitable, {t} total domain")

# Estimate model size
print("\n=== Model Size Estimate (543 effective genes) ===")
n_genes = 543

# If all instructors expanded
avg_instr = sum(qi_counts) / len(qi_counts)
expanded_instr = sum(189 if q < 3 else q for q in qi_counts)
print(f"Avg qualified instructors: {avg_instr:.1f}")
print(f"HC2 optional intervals (original): {n_genes * avg_instr:.0f}")
print(
    f"HC2 optional intervals (with fallback to all 189): ~{expanded_instr * (n_genes / len(qi_counts)):.0f}"
)
avg_rooms = sum(rc_sizes) / len(rc_sizes)
print(f"HC3 optional intervals: {n_genes * avg_rooms:.0f}")
print(f"Total optional intervals: ~{n_genes * (189 + avg_rooms):.0f}")
