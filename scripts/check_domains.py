"""Quick check of instructor/room domain sizes for CP model."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import collections

from src.io.data_store import DataStore
from src.utils.logging_config import quick_setup

quick_setup()
logger = logging.getLogger(__name__)

store = DataStore.from_json("data")
ctx = store.to_context()

# Check qualified instructor counts per course
qi_counts = []
for key, course in ctx.courses.items():
    qi = list(course.qualified_instructor_ids)
    qi_counts.append(len(qi))

cc = collections.Counter(qi_counts)
logger.info("Qualified instructors per course:")
for k in sorted(cc):
    logger.info("  %d qualified: %d courses", k, cc[k])
logger.info(
    "Courses with <3 qualified: %d/%d",
    sum(v for k, v in cc.items() if k < 3),
    len(ctx.courses),
)
logger.info(
    "Total instructors: %d, Total rooms: %d", len(ctx.instructors), len(ctx.rooms)
)

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

logger.info("Room domains per course (suitable + type_compat):")
rc_sizes = [s + t for _, s, t in room_counts]
logger.info(
    "  Min=%d, Max=%d, Avg=%.1f",
    min(rc_sizes),
    max(rc_sizes),
    sum(rc_sizes) / len(rc_sizes),
)
for key, s, t in sorted(room_counts, key=lambda x: -x[2])[:10]:
    logger.info("  %s: %d suitable, %d total domain", key, s, t)

# Estimate model size
logger.info("=== Model Size Estimate (543 effective genes) ===")
n_genes = 543

# If all instructors expanded
avg_instr = sum(qi_counts) / len(qi_counts)
expanded_instr = sum(189 if q < 3 else q for q in qi_counts)
logger.info("Avg qualified instructors: %.1f", avg_instr)
logger.info("HC2 optional intervals (original): %.0f", n_genes * avg_instr)
logger.info(
    "HC2 optional intervals (with fallback to all 189): ~%.0f",
    expanded_instr * (n_genes / len(qi_counts)),
)
avg_rooms = sum(rc_sizes) / len(rc_sizes)
logger.info("HC3 optional intervals: %.0f", n_genes * avg_rooms)
logger.info("Total optional intervals: ~%.0f", n_genes * (189 + avg_rooms))
