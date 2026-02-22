#!/usr/bin/env python3
"""Diagnose the 24 events with 0 suitable rooms."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.io.data_store import DataStore
from src.utils.logging_config import quick_setup
from src.utils.room_compatibility import is_room_suitable_for_course

logger = quick_setup()

store = DataStore.from_json("data")
ctx = store.to_context()

with open(".cache/events_with_domains.pkl", "rb") as f:
    d = pickle.load(f)

events = d["events"]
allowed_rooms = d["allowed_rooms"]
room_ids_sorted = sorted(ctx.rooms.keys())

empties = [i for i, ar in enumerate(allowed_rooms) if len(ar) == 0]
logger.info("Total events with 0 suitable rooms: %d", len(empties))
logger.info("")

# Collect unique course keys for these events
course_keys_seen = set()

for e_idx in empties:
    ev = events[e_idx]
    cid = ev["course_id"]
    ctype = ev["course_type"]
    course_key = (cid, ctype)
    course = ctx.courses.get(course_key)

    required = (
        getattr(course, "required_room_features", "lecture") if course else "lecture"
    )
    req_str = (required if isinstance(required, str) else str(required)).lower().strip()
    course_lab = getattr(course, "specific_lab_features", None) if course else None

    logger.info(
        "Event %d: course=%s type=%s groups=%s dur=%s",
        e_idx,
        cid,
        ctype,
        ev["group_ids"],
        ev["num_quanta"],
    )
    logger.info(
        '  required_room_features="%s"  specific_lab_features=%s',
        req_str,
        course_lab,
    )

    course_keys_seen.add(course_key)

    # Check ALL rooms and categorize failures
    type_fail = 0
    spec_fail = 0
    both_fail = 0
    sample_fails: list[str] = []
    for rid in room_ids_sorted:
        room = ctx.rooms[rid]
        rt = getattr(room, "room_features", "lecture")
        rt_str = (rt if isinstance(rt, str) else str(rt)).lower().strip()
        rsf = getattr(room, "specific_features", None)

        # Check type compatibility separately
        from src.utils.room_compatibility import is_room_type_compatible

        type_ok = is_room_type_compatible(req_str, rt_str)
        full_ok = is_room_suitable_for_course(req_str, rt_str, course_lab, rsf)

        if not full_ok:
            if not type_ok:
                type_fail += 1
                if len(sample_fails) < 2:
                    sample_fails.append(
                        f'    {rid}: type="{rt_str}" spec={rsf} -> TYPE_FAIL'
                    )
            else:
                spec_fail += 1
                if len(sample_fails) < 3:
                    sample_fails.append(
                        f'    {rid}: type="{rt_str}" spec={rsf} -> SPEC_FAIL (type ok, specific features mismatch)'
                    )

    logger.info(
        "  Failures: type_fail=%d spec_feature_fail=%d total=%d/75",
        type_fail,
        spec_fail,
        type_fail + spec_fail,
    )
    for sf in sample_fails:
        logger.info("%s", sf)
    logger.info("")

logger.info("=" * 70)
logger.info("Unique courses affected: %d", len(course_keys_seen))
for ck in sorted(course_keys_seen):
    logger.info("  %s", ck)

# Also show what room types exist
logger.info("")
logger.info("Room type distribution:")
from collections import Counter

rtypes: Counter[str] = Counter()
for rid, room in ctx.rooms.items():
    rt = getattr(room, "room_features", "lecture")
    rt_str = (rt if isinstance(rt, str) else str(rt)).lower().strip()
    rtypes[rt_str] += 1
for rt, cnt in rtypes.most_common():
    logger.info("  %s: %d", rt, cnt)
