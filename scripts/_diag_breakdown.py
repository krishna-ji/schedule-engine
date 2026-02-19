#!/usr/bin/env python3
"""Diagnose post-repair residual violations by constraint type."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.pipeline.fast_evaluator import fast_evaluate_hard
from src.pipeline.repair_operator import SchedulingRepair

repairer = SchedulingRepair()
E = repairer.n_events
with open("events_with_domains.pkl", "rb") as f:
    data = pickle.load(f)

rng = np.random.default_rng(42)
chrom = np.zeros(3 * E, dtype=int)
for e in range(E):
    ai = data["allowed_instructors"][e]
    ar = data["allowed_rooms"][e]
    at = data["allowed_starts"][e]
    chrom[3 * e] = rng.choice(ai) if ai else 0
    chrom[3 * e + 1] = rng.choice(ar) if ar else 0
    chrom[3 * e + 2] = rng.choice(at) if at else 0

repaired = repairer.repair(chrom)
inst = repaired[0::3]
room = repaired[1::3]
time_arr = repaired[2::3]
result = fast_evaluate_hard(
    data["events"],
    inst,
    room,
    time_arr,
    data["allowed_instructors"],
    data["allowed_rooms"],
    data["instructor_available_quanta"],
    data["room_available_quanta"],
)
print("POST-REPAIR BREAKDOWN:")
total = 0
for k, v in sorted(result.items(), key=lambda x: -x[1]):
    print(f"  {k:40s} {v:5d}")
    total += v
print(f"  {'TOTAL':40s} {total:5d}")

# Also show how many events still have each type of violation
print("\nPER-EVENT ANALYSIS:")
# Count events with group conflicts
from collections import defaultdict

group_occ = defaultdict(list)
inst_occ = defaultdict(list)
room_occ = defaultdict(list)
for e in range(E):
    s = int(time_arr[e])
    dur = data["events"][e]["num_quanta"]
    i = int(inst[e])
    r = int(room[e])
    gids = data["events"][e]["group_ids"]
    for q in range(s, s + dur):
        for gid in gids:
            group_occ[(gid, q)].append(e)
        inst_occ[(i, q)].append(e)
        room_occ[(r, q)].append(e)

# Events involved in group conflicts
group_conflict_events = set()
for evts in group_occ.values():
    if len(evts) > 1:
        group_conflict_events.update(evts)
print(f"  Events with group conflicts: {len(group_conflict_events)}")

inst_conflict_events = set()
for evts in inst_occ.values():
    if len(evts) > 1:
        inst_conflict_events.update(evts)
print(f"  Events with instructor conflicts: {len(inst_conflict_events)}")

room_conflict_events = set()
for evts in room_occ.values():
    if len(evts) > 1:
        room_conflict_events.update(evts)
print(f"  Events with room conflicts: {len(room_conflict_events)}")

# Room suitability
suit_bad = sum(1 for e in range(E) if int(room[e]) not in data["allowed_rooms"][e])
print(f"  Events with bad room suitability: {suit_bad}")

# Instructor qualification
qual_bad = sum(
    1 for e in range(E) if int(inst[e]) not in data["allowed_instructors"][e]
)
print(f"  Events with bad instructor qual: {qual_bad}")
