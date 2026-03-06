import pickle
from collections import Counter

import numpy as np

with open(".cache/events_with_domains.pkl", "rb") as f:
    d = pickle.load(f)

events = d["events"]
E = len(events)
ai = d["allowed_instructors"]
ar = d["allowed_rooms"]
ast = d["allowed_starts"]

print(f"E (events)          = {E}")
print("T (quanta)          = 42")
print(f"n_instructors       = {len(d['instructor_to_idx'])}")
print(f"n_rooms             = {len(d['room_to_idx'])}")

groups = set()
for ev in events:
    groups.update(ev["group_ids"])
print(f"n_groups            = {len(groups)}")

durs = [ev["num_quanta"] for ev in events]
print(f"Q (sum durations)   = {sum(durs)}")
print(f"n_var = 3*E         = {3*E}")
print(f"Duration range      = {min(durs)}..{max(durs)}")
print(
    f"Inst domain sizes   = {min(len(x) for x in ai)}..{max(len(x) for x in ai)} (mean {np.mean([len(x) for x in ai]):.1f})"
)
print(
    f"Room domain sizes   = {min(len(x) for x in ar)}..{max(len(x) for x in ar)} (mean {np.mean([len(x) for x in ar]):.1f})"
)
print(
    f"Time domain sizes   = {min(len(x) for x in ast)}..{max(len(x) for x in ast)} (mean {np.mean([len(x) for x in ast]):.1f})"
)
print(f"Paired practicals   = {len(d.get('paired_practical_events', []))} pairs")
print(f"Cohort pairs        = {len(d.get('cohort_pairs', []))} pairs")

ct = Counter(ev["course_type"] for ev in events)
print(f"Course types        = {dict(ct)}")

gc = [len(ev["group_ids"]) for ev in events]
print(f"Groups/event range  = {min(gc)}..{max(gc)} (mean {np.mean(gc):.1f})")
