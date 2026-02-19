#!/usr/bin/env python3
"""Check raw JSON data for 8 courses with 0 suitable rooms."""

import json

with open("data/Course.json") as f:
    courses = json.load(f)
targets = {
    "CT 785 03",
    "CT 80X XX",
    "ENAR 202",
    "ENCE 305",
    "ENIE 325-334",
    "ENME 309",
    "ENSH 204",
    "ENSH 302",
}

for c in courses:
    cc = c.get("course_code", "")
    if cc in targets:
        print(f"  {cc}:")
        print(f"    required_room_features = {c.get('required_room_features')}")
        print(f"    specific_lab_features  = {c.get('specific_lab_features')}")
        print(f"    course_type            = {c.get('course_type')}")
        print()

# Also check: do ANY rooms have 'lecture hall' or 'seminar room' in specific_features?
with open("data/Rooms.json") as f:
    rooms = json.load(f)
for r in rooms:
    sf = r.get("specific_features", [])
    if sf and any("lecture" in f.lower() or "seminar" in f.lower() for f in sf):
        print(f"  Room {r['room_id']}: specific_features={sf}")
