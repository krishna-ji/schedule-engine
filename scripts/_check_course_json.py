#!/usr/bin/env python3
"""Check raw JSON data for 8 courses with 0 suitable rooms."""

import json
import logging

from src.utils.logging_config import quick_setup

logger = quick_setup()

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
        logger.info("  %s:", cc)
        logger.info("    required_room_features = %s", c.get("required_room_features"))
        logger.info("    specific_lab_features  = %s", c.get("specific_lab_features"))
        logger.info("    course_type            = %s", c.get("course_type"))
        logger.info("")

# Also check: do ANY rooms have 'lecture hall' or 'seminar room' in specific_features?
with open("data/Rooms.json") as f:
    rooms = json.load(f)
for r in rooms:
    sf = r.get("specific_features", [])
    if sf and any("lecture" in f.lower() or "seminar" in f.lower() for f in sf):
        logger.info("  Room %s: specific_features=%s", r["room_id"], sf)
