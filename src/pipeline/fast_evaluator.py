#!/usr/bin/env python3
"""Fast numeric evaluator for pymoo migration.

Computes ALL 9 hard constraints that the original Evaluator uses:
  1. student_group_exclusivity   – (group, quantum) double-booking
  2. instructor_exclusivity      – (instructor, quantum) double-booking
  3. room_exclusivity            – (room, quantum) double-booking
  4. instructor_qualifications   – instructor not in course.qualified_instructor_ids
  5. room_suitability            – room not in precomputed suitable rooms
  6. instructor_time_availability – part-time instructor assigned outside available_quanta
  7. room_time_availability      – room assigned outside its available_quanta
  8. course_completeness         – always 0 (gene list is fixed, quanta match by construction)
  9. sibling_same_day            – sub-sessions of same course on same day

Returns per-constraint breakdown dict identical to Evaluator.breakdown().
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def fast_evaluate_hard(
    events: list[dict],
    instructor_assign: np.ndarray,  # shape (E,) instructor index per event
    room_assign: np.ndarray,  # shape (E,) room index per event
    time_assign: np.ndarray,  # shape (E,) start quantum per event
    allowed_instructors: list[list[int]],
    allowed_rooms: list[list[int]],
    instructor_available_quanta: dict[int, set | None],
    room_available_quanta: dict[int, set | None],
) -> dict[str, int]:
    """Compute per-constraint hard violation counts.

    All counts use the SAME convention as the original Evaluator:
    overlapping slots counted as sum(len(bucket)-1 for bucket if len>1).

    Returns dict with the 9 constraint names as keys.
    """
    n_events = len(events)
    instructor_assign = np.asarray(instructor_assign, dtype=int)
    room_assign = np.asarray(room_assign, dtype=int)
    time_assign = np.asarray(time_assign, dtype=int)

    # ---------- occupancy maps: (entity, quantum) -> [event_indices] ----------
    group_occ: dict[tuple, list[int]] = defaultdict(list)
    instructor_occ: dict[tuple, list[int]] = defaultdict(list)
    room_occ: dict[tuple, list[int]] = defaultdict(list)

    for e in range(n_events):
        s = int(time_assign[e])
        dur = events[e]["num_quanta"]
        inst = int(instructor_assign[e])
        rm = int(room_assign[e])
        gids = events[e]["group_ids"]

        for q in range(s, s + dur):
            for gid in gids:
                group_occ[(gid, q)].append(e)
            instructor_occ[(inst, q)].append(e)
            room_occ[(rm, q)].append(e)

    group_violations = sum(len(v) - 1 for v in group_occ.values() if len(v) > 1)
    instructor_violations = sum(
        len(v) - 1 for v in instructor_occ.values() if len(v) > 1
    )
    room_violations = sum(len(v) - 1 for v in room_occ.values() if len(v) > 1)

    # ---------- qualification / suitability -----------------------------------
    qual_violations = 0
    suit_violations = 0
    for e in range(n_events):
        inst = int(instructor_assign[e])
        rm = int(room_assign[e])
        if inst not in allowed_instructors[e]:
            qual_violations += 1
        if rm not in allowed_rooms[e]:
            suit_violations += 1

    # ---------- time-availability ---------------------------------------------
    inst_avail_violations = 0
    room_avail_violations = 0
    for e in range(n_events):
        s = int(time_assign[e])
        dur = events[e]["num_quanta"]
        inst = int(instructor_assign[e])
        rm = int(room_assign[e])

        # Instructor availability
        inst_slots = instructor_available_quanta.get(inst)
        if inst_slots is not None:  # None means full-time = always available
            for q in range(s, s + dur):
                if q not in inst_slots:
                    inst_avail_violations += 1

        # Room availability
        room_slots = room_available_quanta.get(rm)
        if room_slots is not None:  # None means always available
            for q in range(s, s + dur):
                if q not in room_slots:
                    room_avail_violations += 1

    # ---------- sibling same-day ------------------------------------------------
    # Sub-sessions of the same course (same course_id, course_type, group_ids)
    # should not be scheduled on the same day.
    _QPD = 7  # quanta per day (42 total / 6 days)
    course_keys: dict[tuple, list[int]] = defaultdict(list)
    for e in range(n_events):
        ev = events[e]
        ck = (ev["course_id"], ev["course_type"], tuple(sorted(ev["group_ids"])))
        course_keys[ck].append(e)

    sibling_same_day = 0
    for siblings in course_keys.values():
        if len(siblings) < 2:
            continue
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                day_i = int(time_assign[siblings[i]]) // _QPD
                day_j = int(time_assign[siblings[j]]) // _QPD
                if day_i == day_j:
                    sibling_same_day += 1

    return {
        "student_group_exclusivity": group_violations,
        "instructor_exclusivity": instructor_violations,
        "room_exclusivity": room_violations,
        "instructor_qualifications": qual_violations,
        "room_suitability": suit_violations,
        "instructor_time_availability": inst_avail_violations,
        "room_time_availability": room_avail_violations,
        "course_completeness": 0,  # fixed by construction
        "sibling_same_day": sibling_same_day,
    }


def fast_conflict_evaluator(
    start_assign: np.ndarray,
    duration_assign: np.ndarray,
    room_assign: np.ndarray,
    instructor_assign: np.ndarray,
    group_masks: np.ndarray,
    events_data: dict,
) -> tuple[int, int, int, float]:
    """Compatibility wrapper returning (room_conf, inst_conf, group_conf, soft).

    This is a thin adapter around *fast_evaluate_hard* that accepts the
    bitmask-based calling convention used by several standalone scripts.
    """
    events = events_data["events"]
    allowed_instructors = events_data["allowed_instructors"]
    allowed_rooms = events_data["allowed_rooms"]
    inst_avail = events_data.get("instructor_available_quanta", {})
    room_avail = events_data.get("room_available_quanta", {})

    result = fast_evaluate_hard(
        events,
        instructor_assign,
        room_assign,
        start_assign,
        allowed_instructors,
        allowed_rooms,
        inst_avail,
        room_avail,
    )

    room_conf = result["room_exclusivity"]
    inst_conf = result["instructor_exclusivity"]
    group_conf = result["student_group_exclusivity"]
    # Sum remaining hard violations into a soft penalty proxy
    soft_penalty = float(
        result["instructor_qualifications"]
        + result["room_suitability"]
        + result["instructor_time_availability"]
        + result["room_time_availability"]
    )
    return room_conf, inst_conf, group_conf, soft_penalty
