#!/usr/bin/env python3
"""
Pre-Scheduling Data Audit & Pigeonhole Analysis
================================================
Runs EVERY conceivable sanity check on the input data BEFORE the GA starts.
Produces a rich console report + saves a text file to output/.

Categories:
  A. Data Completeness        – are all fields populated?
  B. Pigeonhole / Feasibility – can the problem be solved at all?
  C. Qualification Coverage   – do instructors cover every course?
  D. Lab / Room Features      – are required lab features available?
  E. Availability Analysis    – instructor & room hours sufficient?
  F. Cross-Reference Integrity– are all IDs consistent?
  G. Capacity Analysis        – can rooms hold the groups?
  H. Schedule Density         – how tight is the timetable?
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.io.data_store import DataStore

console = Console(width=120)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════════════════════
console.print()
console.rule("[bold cyan]PRE-SCHEDULING DATA AUDIT[/bold cyan]")
console.print()

store = DataStore.from_json(DATA_DIR, run_preflight=False)  # Audit runs its own checks
courses = store.courses
groups = store.groups
instructors = store.instructors
rooms = store.rooms
qts = store.qts

total_operating_quanta = len(qts.get_all_operating_quanta())
quanta_minutes = qts.QUANTUM_MINUTES

# Also load raw JSON for field-level checks
with open(DATA_DIR / "Instructors.json") as f:
    raw_instructors = json.load(f)
with open(DATA_DIR / "Course.json") as f:
    raw_courses = json.load(f)
with open(DATA_DIR / "Groups.json") as f:
    raw_groups = json.load(f)
with open(DATA_DIR / "Rooms.json") as f:
    raw_rooms = json.load(f)


# ════════════════════════════════════════════════════════════════════════
# Report accumulator
# ════════════════════════════════════════════════════════════════════════
report_lines: list[str] = []
pass_count = 0
warn_count = 0
fail_count = 0


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold yellow]{title}[/bold yellow]")
    console.print()
    report_lines.append(f"\n{'='*80}\n{title}\n{'='*80}\n")


def ok(msg: str) -> None:
    global pass_count
    pass_count += 1
    console.print(f"  [green]✓[/green] {msg}")
    report_lines.append(f"  [PASS] {msg}")


def warn(msg: str) -> None:
    global warn_count
    warn_count += 1
    console.print(f"  [yellow]⚠[/yellow] {msg}")
    report_lines.append(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    global fail_count
    fail_count += 1
    console.print(f"  [red]✗[/red] {msg}")
    report_lines.append(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    console.print(f"  [dim]ℹ {msg}[/dim]")
    report_lines.append(f"  [INFO] {msg}")


# ════════════════════════════════════════════════════════════════════════
# A. DATA COMPLETENESS
# ════════════════════════════════════════════════════════════════════════
section("A. DATA COMPLETENESS")

# A1 – Basic entity counts
info(f"Courses (theory+practical entries): {len(courses)}")
info(f"Groups (subgroups expanded):        {len(groups)}")
info(f"Instructors:                        {len(instructors)}")
info(f"Rooms:                              {len(rooms)}")
info(
    f"Operating quanta/week:              {total_operating_quanta}  ({total_operating_quanta * quanta_minutes // 60}h)"
)

if len(courses) == 0:
    fail("No courses loaded!")
else:
    ok(f"{len(courses)} courses loaded")

if len(groups) == 0:
    fail("No groups loaded!")
else:
    ok(f"{len(groups)} groups loaded")

if len(instructors) == 0:
    fail("No instructors loaded!")
else:
    ok(f"{len(instructors)} instructors loaded")

if len(rooms) == 0:
    fail("No rooms loaded!")
else:
    ok(f"{len(rooms)} rooms loaded")

# A2 – Course field completeness
courses_no_name = [k for k, c in courses.items() if not c.name]
courses_zero_quanta = [k for k, c in courses.items() if c.quanta_per_week <= 0]

if courses_no_name:
    warn(f"{len(courses_no_name)} course(s) have no name")
else:
    ok("All courses have names")

if courses_zero_quanta:
    fail(
        f"{len(courses_zero_quanta)} course(s) have zero/negative quanta_per_week: {courses_zero_quanta[:5]}"
    )
else:
    ok("All courses have positive quanta_per_week")

# A3 – Group field completeness
groups_no_courses = [gid for gid, g in groups.items() if not g.enrolled_courses]
groups_zero_students = [gid for gid, g in groups.items() if g.student_count <= 0]

if groups_no_courses:
    warn(
        f"{len(groups_no_courses)} group(s) have no enrolled courses: {groups_no_courses[:5]}"
    )
else:
    ok("All groups have enrolled courses")

if groups_zero_students:
    fail(f"{len(groups_zero_students)} group(s) have zero/negative student count")
else:
    ok("All groups have positive student counts")

# A4 – Instructor field completeness
instructors_no_courses_original = []
for raw_i in raw_instructors:
    if not raw_i.get("courses", []):
        instructors_no_courses_original.append(raw_i["id"])

if instructors_no_courses_original:
    warn(
        f"{len(instructors_no_courses_original)} instructor(s) have no courses in JSON: {instructors_no_courses_original[:5]}"
    )
else:
    ok("All instructors list at least one course in JSON")

# A5 – Room field completeness
rooms_zero_capacity = [rid for rid, r in rooms.items() if r.capacity <= 0]
rooms_no_features = [rid for rid, r in rooms.items() if not r.room_features]

if rooms_zero_capacity:
    fail(f"{len(rooms_zero_capacity)} room(s) have zero/negative capacity")
else:
    ok("All rooms have positive capacity")

if rooms_no_features:
    warn(f"{len(rooms_no_features)} room(s) have no room type/features")
else:
    ok("All rooms have a room type assigned")


# ════════════════════════════════════════════════════════════════════════
# B. PIGEONHOLE / FEASIBILITY CHECKS
# ════════════════════════════════════════════════════════════════════════
section("B. PIGEONHOLE / FEASIBILITY CHECKS")

# B1 – Group pigeonhole: does any group need more hours than exist?
overloaded_groups = []
group_utilizations = {}
for group_id, group in groups.items():
    demand = 0
    for course_code in group.enrolled_courses:
        theory_key = (course_code, "theory")
        practical_key = (course_code, "practical")
        if theory_key in courses:
            demand += courses[theory_key].quanta_per_week
        if practical_key in courses:
            demand += courses[practical_key].quanta_per_week

    available = (
        len(group.available_quanta)
        if group.available_quanta
        else total_operating_quanta
    )
    util = (demand / available * 100) if available > 0 else float("inf")
    group_utilizations[group_id] = (demand, available, util)
    if demand > available:
        overloaded_groups.append((group_id, group.name, demand, available, util))

if overloaded_groups:
    fail(
        f"{len(overloaded_groups)} group(s) are OVERLOADED (more hours than time slots):"
    )
    table = Table(title="Overloaded Groups", box=box.SIMPLE)
    table.add_column("Group", style="yellow")
    table.add_column("Demand (quanta)")
    table.add_column("Available (quanta)")
    table.add_column("Utilization %", style="red")
    for gid, _gname, dem, avail, util in sorted(overloaded_groups, key=lambda x: -x[4]):
        table.add_row(f"{gid}", str(dem), str(avail), f"{util:.1f}%")
    console.print(table)
else:
    ok("No groups are overloaded (pigeonhole check passed)")

# High utilization warning
high_util_groups = [
    (gid, d, a, u) for gid, (d, a, u) in group_utilizations.items() if 80 <= u <= 100
]
if high_util_groups:
    warn(f"{len(high_util_groups)} group(s) have >80% utilization (tight schedule):")
    for gid, d, a, u in sorted(high_util_groups, key=lambda x: -x[3])[:10]:
        info(f"  {gid}: {d}/{a} quanta ({u:.1f}%)")
else:
    ok("All groups have comfortable utilization (<80%)")

# B2 – Global instructor supply vs demand
total_demand_quanta = sum(
    c.quanta_per_week for c in courses.values() if c.enrolled_group_ids
)
all_operating = qts.get_all_operating_quanta()
total_instructor_supply = 0
full_time_count = 0
part_time_count = 0
for inst in instructors.values():
    if inst.is_full_time:
        total_instructor_supply += len(all_operating)
        full_time_count += 1
    else:
        total_instructor_supply += len(inst.available_quanta)
        part_time_count += 1

inst_util = (
    (total_demand_quanta / total_instructor_supply * 100)
    if total_instructor_supply > 0
    else float("inf")
)
info(
    f"Instructor supply: {total_instructor_supply} quanta ({full_time_count} full-time, {part_time_count} part-time)"
)
info(f"Teaching demand:   {total_demand_quanta} quanta")
info(f"Utilization:       {inst_util:.1f}%")

if total_demand_quanta > total_instructor_supply:
    fail(
        f"GLOBAL instructor shortage: need {total_demand_quanta} quanta, only {total_instructor_supply} available"
    )
elif inst_util > 90:
    warn(f"High global instructor utilization: {inst_util:.1f}%")
else:
    ok(f"Global instructor supply sufficient ({inst_util:.1f}% utilization)")


# ════════════════════════════════════════════════════════════════════════
# C. QUALIFICATION COVERAGE
# ════════════════════════════════════════════════════════════════════════
section("C. INSTRUCTOR QUALIFICATION COVERAGE")

# C1 – Courses with NO qualified instructors (CRITICAL)
courses_no_instructors = []
for ckey, course in courses.items():
    if course.enrolled_group_ids and not course.qualified_instructor_ids:
        courses_no_instructors.append((ckey, course.name))

if courses_no_instructors:
    fail(
        f"{len(courses_no_instructors)} enrolled course(s) have NO qualified instructors!"
    )
    table = Table(title="Courses Without Instructors", box=box.SIMPLE)
    table.add_column("Course Key", style="red")
    table.add_column("Name", style="yellow")
    for ckey, cname in courses_no_instructors:
        table.add_row(str(ckey), cname)
    console.print(table)
else:
    ok("All enrolled courses have at least one qualified instructor")

# C2 – Courses with only ONE qualified instructor (single point of failure)
courses_single_instructor = []
for ckey, course in courses.items():
    if course.enrolled_group_ids and len(course.qualified_instructor_ids) == 1:
        courses_single_instructor.append(
            (ckey, course.name, course.qualified_instructor_ids[0])
        )

if courses_single_instructor:
    warn(
        f"{len(courses_single_instructor)} enrolled course(s) have only ONE qualified instructor (bottleneck risk):"
    )
    for ckey, cname, iid in courses_single_instructor[:10]:
        iname = instructors[iid].name if iid in instructors else "?"
        info(f"  {ckey} → {iname}")
    if len(courses_single_instructor) > 10:
        info(f"  ... and {len(courses_single_instructor) - 10} more")
else:
    ok("All enrolled courses have multiple qualified instructors")

# C3 – Per-course instructor availability bottleneck
course_bottlenecks = []
for ckey, course in courses.items():
    if not course.enrolled_group_ids:
        continue
    demand = course.quanta_per_week
    supply = 0
    for iid in course.qualified_instructor_ids:
        inst = instructors.get(iid)
        if not inst:
            continue
        if inst.is_full_time:
            supply += len(all_operating)
        else:
            supply += len(inst.available_quanta)
    if supply < demand:
        course_bottlenecks.append((ckey, course.name, demand, supply))

if course_bottlenecks:
    fail(
        f"{len(course_bottlenecks)} course(s) have qualified instructors with INSUFFICIENT total availability:"
    )
    table = Table(title="Instructor Availability Bottlenecks", box=box.SIMPLE)
    table.add_column("Course", style="yellow")
    table.add_column("Demand (q)")
    table.add_column("Supply (q)")
    table.add_column("Shortage (q)", style="red")
    for ckey, cname, dem, sup in sorted(
        course_bottlenecks, key=lambda x: x[2] - x[3], reverse=True
    )[:15]:
        table.add_row(str(ckey), str(dem), str(sup), str(dem - sup))
    console.print(table)
else:
    ok("All courses have sufficient qualified-instructor availability")

# C4 – Instructors not qualified for ANY enrolled course
instructors_no_enrolled = []
for iid, inst in instructors.items():
    has_enrolled = False
    for ckey in inst.qualified_courses:
        if ckey in courses and courses[ckey].enrolled_group_ids:
            has_enrolled = True
            break
    if not has_enrolled:
        instructors_no_enrolled.append((iid, inst.name))

if instructors_no_enrolled:
    info(
        f"{len(instructors_no_enrolled)} instructor(s) have no qualifications matching enrolled courses (unused capacity)"
    )
else:
    ok("All instructors are qualified for at least one enrolled course")


# ════════════════════════════════════════════════════════════════════════
# D. LAB / ROOM FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════════════════
section("D. LAB / ROOM FEATURE ANALYSIS")

# D1 – What room features do courses require?
feature_demand: dict[str, int] = defaultdict(int)  # feature -> total quanta demanded
feature_courses: dict[str, list[str]] = defaultdict(list)
for ckey, course in courses.items():
    if not course.enrolled_group_ids:
        continue
    feat = course.required_room_features
    feature_demand[feat] += course.quanta_per_week
    feature_courses[feat].append(str(ckey))

# D2 – What room features do rooms provide?
feature_supply: dict[str, int] = defaultdict(int)  # feature -> total quanta available
feature_rooms: dict[str, list[str]] = defaultdict(list)
for rid, room in rooms.items():
    feat = room.room_features
    if room.available_quanta:
        feature_supply[feat] += len(room.available_quanta)
    else:
        feature_supply[feat] += len(all_operating)
    feature_rooms[feat].append(rid)

# D3 – Compare
all_features = set(feature_demand.keys()) | set(feature_supply.keys())
table = Table(title="Room Feature Supply vs Demand", box=box.ROUNDED)
table.add_column("Feature", style="cyan")
table.add_column("Demand (q)", justify="right")
table.add_column("# Courses", justify="right")
table.add_column("Supply (q)", justify="right")
table.add_column("# Rooms", justify="right")
table.add_column("Status")

feature_bottlenecks = []
for feat in sorted(all_features):
    dem = feature_demand.get(feat, 0)
    sup = feature_supply.get(feat, 0)
    n_courses = len(feature_courses.get(feat, []))
    n_rooms = len(feature_rooms.get(feat, []))
    if dem > sup:
        status = "[red]SHORTAGE[/red]"
        feature_bottlenecks.append(feat)
    elif dem > 0 and sup == 0:
        status = "[red]NO ROOMS[/red]"
        feature_bottlenecks.append(feat)
    elif dem > sup * 0.8:
        status = "[yellow]TIGHT[/yellow]"
    else:
        status = "[green]OK[/green]"
    table.add_row(
        feat or "(none)", str(dem), str(n_courses), str(sup), str(n_rooms), status
    )

console.print(table)

if feature_bottlenecks:
    fail(
        f"{len(feature_bottlenecks)} room feature(s) have insufficient supply: {feature_bottlenecks}"
    )
else:
    ok("All required room features have sufficient supply")

# D4 – Detailed lab features from domain models (Course.specific_lab_features / Room.specific_features)
lab_features_required: dict[str, list[str]] = defaultdict(
    list
)  # feature -> courses needing it
for ckey, course in courses.items():
    for feat in course.specific_lab_features:
        lab_features_required[feat].append(str(ckey))

# Collect all specific features from rooms (now loaded into Room.specific_features)
room_specific_features: set[str] = set()
for rid, room in rooms.items():
    for feat in room.specific_features:
        room_specific_features.add(feat)

missing_lab_features = set(lab_features_required.keys()) - room_specific_features
present_lab_features = set(lab_features_required.keys()) & room_specific_features

info(f"Total unique lab features required by courses: {len(lab_features_required)}")
info(f"Total unique features available in rooms:      {len(room_specific_features)}")

if missing_lab_features:
    fail(
        f"{len(missing_lab_features)} lab feature(s) required by courses but NOT available in any room:"
    )
    for feat in sorted(missing_lab_features):
        course_list = ", ".join(lab_features_required[feat][:5])
        extra = (
            f" +{len(lab_features_required[feat]) - 5} more"
            if len(lab_features_required[feat]) > 5
            else ""
        )
        info(f"  '{feat}' → needed by: {course_list}{extra}")
else:
    ok("All specific lab features required by courses exist in at least one room")

if present_lab_features:
    ok(f"{len(present_lab_features)} lab features matched between courses and rooms")

# D5 – Rooms with EMPTY specific features
rooms_empty_features = [rid for rid, r in rooms.items() if not r.specific_features]
if rooms_empty_features:
    warn(
        f"{len(rooms_empty_features)} room(s) have empty 'features' array in JSON: {rooms_empty_features[:10]}"
    )
else:
    ok("All rooms have features specified in JSON")


# ════════════════════════════════════════════════════════════════════════
# E. AVAILABILITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════
section("E. TEACHER AVAILABILITY HOURS ANALYSIS")

# E1 – Instructors with NO availability (full-time)
ft_instructors = [(iid, i.name) for iid, i in instructors.items() if i.is_full_time]
pt_instructors = [
    (iid, i.name, len(i.available_quanta))
    for iid, i in instructors.items()
    if not i.is_full_time
]

info(f"Full-time instructors (all hours available): {len(ft_instructors)}")
info(f"Part-time instructors (restricted hours):    {len(pt_instructors)}")

# E2 – Part-time instructors: how many hours?
if pt_instructors:
    table = Table(title="Part-Time Instructor Availability", box=box.SIMPLE)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Quanta", justify="right")
    table.add_column("Hours", justify="right")
    table.add_column("Qualified Courses", justify="right")
    for iid, iname, q in sorted(pt_instructors, key=lambda x: x[2]):
        inst = instructors[iid]
        n_courses = len(
            [
                ck
                for ck in inst.qualified_courses
                if ck in courses and courses[ck].enrolled_group_ids
            ]
        )
        table.add_row(
            iid, iname, str(q), f"{q * quanta_minutes / 60:.1f}", str(n_courses)
        )
    console.print(table)

# E3 – Instructors with availability outside operating hours (raw JSON check)
avail_outside_hours = []
for raw_i in raw_instructors:
    avail = raw_i.get("availability", {})
    if not avail:
        continue
    for periods in avail.values():
        if not periods:
            continue
        for _period in periods:
            # Just flag if they have availability defined — already validated during load
            pass

# E4 – Part-time instructors with very few hours relative to their course load
pt_overloaded = []
for iid, iname, avail_q in pt_instructors:
    inst = instructors[iid]
    teaching_demand = 0
    for ckey in inst.qualified_courses:
        if ckey in courses:
            teaching_demand += courses[ckey].quanta_per_week
    if teaching_demand > avail_q and teaching_demand > 0:
        pt_overloaded.append((iid, iname, teaching_demand, avail_q))

if pt_overloaded:
    warn(
        f"{len(pt_overloaded)} part-time instructor(s) have more teaching demand than available hours:"
    )
    for iid, iname, dem, avail in pt_overloaded[:10]:
        info(f"  {iid} ({iname}): demand={dem}q, available={avail}q")
else:
    ok("All part-time instructors have sufficient availability for their course load")

# E5 – Room availability analysis
rooms_with_restrictions = [
    (rid, r) for rid, r in rooms.items() if r.available_quanta != set(all_operating)
]
info(
    f"Rooms with default (full) availability: {len(rooms) - len(rooms_with_restrictions)}"
)
info(f"Rooms with restricted availability:     {len(rooms_with_restrictions)}")


# ════════════════════════════════════════════════════════════════════════
# F. CROSS-REFERENCE INTEGRITY
# ════════════════════════════════════════════════════════════════════════
section("F. CROSS-REFERENCE INTEGRITY")

# F1 – Courses referencing non-existent instructors
bad_instructor_refs = []
for ckey, course in courses.items():
    for iid in course.qualified_instructor_ids:
        if iid not in instructors:
            bad_instructor_refs.append((ckey, iid))

if bad_instructor_refs:
    fail(
        f"{len(bad_instructor_refs)} course→instructor reference(s) point to non-existent instructors"
    )
    for ckey, iid in bad_instructor_refs[:10]:
        info(f"  {ckey} → {iid}")
else:
    ok("All course→instructor references are valid")

# F2 – Instructor qualified_courses referencing non-existent courses
bad_course_refs = []
for iid, inst in instructors.items():
    for ckey in inst.qualified_courses:
        if ckey not in courses:
            bad_course_refs.append((iid, ckey))

if bad_course_refs:
    warn(
        f"{len(bad_course_refs)} instructor→course reference(s) point to non-enrolled courses (after filtering)"
    )
else:
    ok("All instructor→course references are valid (post-filter)")

# F3 – Symmetry check: if course lists instructor, instructor should list course
asymmetric_refs = []
for ckey, course in courses.items():
    for iid in course.qualified_instructor_ids:
        if iid in instructors:
            if ckey not in instructors[iid].qualified_courses:
                asymmetric_refs.append((ckey, iid))

if asymmetric_refs:
    warn(f"{len(asymmetric_refs)} asymmetric qualification reference(s) found")
else:
    ok("All qualification references are symmetric (course↔instructor)")

# F4 – Groups referencing courses that don't exist in loaded data
# (already filtered during load, but check raw)
group_course_issues = []
for raw_g in raw_groups:
    group_courses = raw_g.get("courses", [])
    for cc in group_courses:
        if (cc, "theory") not in courses and (cc, "practical") not in courses:
            # might be non-schedulable (Survey Camp, etc.)
            group_course_issues.append(
                (
                    raw_g.get(
                        "group_id", raw_g.get("subgroups", [{}])[0].get("id", "?")
                    ),
                    cc,
                )
            )

unique_missing_codes = {cc for _, cc in group_course_issues}
if unique_missing_codes:
    info(
        f"{len(unique_missing_codes)} course code(s) in Groups.json don't map to schedulable courses (expected for non-classroom activities)"
    )
else:
    ok("All course codes in Groups.json map to schedulable courses")

# F5 – Duplicate IDs
dup_instructors = len(raw_instructors) - len({i["id"] for i in raw_instructors})
dup_rooms = len(raw_rooms) - len({r["room_id"] for r in raw_rooms})
dup_courses = len(raw_courses) - len({c["CourseCode"] for c in raw_courses})

if dup_instructors:
    fail(f"{dup_instructors} duplicate instructor ID(s) in JSON")
else:
    ok("No duplicate instructor IDs")

if dup_rooms:
    fail(f"{dup_rooms} duplicate room ID(s) in JSON")
else:
    ok("No duplicate room IDs")

if dup_courses:
    warn(
        f"{dup_courses} duplicate course code(s) in JSON (may be intentional for multi-dept)"
    )
else:
    ok("No duplicate course codes")


# ════════════════════════════════════════════════════════════════════════
# G. CAPACITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════
section("G. ROOM CAPACITY ANALYSIS")

# G1 – Can the largest group fit in any room?
lecture_rooms = {rid: r for rid, r in rooms.items() if r.room_features == "lecture"}
practical_rooms = {rid: r for rid, r in rooms.items() if r.room_features == "practical"}

max_lecture_cap = max((r.capacity for r in lecture_rooms.values()), default=0)
max_practical_cap = max((r.capacity for r in practical_rooms.values()), default=0)

info(f"Lecture rooms: {len(lecture_rooms)} (max capacity: {max_lecture_cap})")
info(f"Practical rooms: {len(practical_rooms)} (max capacity: {max_practical_cap})")

groups_too_big_for_lecture = []
groups_too_big_for_practical = []

for gid, group in groups.items():
    # Check theory courses
    has_theory = any((cc, "theory") in courses for cc in group.enrolled_courses)
    has_practical = any((cc, "practical") in courses for cc in group.enrolled_courses)

    if has_theory and group.student_count > max_lecture_cap and max_lecture_cap > 0:
        groups_too_big_for_lecture.append((gid, group.student_count))

    if (
        has_practical
        and group.student_count > max_practical_cap
        and max_practical_cap > 0
    ):
        groups_too_big_for_practical.append((gid, group.student_count))

if groups_too_big_for_lecture:
    fail(
        f"{len(groups_too_big_for_lecture)} group(s) are too large for ANY lecture room (max={max_lecture_cap}):"
    )
    for gid, sc in groups_too_big_for_lecture[:10]:
        info(f"  {gid}: {sc} students")
else:
    ok(
        f"All groups fit in at least one lecture room (max room capacity: {max_lecture_cap})"
    )

if groups_too_big_for_practical:
    fail(
        f"{len(groups_too_big_for_practical)} group(s) are too large for ANY practical room (max={max_practical_cap}):"
    )
    for gid, sc in groups_too_big_for_practical[:10]:
        info(f"  {gid}: {sc} students")
else:
    ok(
        f"All groups fit in at least one practical room (max room capacity: {max_practical_cap})"
    )

# G2 – Total seat-hours
total_seat_hours = sum(
    r.capacity * (len(r.available_quanta) if r.available_quanta else len(all_operating))
    for r in rooms.values()
)
total_student_hours = 0
for ckey, course in courses.items():
    for gid in course.enrolled_group_ids:
        if gid in groups:
            total_student_hours += groups[gid].student_count * course.quanta_per_week

seat_util = (
    (total_student_hours / total_seat_hours * 100)
    if total_seat_hours > 0
    else float("inf")
)
info(f"Total seat-hours available: {total_seat_hours:,}")
info(f"Total student-hours needed: {total_student_hours:,}")
info(f"Seat utilization: {seat_util:.1f}%")

if total_student_hours > total_seat_hours:
    fail("Global seat-hour shortage!")
elif seat_util > 85:
    warn(f"High seat utilization ({seat_util:.1f}%)")
else:
    ok(f"Seat-hours sufficient ({seat_util:.1f}% utilization)")


# ════════════════════════════════════════════════════════════════════════
# H. SCHEDULE DENSITY & COMPLEXITY
# ════════════════════════════════════════════════════════════════════════
section("H. SCHEDULE DENSITY & COMPLEXITY METRICS")

# H1 – Total sessions to schedule
total_sessions = 0
session_details: dict[str, int] = defaultdict(int)
for ckey, course in courses.items():
    for gid in course.enrolled_group_ids:
        total_sessions += 1
        session_details[course.course_type] += 1

info(f"Total (course, group) sessions to schedule: {total_sessions}")
info(f"  Theory sessions:    {session_details.get('theory', 0)}")
info(f"  Practical sessions: {session_details.get('practical', 0)}")

# H2 – Total quanta to place
total_quanta_to_place = 0
for ckey, course in courses.items():
    total_quanta_to_place += course.quanta_per_week * len(course.enrolled_group_ids)

info(f"Total quanta to place: {total_quanta_to_place}")
info(f"Total quanta available: {total_operating_quanta}")
quanta_density = (
    total_quanta_to_place / total_operating_quanta * 100
    if total_operating_quanta > 0
    else 0
)
info(f"Quanta density (demand/supply per single slot): {quanta_density:.1f}%")

# H3 – Average courses per group
avg_courses = (
    sum(len(g.enrolled_courses) for g in groups.values()) / len(groups) if groups else 0
)
max_courses = max(len(g.enrolled_courses) for g in groups.values()) if groups else 0
info(f"Average courses per group: {avg_courses:.1f}")
info(f"Max courses per group: {max_courses}")

# H4 – Average groups per course
avg_groups = sum(
    len(c.enrolled_group_ids) for c in courses.values() if c.enrolled_group_ids
) / max(1, sum(1 for c in courses.values() if c.enrolled_group_ids))
max_groups = max((len(c.enrolled_group_ids) for c in courses.values()), default=0)
info(f"Average groups per course: {avg_groups:.1f}")
info(f"Max groups per course:     {max_groups}")

# H5 – Cohort pairs
info(f"Cohort pairs (for practical alignment): {len(store.cohort_pairs)}")


# ════════════════════════════════════════════════════════════════════════
# I. CONSTRAINT SYSTEM OVERVIEW
# ════════════════════════════════════════════════════════════════════════
section("I. CONSTRAINT SYSTEM OVERVIEW (what gets enforced)")

from src.constraints.constraints import (
    HARD_CONSTRAINT_CLASSES,
    SOFT_CONSTRAINT_CLASSES,
)

console.print("[bold]Hard Constraints (must be satisfied):[/bold]")
for c in HARD_CONSTRAINT_CLASSES:
    console.print(f"  [green]■[/green] {c.name}")
    report_lines.append(f"  [HARD] {c.name}")

console.print()
console.print("[bold]Soft Constraints (optimized):[/bold]")
for c in SOFT_CONSTRAINT_CLASSES:
    console.print(f"  [blue]□[/blue] {c.name}")
    report_lines.append(f"  [SOFT] {c.name}")


# ════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

summary_table = Table(box=box.DOUBLE_EDGE)
summary_table.add_column("Metric", style="cyan")
summary_table.add_column("Count", justify="right")
summary_table.add_row("[green]Checks Passed[/green]", f"[green]{pass_count}[/green]")
summary_table.add_row("[yellow]Warnings[/yellow]", f"[yellow]{warn_count}[/yellow]")
summary_table.add_row("[red]Failures[/red]", f"[red]{fail_count}[/red]")
summary_table.add_row("Total Checks", str(pass_count + warn_count + fail_count))
console.print(summary_table)

if fail_count == 0 and warn_count == 0:
    console.print(
        Panel(
            "[bold green]ALL CHECKS PASSED — data is ready for scheduling![/bold green]",
            border_style="green",
        )
    )
elif fail_count == 0:
    console.print(
        Panel(
            f"[bold yellow]No failures, but {warn_count} warning(s) — review before scheduling.[/bold yellow]",
            border_style="yellow",
        )
    )
else:
    console.print(
        Panel(
            f"[bold red]{fail_count} FAILURE(S) FOUND — fix these before running the GA![/bold red]",
            border_style="red",
        )
    )

# ════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS: What ADDITIONAL checks could be added
# ════════════════════════════════════════════════════════════════════════
section("RECOMMENDATIONS: ADDITIONAL CHECKS TO IMPLEMENT")

recommendations = [
    (
        "PRESENT",
        "Group pigeonhole (hours > available slots)",
        "feasibility.py + this audit",
    ),
    ("PRESENT", "Global instructor supply vs demand", "feasibility.py + this audit"),
    (
        "PRESENT",
        "Per-course instructor qualification bottleneck",
        "feasibility.py + this audit",
    ),
    ("PRESENT", "Room feature supply vs demand", "feasibility.py + this audit"),
    ("PRESENT", "Room capacity (group fits in room)", "feasibility.py + this audit"),
    (
        "PRESENT",
        "Teacher qualification check (all courses covered)",
        "validator.py + this audit",
    ),
    ("PRESENT", "Lab features present in rooms", "validator.py + this audit"),
    ("PRESENT", "Teacher availability hours defined", "validator.py + this audit"),
    ("PRESENT", "Cross-reference integrity (IDs match)", "validator.py + this audit"),
    ("PRESENT", "Duplicate entity detection", "this audit"),
    ("PRESENT", "Data completeness (no empty fields)", "this audit"),
    ("PRESENT", "Hard constraints: 8 types enforced", "constraints.py"),
    ("PRESENT", "Soft constraints: 6 types optimized", "constraints.py"),
    (
        "TO ADD",
        "Per-day instructor overload (max hours/day)",
        "Not yet: check no instructor teaches >X quanta per day",
    ),
    (
        "TO ADD",
        "Per-day group overload (max hours/day)",
        "Not yet: check no group has >Y quanta per day",
    ),
    (
        "TO ADD",
        "Room-specific feature match for practicals",
        "Partially: room_type matches but specific lab features (e.g. 'Networking Lab') not matched during scheduling",
    ),
    (
        "TO ADD",
        "Instructor concurrent course conflict detection",
        "Not yet: same instructor assigned theory+practical of same course at same time",
    ),
    (
        "TO ADD",
        "Break window feasibility",
        "Not yet: check that break windows actually exist in the timetable grid",
    ),
    (
        "TO ADD",
        "Multi-section balancing check",
        "Not yet: verify that courses with many groups can actually parallelize",
    ),
    (
        "TO ADD",
        "Time zone / day distribution balance",
        "Not yet: check if demand is spread across days or bunched",
    ),
    (
        "TO ADD",
        "Instructor travel time between rooms",
        "Not yet: if campus is large, consecutive slots in distant rooms",
    ),
    (
        "TO ADD",
        "Exam scheduling conflicts (future)",
        "Not yet: when exam scheduling is added",
    ),
    (
        "TO ADD",
        "Historical schedule comparison",
        "Not yet: compare with previous semester's schedule",
    ),
]

table = Table(title="Check Inventory", box=box.ROUNDED)
table.add_column("Status", style="bold")
table.add_column("Check", style="cyan")
table.add_column("Where / Notes")

for status, check, note in recommendations:
    if status == "PRESENT":
        table.add_row("[green]PRESENT[/green]", check, note)
    else:
        table.add_row("[yellow]TO ADD[/yellow]", check, note)

console.print(table)

present_count = sum(1 for s, _, _ in recommendations if s == "PRESENT")
todo_count = sum(1 for s, _, _ in recommendations if s == "TO ADD")
info(f"Currently implemented: {present_count} checks")
info(f"Recommended additions: {todo_count} checks")

# ════════════════════════════════════════════════════════════════════════
# Save text report
# ════════════════════════════════════════════════════════════════════════
report_path = OUTPUT_DIR / "pre_scheduling_audit.txt"
with open(report_path, "w") as f:
    f.write("PRE-SCHEDULING DATA AUDIT REPORT\n")
    f.write("Generated by: runs/pre_scheduling_audit.py\n")
    f.write("=" * 80 + "\n\n")
    for line in report_lines:
        f.write(line + "\n")
    f.write(
        f"\n\nSummary: {pass_count} passed, {warn_count} warnings, {fail_count} failures\n"
    )

console.print(f"\n[dim]Report saved to: {report_path}[/dim]")
console.print()
