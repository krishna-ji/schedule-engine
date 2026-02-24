"""Pre-feasibility report generator.

Analyses the mathematical topology of the scheduling input data *before*
the evolutionary optimisation begins.  The report identifies absolute
structural bottlenecks — resource deficits, forced lunch collisions,
and cohort cascade risks — that no amount of GA tuning can overcome.

Public API
----------
    generate_pre_feasibility_report(pkl_data, output_dir)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Time-system constants (must match bitset_time / instance_config)
_QUANTA_PER_DAY = 7
_N_DAYS = 6
_T = _QUANTA_PER_DAY * _N_DAYS  # 42
_LUNCH_QUANTA = {2, 3}  # within-day quanta forming the lunch window


def generate_pre_feasibility_report(
    pkl_data: dict,
    output_dir: Path | str,
) -> Path:
    """Generate ``pre_feasibility_report.md`` inside *output_dir*.

    Parameters
    ----------
    pkl_data : dict
        Loaded ``events_with_domains.pkl`` dictionary.
    output_dir : Path or str
        Directory where the report file is written.

    Returns
    -------
    Path
        Absolute path to the generated report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pre_feasibility_report.md"

    events = pkl_data["events"]
    allowed_rooms = pkl_data["allowed_rooms"]
    allowed_instructors = pkl_data["allowed_instructors"]
    inst_avail = pkl_data.get("instructor_available_quanta", {})
    room_avail = pkl_data.get("room_available_quanta", {})
    idx_to_room = pkl_data.get("idx_to_room", {})
    idx_to_inst = pkl_data.get("idx_to_instructor", {})
    cohort_pairs = pkl_data.get("cohort_pairs", [])

    sections: list[str] = []
    sections.append("# Pre-Feasibility Analysis Report\n")
    sections.append(
        "*Auto-generated before GA execution. "
        "Identifies structural bottlenecks in the input data.*\n"
    )
    sections.append(f"- **Events**: {len(events)}")
    sections.append(f"- **Rooms**: {len(idx_to_room)}")
    sections.append(f"- **Instructors**: {len(idx_to_inst)}")
    sections.append(f"- **Cohort Pairs**: {len(cohort_pairs)}")
    sections.append(
        f"- **Time Quanta**: {_T} ({_N_DAYS} days × {_QUANTA_PER_DAY}/day)\n"
    )

    # ==================================================================
    # Section 1: Spatial Supply vs. Demand
    # ==================================================================
    sections.append("## 1. Spatial Supply vs. Demand (Subgroup-Aware)\n")

    # Group events by their allowed-room set (room "feature class")
    room_class_demand: dict[frozenset[int], int] = defaultdict(int)  # quanta demanded
    room_class_events: dict[frozenset[int], int] = defaultdict(int)  # event count
    room_class_label: dict[frozenset[int], list[str]] = defaultdict(list)

    for e, ev in enumerate(events):
        ar = allowed_rooms[e]
        if not ar:
            continue
        key = frozenset(ar)
        dur = ev["num_quanta"]
        n_groups = len(ev["group_ids"])
        room_class_demand[key] += dur  # each event needs `dur` quanta of room time
        room_class_events[key] += 1
        if len(room_class_label[key]) < 3:
            room_class_label[key].append(ev["course_id"])

    # Compute supply per room class: each room in the set provides _T quanta
    # (adjusted by room availability if restricted)
    deficits: list[tuple[str, int, int, int, int]] = []

    sections.append(
        "| Room Class (sample courses) | Rooms | Events | Demand (quanta) "
        "| Supply (quanta) | Surplus | Status |"
    )
    sections.append("|---|---:|---:|---:|---:|---:|---|")

    for key in sorted(room_class_demand, key=lambda k: -room_class_demand[k]):
        n_rooms = len(key)
        demand = room_class_demand[key]
        n_ev = room_class_events[key]

        # Supply = sum of available quanta for each room in class
        supply = 0
        for r_idx in key:
            avail = room_avail.get(r_idx)
            if avail is None:
                supply += _T  # fully available
            else:
                supply += len(avail)

        surplus = supply - demand
        status = "OK" if surplus >= 0 else "**DEFICIT**"
        label = ", ".join(sorted(set(room_class_label[key])))
        sections.append(
            f"| {label} | {n_rooms} | {n_ev} | {demand} | {supply} | {surplus} | {status} |"
        )
        if surplus < 0:
            deficits.append((label, n_rooms, n_ev, demand, supply))

    if deficits:
        sections.append(
            f"\n> **Warning**: {len(deficits)} room class(es) have structural "
            f"deficits where demand exceeds available supply."
        )
    else:
        sections.append("\n> All room classes have sufficient supply to cover demand.")

    # Utilisation summary
    total_demand = sum(room_class_demand.values())
    unique_rooms = set()
    for key in room_class_demand:
        unique_rooms.update(key)
    total_supply = 0
    for r_idx in unique_rooms:
        avail = room_avail.get(r_idx)
        total_supply += len(avail) if avail is not None else _T
    util_pct = (total_demand / total_supply * 100) if total_supply > 0 else 0
    sections.append(
        f"\n**Aggregate room utilisation**: {total_demand}/{total_supply} "
        f"quanta ({util_pct:.1f}%)\n"
    )

    # ==================================================================
    # Section 2: Faculty Meridian Collisions
    # ==================================================================
    sections.append("## 2. Faculty Meridian Collisions\n")
    sections.append(
        "Instructors whose *entire* availability falls within the lunch "
        "window (within-day quanta 2–3) are mathematically forced to incur "
        "``sc_lunch`` penalties on every assigned event.\n"
    )

    # Absolute lunch quanta across all days
    lunch_abs = set()
    for day in range(_N_DAYS):
        for q in _LUNCH_QUANTA:
            lunch_abs.add(day * _QUANTA_PER_DAY + q)

    forced_lunch: list[tuple[str, int, int]] = []
    lunch_heavy: list[tuple[str, int, int, float]] = []

    # Build instructor→event count
    inst_event_count: dict[int, int] = defaultdict(int)
    for ai in allowed_instructors:
        for i in ai:
            inst_event_count[i] += 1

    for i_idx, slots in inst_avail.items():
        i_idx = int(i_idx)
        if slots is None:
            continue
        slot_set = set(slots)
        non_lunch = slot_set - lunch_abs
        lunch_overlap = slot_set & lunch_abs
        name = idx_to_inst.get(i_idx, f"Inst#{i_idx}")
        n_events = inst_event_count.get(i_idx, 0)

        if len(non_lunch) == 0 and len(lunch_overlap) > 0:
            forced_lunch.append((name, len(slot_set), n_events))
        elif len(slot_set) > 0:
            ratio = len(lunch_overlap) / len(slot_set)
            if ratio > 0.5:
                lunch_heavy.append((name, len(slot_set), n_events, ratio))

    if forced_lunch:
        sections.append(
            f"### Forced Lunch Collisions ({len(forced_lunch)} instructors)\n"
        )
        sections.append("| Instructor | Avail Quanta | Events | Issue |")
        sections.append("|---|---:|---:|---|")
        for name, n_avail, n_ev in forced_lunch:
            sections.append(
                f"| {name} | {n_avail} | {n_ev} | "
                f"All availability is within lunch window |"
            )
    else:
        sections.append("> No instructors are *entirely* confined to the lunch window.")

    if lunch_heavy:
        sections.append(f"\n### High Lunch Exposure ({len(lunch_heavy)} instructors)\n")
        sections.append("| Instructor | Avail Quanta | Events | Lunch Overlap % |")
        sections.append("|---|---:|---:|---:|")
        for name, n_avail, n_ev, ratio in sorted(lunch_heavy, key=lambda x: -x[3]):
            sections.append(f"| {name} | {n_avail} | {n_ev} | {ratio * 100:.0f}% |")

    sections.append("")

    # ==================================================================
    # Section 3: Subcohort Topology
    # ==================================================================
    sections.append("## 3. Subcohort Topology\n")

    if not cohort_pairs:
        sections.append("> No cohort pairs defined in the data.\n")
    else:
        sections.append(
            f"**{len(cohort_pairs)} cohort pairs** define subgroup pairings "
            f"for practical alignment.\n"
        )

        # Build group→practical events
        group_to_idx: dict[str, int] = {}
        for ev in events:
            for gid in ev["group_ids"]:
                if gid not in group_to_idx:
                    group_to_idx[gid] = len(group_to_idx)

        group_prac_load: dict[str, int] = defaultdict(int)  # total prac quanta
        group_prac_events: dict[str, int] = defaultdict(int)
        group_total_load: dict[str, int] = defaultdict(int)

        for ev in events:
            dur = ev["num_quanta"]
            is_prac = ev.get("course_type", "theory").lower() == "practical"
            for gid in ev["group_ids"]:
                group_total_load[gid] += dur
                if is_prac:
                    group_prac_load[gid] += dur
                    group_prac_events[gid] += dur

        sections.append(
            "| Left | Right | Prac Quanta (L) | Prac Quanta (R) "
            "| Total Load (L) | Total Load (R) | Cascade Risk |"
        )
        sections.append("|---|---|---:|---:|---:|---:|---|")

        cascade_risks = 0
        for left_id, right_id in cohort_pairs:
            l_prac = group_prac_load.get(left_id, 0)
            r_prac = group_prac_load.get(right_id, 0)
            l_total = group_total_load.get(left_id, 0)
            r_total = group_total_load.get(right_id, 0)

            # For practicals to align, both subgroups need the same
            # timeslots free.  If combined load > T, cascade is certain.
            combined = l_total + r_total
            # Max concurrent capacity: each pair needs aligned practicals
            # meaning their practical quanta consume the SAME slots.
            # Net unique load = l_total + r_total - min(l_prac, r_prac)
            # (the aligned practicals share slots)
            aligned_prac = min(l_prac, r_prac)
            net_load = l_total + r_total - aligned_prac

            risk = (
                "HIGH"
                if net_load > _T
                else ("MEDIUM" if net_load > _T * 0.85 else "LOW")
            )
            if risk == "HIGH":
                cascade_risks += 1

            sections.append(
                f"| {left_id} | {right_id} | {l_prac} | {r_prac} "
                f"| {l_total} | {r_total} | {risk} |"
            )

        if cascade_risks > 0:
            sections.append(
                f"\n> **Warning**: {cascade_risks} pair(s) have combined load "
                f"exceeding T={_T}, forcing cascading slot conflicts."
            )
        else:
            sections.append(
                "\n> No cohort pairs have combined loads that mathematically "
                "force cascade conflicts."
            )

    # ==================================================================
    # Write report
    # ==================================================================
    report_text = "\n".join(sections) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Pre-feasibility report written to %s", report_path)
    return report_path
