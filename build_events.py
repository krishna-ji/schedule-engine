#!/usr/bin/env python3
"""Event builder with precomputed domains for pymoo migration.

Guarantees:
- Deterministic event ordering via stable event_key sort
- Integer-quanta overlap assertions
- Instructor/room availability data exported per event
"""

import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_event_key(gene) -> tuple:
    """Stable, deterministic sort key for an event (gene).

    Key: (course_id, course_type, tuple(sorted(group_ids)), num_quanta)
    This is independent of dict iteration order.
    """
    return (
        gene.course_id,
        gene.course_type,
        tuple(sorted(gene.group_ids)),
        gene.num_quanta,
    )


def build_events_with_domains(data_dir: str = "data") -> dict:
    """Build events with precomputed allowed domains for encoding.

    Returns the export_data dict, or raises on failure.
    """
    from src.ga.core.population import generate_pure_random_population
    from src.io.data_store import DataStore
    from src.io.time_system import QuantumTimeSystem
    from src.utils.room_compatibility import is_room_suitable_for_course

    print("Loading data and generating reference individual...")
    store = DataStore.from_json(data_dir)
    ctx = store.to_context()
    qts = QuantumTimeSystem()

    pop = generate_pure_random_population(1, ctx, parallel=False)
    raw_genes = pop[0]
    print(f"Raw genes from generator: {len(raw_genes)}")

    # --- 1. Sort genes by stable event_key ---
    indexed_genes = list(enumerate(raw_genes))
    indexed_genes.sort(key=lambda pair: _make_event_key(pair[1]))
    genes = [g for _, g in indexed_genes]
    print(f"Sorted {len(genes)} events by stable event_key")

    # --- 2. Build mapping tables (sorted for determinism) ---
    room_ids_sorted = sorted(ctx.rooms.keys())
    instructor_ids_sorted = sorted(ctx.instructors.keys())

    room_to_idx = {rid: i for i, rid in enumerate(room_ids_sorted)}
    idx_to_room = {i: rid for rid, i in room_to_idx.items()}
    instructor_to_idx = {iid: i for i, iid in enumerate(instructor_ids_sorted)}
    idx_to_instructor = {i: iid for iid, i in instructor_to_idx.items()}

    # --- 3. Instructor available quanta (as sets keyed by idx) ---
    instructor_available_quanta: dict[int, set | None] = {}
    for iid, inst in ctx.instructors.items():
        idx = instructor_to_idx[iid]
        if inst.is_full_time:
            instructor_available_quanta[idx] = None  # None means always available
        else:
            instructor_available_quanta[idx] = set(inst.available_quanta)

    # --- 4. Room available quanta (as sets keyed by idx) ---
    room_available_quanta: dict[int, set | None] = {}
    for rid, room in ctx.rooms.items():
        idx = room_to_idx[rid]
        if room.available_quanta:
            room_available_quanta[idx] = set(room.available_quanta)
        else:
            room_available_quanta[idx] = None  # None means always available

    max_quantum = qts.total_quanta

    # --- 5. Build events with domains ---
    events = []
    event_keys = []
    allowed_rooms = []
    allowed_instructors = []
    allowed_starts = []

    t0 = time.time()
    for e, gene in enumerate(genes):
        if e % 100 == 0:
            print(f"  Processing event {e}/{len(genes)}")

        # Overlap-model assertions: all durations/starts are integer quanta
        assert isinstance(
            gene.num_quanta, int
        ), f"Event {e}: num_quanta={gene.num_quanta!r} is not int"
        assert gene.num_quanta >= 1, f"Event {e}: num_quanta={gene.num_quanta} < 1"
        assert isinstance(
            gene.start_quanta, int
        ), f"Event {e}: start_quanta={gene.start_quanta!r} is not int"

        course_key = (gene.course_id, gene.course_type)
        course = ctx.courses.get(course_key)

        ekey = _make_event_key(gene)
        event_keys.append(ekey)

        event = {
            "idx": e,
            "course_id": gene.course_id,
            "course_type": gene.course_type,
            "group_ids": sorted(gene.group_ids),
            "num_quanta": gene.num_quanta,
        }
        events.append(event)

        # Allowed rooms – EXACTLY the same logic as RoomSuitability constraint
        # (type compatibility only, NO capacity check, NO fallback to all rooms)
        required = (
            getattr(course, "required_room_features", "lecture")
            if course
            else "lecture"
        )
        required_str = (
            (required if isinstance(required, str) else str(required)).lower().strip()
        )
        course_lab_feats = (
            getattr(course, "specific_lab_features", None) if course else None
        )

        room_indices = []
        for rid in room_ids_sorted:
            room = ctx.rooms[rid]
            room_type = getattr(room, "room_features", "lecture")
            room_str = (
                (room_type if isinstance(room_type, str) else str(room_type))
                .lower()
                .strip()
            )
            room_spec_feats = getattr(room, "specific_features", None)
            if is_room_suitable_for_course(
                required_str, room_str, course_lab_feats, room_spec_feats
            ):
                room_indices.append(room_to_idx[rid])
        allowed_rooms.append(room_indices)

        # Allowed instructors (qualification)
        qualified: list[str] = []
        if course:
            qualified = getattr(course, "qualified_instructor_ids", [])
        inst_indices = sorted(
            instructor_to_idx[iid] for iid in qualified if iid in instructor_to_idx
        )
        allowed_instructors.append(inst_indices)

        # Allowed start times (start + duration <= max_quantum)
        duration = gene.num_quanta
        max_start = max_quantum - duration
        assert (
            max_start >= 0
        ), f"Event {e}: duration {duration} > max_quantum {max_quantum}"
        start_indices = list(range(max_start + 1))
        allowed_starts.append(start_indices)

    elapsed = time.time() - t0
    print(f"Domain computation: {elapsed:.2f}s for {len(genes)} events")

    rlens = [len(r) for r in allowed_rooms]
    ilens = [len(i) for i in allowed_instructors]
    slens = [len(s) for s in allowed_starts]
    print(f"Rooms   min={min(rlens)} max={max(rlens)} avg={sum(rlens)/len(rlens):.1f}")
    print(f"Instr   min={min(ilens)} max={max(ilens)} avg={sum(ilens)/len(ilens):.1f}")
    print(f"Starts  min={min(slens)} max={max(slens)} avg={sum(slens)/len(slens):.1f}")

    # --- 6. Export ---
    export_data = {
        "events": events,
        "event_keys": event_keys,
        "allowed_rooms": allowed_rooms,
        "allowed_instructors": allowed_instructors,
        "allowed_starts": allowed_starts,
        "room_to_idx": room_to_idx,
        "idx_to_room": idx_to_room,
        "instructor_to_idx": instructor_to_idx,
        "idx_to_instructor": idx_to_instructor,
        "instructor_available_quanta": instructor_available_quanta,
        "room_available_quanta": room_available_quanta,
        "metadata": {
            "n_events": len(genes),
            "n_rooms": len(ctx.rooms),
            "n_instructors": len(ctx.instructors),
            "max_quanta": max_quantum,
        },
    }

    pkl_path = "events_with_domains.pkl"
    print(f"Saving {pkl_path} ...")
    with open(pkl_path, "wb") as f:
        pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {pkl_path}: {Path(pkl_path).stat().st_size / 1024:.1f} KB")

    return export_data


def load_events(pkl_path: str = "events_with_domains.pkl") -> dict:
    """Load events and loudly fail if event_keys absent."""
    with open(pkl_path, "rb") as f:
        data: dict = pickle.load(f)
    if "event_keys" not in data:
        raise RuntimeError(
            "events_with_domains.pkl has no event_keys. "
            "Re-run: python build_events.py"
        )
    if "instructor_available_quanta" not in data:
        raise RuntimeError(
            "events_with_domains.pkl missing instructor availability. "
            "Re-run: python build_events.py"
        )
    return data


if __name__ == "__main__":
    build_events_with_domains()
