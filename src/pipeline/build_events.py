#!/usr/bin/env python3
"""Event builder with precomputed domains for pymoo migration.

Guarantees:
- Deterministic event ordering via stable event_key sort
- Integer-quanta overlap assertions
- Instructor/room availability data exported per event
"""

import hashlib
import logging
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # Bump when export format changes

# Canonical cache location for the events pickle
_CACHE_DIR = Path(".cache")
PKL_DEFAULT_PATH = str(_CACHE_DIR / "events_with_domains.pkl")


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


def _compute_data_hash(data_dir: str) -> str:
    """SHA-256 hash of all input JSON files for change detection."""
    h = hashlib.sha256()
    data_path = Path(data_dir)
    for name in sorted(
        ["Course.json", "Groups.json", "Instructors.json", "Rooms.json"]
    ):
        fp = data_path / name
        if fp.exists():
            h.update(fp.read_bytes())
    return h.hexdigest()


def build_events_with_domains(
    data_dir: str = "data",
    *,
    fix_tutorial_practicals: bool = True,
) -> dict:
    """Build events with precomputed allowed domains for encoding.

    Args:
        data_dir: Path to the data directory containing JSON files.
        fix_tutorial_practicals: If True, clear specific_lab_features for
            courses whose PracticalRoomFeatures is 'Lecture Hall' or
            'Seminar Room' (tutorial-style practicals that don't need
            a specific lab).  This fixes the 24-event structural
            infeasibility.

    Returns the export_data dict, or raises on failure.
    """
    from src.ga.core.population import generate_pure_random_population
    from src.io.data_store import DataStore
    from src.io.time_system import QuantumTimeSystem
    from src.utils.room_compatibility import is_room_suitable_for_course

    data_hash = _compute_data_hash(data_dir)
    logger.info("Data hash: %s...", data_hash[:16])

    logger.info("Loading data and generating reference individual...")
    store = DataStore.from_json(data_dir, run_preflight=False)
    ctx = store.to_context()
    qts = QuantumTimeSystem()

    # --- 0. Fix tutorial-style practicals (structural infeasibility) ---
    if fix_tutorial_practicals:
        n_fixed = 0
        for course in ctx.courses.values():
            lab_feats = getattr(course, "specific_lab_features", None)
            if lab_feats:
                # Normalize to list of lowercase strings
                feats_lower = [
                    (f if isinstance(f, str) else str(f)).lower().strip()
                    for f in lab_feats
                ]
                if any(f in ("lecture hall", "seminar room") for f in feats_lower):
                    course.specific_lab_features = []
                    n_fixed += 1
        if n_fixed:
            logger.info(
                "  Fixed %d courses with tutorial-practical room mismatch", n_fixed
            )

    pop = generate_pure_random_population(1, ctx, parallel=False)
    raw_genes = pop[0]
    logger.info("Raw genes from generator: %d", len(raw_genes))

    # --- 1. Sort genes by stable event_key ---
    indexed_genes = list(enumerate(raw_genes))
    indexed_genes.sort(key=lambda pair: _make_event_key(pair[1]))
    genes = [g for _, g in indexed_genes]
    logger.info("Sorted %d events by stable event_key", len(genes))

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
            logger.debug("  Processing event %d/%d", e, len(genes))

        # Overlap-model assertions: all durations/starts are integer quanta
        assert isinstance(
            gene.num_quanta, int
        ), f"Event {e}: num_quanta={gene.num_quanta!r} is not int"
        assert gene.num_quanta >= 1, f"Event {e}: num_quanta={gene.num_quanta} < 1"
        assert isinstance(
            gene.start_quanta, int
        ), f"Event {e}: start_quanta={gene.start_quanta!r} is not int"

        course_key = (gene.course_id, gene.course_type)
        ev_course = ctx.courses.get(course_key)  # may be None

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

        # Allowed rooms – type compatibility only
        # (Room capacity is non-binding in the target deployment;
        #  campus rooms have sufficient excess capacity for all groups.)
        required = (
            getattr(ev_course, "required_room_features", "lecture")
            if ev_course
            else "lecture"
        )
        required_str = (
            (required if isinstance(required, str) else str(required)).lower().strip()
        )
        course_lab_feats = (
            getattr(ev_course, "specific_lab_features", None) if ev_course else None
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
        if ev_course:
            qualified = getattr(ev_course, "qualified_instructor_ids", [])
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
    logger.info("Domain computation: %.2fs for %d events", elapsed, len(genes))

    # --- Domain-integrity assertions ---
    empty_room_events = [e for e, r in enumerate(allowed_rooms) if not r]
    empty_inst_events = [e for e, i in enumerate(allowed_instructors) if not i]
    if empty_room_events:
        logger.warning(
            "  %d events have EMPTY room domains: %s",
            len(empty_room_events),
            empty_room_events[:20],
        )
    if empty_inst_events:
        logger.warning(
            "  %d events have EMPTY instructor domains: %s",
            len(empty_inst_events),
            empty_inst_events[:20],
        )

    rlens = [len(r) for r in allowed_rooms]
    ilens = [len(i) for i in allowed_instructors]
    slens = [len(s) for s in allowed_starts]
    logger.info(
        "Rooms   min=%d max=%d avg=%.1f",
        min(rlens),
        max(rlens),
        sum(rlens) / len(rlens),
    )
    logger.info(
        "Instr   min=%d max=%d avg=%.1f",
        min(ilens),
        max(ilens),
        sum(ilens) / len(ilens),
    )
    logger.info(
        "Starts  min=%d max=%d avg=%.1f",
        min(slens),
        max(slens),
        sum(slens) / len(slens),
    )

    # --- 6. Paired practical events for cohort alignment ---
    paired_practical_events: list[tuple[int, int]] = []
    cohort_pairs = getattr(ctx, "cohort_pairs", None) or []
    if cohort_pairs:
        # Index: (course_id, group_id) -> list of event indices for practicals
        _prac_idx: dict[tuple[str, str], list[int]] = {}
        for e, ev in enumerate(events):
            if ev.get("course_type", "theory").lower() != "practical":
                continue
            for gid in ev["group_ids"]:
                _prac_idx.setdefault((ev["course_id"], gid), []).append(e)

        for left_id, right_id in cohort_pairs:
            # Find all course_ids that have practicals for the left cohort
            left_courses = {cid for (cid, gid) in _prac_idx if gid == left_id}
            for cid in left_courses:
                left_evts = _prac_idx.get((cid, left_id), [])
                right_evts = _prac_idx.get((cid, right_id), [])
                # Pair them positionally (same course, same sub-session order)
                for a, b in zip(left_evts, right_evts):
                    paired_practical_events.append((a, b))

        logger.info(
            "Cohort pairs: %d pairs -> %d paired practical event tuples",
            len(cohort_pairs),
            len(paired_practical_events),
        )

    # --- 7. Export ---
    export_data = {
        "schema_version": SCHEMA_VERSION,
        "data_hash": data_hash,
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
        "paired_practical_events": paired_practical_events,
        "cohort_pairs": [(l, r) for l, r in (cohort_pairs or [])],
        "fix_tutorial_practicals": fix_tutorial_practicals,
        "metadata": {
            "n_events": len(genes),
            "n_rooms": len(ctx.rooms),
            "n_instructors": len(ctx.instructors),
            "max_quanta": max_quantum,
            "empty_room_domains": len(empty_room_events),
            "empty_inst_domains": len(empty_inst_events),
        },
    }

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = PKL_DEFAULT_PATH
    logger.info("Saving %s ...", pkl_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  %s: %.1f KB", pkl_path, Path(pkl_path).stat().st_size / 1024)

    return export_data


def load_events(
    pkl_path: str = PKL_DEFAULT_PATH,
    data_dir: str = "data",
    *,
    verify: bool = True,
) -> dict:
    """Load events with integrity checks.

    Args:
        pkl_path: Path to the pickle file.
        data_dir: Path to the data directory (for hash verification).
        verify: If True (default), recompute event_keys from the stored
            events and assert they match the saved keys.  Also verify
            the data_hash matches the current JSON files.

    Raises:
        RuntimeError: If the pkl is missing required fields, event_keys
            don't match, or data_hash changed.
    """
    with open(pkl_path, "rb") as f:
        data: dict = pickle.load(f)

    # --- Required fields ---
    for field in ("event_keys", "instructor_available_quanta", "schema_version"):
        if field not in data:
            raise RuntimeError(
                f"events_with_domains.pkl missing '{field}'. "
                "Re-run: python build_events.py"
            )

    if data["schema_version"] < SCHEMA_VERSION:
        raise RuntimeError(
            f"PKL schema_version={data['schema_version']} < expected {SCHEMA_VERSION}. "
            "Re-run: python build_events.py"
        )

    if not verify:
        return data

    # --- Data hash check ---
    if "data_hash" in data:
        current_hash = _compute_data_hash(data_dir)
        if current_hash != data["data_hash"]:
            raise RuntimeError(
                "Input JSON files changed since events_with_domains.pkl was built!\n"
                f"  Saved hash: {data['data_hash'][:16]}...\n"
                f"  Current hash: {current_hash[:16]}...\n"
                "Re-run: python build_events.py"
            )

    # --- Event key integrity ---
    stored_keys = data["event_keys"]
    events = data["events"]
    if len(stored_keys) != len(events):
        raise RuntimeError(
            f"event_keys length {len(stored_keys)} != events length {len(events)}"
        )

    # Recompute keys from stored event dicts and verify
    for i, ev in enumerate(events):
        recomputed = (
            ev["course_id"],
            ev["course_type"],
            tuple(sorted(ev["group_ids"])),
            ev["num_quanta"],
        )
        if recomputed != stored_keys[i]:
            raise RuntimeError(
                f"Event key mismatch at index {i}:\n"
                f"  Stored:     {stored_keys[i]}\n"
                f"  Recomputed: {recomputed}\n"
                "The pkl file is corrupted. Re-run: python build_events.py"
            )

    # Verify ordering is sorted
    for i in range(len(stored_keys) - 1):
        if stored_keys[i] > stored_keys[i + 1]:
            raise RuntimeError(
                f"Event keys not sorted at index {i}->{i + 1}:\n"
                f"  {stored_keys[i]} > {stored_keys[i + 1]}\n"
                "Re-run: python build_events.py"
            )

    return data


if __name__ == "__main__":
    build_events_with_domains()
