"""CI gate: evaluator equivalence and encoding roundtrip.

These tests MUST pass before any pymoo migration code is merged.
Run: pytest tests/test_migration_gates.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PKL_PATH = PROJECT_ROOT / ".cache" / "events_with_domains.pkl"
SKIP_REASON = ".cache/events_with_domains.pkl not found — run `python build_events.py`"


@pytest.fixture(scope="module")
def pkl_data():
    if not PKL_PATH.exists():
        pytest.skip(SKIP_REASON)
    from src.pipeline.build_events import load_events

    return load_events(str(PKL_PATH), verify=True)


@pytest.fixture(scope="module")
def ctx_qts(pkl_data):
    try:
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem

        store = DataStore.from_json("data")
        ctx = store.to_context()

        # Apply the same tutorial-practical fix used when the pkl was built
        if pkl_data.get("fix_tutorial_practicals", False):
            for course in ctx.courses.values():
                lab_feats = getattr(course, "specific_lab_features", None)
                if lab_feats:
                    feats_lower = [
                        (f if isinstance(f, str) else str(f)).lower().strip()
                        for f in lab_feats
                    ]
                    if any(f in ("lecture hall", "seminar room") for f in feats_lower):
                        course.specific_lab_features = []

        return ctx, QuantumTimeSystem()
    except Exception as e:
        pytest.skip(f"Cannot load data: {e}")


# =====================================================================
# Gate A: Encoding roundtrip
# =====================================================================


class TestEncodingRoundtrip:
    def test_roundtrip(self):
        from src.pipeline.encoding import GeneAssignment, decode, encode

        genes = [GeneAssignment(i, i + 1, i * 2) for i in range(50)]
        x = encode(genes)
        assert x.shape == (150,)
        assert decode(x) == genes

    def test_chromosome_views(self):
        from src.pipeline.encoding import GeneAssignment, chromosome_views, encode

        genes = [GeneAssignment(10, 20, 30), GeneAssignment(40, 50, 60)]
        x = encode(genes)
        inst, room, time = chromosome_views(x)
        assert list(inst) == [10, 40]
        assert list(room) == [20, 50]
        assert list(time) == [30, 60]
        # Views modify x
        inst[0] = 99
        assert x[0] == 99

    def test_encoding_spec(self, pkl_data):
        from src.pipeline.encoding import EncodingSpec

        spec = EncodingSpec.from_pkl_data(pkl_data)
        assert spec.n_vars == 3 * spec.n_events
        xl = spec.xl()
        xu = spec.xu()
        assert xl.shape == (spec.n_vars,)
        assert xu.shape == (spec.n_vars,)
        assert (xl >= 0).all()
        assert (xu >= xl).all()


# =====================================================================
# Gate B: Evaluator equivalence (fast vs original)
# =====================================================================


class TestEvaluatorEquivalence:
    """fast_evaluate_hard must produce IDENTICAL results to the original Evaluator."""

    N = 10  # Reduced for CI speed; validate_migration.py tests 50

    def test_equivalence_on_random_individuals(self, pkl_data, ctx_qts):
        from src.constraints.evaluator import Evaluator
        from src.domain.timetable import Timetable
        from src.ga.core.population import generate_pure_random_population
        from src.pipeline.build_events import _make_event_key
        from src.pipeline.fast_evaluator import fast_evaluate_hard

        ctx, qts = ctx_qts
        events = pkl_data["events"]
        allowed_instructors = pkl_data["allowed_instructors"]
        allowed_rooms = pkl_data["allowed_rooms"]
        inst_avail = pkl_data["instructor_available_quanta"]
        room_avail = pkl_data["room_available_quanta"]
        instructor_to_idx = pkl_data["instructor_to_idx"]
        room_to_idx = pkl_data["room_to_idx"]

        evaluator = Evaluator()
        constraint_names = [
            "student_group_exclusivity",
            "instructor_exclusivity",
            "room_exclusivity",
            "instructor_qualifications",
            "room_suitability",
            "instructor_time_availability",
            "room_time_availability",
            "course_completeness",
        ]

        pop = generate_pure_random_population(self.N, ctx, parallel=False)
        mismatches = []

        for idx, genes in enumerate(pop):
            # Original evaluator
            tt = Timetable(genes=genes, context=ctx, qts=qts)
            orig = {c.name: int(c.weight * c.evaluate(tt)) for c in evaluator.hard}

            # Fast evaluator
            sorted_genes = sorted(genes, key=_make_event_key)
            n = len(sorted_genes)
            inst = np.zeros(n, dtype=int)
            room = np.zeros(n, dtype=int)
            time_ = np.zeros(n, dtype=int)
            for i, g in enumerate(sorted_genes):
                inst[i] = instructor_to_idx[g.instructor_id]
                room[i] = room_to_idx[g.room_id]
                time_[i] = g.start_quanta

            fast = fast_evaluate_hard(
                events,
                inst,
                room,
                time_,
                allowed_instructors,
                allowed_rooms,
                inst_avail,
                room_avail,
            )

            mismatches.extend(
                f"Ind#{idx} {cn}: orig={orig.get(cn, 0)} fast={fast.get(cn, 0)}"
                for cn in constraint_names
                if orig.get(cn, 0) != fast.get(cn, 0)
            )

        assert mismatches == [], "Equivalence mismatches:\n" + "\n".join(mismatches)


# =====================================================================
# Gate C: Event key integrity
# =====================================================================


class TestEventKeyIntegrity:
    def test_keys_are_sorted(self, pkl_data):
        keys = pkl_data["event_keys"]
        for i in range(len(keys) - 1):
            assert (
                keys[i] <= keys[i + 1]
            ), f"Keys not sorted at {i}: {keys[i]} > {keys[i + 1]}"

    def test_keys_match_events(self, pkl_data):
        events = pkl_data["events"]
        keys = pkl_data["event_keys"]
        assert len(events) == len(keys)
        for i, ev in enumerate(events):
            recomputed = (
                ev["course_id"],
                ev["course_type"],
                tuple(sorted(ev["group_ids"])),
                ev["num_quanta"],
            )
            assert recomputed == keys[i], f"Key mismatch at {i}"

    def test_schema_version(self, pkl_data):
        assert pkl_data["schema_version"] >= 2

    def test_data_hash_present(self, pkl_data):
        assert "data_hash" in pkl_data
        assert len(pkl_data["data_hash"]) == 64  # SHA-256 hex

    def test_no_zero_room_domains(self, pkl_data):
        """After tutorial-practical fix, no event should have 0 suitable rooms."""
        for i, ar in enumerate(pkl_data["allowed_rooms"]):
            assert (
                len(ar) > 0
            ), f"Event {i} has 0 suitable rooms: {pkl_data['events'][i]}"

    def test_no_zero_instructor_domains(self, pkl_data):
        for i, ai in enumerate(pkl_data["allowed_instructors"]):
            assert (
                len(ai) > 0
            ), f"Event {i} has 0 qualified instructors: {pkl_data['events'][i]}"
