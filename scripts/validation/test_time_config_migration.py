#!/usr/bin/env python3
"""
Test: TimeConfig merged into QuantumTimeSystem, dead sub-configs deleted.

Verifies:
1. QTS stores all time/break/penalty params
2. Consumers read from QTS (not cfg.time)
3. Config no longer has time/gpu/metrics/export/calendar/feasibility/performance
4. Config has cohort_pairs at top level
5. Config round-trip (from_dict) still works
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schedule_engine.config.models import Config
from schedule_engine.io.time_system import QuantumTimeSystem

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {label}")
    else:
        failed += 1
        print(f"  ✗ {label} {detail}")


print("TEST 1: QTS default params")
qts = QuantumTimeSystem()
check("midday_break_start", qts.midday_break_start == "12:00")
check("midday_break_end", qts.midday_break_end == "13:00")
check("break_window_start", qts.break_window_start == "12:00")
check("break_window_end", qts.break_window_end == "14:00")
check("enforce_break_placement", qts.enforce_break_placement is True)
check("break_min_quanta", qts.break_min_quanta == 1)
check("theory_isolated_penalty", qts.theory_isolated_penalty == 1)
check("theory_max_excused_isolated", qts.theory_max_excused_isolated == 1)
check("preferred_block_size_max", qts.preferred_block_size_max == 3)
check("max_session_coalescence", qts.max_session_coalescence == 3)
check("max_sessions_per_day", qts.max_sessions_per_day == 4)
check("earliest_preferred_time", qts.earliest_preferred_time == "07:00")
check("latest_preferred_time", qts.latest_preferred_time == "21:00")
print()

print("TEST 2: QTS custom params")
qts2 = QuantumTimeSystem(
    midday_break_start="11:30",
    break_min_quanta=2,
    preferred_block_size_max=4,
    enforce_break_placement=False,
)
check("custom break_start", qts2.midday_break_start == "11:30")
check("custom break_min_quanta", qts2.break_min_quanta == 2)
check("custom block_size_max", qts2.preferred_block_size_max == 4)
check("custom enforce=False", qts2.enforce_break_placement is False)
print()

print("TEST 3: Config has NO deleted sub-configs")
cfg = Config()
for attr in (
    "time",
    "gpu",
    "metrics",
    "export",
    "calendar",
    "feasibility",
    "performance",
):
    check(f"no config.{attr}", not hasattr(cfg, attr))
print()

print("TEST 4: Config has cohort_pairs at top level")
check("cohort_pairs exists", hasattr(cfg, "cohort_pairs"))
check("cohort_pairs default empty", cfg.cohort_pairs == [])
cfg.cohort_pairs = [("bei1a", "bei1b")]
check("cohort_pairs mutable", cfg.cohort_pairs == [("bei1a", "bei1b")])
print()

print("TEST 5: Config still has core fields")
for attr in (
    "ga",
    "repair",
    "lns",
    "enhancements",
    "heuristics",
    "rl",
    "hard_constraints",
    "soft_constraints",
    "parallel",
    "io",
):
    check(f"config.{attr} exists", hasattr(cfg, attr))
print()

print("TEST 6: Config round-trip via from_dict")
cfg3 = Config(name="test-roundtrip")
cfg3.cohort_pairs = [("a", "b")]
d = dataclasses.asdict(cfg3)
cfg3_rt = Config.from_dict(d)
check("round-trip name", cfg3_rt.name == "test-roundtrip")
check("round-trip cohort_pairs", cfg3_rt.cohort_pairs == [("a", "b")])
check("round-trip ga.ngen", cfg3_rt.ga.ngen == cfg3.ga.ngen)
print()

print("TEST 7: Config.summary() works")
summary = cfg.summary()
check("summary contains name", "default" in summary)
check("summary no GPU line", "GPU" not in summary)
print()

print("TEST 8: time_helpers reads from QTS")
from schedule_engine.utils.time_helpers import get_midday_break_quanta

qts_test = QuantumTimeSystem(midday_break_start="11:00", midday_break_end="12:00")
break_q = get_midday_break_quanta(qts_test)
# Should have break quanta for operational days
check("break_quanta non-empty", len(break_q) > 0)
# Verify it uses QTS params (11:00-12:00 = 1 quantum for 60-min quanta)
for day, quanta in break_q.items():
    check(f"break_{day}_size=1", len(quanta) == 1, f"got {len(quanta)}")
    break  # Just check first day
print()

print(f"\n{'='*40}")
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
