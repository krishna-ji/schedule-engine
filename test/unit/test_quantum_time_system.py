import pytest

from src.encoder.quantum_time_system import QuantumTimeSystem


@pytest.fixture
def default_qts() -> QuantumTimeSystem:
    """Quantum time system with explicit operating hours for test determinism."""
    hours: dict[str, tuple[str, str] | None] = dict.fromkeys(
        QuantumTimeSystem.DAY_NAMES, ("10:00", "17:00")
    )
    hours["Saturday"] = None
    return QuantumTimeSystem(operating_hours=hours)


def test_decode_schedule_uses_day_end_when_period_reaches_boundary(
    default_qts: QuantumTimeSystem,
) -> None:
    quanta = {
        default_qts.time_to_quanta("Sunday", "15:00"),
        default_qts.time_to_quanta("Sunday", "16:00"),
    }

    schedule = default_qts.decode_schedule(quanta)

    assert schedule["Sunday"] == [{"start": "15:00", "end": "17:00"}]


def test_decode_schedule_preserves_regular_period_end(
    default_qts: QuantumTimeSystem,
) -> None:
    quanta = {default_qts.time_to_quanta("Monday", "11:00")}

    schedule = default_qts.decode_schedule(quanta)

    assert schedule["Monday"] == [{"start": "11:00", "end": "12:00"}]


def test_decode_schedule_handles_final_operational_day(
    default_qts: QuantumTimeSystem,
) -> None:
    quanta = {default_qts.time_to_quanta("Friday", "16:00")}

    schedule = default_qts.decode_schedule(quanta)

    assert schedule["Friday"] == [{"start": "16:00", "end": "17:00"}]
