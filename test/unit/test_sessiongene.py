from src.ga.sessiongene import SessionGene


def test_practical_session_keeps_full_duration_within_day():
    """Practical blocks must stay continuous when they fit in the same day."""
    gene = SessionGene(
        course_id="ENSH 153",
        course_type="practical",
        instructor_id="INS-1",
        group_ids=["BAM2B"],
        room_id="LAB-1",
        start_quanta=38,
        num_quanta=3,
    )

    assert gene.num_quanta == 3


def test_long_session_allowed_to_span_multiple_days():
    """Sessions longer than a day should remain multi-day and untrimmed."""
    gene = SessionGene(
        course_id="AR701",
        course_type="practical",
        instructor_id="INS-2",
        group_ids=["ARCH"],
        room_id="STUDIO",
        start_quanta=0,
        num_quanta=9,
    )

    assert gene.num_quanta == 9


def test_short_session_that_overflows_day_is_shifted():
    """Short sessions should shift earlier inside the same day to keep their duration."""
    gene = SessionGene(
        course_id="ENSH 151",
        course_type="theory",
        instructor_id="INS-3",
        group_ids=["BAM2A"],
        room_id="ROOM-1",
        start_quanta=5,
        num_quanta=4,
    )

    assert gene.start_quanta == 3
    assert gene.num_quanta == 4
