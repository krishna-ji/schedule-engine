"""
Simplified continuity validation for new SessionGene structure.

With start_quanta + num_quanta, continuity is structurally guaranteed.
Validation now just checks:
1. Valid ranges (num_quanta > 0, within bounds)
2. Day boundary compliance (no midnight wrap)
"""

from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.sessiongene import SessionGene


def validate_continuity(individual: list[SessionGene]) -> tuple[bool, list[str]]:
    """
    Verify all SessionGenes have valid contiguous quanta.

    With new structure, this is TRIVIAL - just validate ranges!

    Returns:
        (is_valid, violation_messages)
    """
    qts = QuantumTimeSystem()
    violations = []

    for gene in individual:
        # Check 1: Positive duration
        if gene.num_quanta <= 0:
            violations.append(f"{gene.course_id}: Invalid num_quanta={gene.num_quanta}")

        # Check 2: Within quantum bounds
        if gene.start_quanta < 0:
            violations.append(
                f"{gene.course_id}: Negative start_quanta={gene.start_quanta}"
            )
        if gene.end_quanta > qts.total_quanta:
            violations.append(
                f"{gene.course_id}: Overflow "
                f"end_quanta={gene.end_quanta} > total={qts.total_quanta}"
            )

        # Check 3: Same day (no midnight wrap)
        start_day = gene.start_quanta // qts.quanta_per_day
        end_day = (gene.end_quanta - 1) // qts.quanta_per_day
        if start_day != end_day:
            violations.append(f"{gene.course_id}: Spans days {start_day} → {end_day}")

    return (len(violations) == 0, violations)
