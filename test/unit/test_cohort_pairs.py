from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.encoder.input_encoder import derive_cohort_pairs_from_groups
from src.workflows.standard_run import _merge_cohort_pairs


def _write_groups_file(tmp_path: Path, payload: list[dict[str, Any]]) -> str:
    """Write a temporary Groups.json payload to disk for test derivations."""

    groups_path = tmp_path / "Groups.json"
    groups_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(groups_path)


def test_derive_cohort_pairs_from_groups_handles_mixed_subgroup_formats(
    tmp_path: Path,
) -> None:
    groups_payload: list[dict[str, Any]] = [
        {
            "group_id": "BAM2AB",
            "subgroups": [
                {"id": "BAM2A", "student_count": 24},
                {"id": " BAM2B ", "student_count": 24},
                "BAM2C",
                {"id": "BAM2B"},  # duplicate that should be ignored
            ],
        },
        {
            "group_id": "BCT5AB",
            "subgroups": ["BCT5A", "BCT5B"],
        },
    ]

    groups_file = _write_groups_file(tmp_path, groups_payload)
    derived_pairs = derive_cohort_pairs_from_groups(groups_file)

    assert derived_pairs == [
        ("BAM2A", "BAM2B"),
        ("BAM2A", "BAM2C"),
        ("BCT5A", "BCT5B"),
    ]


def test_derive_cohort_pairs_from_groups_deduplicates_case_insensitive_pairs(
    tmp_path: Path,
) -> None:
    groups_payload: list[dict[str, Any]] = [
        {
            "group_id": "BME2AB",
            "subgroups": ["BME2A", "BME2B"],
        },
        {
            "group_id": "duplicate-case",
            "subgroups": ["bme2b", "bme2a", "BME2C"],
        },
    ]

    groups_file = _write_groups_file(tmp_path, groups_payload)
    derived_pairs = derive_cohort_pairs_from_groups(groups_file)

    assert derived_pairs == [
        ("BME2A", "BME2B"),
        ("bme2b", "BME2C"),
    ]


def test_merge_cohort_pairs_adds_clean_manual_overrides() -> None:
    derived_pairs = [("BAM2A", "BAM2B"), ("BCT5A", "BCT5B")]
    configured_pairs = [("bam2b", "bam2c"), ("  ", ""), ("BCT5A", "bct5b")]

    merged = _merge_cohort_pairs(derived_pairs, configured_pairs)

    assert merged == [
        ("BAM2A", "BAM2B"),
        ("BCT5A", "BCT5B"),
        ("bam2b", "bam2c"),
    ]
