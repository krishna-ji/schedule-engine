"""
Group Hierarchy Analyzer

Identifies parent groups and their subgroups to enable proper scheduling:
- Theory sessions for parent (whole class together)
- Practical sessions for subgroups separately
"""

import json
from typing import Any

from schedule_engine.domain.group import Group


def analyze_group_hierarchy_from_json(
    json_path: str,
) -> dict[str, list[str] | dict[str, list[str]] | dict[str, str]]:
    """
    Analyzes group hierarchy by reading explicit subgroups from Groups.json.

    The JSON structure has parent groups with a "subgroups" array, e.g.:
        {"group_id": "BME1AB", "subgroups": [{"id": "BME1A"}, {"id": "BME1B"}]}

    Args:
        json_path: Path to Groups.json file

    Returns:
        Dictionary with:
        - "parents": List of parent group IDs
        - "subgroups": Dict mapping parent_id -> [subgroup_ids]
        - "parent_map": Dict mapping subgroup_id -> parent_id
        - "standalone": List of groups with no subgroups

    Example:
        {
            "parents": ["BME1AB", "BME2AB"],
            "subgroups": {"BME1AB": ["BME1A", "BME1B"]},
            "parent_map": {"BME1A": "BME1AB", "BME1B": "BME1AB"},
            "standalone": ["BCE8"]  # Groups with no subgroups
        }
    """
    with open(json_path) as f:
        raw_data = json.load(f)

    parents = set()
    subgroups_dict: dict[str, list[str]] = {}
    parent_map: dict[str, str] = {}
    all_group_ids: set[str] = set()

    # First pass: collect all group IDs and identify parent-subgroup relationships
    for item in raw_data:
        group_id = item.get("group_id", "")
        if group_id:
            all_group_ids.add(group_id)

        subgroups_raw = item.get("subgroups")
        if subgroups_raw and len(subgroups_raw) >= 1:
            # This group is a parent!
            parent_id = group_id
            parents.add(parent_id)

            subgroup_ids = _extract_subgroup_ids(subgroups_raw)
            if subgroup_ids:
                subgroups_dict[parent_id] = subgroup_ids
                for sg_id in subgroup_ids:
                    parent_map[sg_id] = parent_id
                    all_group_ids.add(sg_id)

    # Identify standalone groups
    all_subgroups = set(parent_map.keys())
    standalone = sorted(all_group_ids - parents - all_subgroups)

    return {
        "parents": sorted(parents),
        "subgroups": subgroups_dict,
        "parent_map": parent_map,
        "standalone": standalone,
    }


def _extract_subgroup_ids(subgroups: list[Any]) -> list[str]:
    """Normalize subgroup entries to clean string identifiers."""
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_entry in subgroups:
        subgroup_id: str | None
        if isinstance(raw_entry, dict):
            subgroup_id = raw_entry.get("id")
        else:
            subgroup_id = str(raw_entry)

        if subgroup_id is None:
            continue

        clean_id = subgroup_id.strip()
        if not clean_id:
            continue

        canonical = clean_id.lower()
        if canonical in seen:
            continue

        seen.add(canonical)
        normalized.append(clean_id)

    return normalized


def analyze_group_hierarchy(
    groups: dict[str, Group],
) -> dict[str, list[str] | dict[str, list[str]] | dict[str, str]]:
    """
    DEPRECATED: Use analyze_group_hierarchy_from_json instead.

    This fallback uses pattern matching which may not work for all naming conventions.
    For accurate hierarchy detection, use analyze_group_hierarchy_from_json
    which reads explicit subgroup relationships from Groups.json.
    """
    parents = set()
    subgroups_dict: dict[str, list[str]] = {}  # parent_id -> [subgroup_ids]
    parent_map: dict[str, str] = {}  # subgroup_id -> parent_id
    all_group_ids = set(groups.keys())

    # Identify parent-subgroup relationships (FALLBACK: pattern matching)
    for group_id in all_group_ids:
        # Check if this could be a subgroup (ends with letter)
        if len(group_id) > 1 and group_id[-1].isalpha():
            # Try to find parent by removing last character
            potential_parent = group_id[:-1]

            if potential_parent in all_group_ids:
                # This is a subgroup!
                parents.add(potential_parent)
                parent_map[group_id] = potential_parent

                if potential_parent not in subgroups_dict:
                    subgroups_dict[potential_parent] = []
                subgroups_dict[potential_parent].append(group_id)

    # Identify standalone groups (neither parent nor subgroup)
    parents_list = sorted(parents)
    all_subgroups = set(parent_map.keys())
    standalone = sorted(all_group_ids - parents - all_subgroups)

    return {
        "parents": parents_list,
        "subgroups": subgroups_dict,
        "parent_map": parent_map,
        "standalone": standalone,
    }


def is_parent_group(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> bool:
    """Check if a group is a parent group."""
    return group_id in hierarchy["parents"]


def is_subgroup(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> bool:
    """Check if a group is a subgroup."""
    return group_id in hierarchy["parent_map"]


def get_parent(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> str:
    """Get parent group ID for a subgroup."""
    parent_map: dict[str, str] = hierarchy["parent_map"]  # type: ignore[assignment]
    result = parent_map.get(group_id)
    return str(result) if result is not None else ""


def get_subgroups(
    parent_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> list[str]:
    """Get list of subgroup IDs for a parent."""
    subgroups: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
    result: list[str] = subgroups.get(parent_id, [])
    return result


def has_subgroups(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> bool:
    """Check if a group has subgroups."""
    return group_id in hierarchy["subgroups"]


def get_sibling_groups(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> list[str]:
    """
    Get sibling groups (other subgroups of the same parent).

    E.g., for BME1A, returns [BME1B] if both are subgroups of BME1AB.
    Returns empty list if group is standalone or parent.
    """
    parent_id = get_parent(group_id, hierarchy)
    if not parent_id:
        return []

    all_subgroups = get_subgroups(parent_id, hierarchy)
    return [g for g in all_subgroups if g != group_id]


def get_all_related_groups(
    group_id: str,
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> set[str]:
    """
    Get all groups related to this one (parent, siblings, and self).

    This is crucial for conflict detection: if any related group has a session,
    this group is also busy (for theory sessions where all attend together).

    E.g., for BME1A returns {BME1A, BME1B, BME1AB} if structure exists.
    """
    related = {group_id}

    # Add parent if exists
    parent_id = get_parent(group_id, hierarchy)
    if parent_id:
        related.add(parent_id)
        # Add all siblings (they share parent)
        subgroups: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
        related.update(subgroups.get(parent_id, []))

    # If this is a parent, add all subgroups
    if has_subgroups(group_id, hierarchy):
        subgroups_dict: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]
        related.update(subgroups_dict.get(group_id, []))

    return related


def build_group_family_map(
    hierarchy: dict[str, list[str] | dict[str, list[str]] | dict[str, str]],
) -> dict[str, set[str]]:
    """
    Pre-compute a map from each group_id to all related groups.

    This is used for fast conflict checking during repair operations.

    Returns:
        Dict mapping group_id -> set of all related group_ids

    Example:
        {
            "BME1A": {"BME1A", "BME1B", "BME1AB"},
            "BME1B": {"BME1A", "BME1B", "BME1AB"},
            "BME1AB": {"BME1A", "BME1B", "BME1AB"},
            "BCE8": {"BCE8"},  # Standalone
        }
    """
    all_groups: set[str] = set()

    # Collect all group IDs
    parents: list[str] = hierarchy["parents"]  # type: ignore[assignment]
    parent_map: dict[str, str] = hierarchy["parent_map"]  # type: ignore[assignment]
    standalone: list[str] = hierarchy["standalone"]  # type: ignore[assignment]
    subgroups_dict: dict[str, list[str]] = hierarchy["subgroups"]  # type: ignore[assignment]

    all_groups.update(parents)
    all_groups.update(parent_map.keys())  # All subgroups
    all_groups.update(standalone)

    family_map: dict[str, set[str]] = {}

    for group_id in all_groups:
        family_map[group_id] = get_all_related_groups(group_id, hierarchy)

    return family_map


def groups_conflict(
    group_ids_a: list[str],
    group_ids_b: list[str],
    family_map: dict[str, set[str]],
) -> bool:
    """
    Check if two sets of groups have any family overlap (conflict).

    This correctly handles the parent-subgroup relationship:
    - BME1A and BME1B conflict (same parent, can't be in two places)
    - BME1A and BME1AB conflict (subgroup can't attend if already in parent session)
    - BME1A and BCE1A don't conflict (different families)

    Args:
        group_ids_a: First list of group IDs
        group_ids_b: Second list of group IDs
        family_map: Pre-computed family relationships

    Returns:
        True if any group in A is related to any group in B
    """
    # Get all related groups for set A
    a_family: set[str] = set()
    for gid in group_ids_a:
        a_family.update(family_map.get(gid, {gid}))

    # Check if any group in B (or its family) overlaps with A's family
    for gid in group_ids_b:
        b_family = family_map.get(gid, {gid})
        if a_family & b_family:  # Set intersection
            return True

    return False


# ============================================================================
# CACHED HIERARCHY LOADER
# ============================================================================

# Module-level cache for hierarchy and family_map
_cached_hierarchy: (
    dict[str, list[str] | dict[str, list[str]] | dict[str, str]] | None
) = None
_cached_family_map: dict[str, set[str]] | None = None
_cached_json_path: str | None = None


def get_hierarchy_from_json(
    json_path: str = "data/Groups.json",
) -> dict[str, list[str] | dict[str, list[str]] | dict[str, str]]:
    """
    Get the group hierarchy, loading from JSON and caching the result.

    This is the preferred way to get hierarchy information - it uses the
    explicit subgroup relationships from Groups.json.

    Args:
        json_path: Path to Groups.json (default: "data/Groups.json")

    Returns:
        Hierarchy dict with parents, subgroups, parent_map, standalone
    """
    global _cached_hierarchy, _cached_json_path

    if _cached_hierarchy is None or _cached_json_path != json_path:
        _cached_hierarchy = analyze_group_hierarchy_from_json(json_path)
        _cached_json_path = json_path

    return _cached_hierarchy


def get_family_map_from_json(
    json_path: str = "data/Groups.json",
) -> dict[str, set[str]]:
    """
    Get the pre-computed family map, loading from JSON and caching.

    The family map provides O(1) lookup for all groups related to any given group.

    Args:
        json_path: Path to Groups.json (default: "data/Groups.json")

    Returns:
        Dict mapping group_id -> set of all related group_ids
    """
    global _cached_family_map, _cached_json_path

    hierarchy = get_hierarchy_from_json(json_path)

    if _cached_family_map is None or _cached_json_path != json_path:
        _cached_family_map = build_group_family_map(hierarchy)

    return _cached_family_map


def clear_hierarchy_cache() -> None:
    """Clear the cached hierarchy data (useful for testing or data reload)."""
    global _cached_hierarchy, _cached_family_map, _cached_json_path
    _cached_hierarchy = None
    _cached_family_map = None
    _cached_json_path = None
