"""
Group Hierarchy Analyzer

Identifies parent groups and their subgroups to enable proper scheduling:
- Theory sessions for parent (whole class together)
- Practical sessions for subgroups separately
"""

from src.entities.group import Group


def analyze_group_hierarchy(groups: dict[str, Group]) -> dict:
    """
    Analyzes group relationships to identify parents and subgroups.

    A subgroup is identified by:
    - Having a parent group ID as prefix (e.g., BAE2A has parent BAE2)
    - Parent group must exist in groups dict

    Args:
        groups: Dictionary of group_id -> Group objects

    Returns:
        Dictionary with:
        - "parents": List of parent group IDs
        - "subgroups": Dict mapping parent_id -> [subgroup_ids]
        - "parent_map": Dict mapping subgroup_id -> parent_id
        - "standalone": List of groups with no subgroups

    Example:
        {
            "parents": ["BAE2", "BAE4"],
            "subgroups": {"BAE2": ["BAE2A", "BAE2B"], "BAE4": ["BAE4A", "BAE4B"]},
            "parent_map": {"BAE2A": "BAE2", "BAE2B": "BAE2", "BAE4A": "BAE4", "BAE4B": "BAE4"},
            "standalone": ["BAE8"]  # Groups with no subgroups
        }
    """
    parents = set()
    subgroups_dict: dict[str, list[str]] = {}  # parent_id -> [subgroup_ids]
    parent_map = {}  # subgroup_id -> parent_id
    all_group_ids = set(groups.keys())

    # Identify parent-subgroup relationships
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


def is_parent_group(group_id: str, hierarchy: dict) -> bool:
    """Check if a group is a parent group."""
    return group_id in hierarchy["parents"]


def is_subgroup(group_id: str, hierarchy: dict) -> bool:
    """Check if a group is a subgroup."""
    return group_id in hierarchy["parent_map"]


def get_parent(group_id: str, hierarchy: dict) -> str:
    """Get parent group ID for a subgroup."""
    result = hierarchy["parent_map"].get(group_id)
    return str(result) if result is not None else ""


def get_subgroups(parent_id: str, hierarchy: dict) -> list[str]:
    """Get list of subgroup IDs for a parent."""
    result = hierarchy["subgroups"].get(parent_id, [])
    return list(result) if result is not None else []


def has_subgroups(group_id: str, hierarchy: dict) -> bool:
    """Check if a group has subgroups."""
    return group_id in hierarchy["subgroups"]
