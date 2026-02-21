"""Hierarchical community parsing for multi-level GraphRAG.

## RAG Theory: Hierarchical Community Structure

Microsoft GraphRAG (arXiv:2404.16130) uses Leiden's multi-level output:
- Level 0 (C0): Coarsest granularity - corpus-wide themes
- Level 1 (C1): Medium granularity - domain-level themes
- Level 2 (C2): Finest granularity - specific topics within domains

Query routing uses hierarchy:
- Global queries (thematic): Use L0 (coarsest) communities with DRIFT search
- Local queries (entity-specific): Use L2 (finest) communities

## Data Flow

1. Leiden returns `intermediateCommunityIds` per node
2. `parse_leiden_hierarchy()` extracts C0, C1, C2 structure
3. Each level stored separately in Weaviate/JSON
4. Query time selects appropriate level based on query type
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

from neo4j import Driver

from src.shared.files import setup_logging

logger = setup_logging(__name__)


@dataclass
class CommunityLevel:
    """Single level in the community hierarchy.

    Attributes:
        level: Hierarchy depth (0=coarsest, higher=finer).
        communities: Map from community_id to set of entity node IDs.
        parent_map: Map from community_id to parent community_id at next level.
        child_map: Map from community_id to list of child community_ids.
        member_counts: Map from community_id to entity count.

    Example:
        >>> level = CommunityLevel(level=0)
        >>> level.communities[42] = {1, 2, 3}  # Community 42 has nodes 1, 2, 3
        >>> level.parent_map[42] = 10  # Community 42's parent at L1 is 10
    """

    level: int
    communities: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    parent_map: dict[int, Optional[int]] = field(default_factory=dict)
    child_map: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    member_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert to proper defaultdict types."""
        if not isinstance(self.communities, defaultdict):
            self.communities = defaultdict(set, self.communities)
        if not isinstance(self.child_map, defaultdict):
            self.child_map = defaultdict(list, self.child_map)


def parse_leiden_hierarchy(
    leiden_result: dict[str, Any],
    max_levels: int = 3,
) -> dict[int, CommunityLevel]:
    """Parse Leiden result into multi-level hierarchy.

    Extracts L0, L1, L2 (and optionally more) levels from the
    `intermediateCommunityIds` returned by Neo4j GDS Leiden.

    The intermediate IDs array represents the hierarchy (matching Microsoft convention):
    - Index 0: Level 0 community (coarsest, corpus-wide themes)
    - Index 1: Level 1 community (medium granularity)
    - Index 2: Level 2 community (finest, same as Leiden's `communityId`)

    Note: Entities in Neo4j store their community_id from the FINEST level (L2).
    Global queries use L0 (coarsest). Local queries use L2 (finest) for entity lookup.

    Args:
        leiden_result: Result from run_leiden() containing node_communities
            with intermediateCommunityIds.
        max_levels: Maximum hierarchy levels to extract (default 3).

    Returns:
        Dict mapping level (0, 1, 2) to CommunityLevel objects.

    Example:
        >>> result = run_leiden(gds, graph)
        >>> levels = parse_leiden_hierarchy(result)
        >>> print(f"L0: {len(levels[0].communities)} communities")
        >>> print(f"L1: {len(levels[1].communities)} communities")
        >>> print(f"L2: {len(levels[2].communities)} communities")
    """
    node_communities = leiden_result.get("node_communities", [])

    if not node_communities:
        logger.warning("No node_communities in Leiden result")
        return {i: CommunityLevel(level=i) for i in range(max_levels)}

    # Initialize levels
    levels: dict[int, CommunityLevel] = {
        i: CommunityLevel(level=i) for i in range(max_levels)
    }

    # Track parent relationships across levels
    # Key: (child_level, child_community_id), Value: parent_community_id
    parent_relationships: dict[tuple[int, int], int] = {}

    for nc in node_communities:
        node_id = nc["node_id"]
        final_community_id = nc["community_id"]
        intermediate_ids = nc.get("intermediate_ids", [])

        # Build the full hierarchy for this node
        # The intermediate_ids array may be shorter than max_levels
        # if Leiden didn't produce that many levels
        hierarchy = []

        # Intermediate IDs from Neo4j GDS: index 0 = coarsest, last = finest
        # Keep this order to match Microsoft GraphRAG convention (L0 = coarsest)
        if intermediate_ids:
            # intermediate_ids[0] = coarsest (L0), last element = finest (L2)
            hierarchy = list(intermediate_ids)
        else:
            # No hierarchy data, just use the final community ID for L0
            hierarchy = [final_community_id]

        # Ensure we have entries for each level
        # If hierarchy is shorter, the node belongs to same community at coarser levels
        while len(hierarchy) < max_levels:
            hierarchy.append(hierarchy[-1] if hierarchy else final_community_id)

        # Assign node to communities at each level
        for level_idx in range(min(max_levels, len(hierarchy))):
            community_id = hierarchy[level_idx]
            levels[level_idx].communities[community_id].add(node_id)

            # Track parent relationship (L0 -> L1, L1 -> L2)
            if level_idx < max_levels - 1 and level_idx + 1 < len(hierarchy):
                parent_id = hierarchy[level_idx + 1]
                parent_relationships[(level_idx, community_id)] = parent_id

    # Build parent_map and child_map for each level
    for (child_level, child_id), parent_id in parent_relationships.items():
        levels[child_level].parent_map[child_id] = parent_id
        if child_level + 1 < max_levels:
            levels[child_level + 1].child_map[parent_id].append(child_id)

    # Calculate member counts
    for level_idx in range(max_levels):
        for community_id, node_ids in levels[level_idx].communities.items():
            levels[level_idx].member_counts[community_id] = len(node_ids)

    # Log summary
    for level_idx in range(max_levels):
        num_communities = len(levels[level_idx].communities)
        total_members = sum(levels[level_idx].member_counts.values())
        logger.info(
            f"Level {level_idx}: {num_communities} communities, "
            f"{total_members} total members"
        )

    return levels


def get_community_node_ids(
    level: CommunityLevel,
    community_id: int,
) -> set[int]:
    """Get Neo4j node IDs for a specific community.

    Args:
        level: CommunityLevel object.
        community_id: Community ID to lookup.

    Returns:
        Set of Neo4j internal node IDs.
    """
    return level.communities.get(community_id, set())


def get_parent_community_id(
    levels: dict[int, CommunityLevel],
    current_level: int,
    community_id: int,
) -> Optional[int]:
    """Get parent community ID at the next coarser level.

    Args:
        levels: Dict of all CommunityLevel objects.
        current_level: Current hierarchy level (0, 1, etc.).
        community_id: Community ID at current level.

    Returns:
        Parent community ID at level+1, or None if top level.
    """
    if current_level >= len(levels) - 1:
        return None

    return levels[current_level].parent_map.get(community_id)


def filter_communities_by_size(
    level: CommunityLevel,
    min_size: int,
) -> list[int]:
    """Get community IDs that meet minimum size requirement.

    Args:
        level: CommunityLevel object.
        min_size: Minimum number of members.

    Returns:
        List of community IDs with >= min_size members.
    """
    return [
        community_id
        for community_id, count in level.member_counts.items()
        if count >= min_size
    ]


def build_community_key(level: int, community_id: int) -> str:
    """Build unique community key for storage.

    Format: community_L{level}_{id}

    Args:
        level: Hierarchy level (0, 1, 2).
        community_id: Leiden-assigned community ID.

    Returns:
        Unique string key for JSON/Weaviate storage.

    Example:
        >>> build_community_key(0, 42)
        'community_L0_42'
        >>> build_community_key(2, 5)
        'community_L2_5'
    """
    return f"community_L{level}_{community_id}"


def parse_community_key(key: str) -> tuple[int, int]:
    """Parse community key back to level and ID.

    Args:
        key: Community key string (e.g., "community_L0_42").

    Returns:
        Tuple of (level, community_id).

    Raises:
        ValueError: If key format is invalid.

    Example:
        >>> parse_community_key("community_L0_42")
        (0, 42)
    """
    # Handle legacy format (community_42) - treat as L0
    if key.startswith("community_") and "_L" not in key:
        community_id = int(key.replace("community_", ""))
        return (0, community_id)

    # New format: community_L{level}_{id}
    parts = key.replace("community_L", "").split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid community key format: {key}")

    level = int(parts[0])
    community_id = int(parts[1])
    return (level, community_id)
