"""Leiden community detection and checkpoint management.

## RAG Theory: Leiden Algorithm

The Leiden algorithm (improvement over Louvain) detects communities of
related entities in the knowledge graph. It guarantees well-connected
communities, unlike Louvain which can produce disconnected clusters.

Uses Neo4j GDS (Graph Data Science) for execution:
- gds.graph.project() - Create in-memory graph projection
- gds.leiden.stream() - Run Leiden, get community assignments

## Data Flow

1. Project graph -> GDS in-memory graph
2. Run Leiden -> Community assignments per node
3. Save checkpoint for crash recovery
4. Write community IDs back to Neo4j nodes
"""

from typing import Any, Optional
from pathlib import Path
import json

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_LEIDEN_RESOLUTION,
    GRAPHRAG_LEIDEN_MAX_LEVELS,
    GRAPHRAG_LEIDEN_SEED,
    GRAPHRAG_LEIDEN_CONCURRENCY,
    DIR_GRAPH_DATA,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def project_graph(gds: GraphDataScience, graph_name: str = "graphrag") -> Any:
    """Create GDS graph projection for community detection.

    Projects Entity nodes and RELATED_TO relationships into
    GDS in-memory format for algorithm execution.

    Args:
        gds: GraphDataScience client instance.
        graph_name: Name for the projected graph.

    Returns:
        GDS Graph object.

    Raises:
        Exception: If projection fails (e.g., no data).
    """
    # Drop existing projection if exists
    if gds.graph.exists(graph_name).exists:
        gds.graph.drop(graph_name)
        logger.info(f"Dropped existing graph projection: {graph_name}")

    # Project the graph
    # Using native projection for Entity nodes and RELATED_TO relationships
    # Note: GDS only supports numeric properties, so we don't project entity_type (string)
    graph, result = gds.graph.project(
        graph_name,
        "Entity",  # Node label
        {
            "RELATED_TO": {
                "orientation": "UNDIRECTED",  # Leiden works on undirected
            }
        },
    )

    logger.info(
        f"Projected graph '{graph_name}': "
        f"{result['nodeCount']} nodes, {result['relationshipCount']} relationships"
    )

    return graph


def run_leiden(
    gds: GraphDataScience,
    graph: Any,
    resolution: float = GRAPHRAG_LEIDEN_RESOLUTION,
    max_levels: int = GRAPHRAG_LEIDEN_MAX_LEVELS,
    seed: int = GRAPHRAG_LEIDEN_SEED,
    concurrency: int = GRAPHRAG_LEIDEN_CONCURRENCY,
) -> dict[str, Any]:
    """Run Leiden community detection algorithm.

    Leiden improves on Louvain by guaranteeing well-connected communities.
    Returns hierarchical community assignments.

    Uses randomSeed + concurrency=1 for DETERMINISTIC results.
    Same graph + same seed = same community assignments (guaranteed).
    This enables crash recovery without community ID mismatches.

    Args:
        gds: GraphDataScience client instance.
        graph: GDS Graph object from project_graph().
        resolution: Higher = more, smaller communities (default 1.0).
        max_levels: Maximum hierarchy depth.
        seed: Random seed for deterministic results (default 42).
        concurrency: Thread count (1 for determinism, default 1).

    Returns:
        Dict with:
        - community_count: Number of communities found
        - levels: Number of hierarchy levels
        - node_communities: List of (node_id, community_id) tuples
        - seed: The seed used (for checkpoint verification)

    Example:
        >>> result = run_leiden(gds, graph)
        >>> print(result["community_count"])
        12
    """
    logger.info(f"Running Leiden with seed={seed}, concurrency={concurrency}")

    # Run Leiden in stream mode with deterministic settings
    result = gds.leiden.stream(
        graph,
        gamma=resolution,  # Resolution parameter
        maxLevels=max_levels,
        includeIntermediateCommunities=True,  # Get hierarchy
        randomSeed=seed,  # Fixed seed for determinism
        concurrency=concurrency,  # Single-threaded for reproducibility
    )

    # Convert to list of dicts
    node_communities = []
    for record in result.itertuples():
        node_communities.append({
            "node_id": record.nodeId,
            "community_id": record.communityId,
            "intermediate_ids": list(record.intermediateCommunityIds) if hasattr(record, 'intermediateCommunityIds') else [],
        })

    # Get unique community count
    unique_communities = set(nc["community_id"] for nc in node_communities)

    logger.info(
        f"Leiden found {len(unique_communities)} communities "
        f"across {len(node_communities)} nodes"
    )

    return {
        "community_count": len(unique_communities),
        "node_count": len(node_communities),
        "node_communities": node_communities,
        "seed": seed,  # Include for checkpoint verification
    }


# ============================================================================
# Leiden Checkpoint (for crash recovery)
# ============================================================================


def save_leiden_checkpoint(
    leiden_result: dict[str, Any],
    output_name: str = "leiden_checkpoint.json",
) -> Path:
    """Save Leiden result to checkpoint file for crash recovery.

    Stores the seed and node->community assignments so that:
    1. We can verify Leiden produces same results on re-run
    2. We can resume summarization without re-running Leiden

    Args:
        leiden_result: Result from run_leiden() with node_communities and seed.
        output_name: Checkpoint filename.

    Returns:
        Path to saved checkpoint file.

    Example:
        >>> result = run_leiden(gds, graph)
        >>> path = save_leiden_checkpoint(result)
        >>> print(f"Saved checkpoint to {path}")
    """
    from datetime import datetime

    output_path = DIR_GRAPH_DATA / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "seed": leiden_result["seed"],
        "timestamp": datetime.now().isoformat(),
        "community_count": leiden_result["community_count"],
        "node_count": leiden_result["node_count"],
        "assignments": leiden_result["node_communities"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Saved Leiden checkpoint to {output_path}")
    return output_path


def load_leiden_checkpoint(
    input_name: str = "leiden_checkpoint.json",
) -> Optional[dict[str, Any]]:
    """Load Leiden checkpoint from file.

    Args:
        input_name: Checkpoint filename.

    Returns:
        Checkpoint dict with seed, timestamp, and assignments,
        or None if file doesn't exist.

    Example:
        >>> checkpoint = load_leiden_checkpoint()
        >>> if checkpoint:
        ...     print(f"Loaded {len(checkpoint['assignments'])} assignments")
    """
    input_path = DIR_GRAPH_DATA / input_name

    if not input_path.exists():
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    logger.info(
        f"Loaded Leiden checkpoint: seed={checkpoint['seed']}, "
        f"{checkpoint['community_count']} communities, "
        f"{checkpoint['node_count']} nodes"
    )
    return checkpoint


def write_communities_to_neo4j(
    driver: Driver,
    node_communities: list[dict[str, Any]],
) -> int:
    """Write community assignments back to Neo4j nodes.

    Stores community_id as a property on each Entity node
    for later querying.

    Args:
        driver: Neo4j driver instance.
        node_communities: List from run_leiden() with node_id and community_id.

    Returns:
        Number of nodes updated.
    """
    query = """
    UNWIND $assignments AS assignment
    MATCH (e:Entity)
    WHERE id(e) = assignment.node_id
    SET e.community_id = assignment.community_id
    RETURN count(e) as count
    """

    result = driver.execute_query(query, assignments=node_communities)
    count = result.records[0]["count"]

    logger.info(f"Updated {count} nodes with community IDs")
    return count
