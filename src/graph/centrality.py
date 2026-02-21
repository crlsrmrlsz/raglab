"""PageRank centrality computation for GraphRAG entity importance ranking.

## RAG Theory: Entity Importance in Communities

PageRank identifies "hub" entities that are well-connected to other
important entities. This is more informative than simple degree count
because it considers the quality of connections, not just quantity.

Used for:
1. Prioritizing entities in community summaries (top-k by PageRank)
2. Ordering members in generation context
3. Selecting representative entities for community search

## Library Usage

Uses Neo4j GDS (Graph Data Science) library:
- gds.pageRank.stream() - Compute PageRank scores for all nodes
- gds.graph.project() - Create subgraph projection if needed

## Data Flow

1. After Leiden: Run PageRank on full graph projection
2. Store scores in Neo4j as `e.pagerank` property
3. Retrieve scores when building CommunityMember objects
4. Sort members by PageRank for summarization focus
"""

from typing import Any

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_PAGERANK_DAMPING,
    GRAPHRAG_PAGERANK_ITERATIONS,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def compute_pagerank(
    gds: GraphDataScience,
    graph: Any,
    damping_factor: float = GRAPHRAG_PAGERANK_DAMPING,
    max_iterations: int = GRAPHRAG_PAGERANK_ITERATIONS,
) -> dict[int, float]:
    """Compute PageRank centrality for all nodes in the graph.

    PageRank measures node importance based on the structure of
    incoming links. Higher scores indicate more influential nodes.

    Args:
        gds: GraphDataScience client instance.
        graph: GDS graph projection from project_graph().
        damping_factor: Probability of continuing walk (default 0.85).
        max_iterations: Maximum iterations for convergence (default 20).

    Returns:
        Dict mapping Neo4j internal node_id to PageRank score.

    Example:
        >>> scores = compute_pagerank(gds, graph)
        >>> top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        >>> for node_id, score in top_nodes:
        ...     print(f"Node {node_id}: {score:.4f}")
    """
    logger.info(
        f"Computing PageRank with damping={damping_factor}, "
        f"maxIterations={max_iterations}"
    )

    result = gds.pageRank.stream(
        graph,
        dampingFactor=damping_factor,
        maxIterations=max_iterations,
    )

    scores = {}
    for record in result.itertuples():
        scores[record.nodeId] = record.score

    # Log statistics
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        logger.info(
            f"PageRank computed for {len(scores)} nodes: "
            f"min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}"
        )

    return scores


def write_pagerank_to_neo4j(
    driver: Driver,
    pagerank_scores: dict[int, float],
    batch_size: int = 1000,
) -> int:
    """Write PageRank scores to Neo4j Entity nodes.

    Stores the PageRank score as a property on each Entity node
    for later retrieval during community summarization.

    Args:
        driver: Neo4j driver instance.
        pagerank_scores: Dict from compute_pagerank().
        batch_size: Batch size for UNWIND operations.

    Returns:
        Number of nodes updated.

    Example:
        >>> scores = compute_pagerank(gds, graph)
        >>> count = write_pagerank_to_neo4j(driver, scores)
        >>> print(f"Updated {count} nodes with PageRank scores")
    """
    if not pagerank_scores:
        logger.warning("No PageRank scores to write")
        return 0

    # Convert to list of dicts for UNWIND
    assignments = [
        {"node_id": node_id, "score": score}
        for node_id, score in pagerank_scores.items()
    ]

    query = """
    UNWIND $assignments AS assignment
    MATCH (e:Entity)
    WHERE id(e) = assignment.node_id
    SET e.pagerank = assignment.score
    RETURN count(e) as count
    """

    total_updated = 0

    # Process in batches
    for i in range(0, len(assignments), batch_size):
        batch = assignments[i : i + batch_size]
        result = driver.execute_query(query, assignments=batch)
        count = result.records[0]["count"]
        total_updated += count

    logger.info(f"Wrote PageRank scores to {total_updated} Entity nodes")
    return total_updated


def compute_and_store_degree(driver: Driver) -> int:
    """Compute entity degree and store on Entity nodes.

    Degree is the number of RELATED_TO relationships connected to an entity.
    Used for combined_degree ranking in local search (Microsoft GraphRAG approach):
    combined_degree = degree(source_entity) + degree(neighbor)

    Higher-degree entities are "hub" nodes that carry more information value.

    Args:
        driver: Neo4j driver instance.

    Returns:
        Number of nodes updated with degree property.

    Example:
        >>> count = compute_and_store_degree(driver)
        >>> print(f"Updated {count} nodes with degree")
    """
    query = """
    MATCH (e:Entity)
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    SET e.degree = degree
    RETURN count(e) as count
    """

    result = driver.execute_query(query)
    count = result.records[0]["count"]

    # Log statistics
    stats_query = """
    MATCH (e:Entity)
    WHERE e.degree IS NOT NULL
    RETURN min(e.degree) as min_deg, max(e.degree) as max_deg, avg(e.degree) as avg_deg
    """
    stats = driver.execute_query(stats_query)
    if stats.records:
        rec = stats.records[0]
        logger.info(
            f"Stored degree for {count} Entity nodes: "
            f"min={rec['min_deg']}, max={rec['max_deg']}, avg={rec['avg_deg']:.2f}"
        )
    else:
        logger.info(f"Stored degree for {count} Entity nodes")

    return count
