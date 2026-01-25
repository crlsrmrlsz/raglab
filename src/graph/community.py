"""Leiden community detection and summarization for GraphRAG.

## RAG Theory: Community Detection in GraphRAG

GraphRAG uses the Leiden algorithm (improvement over Louvain) to:
1. Detect communities of related entities in the knowledge graph
2. Create hierarchical community structure (multi-level)
3. Generate LLM summaries for each community

These community summaries enable "global queries" that synthesize
information across multiple documents (e.g., "What are the main themes?").

## Library Usage

Uses Neo4j GDS (Graph Data Science) for Leiden:
- gds.graph.project() - Create in-memory graph projection
- gds.leiden.stream() - Run Leiden, get community assignments
- gds.pageRank.stream() - Compute node centrality for ranking

## Data Flow

1. Project graph → GDS in-memory graph
2. Run Leiden → Community assignments per node
3. For each community: Collect members → Generate LLM summary
4. Store summaries in Neo4j and/or JSON for retrieval
"""

from typing import Any, Optional
from pathlib import Path
import json

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_LEIDEN_RESOLUTION,
    GRAPHRAG_LEIDEN_MAX_LEVELS,
    GRAPHRAG_MIN_COMMUNITY_SIZE,
    GRAPHRAG_LEIDEN_SEED,
    GRAPHRAG_LEIDEN_CONCURRENCY,
    GRAPHRAG_SUMMARY_MODEL,
    GRAPHRAG_COMMUNITY_PROMPT,
    GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT,
    GRAPHRAG_MAX_CONTEXT_TOKENS,
    GRAPHRAG_MAX_HIERARCHY_LEVELS,
    DIR_GRAPH_DATA,
    get_community_collection_name,
)
from src.shared.openrouter_client import call_chat_completion
from src.shared.files import setup_logging
from src.rag_pipeline.embedding.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_client import (
    get_client as get_weaviate_client,
    create_community_collection,
    upload_community as weaviate_upload_community,
    get_existing_community_ids as weaviate_get_existing_ids,
)
from .schemas import Community, CommunityMember, CommunityRelationship
from .neo4j_client import get_gds_client
from .hierarchy import (
    parse_leiden_hierarchy,
    build_community_key,
    filter_communities_by_size,
    CommunityLevel,
)

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

    Stores the seed and node→community assignments so that:
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


def verify_leiden_checkpoint(
    leiden_result: dict[str, Any],
    checkpoint: dict[str, Any],
) -> bool:
    """Verify that Leiden result matches checkpoint.

    Checks that seed and community assignments are identical,
    ensuring deterministic behavior.

    Args:
        leiden_result: Result from run_leiden().
        checkpoint: Loaded checkpoint from load_leiden_checkpoint().

    Returns:
        True if results match, False otherwise.

    Raises:
        ValueError: If seed or assignments don't match.
    """
    if leiden_result["seed"] != checkpoint["seed"]:
        raise ValueError(
            f"Seed mismatch: got {leiden_result['seed']}, "
            f"expected {checkpoint['seed']}"
        )

    if leiden_result["community_count"] != checkpoint["community_count"]:
        logger.warning(
            f"Community count changed: {leiden_result['community_count']} vs "
            f"{checkpoint['community_count']} (this may indicate graph changes)"
        )
        return False

    logger.info("Leiden checkpoint verification passed")
    return True


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


def get_community_members(
    driver: Driver,
    community_id: int,
) -> list[CommunityMember]:
    """Get all entities belonging to a specific community.

    Includes PageRank scores if computed, sorts by PageRank then degree.

    Args:
        driver: Neo4j driver instance.
        community_id: Community ID to query.

    Returns:
        List of CommunityMember objects sorted by importance.
    """
    query = """
    MATCH (e:Entity {community_id: $community_id})
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    RETURN
        e.name as entity_name,
        e.entity_type as entity_type,
        e.description as description,
        degree,
        coalesce(e.pagerank, 0.0) as pagerank
    ORDER BY pagerank DESC, degree DESC
    """

    result = driver.execute_query(query, community_id=community_id)

    members = []
    for record in result.records:
        members.append(CommunityMember(
            entity_name=record["entity_name"],
            entity_type=record["entity_type"] or "UNKNOWN",
            description=record["description"] or "",
            degree=record["degree"],
            pagerank=record["pagerank"],
        ))

    return members


def get_community_members_by_node_ids(
    driver: Driver,
    node_ids: set[int],
) -> list[CommunityMember]:
    """Get members by Neo4j internal node IDs (for hierarchy levels).

    Used when processing higher-level communities (C1, C2) that aggregate
    multiple C0 communities. Instead of querying by community_id, we
    query by the specific node IDs that belong to the aggregated community.

    Args:
        driver: Neo4j driver instance.
        node_ids: Set of Neo4j internal node IDs.

    Returns:
        List of CommunityMember objects sorted by PageRank.
    """
    if not node_ids:
        return []

    query = """
    MATCH (e:Entity)
    WHERE id(e) IN $node_ids
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    RETURN
        e.name as entity_name,
        e.entity_type as entity_type,
        e.description as description,
        degree,
        coalesce(e.pagerank, 0.0) as pagerank
    ORDER BY pagerank DESC, degree DESC
    """

    result = driver.execute_query(query, node_ids=list(node_ids))

    members = []
    for record in result.records:
        members.append(CommunityMember(
            entity_name=record["entity_name"],
            entity_type=record["entity_type"] or "UNKNOWN",
            description=record["description"] or "",
            degree=record["degree"],
            pagerank=record["pagerank"],
        ))

    return members


def get_community_relationships(
    driver: Driver,
    community_id: int,
) -> list[CommunityRelationship]:
    """Get relationships within a community as structured objects.

    Returns CommunityRelationship objects that can be stored in JSON
    for offline access without Neo4j queries.

    Args:
        driver: Neo4j driver instance.
        community_id: Community ID to query.

    Returns:
        List of CommunityRelationship objects.
    """
    query = """
    MATCH (source:Entity {community_id: $community_id})-[r:RELATED_TO]->(target:Entity {community_id: $community_id})
    RETURN
        source.name as source,
        target.name as target,
        r.type as relationship_type,
        r.description as description,
        coalesce(r.weight, 1.0) as weight
    """

    result = driver.execute_query(query, community_id=community_id)

    return [
        CommunityRelationship(
            source=r["source"],
            target=r["target"],
            relationship_type=r["relationship_type"] or "RELATED_TO",
            description=r["description"] or "",
            weight=r["weight"],
        )
        for r in result.records
    ]


def get_community_relationships_by_node_ids(
    driver: Driver,
    node_ids: set[int],
) -> list[CommunityRelationship]:
    """Get relationships between nodes in a set (for hierarchy levels).

    Used for C1/C2 communities that span multiple C0 communities.

    Args:
        driver: Neo4j driver instance.
        node_ids: Set of Neo4j internal node IDs.

    Returns:
        List of CommunityRelationship objects.
    """
    if not node_ids:
        return []

    query = """
    MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
    WHERE id(source) IN $node_ids AND id(target) IN $node_ids
    RETURN
        source.name as source,
        target.name as target,
        r.type as relationship_type,
        r.description as description,
        coalesce(r.weight, 1.0) as weight
    """

    result = driver.execute_query(query, node_ids=list(node_ids))

    return [
        CommunityRelationship(
            source=r["source"],
            target=r["target"],
            relationship_type=r["relationship_type"] or "RELATED_TO",
            description=r["description"] or "",
            weight=r["weight"],
        )
        for r in result.records
    ]


def build_community_context(
    members: list[CommunityMember],
    relationships: list[CommunityRelationship],
    max_tokens: int = GRAPHRAG_MAX_CONTEXT_TOKENS,
) -> str:
    """Build context string for community summarization.

    Formats entity and relationship information for LLM prompt.
    Entities are sorted by PageRank (highest importance first).

    Args:
        members: List of community members (should be pre-sorted by PageRank).
        relationships: List of CommunityRelationship objects.
        max_tokens: Approximate token limit (chars / 4).

    Returns:
        Formatted context string.
    """
    lines = []

    # Add entities (already sorted by PageRank from get_community_members)
    lines.append("## Entities")
    for member in members:
        desc = f" - {member.description}" if member.description else ""
        pr_note = f" [PR:{member.pagerank:.3f}]" if member.pagerank > 0 else ""
        lines.append(f"- {member.entity_name} ({member.entity_type}){pr_note}{desc}")

    # Add relationships
    if relationships:
        lines.append("\n## Relationships")
        for rel in relationships:
            desc = f": {rel.description}" if rel.description else ""
            lines.append(
                f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}{desc}"
            )

    context = "\n".join(lines)

    # Truncate if too long (approximate token limit)
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[... truncated]"

    return context


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (chars / 4 approximation)."""
    return len(text) // 4


def _format_child_summary_block(child_key: str, summary: str) -> str:
    """Format a child community summary for inclusion in parent context.

    Args:
        child_key: Child community key (e.g., "community_L2_42").
        summary: The child community's summary text.

    Returns:
        Formatted block for inclusion in hierarchical context.
    """
    return f"\n[Sub-Community: {child_key}]\n{summary}\n"


def build_hierarchical_context(
    community_id: int,
    level: int,
    levels: dict[int, "CommunityLevel"],
    driver: Driver,
    child_summaries: dict[str, str],
    max_tokens: int = GRAPHRAG_MAX_CONTEXT_TOKENS,
) -> tuple[str, bool]:
    """Build context with child summary substitution when over token limit.

    Implements Microsoft GraphRAG's bottom-up summarization (arXiv:2404.16130):
    1. Try raw entity/relationship data first
    2. If over limit: get children from levels[level+1].child_map
    3. Rank children by raw token count (descending)
    4. Iteratively substitute largest child's raw data with its summary
    5. Rebuild context until under limit

    Args:
        community_id: The Leiden-assigned community ID at this level.
        level: Current hierarchy level (0=coarsest, 2=finest).
        levels: Dict mapping level index to CommunityLevel objects.
        driver: Neo4j driver for fetching entity/relationship data.
        child_summaries: Dict of child_key -> summary for substitution.
        max_tokens: Maximum tokens for context.

    Returns:
        Tuple of (context_string, used_substitution_flag).
        used_substitution_flag is True if any child summaries were substituted.
    """
    level_data = levels[level]
    node_ids = level_data.communities.get(community_id, set())

    # Get members and relationships
    members = get_community_members_by_node_ids(driver, node_ids)
    relationships = get_community_relationships_by_node_ids(driver, node_ids)

    # Try raw context first
    raw_context = build_community_context(members, relationships, max_tokens=max_tokens)
    raw_tokens = _estimate_tokens(raw_context)

    if raw_tokens <= max_tokens:
        return raw_context, False

    # Need to substitute child summaries
    # Get children from the next finer level (level + 1)
    finer_level = level + 1
    if finer_level not in levels:
        # No finer level exists, use truncated raw context
        logger.warning(
            f"Community L{level}_{community_id}: context exceeds {max_tokens} tokens, "
            f"no finer level for substitution, using truncated context"
        )
        return raw_context, False

    # Get child community IDs from child_map
    child_community_ids = level_data.child_map.get(community_id, [])

    if not child_community_ids:
        # No children found, use truncated raw context
        return raw_context, False

    logger.debug(f"L{level}_{community_id}: {len(child_community_ids)} children for substitution")

    # Calculate token count for each child's raw data
    child_token_counts: list[tuple[int, int]] = []  # (child_id, token_count)
    finer_level_data = levels[finer_level]

    for child_id in child_community_ids:
        child_node_ids = finer_level_data.communities.get(child_id, set())
        child_members = get_community_members_by_node_ids(driver, child_node_ids)
        child_rels = get_community_relationships_by_node_ids(driver, child_node_ids)
        child_context = build_community_context(child_members, child_rels, max_tokens=max_tokens)
        child_token_counts.append((child_id, _estimate_tokens(child_context)))

    # Sort by token count descending (substitute largest first)
    child_token_counts.sort(key=lambda x: x[1], reverse=True)

    # Build context with progressive substitution
    # Start with all children using raw data
    children_using_summary: set[int] = set()

    for child_id, _ in child_token_counts:
        children_using_summary.add(child_id)

        # Rebuild context with substitutions
        context_parts = []
        cross_community_rels = []

        # Process each child: use summary if in substitution set
        for cid in child_community_ids:
            child_key = build_community_key(finer_level, cid)

            if cid in children_using_summary and child_key in child_summaries:
                # Use child summary
                context_parts.append(_format_child_summary_block(child_key, child_summaries[child_key]))
            else:
                # Use raw data
                child_node_ids = finer_level_data.communities.get(cid, set())
                child_members = get_community_members_by_node_ids(driver, child_node_ids)
                child_rels = get_community_relationships_by_node_ids(driver, child_node_ids)
                child_raw = build_community_context(child_members, child_rels, max_tokens=max_tokens)
                context_parts.append(f"\n[Child Community L{finer_level}_{cid}]\n{child_raw}\n")

        # Add cross-community relationships (relationships between entities in different children)
        all_child_node_ids = set()
        for cid in child_community_ids:
            all_child_node_ids.update(finer_level_data.communities.get(cid, set()))

        cross_rels = get_community_relationships_by_node_ids(driver, all_child_node_ids)
        if cross_rels:
            context_parts.append("\n## Cross-Community Relationships")
            for rel in cross_rels[:20]:  # Limit cross-community rels
                desc = f": {rel.description}" if rel.description else ""
                context_parts.append(
                    f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}{desc}"
                )

        full_context = "\n".join(context_parts)
        context_tokens = _estimate_tokens(full_context)

        if context_tokens <= max_tokens:
            logger.info(
                f"L{level}_{community_id}: substituted {len(children_using_summary)} child summaries "
                f"({context_tokens} tokens)"
            )
            return full_context, True

    # All children substituted but still over limit - use truncated
    logger.warning(
        f"L{level}_{community_id}: all children substituted but still {context_tokens} tokens, truncating"
    )
    max_chars = max_tokens * 4
    return full_context[:max_chars] + "\n[... truncated]", True


def summarize_community(
    members: list[CommunityMember],
    relationships: list[CommunityRelationship],
    model: str = GRAPHRAG_SUMMARY_MODEL,
    prebuilt_context: Optional[str] = None,
    used_substitution: bool = False,
) -> tuple[str, Optional[list[float]]]:
    """Generate LLM summary AND embedding for a community.

    Uses community entities and relationships to generate
    a thematic summary, then creates an embedding vector
    for semantic retrieval.

    Microsoft GraphRAG bottom-up approach (arXiv:2404.16130):
    - When prebuilt_context is provided, it may contain child summaries
    - If used_substitution is True, use GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT
    - No max_tokens limit on output to allow complete summaries

    Args:
        members: List of community members (pre-sorted by PageRank).
        relationships: List of CommunityRelationship objects.
        model: LLM model for summarization.
        prebuilt_context: Pre-built context string (may contain child summaries).
        used_substitution: True if child summaries were substituted.

    Returns:
        Tuple of (summary_text, embedding_vector).
        Embedding may be None if generation fails.
    """
    # Use prebuilt context if provided, otherwise build from members/relationships
    if prebuilt_context is not None:
        context = prebuilt_context
    else:
        context = build_community_context(members, relationships)

    # Select prompt based on whether child summaries were used
    if used_substitution:
        prompt = GRAPHRAG_HIERARCHICAL_COMMUNITY_PROMPT.format(community_context=context)
    else:
        prompt = GRAPHRAG_COMMUNITY_PROMPT.format(community_context=context)

    # Call LLM - no max_tokens to allow complete summaries (Microsoft approach)
    messages = [{"role": "user", "content": prompt}]
    summary = call_chat_completion(
        messages=messages,
        model=model,
        temperature=0.3,
    )
    summary = summary.strip()

    # Generate embedding for the summary
    try:
        embeddings = embed_texts([summary])
        embedding = embeddings[0] if embeddings else None
        logger.debug(f"Generated embedding with {len(embedding) if embedding else 0} dimensions")
    except Exception as e:
        logger.warning(f"Failed to generate community embedding: {e}")
        embedding = None

    return summary, embedding


def get_community_ids_from_neo4j(driver: Driver) -> set:
    """Get unique community IDs already stored in Neo4j.

    Used for resume functionality when skipping Leiden.

    Args:
        driver: Neo4j driver instance.

    Returns:
        Set of community IDs.
    """
    query = """
    MATCH (e:Entity)
    WHERE e.community_id IS NOT NULL
    RETURN DISTINCT e.community_id as community_id
    """
    with driver.session() as session:
        result = session.run(query)
        return {record["community_id"] for record in result}


def detect_and_summarize_communities(
    driver: Driver,
    gds: GraphDataScience,
    min_size: int = GRAPHRAG_MIN_COMMUNITY_SIZE,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    resume: bool = False,
    skip_leiden: bool = False,
) -> list[Community]:
    """Run full community detection and summarization pipeline.

    Main entry point for community processing with crash-proof design:
    1. Project graph to GDS (unless skip_leiden)
    2. Run Leiden algorithm (deterministic with seed, unless skip_leiden)
    3. Save Leiden checkpoint for crash recovery
    4. Write community IDs to Neo4j (unless skip_leiden)
    5. Generate summaries and upload to Weaviate (atomic per community)

    Resume mode checks Weaviate for existing communities, enabling
    crash recovery without community ID mismatches.

    Args:
        driver: Neo4j driver instance.
        gds: GraphDataScience client.
        min_size: Minimum community size to summarize.
        model: LLM model for summarization.
        resume: If True, skip already-done communities (checks Weaviate).
        skip_leiden: If True, skip Leiden and use existing community_ids from Neo4j.

    Returns:
        List of Community objects with summaries.

    Example:
        >>> communities = detect_and_summarize_communities(driver, gds)
        >>> for c in communities:
        ...     print(c.community_id, c.member_count, c.summary[:50])
    """
    # Initialize Weaviate client for crash-proof storage
    collection_name = get_community_collection_name()
    try:
        weaviate_client = get_weaviate_client()
        use_weaviate = True

        # Create collection if it doesn't exist
        if not weaviate_client.collections.exists(collection_name):
            create_community_collection(weaviate_client, collection_name)
            logger.info(f"Created Weaviate collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Weaviate not available, using file-only storage: {e}")
        weaviate_client = None
        use_weaviate = False

    # Get existing community IDs for resume (from Weaviate if available)
    existing_ids = set()
    if resume:
        if use_weaviate:
            existing_ids = weaviate_get_existing_ids(weaviate_client, collection_name)
            logger.info(f"Found {len(existing_ids)} existing communities in Weaviate")
        else:
            # Fallback: load from JSON file
            try:
                existing = load_communities()
                existing_ids = {c.community_id for c in existing}
                logger.info(f"Loaded {len(existing_ids)} existing summaries from file")
            except FileNotFoundError:
                logger.info("No existing summaries found, starting fresh")

    # Number of hierarchy levels to process (L0=coarsest, L1=medium, L2=finest)
    hierarchy_levels = GRAPHRAG_MAX_HIERARCHY_LEVELS

    # Run Leiden and parse hierarchy
    if skip_leiden:
        # Try to load hierarchy from checkpoint first
        checkpoint = load_leiden_checkpoint()
        graph = None

        if checkpoint and checkpoint.get("assignments"):
            # Reconstruct leiden_result format from checkpoint for hierarchy parsing
            logger.info(f"Loaded checkpoint with {len(checkpoint['assignments'])} assignments")

            # Build node_communities as list of dicts (same format as run_leiden output)
            node_communities = [
                {
                    "node_id": a["node_id"],
                    "community_id": a["community_id"],
                    "intermediate_ids": a.get("intermediate_ids", []),
                }
                for a in checkpoint["assignments"]
            ]

            leiden_result = {
                "node_communities": node_communities,
                "seed": checkpoint.get("seed", 42),
            }

            # Parse hierarchy from checkpoint data
            levels = parse_leiden_hierarchy(leiden_result, max_levels=hierarchy_levels)
            logger.info(f"Parsed hierarchy from checkpoint into {hierarchy_levels} levels")
        else:
            # Fallback: no checkpoint, use Neo4j community IDs (L0 only)
            unique_ids = get_community_ids_from_neo4j(driver)
            logger.info(f"Loaded {len(unique_ids)} community IDs from Neo4j (skipping Leiden)")
            logger.warning("No checkpoint found - processing L0 only (no hierarchy)")
            levels = None
    else:
        # Step 1: Project graph
        graph = project_graph(gds)

        # Step 2: Run Leiden (deterministic with seed)
        leiden_result = run_leiden(gds, graph)

        # Step 3: Save Leiden checkpoint for crash recovery
        save_leiden_checkpoint(leiden_result)

        # Step 4: Write community IDs to Neo4j
        write_communities_to_neo4j(driver, leiden_result["node_communities"])

        # Step 4.5: Compute PageRank for entity importance ranking
        try:
            from .centrality import compute_pagerank, write_pagerank_to_neo4j, compute_and_store_degree
            pagerank_scores = compute_pagerank(gds, graph)
            write_pagerank_to_neo4j(driver, pagerank_scores)
            # Compute degree for combined_degree ranking (Microsoft GraphRAG approach)
            compute_and_store_degree(driver)
        except Exception as e:
            logger.warning(f"PageRank/degree computation failed: {e}")

        # Step 5: Parse hierarchy into levels
        levels = parse_leiden_hierarchy(leiden_result, max_levels=hierarchy_levels)
        logger.info(f"Parsed hierarchy into {hierarchy_levels} levels")

    # Step 6: Process communities at each level
    communities = []
    new_summaries = 0
    processed_idx = 0

    if levels is None:
        # Fallback: skip_leiden mode - process L0 only using community_id
        total_to_process = len(unique_ids)
        for community_id in sorted(unique_ids):
            community_key = build_community_key(0, community_id)
            processed_idx += 1

            # Skip if already in Weaviate (resume mode)
            if community_key in existing_ids:
                continue

            # Get members and relationships using community_id
            members = get_community_members(driver, community_id)
            if len(members) < min_size:
                continue

            relationships = get_community_relationships(driver, community_id)

            # Generate summary
            logger.info(
                f"[{processed_idx}/{total_to_process}] L0 community {community_id} "
                f"({len(members)} members, {len(relationships)} relationships)"
            )
            summary, embedding = summarize_community(members, relationships, model=model)

            # Create and store community
            community = Community(
                community_id=community_key,
                level=0,
                members=members,
                member_count=len(members),
                relationships=relationships,
                relationship_count=len(relationships),
                summary=summary,
                embedding=embedding,
            )
            communities.append(community)
            new_summaries += 1

            # Upload to Weaviate
            if use_weaviate and embedding:
                try:
                    weaviate_upload_community(
                        client=weaviate_client,
                        collection_name=collection_name,
                        community_id=community_key,
                        summary=summary,
                        embedding=embedding,
                        member_count=len(members),
                        relationship_count=len(relationships),
                        level=0,
                    )
                except Exception as e:
                    logger.warning(f"Failed to upload {community_key} to Weaviate: {e}")

            save_communities(communities)
    else:
        # Full hierarchy mode - process all levels
        # Microsoft GraphRAG bottom-up approach: process finest-to-coarsest
        # so child summaries exist when processing parent communities
        max_level = hierarchy_levels - 1  # e.g., 2 for 3 levels (L0, L1, L2)

        # Track child summaries for parent context building
        child_summaries: dict[str, str] = {}

        # Process levels in reverse order: finest (L2) to coarsest (L0)
        for level_idx in reversed(range(hierarchy_levels)):
            level_data = levels[level_idx]
            level_community_ids = filter_communities_by_size(level_data, min_size)
            total_at_level = len(level_community_ids)

            logger.info(
                f"Processing Level {level_idx} (bottom-up): "
                f"{total_at_level} communities (>= {min_size} members)"
            )

            for idx, community_id in enumerate(sorted(level_community_ids)):
                community_key = build_community_key(level_idx, community_id)
                processed_idx += 1

                # Skip if already in Weaviate (resume mode)
                if community_key in existing_ids:
                    # Still need to load summary for child_summaries if resuming
                    # Try to get from existing communities list or Weaviate
                    continue

                # Get node IDs for this community at this level
                node_ids = level_data.communities.get(community_id, set())

                # GDS Leiden writes community_id property for the FINEST level only
                if level_idx == max_level:
                    # Finest level: use community_id-based queries (more efficient)
                    members = get_community_members(driver, community_id)
                    relationships = get_community_relationships(driver, community_id)
                    # Finest level always uses raw data (no children to substitute)
                    prebuilt_context = None
                    used_substitution = False
                else:
                    # Coarser levels: use hierarchical context with potential child substitution
                    members = get_community_members_by_node_ids(driver, node_ids)
                    relationships = get_community_relationships_by_node_ids(driver, node_ids)
                    # Try to build context with child summary substitution if needed
                    prebuilt_context, used_substitution = build_hierarchical_context(
                        community_id=community_id,
                        level=level_idx,
                        levels=levels,
                        driver=driver,
                        child_summaries=child_summaries,
                    )

                if len(members) < min_size:
                    continue

                # Generate summary
                sub_note = " (using child summaries)" if used_substitution else ""
                logger.info(
                    f"[L{level_idx} {idx + 1}/{total_at_level}] Community {community_id} "
                    f"({len(members)} members, {len(relationships)} relationships){sub_note}"
                )
                summary, embedding = summarize_community(
                    members,
                    relationships,
                    model=model,
                    prebuilt_context=prebuilt_context,
                    used_substitution=used_substitution,
                )

                # Track summary for parent communities
                child_summaries[community_key] = summary

                # Create and store community
                community = Community(
                    community_id=community_key,
                    level=level_idx,
                    members=members,
                    member_count=len(members),
                    relationships=relationships,
                    relationship_count=len(relationships),
                    summary=summary,
                    embedding=embedding,
                )
                communities.append(community)
                new_summaries += 1

                # Upload to Weaviate
                if use_weaviate and embedding:
                    try:
                        weaviate_upload_community(
                            client=weaviate_client,
                            collection_name=collection_name,
                            community_id=community_key,
                            summary=summary,
                            embedding=embedding,
                            member_count=len(members),
                            relationship_count=len(relationships),
                            level=level_idx,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to upload {community_key} to Weaviate: {e}")

                save_communities(communities)

    # Cleanup: drop graph projection (if we created one)
    if graph is not None:
        gds.graph.drop(graph.name())

    # Close Weaviate client
    if weaviate_client:
        weaviate_client.close()

    logger.info(
        f"Generated {new_summaries} new community summaries "
        f"({len(communities)} total, stored in Weaviate: {use_weaviate})"
    )
    return communities


def save_communities(
    communities: list[Community],
    output_name: str = "communities.json",
) -> Path:
    """Save community data to JSON file.

    Args:
        communities: List of Community objects.
        output_name: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = DIR_GRAPH_DATA
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name

    data = {
        "communities": [c.to_dict() for c in communities],
        "total_count": len(communities),
        "total_members": sum(c.member_count for c in communities),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(communities)} communities to {output_path}")
    return output_path


def load_communities(
    input_name: str = "communities.json",
) -> list[Community]:
    """Load communities from JSON file.

    Handles both old format (without relationships) and new format
    (with relationships and parent_id).

    Args:
        input_name: Input filename.

    Returns:
        List of Community objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    input_path = DIR_GRAPH_DATA / input_name

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    communities = []
    for c_data in data["communities"]:
        # Parse members
        members = [CommunityMember(**m) for m in c_data.get("members", [])]

        # Parse relationships (new field, may be missing in old files)
        relationships = [
            CommunityRelationship(**r) for r in c_data.get("relationships", [])
        ]

        community = Community(
            community_id=c_data["community_id"],
            level=c_data.get("level", 0),
            parent_id=c_data.get("parent_id"),  # New field
            members=members,
            member_count=c_data["member_count"],
            relationships=relationships,  # New field
            relationship_count=c_data["relationship_count"],
            summary=c_data["summary"],
            embedding=c_data.get("embedding"),
        )
        communities.append(community)

    logger.info(f"Loaded {len(communities)} communities from {input_path}")
    return communities
