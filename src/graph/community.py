"""Community summarization for GraphRAG.

## RAG Theory: Community Summarization

After Leiden community detection (see leiden.py), this module:
1. Queries Neo4j for community members and relationships
2. Generates LLM summaries for each community
3. Uploads summaries + embeddings to Weaviate for retrieval

These community summaries enable "global queries" that synthesize
information across multiple documents (e.g., "What are the main themes?").

## Data Flow

1. Leiden results (from leiden.py) -> Community assignments
2. For each community: Collect members -> Generate LLM summary
3. Store summaries in Weaviate and/or JSON for retrieval
"""

from typing import Any, Optional
from pathlib import Path
import json

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_MIN_COMMUNITY_SIZE,
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
from src.rag_pipeline.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_client import (
    get_client as get_weaviate_client,
    create_community_collection,
    upload_community as weaviate_upload_community,
    get_existing_community_ids as weaviate_get_existing_ids,
)
from .schemas import Community, CommunityMember, CommunityRelationship
from .neo4j_client import get_gds_client
from .leiden import (
    project_graph,
    run_leiden,
    save_leiden_checkpoint,
    load_leiden_checkpoint,
    write_communities_to_neo4j,
)
from .hierarchy import (
    parse_leiden_hierarchy,
    build_community_key,
    filter_communities_by_size,
    CommunityLevel,
)

logger = setup_logging(__name__)

# Maximum cross-community relationships to include in hierarchical context
# Keeps context manageable without truncating important intra-community data
MAX_CROSS_COMMUNITY_RELS = 20


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
            for rel in cross_rels[:MAX_CROSS_COMMUNITY_RELS]:
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


# ============================================================================
# COMMUNITY PROCESSING - HELPER FUNCTIONS
# ============================================================================


def _init_weaviate_storage(
    collection_name: str,
) -> tuple[Any, bool]:
    """Initialize Weaviate client for community storage.

    Args:
        collection_name: Name of the Weaviate collection.

    Returns:
        Tuple of (weaviate_client, use_weaviate_flag).
        weaviate_client may be None if Weaviate is unavailable.
    """
    try:
        weaviate_client = get_weaviate_client()
        use_weaviate = True

        if not weaviate_client.collections.exists(collection_name):
            create_community_collection(weaviate_client, collection_name)
            logger.info(f"Created Weaviate collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Weaviate not available, using file-only storage: {e}")
        weaviate_client = None
        use_weaviate = False

    return weaviate_client, use_weaviate


def _get_existing_ids_for_resume(
    resume: bool,
    use_weaviate: bool,
    weaviate_client: Any,
    collection_name: str,
) -> set:
    """Get existing community IDs for resume mode.

    Args:
        resume: Whether resume mode is enabled.
        use_weaviate: Whether Weaviate is available.
        weaviate_client: Weaviate client instance.
        collection_name: Name of the Weaviate collection.

    Returns:
        Set of existing community IDs to skip.

    Raises:
        RuntimeError: If resume mode is requested but Weaviate is unavailable.
    """
    if not resume:
        return set()

    if not use_weaviate:
        raise RuntimeError(
            "Resume mode requires Weaviate but it is unavailable. "
            "Ensure Weaviate is running: docker compose up -d weaviate"
        )

    existing_ids = weaviate_get_existing_ids(weaviate_client, collection_name)
    logger.info(f"Found {len(existing_ids)} existing communities in Weaviate")
    return existing_ids


def _run_leiden_phase(
    driver: Driver,
    gds: GraphDataScience,
    skip_leiden: bool,
    hierarchy_levels: int,
) -> tuple[Any, Optional[dict[int, "CommunityLevel"]], Optional[set]]:
    """Run Leiden algorithm and parse hierarchy.

    Args:
        driver: Neo4j driver instance.
        gds: GraphDataScience client.
        skip_leiden: If True, load from checkpoint instead of running Leiden.
        hierarchy_levels: Number of hierarchy levels to process.

    Returns:
        Tuple of (graph, levels, unique_ids).
        - graph: GDS graph projection (None if skip_leiden).
        - levels: Dict of level index to CommunityLevel (None if skip_leiden fallback).
        - unique_ids: Set of community IDs (only used if levels is None).
    """
    if skip_leiden:
        checkpoint = load_leiden_checkpoint()
        graph = None

        if checkpoint and checkpoint.get("assignments"):
            logger.info(f"Loaded checkpoint with {len(checkpoint['assignments'])} assignments")

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

            levels = parse_leiden_hierarchy(leiden_result, max_levels=hierarchy_levels)
            logger.info(f"Parsed hierarchy from checkpoint into {hierarchy_levels} levels")
            return graph, levels, None
        else:
            unique_ids = get_community_ids_from_neo4j(driver)
            logger.info(f"Loaded {len(unique_ids)} community IDs from Neo4j (skipping Leiden)")
            logger.warning("No checkpoint found - processing L0 only (no hierarchy)")
            return graph, None, unique_ids
    else:
        # Run full Leiden pipeline
        graph = project_graph(gds)
        leiden_result = run_leiden(gds, graph)
        save_leiden_checkpoint(leiden_result)
        write_communities_to_neo4j(driver, leiden_result["node_communities"])

        # Compute PageRank for entity importance ranking
        try:
            from .centrality import compute_pagerank, write_pagerank_to_neo4j, compute_and_store_degree
            pagerank_scores = compute_pagerank(gds, graph)
            write_pagerank_to_neo4j(driver, pagerank_scores)
            compute_and_store_degree(driver)
        except Exception as e:
            logger.warning(f"PageRank/degree computation failed: {e}")

        levels = parse_leiden_hierarchy(leiden_result, max_levels=hierarchy_levels)
        logger.info(f"Parsed hierarchy into {hierarchy_levels} levels")
        return graph, levels, None


def _process_single_community(
    driver: Driver,
    community_id: int,
    level_idx: int,
    max_level: int,
    level_data: "CommunityLevel",
    levels: dict[int, "CommunityLevel"],
    child_summaries: dict[str, str],
    min_size: int,
    model: str,
) -> Optional[tuple[Community, str]]:
    """Process a single community: get members, generate summary.

    Args:
        driver: Neo4j driver instance.
        community_id: Leiden community ID.
        level_idx: Current hierarchy level.
        max_level: Maximum (finest) level index.
        level_data: CommunityLevel for current level.
        levels: Dict of all levels.
        child_summaries: Dict of child community summaries for substitution.
        min_size: Minimum community size.
        model: LLM model for summarization.

    Returns:
        Tuple of (Community, summary) if successful, None if skipped.
    """
    node_ids = level_data.communities.get(community_id, set())

    # GDS Leiden writes community_id property for the FINEST level only
    if level_idx == max_level:
        members = get_community_members(driver, community_id)
        relationships = get_community_relationships(driver, community_id)
        prebuilt_context = None
        used_substitution = False
    else:
        members = get_community_members_by_node_ids(driver, node_ids)
        relationships = get_community_relationships_by_node_ids(driver, node_ids)
        prebuilt_context, used_substitution = build_hierarchical_context(
            community_id=community_id,
            level=level_idx,
            levels=levels,
            driver=driver,
            child_summaries=child_summaries,
        )

    if len(members) < min_size:
        return None

    # Generate summary
    sub_note = " (using child summaries)" if used_substitution else ""
    logger.info(
        f"[L{level_idx}] Community {community_id} "
        f"({len(members)} members, {len(relationships)} relationships){sub_note}"
    )
    summary, embedding = summarize_community(
        members,
        relationships,
        model=model,
        prebuilt_context=prebuilt_context,
        used_substitution=used_substitution,
    )

    community = Community(
        community_id=build_community_key(level_idx, community_id),
        level=level_idx,
        members=members,
        member_count=len(members),
        relationships=relationships,
        relationship_count=len(relationships),
        summary=summary,
        embedding=embedding,
    )

    return community, summary


def _upload_community_to_weaviate(
    weaviate_client: Any,
    collection_name: str,
    community: Community,
) -> None:
    """Upload a community to Weaviate with error handling.

    Serializes members and relationships as JSON strings so that
    query-time functions can reconstruct full Community objects
    from Weaviate without needing communities.json.

    Args:
        weaviate_client: Weaviate client instance.
        collection_name: Name of the collection.
        community: Community to upload.
    """
    if not community.embedding:
        logger.warning(f"Skipping Weaviate upload for {community.community_id}: no embedding")
        return

    try:
        members_json = json.dumps([m.model_dump() for m in community.members])
        relationships_json = json.dumps([r.model_dump() for r in community.relationships])

        weaviate_upload_community(
            client=weaviate_client,
            collection_name=collection_name,
            community_id=community.community_id,
            summary=community.summary,
            embedding=community.embedding,
            member_count=community.member_count,
            relationship_count=community.relationship_count,
            level=community.level,
            members_json=members_json,
            relationships_json=relationships_json,
        )
    except Exception as e:
        logger.warning(f"Failed to upload {community.community_id} to Weaviate: {e}")


# ============================================================================
# COMMUNITY PROCESSING - MAIN FUNCTION
# ============================================================================


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
    # Phase 1: Initialize storage
    collection_name = get_community_collection_name()
    weaviate_client, use_weaviate = _init_weaviate_storage(collection_name)
    existing_ids = _get_existing_ids_for_resume(
        resume, use_weaviate, weaviate_client, collection_name
    )

    # Phase 2: Run Leiden (or load from checkpoint)
    hierarchy_levels = GRAPHRAG_MAX_HIERARCHY_LEVELS
    graph, levels, unique_ids = _run_leiden_phase(
        driver, gds, skip_leiden, hierarchy_levels
    )

    # Phase 3: Process communities
    communities = []
    new_summaries = 0

    if levels is None:
        # Fallback: skip_leiden mode - process L0 only
        for community_id in sorted(unique_ids):
            community_key = build_community_key(0, community_id)

            if community_key in existing_ids:
                continue

            members = get_community_members(driver, community_id)
            if len(members) < min_size:
                continue

            relationships = get_community_relationships(driver, community_id)

            logger.info(
                f"[L0] Community {community_id} "
                f"({len(members)} members, {len(relationships)} relationships)"
            )
            summary, embedding = summarize_community(members, relationships, model=model)

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

            if use_weaviate:
                _upload_community_to_weaviate(weaviate_client, collection_name, community)
    else:
        # Full hierarchy mode - process finest-to-coarsest (Microsoft GraphRAG approach)
        max_level = hierarchy_levels - 1
        child_summaries: dict[str, str] = {}

        for level_idx in reversed(range(hierarchy_levels)):
            level_data = levels[level_idx]
            level_community_ids = filter_communities_by_size(level_data, min_size)

            logger.info(
                f"Processing Level {level_idx} (bottom-up): "
                f"{len(level_community_ids)} communities (>= {min_size} members)"
            )

            for community_id in sorted(level_community_ids):
                community_key = build_community_key(level_idx, community_id)

                if community_key in existing_ids:
                    continue

                result = _process_single_community(
                    driver, community_id, level_idx, max_level,
                    level_data, levels, child_summaries, min_size, model
                )

                if result is None:
                    continue

                community, summary = result
                child_summaries[community_key] = summary
                communities.append(community)
                new_summaries += 1

                if use_weaviate:
                    _upload_community_to_weaviate(weaviate_client, collection_name, community)

    # Cleanup
    if graph is not None:
        gds.graph.drop(graph.name())

    if weaviate_client:
        weaviate_client.close()

    logger.info(
        f"Generated {new_summaries} new community summaries "
        f"({len(communities)} total, stored in Weaviate: {use_weaviate})"
    )
    return communities
