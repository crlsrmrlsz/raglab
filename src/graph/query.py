"""Graph retrieval strategies for GraphRAG.

## RAG Theory: Graph-Based Retrieval (Microsoft GraphRAG)

GraphRAG uses two retrieval methods:
1. **Local search**: Entity matching → Graph traversal → Related chunks
   - Ranks by combined_degree (relationship hub importance)
2. **Global search**: Query → DRIFT search → Theme context
   - HNSW top-K communities → primer folds → reduce

For local queries, pure graph traversal (Microsoft's design):
- Leverages relationship structure (hub entities = more informative)
- No vector search on chunks (entity descriptions only for matching)
- Faster retrieval (single graph traversal path)

## Library Usage

Uses existing infrastructure:
- Neo4j for graph queries and entity matching
- Weaviate for chunk fetching (batch filter, not vector search)

## Data Flow (Local Queries - Graph-Only)

1. Query → Extract entity mentions (embedding similarity)
2. Match entities in Neo4j → Traverse graph → Get related chunk IDs
3. Fetch chunks from Weaviate by ID (batch filter)
4. Rank by combined_degree (Microsoft approach: hub entities = more informative)
5. Return top-k chunks

## Data Flow (Global Queries - DRIFT Search)

1. Query → LLM classifies as local or global
2. Embed query → Weaviate HNSW → top-K relevant communities
3. Primer: Parallel LLM calls over community folds → intermediate answers
4. Reduce: Single LLM call → final synthesized answer
"""

from typing import Any, Optional
import json

from neo4j import Driver

from src.config import (
    GRAPHRAG_TRAVERSE_DEPTH,
    GRAPHRAG_MAX_HIERARCHY_LEVELS,
    GRAPHRAG_SUMMARY_MODEL,
    get_community_collection_name,
    get_collection_name,
)
from src.shared.files import setup_logging
from src.rag_pipeline.indexing.weaviate_client import (
    get_client as get_weaviate_client,
    fetch_communities_by_ids,
)
from src.rag_pipeline.indexing.weaviate_query import SearchResult
from .neo4j_client import find_entity_neighbors, get_entity_community_ids
from .schemas import Community, CommunityMember, CommunityRelationship
from .query_entities import extract_query_entities

logger = setup_logging(__name__)


def _deserialize_community(data: dict[str, Any]) -> Community:
    """Deserialize a Weaviate community dict into a Community object.

    Parses the members_json and relationships_json TEXT fields
    back into CommunityMember and CommunityRelationship objects.

    Args:
        data: Dict from Weaviate query with community_id, summary,
              members_json, relationships_json, etc.

    Returns:
        Full Community object with members and relationships.
    """
    community_id = data.get("community_id", "unknown")

    try:
        members = [
            CommunityMember(**m)
            for m in json.loads(data.get("members_json", "[]"))
        ]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to deserialize members for {community_id}: {e}")
        members = []

    try:
        relationships = [
            CommunityRelationship(**r)
            for r in json.loads(data.get("relationships_json", "[]"))
        ]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to deserialize relationships for {community_id}: {e}")
        relationships = []

    return Community(
        community_id=community_id,
        level=data.get("level", 0),
        members=members,
        member_count=data.get("member_count", len(members)),
        relationships=relationships,
        relationship_count=data.get("relationship_count", len(relationships)),
        summary=data.get("summary", ""),
    )


# ============================================================================
# Graph Context Retrieval
# ============================================================================


def retrieve_graph_context(
    query: str,
    driver: Driver,
    max_hops: int = GRAPHRAG_TRAVERSE_DEPTH,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Retrieve context from knowledge graph based on query.

    Extracts entities from query, traverses graph, returns
    related entities and their source chunks.

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        max_hops: Maximum traversal depth.
        limit: Maximum results per entity.

    Returns:
        List of dicts with entity info and source_chunk_ids (list).

    Example:
        >>> context = retrieve_graph_context("What causes stress?", driver)
        >>> for c in context:
        ...     print(c["name"], c["source_chunk_ids"])
    """
    # Extract entities from query
    query_entities = extract_query_entities(query)

    if not query_entities:
        logger.debug("No entities found in query for graph traversal")
        return []

    logger.info(f"Graph traversal from entities: {query_entities}")

    # Traverse from each entity
    all_neighbors = []
    seen_names = set()

    for entity_name in query_entities:
        neighbors = find_entity_neighbors(
            driver, entity_name, max_hops=max_hops, limit=limit
        )
        for neighbor in neighbors:
            if neighbor["name"].lower() not in seen_names:
                seen_names.add(neighbor["name"].lower())
                all_neighbors.append(neighbor)

    logger.info(f"Graph traversal found {len(all_neighbors)} related entities")
    return all_neighbors


def get_chunk_ids_from_graph(
    graph_context: list[dict[str, Any]],
) -> list[str]:
    """Extract unique chunk IDs from graph context.

    Used to fetch full chunk content from Weaviate or files.

    Args:
        graph_context: List from retrieve_graph_context().

    Returns:
        List of unique chunk IDs.
    """
    chunk_ids = set()
    for entity in graph_context:
        for cid in entity.get("source_chunk_ids", []):
            if cid:
                chunk_ids.add(cid)
    return list(chunk_ids)


def fetch_chunks_by_ids(
    chunk_ids: list[str],
    collection_name: Optional[str] = None,
) -> list[SearchResult]:
    """Fetch specific chunks from Weaviate by chunk_id.

    Used to retrieve graph-discovered chunks that weren't in vector results.
    Returns SearchResult objects for RRF compatibility.

    Uses batch filtering (ContainsAny) for efficient retrieval instead of
    individual queries per chunk.

    Args:
        chunk_ids: List of chunk IDs to fetch.
        collection_name: Weaviate collection name (default: from config).

    Returns:
        List of SearchResult objects for found chunks.

    Example:
        >>> chunks = fetch_chunks_by_ids(["book::chunk_42", "book::chunk_43"])
        >>> for c in chunks:
        ...     print(c.chunk_id, c.text[:50])
    """
    from weaviate.classes.query import Filter

    if not chunk_ids:
        return []

    collection_name = collection_name or get_collection_name()
    client = get_weaviate_client()

    try:
        collection = client.collections.get(collection_name)
        results = []

        # Use batch filter (ContainsAny) for efficient retrieval
        # instead of N+1 individual queries
        try:
            response = collection.query.fetch_objects(
                filters=Filter.by_property("chunk_id").contains_any(chunk_ids),
                limit=len(chunk_ids),
            )

            for obj in response.objects:
                props = obj.properties
                results.append(
                    SearchResult(
                        chunk_id=props.get("chunk_id", ""),
                        book_id=props.get("book_id", ""),
                        section=props.get("section", ""),
                        context=props.get("context", ""),
                        text=props.get("text", ""),
                        token_count=props.get("token_count", 0),
                        score=0.0,  # No vector score for graph-only chunks
                        is_summary=props.get("is_summary", False),
                        tree_level=props.get("tree_level", 0),
                    )
                )
        except Exception as e:
            logger.warning(f"Batch fetch failed: {e}, returning empty list")

        logger.info(f"Fetched {len(results)} graph-only chunks from Weaviate")
        return results

    finally:
        client.close()


def retrieve_community_context_by_membership(
    entity_names: list[str],
    driver: Driver,
) -> list[dict[str, Any]]:
    """Retrieve community summaries by entity membership (Microsoft approach).

    Instead of embedding similarity, gets communities that CONTAIN the
    matched query entities. This is how Microsoft GraphRAG does local
    search community context.

    Queries Neo4j for community IDs, then fetches full community data
    from Weaviate by community_id filter.

    Args:
        entity_names: Entity names matched from query (already validated).
        driver: Neo4j driver instance.

    Returns:
        List of community dicts with summary, member info, and score.

    Example:
        >>> context = retrieve_community_context_by_membership(
        ...     ["dopamine", "motivation"], driver
        ... )
        >>> for c in context:
        ...     print(c["community_id"], c["member_count"])
    """
    if not entity_names:
        return []

    # Get community IDs from Neo4j for the matched entities
    community_ids = get_entity_community_ids(driver, entity_names)

    if not community_ids:
        logger.debug("No community IDs found for entities, skipping community context")
        return []

    logger.info(f"Found {len(community_ids)} communities for entities {entity_names}")

    # Build Weaviate community_id keys for the finest level
    # Entities in Neo4j store community_id from Leiden's finest level
    from .hierarchy import build_community_key

    finest_level = GRAPHRAG_MAX_HIERARCHY_LEVELS - 1
    community_keys = [
        build_community_key(finest_level, cid) for cid in community_ids
    ]

    # Fetch from Weaviate by community_id filter
    try:
        collection_name = get_community_collection_name()
        client = get_weaviate_client()
        try:
            weaviate_results = fetch_communities_by_ids(
                client=client,
                collection_name=collection_name,
                community_ids=community_keys,
            )
        finally:
            client.close()
    except Exception as e:
        logger.warning(f"Weaviate community fetch failed: {e}")
        return []

    results = [
        {
            "community_id": r["community_id"],
            "summary": r["summary"],
            "member_count": r["member_count"],
            "score": 1.0,  # Full membership match
        }
        for r in weaviate_results
    ]

    logger.info(f"Retrieved {len(results)} communities by entity membership")
    return results


def get_graph_chunk_ids(
    query: str,
    driver: Driver,
) -> tuple[list[str], dict[str, Any]]:
    """Get chunk IDs from graph traversal for a query.

    Extracts entities from query, traverses graph, and returns
    source chunk IDs from related entities. These can be used
    to boost or add chunks to vector search results.

    Args:
        query: User query string.
        driver: Neo4j driver instance.

    Returns:
        Tuple of:
        - List of chunk IDs found via graph traversal
        - Metadata dict with query_entities, extracted_entities, and graph_context

    Raises:
        neo4j.exceptions.ServiceUnavailable: If Neo4j connection fails.

    Example:
        >>> chunk_ids, meta = get_graph_chunk_ids("What is dopamine?", driver)
        >>> print(chunk_ids[:3])
        ["behave::chunk_42", "behave::chunk_43", ...]
    """
    metadata = {
        "extracted_entities": [],
        "query_entities": [],
        "graph_context": [],
    }

    # Step 1: Extract entities from query using embedding similarity
    # All returned entities exist in Neo4j (same indexing source), no validation needed
    query_entities = extract_query_entities(query)
    metadata["extracted_entities"] = query_entities
    metadata["query_entities"] = query_entities
    logger.info(f"Embedding extraction from query: {query_entities}")

    if not query_entities:
        logger.info("No entities found in query, skipping traversal")
        return [], metadata

    # Step 2: Traverse graph from matched entities
    graph_context = retrieve_graph_context(query, driver)
    metadata["graph_context"] = graph_context

    # Extract unique chunk IDs
    chunk_ids = get_chunk_ids_from_graph(graph_context)

    logger.info(
        f"Graph retrieval: {len(metadata['query_entities'])} matched entities -> "
        f"{len(graph_context)} neighbors -> {len(chunk_ids)} chunks"
    )

    return chunk_ids, metadata


def _build_graph_ranked_list(
    graph_context: list[dict[str, Any]],
    fetched_chunks: list[SearchResult],
) -> list[SearchResult]:
    """Build a ranked list of graph results ordered by combined_degree.

    Microsoft GraphRAG approach: combined_degree = start_degree + neighbor_degree
    Higher combined_degree = relationship involves hub entities = more informative.
    This list is used for RRF merging with vector results.

    Args:
        graph_context: List of entities with degree, start_degree, and source_chunk_ids.
        fetched_chunks: SearchResult objects fetched from Weaviate.

    Returns:
        List of SearchResult ordered by graph relevance (combined_degree descending).
    """
    # Build chunk_id -> maximum combined_degree mapping
    # A chunk might be reached via multiple entities; use highest combined_degree
    chunk_combined_degrees: dict[str, int] = {}
    for entity in graph_context:
        # combined_degree = start_degree + neighbor_degree (Microsoft approach)
        start_degree = entity.get("start_degree", 0)
        neighbor_degree = entity.get("degree", 0)
        combined_degree = start_degree + neighbor_degree

        for chunk_id in entity.get("source_chunk_ids", []):
            if chunk_id:
                if chunk_id not in chunk_combined_degrees:
                    chunk_combined_degrees[chunk_id] = combined_degree
                else:
                    # Use maximum combined_degree if chunk reached via multiple paths
                    chunk_combined_degrees[chunk_id] = max(
                        chunk_combined_degrees[chunk_id], combined_degree
                    )

    # Create lookup for fetched chunks
    fetched_by_id = {c.chunk_id: c for c in fetched_chunks}

    # Build ranked list: sort by combined_degree (DESCENDING - higher = better)
    ranked_chunk_ids = sorted(
        chunk_combined_degrees.keys(),
        key=lambda x: chunk_combined_degrees[x],
        reverse=True,
    )

    # Create SearchResult list in ranked order
    ranked_results = []
    for chunk_id in ranked_chunk_ids:
        if chunk_id in fetched_by_id:
            ranked_results.append(fetched_by_id[chunk_id])

    return ranked_results


def format_graph_context_for_generation(
    metadata: dict[str, Any],
    max_chars: int = 2000,
) -> str:
    """Format graph metadata as additional context for answer generation.

    Includes entity relationships and community summaries
    to augment the retrieved chunks.

    Args:
        metadata: Dict from graph_retrieval().
        max_chars: Maximum characters for context.

    Returns:
        Formatted context string for LLM prompt.
    """
    lines = []

    # Add community summaries if available
    if metadata.get("community_context"):
        lines.append("## Relevant Themes (from document corpus)")
        for comm in metadata["community_context"][:2]:  # Top 2 communities
            lines.append(f"\n{comm['summary']}")

    # Add entity relationships if available
    if metadata.get("graph_context"):
        lines.append("\n## Related Concepts (from knowledge graph)")
        for entity in metadata["graph_context"][:10]:  # Top 10 entities
            if entity.get("description"):
                lines.append(f"- {entity['name']}: {entity['description']}")
            else:
                lines.append(f"- {entity['name']} ({entity.get('entity_type', 'concept')})")

    context = "\n".join(lines)

    # Truncate if needed
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[... truncated]"

    return context


# ============================================================================
# Graph-Only Retrieval (Microsoft GraphRAG Design)
# ============================================================================


def graph_retrieval(
    query: str,
    driver: Driver,
    top_k: int = 10,
    collection_name: Optional[str] = None,
    _precomputed: Optional[tuple[list[str], dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure graph retrieval without vector search (Microsoft GraphRAG design).

    Flow:
        1. Extract entities from query (embedding similarity)
        2. Traverse graph from matched entities
        3. Fetch ALL chunks from graph traversal
        4. Rank by combined_degree (start_degree + neighbor_degree)
        5. Add community summaries (by entity membership)
        6. Return top-k chunks

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        top_k: Number of results to return.
        collection_name: Weaviate collection (default: from config).
        _precomputed: Pre-computed (graph_chunk_ids, graph_meta) from caller
            to avoid redundant entity extraction. Internal use only.

    Returns:
        Tuple of:
        - Results list (dicts with chunk data, ranked by combined_degree)
        - Metadata dict (entities, graph context, communities)

    Raises:
        neo4j.exceptions.ServiceUnavailable: If Neo4j connection fails.

    Example:
        >>> driver = get_driver()
        >>> results, meta = graph_retrieval("What is dopamine?", driver, top_k=10)
        >>> len(results) <= 10
        True
    """
    # Use pre-computed data if available, otherwise compute
    if _precomputed is not None:
        graph_chunk_ids, graph_meta = _precomputed
    else:
        graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Get community context by entity membership (Microsoft approach)
    query_entities = graph_meta.get("query_entities", [])
    community_context = retrieve_community_context_by_membership(
        entity_names=query_entities,
        driver=driver,
    )

    # Build metadata
    metadata = {
        "extracted_entities": graph_meta.get("extracted_entities", []),
        "query_entities": query_entities,
        "graph_context": graph_meta.get("graph_context", []),
        "community_context": community_context,
        "graph_chunk_count": len(graph_chunk_ids),
    }

    if not graph_chunk_ids:
        logger.info("No graph chunks found, returning empty results")
        return [], metadata

    # Fetch ALL graph-discovered chunks from Weaviate (batch filter, not vector search)
    all_graph_chunks = fetch_chunks_by_ids(graph_chunk_ids, collection_name)

    if not all_graph_chunks:
        logger.warning(f"No chunks fetched for {len(graph_chunk_ids)} graph IDs")
        return [], metadata

    # Rank by combined_degree (Microsoft approach: hub entities = more informative)
    graph_context = graph_meta.get("graph_context", [])
    ranked_results = _build_graph_ranked_list(graph_context, all_graph_chunks)

    # Convert to dicts for consistency with existing interface
    result_dicts = []
    for r in ranked_results[:top_k]:
        result_dicts.append({
            "chunk_id": r.chunk_id,
            "book_id": r.book_id,
            "section": r.section,
            "context": r.context,
            "text": r.text,
            "token_count": r.token_count,
            "similarity": r.score,
            "is_summary": r.is_summary,
            "tree_level": r.tree_level,
            "graph_found": True,
        })

    metadata["graph_fetched"] = len(all_graph_chunks)

    logger.info(
        f"Graph retrieval: {len(query_entities)} entities -> "
        f"{len(graph_chunk_ids)} chunks -> {len(result_dicts)} results"
    )

    return result_dicts, metadata


def graph_retrieval_with_drift(
    query: str,
    driver: Driver,
    top_k: int = 10,
    collection_name: Optional[str] = None,
    use_drift: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Graph-only retrieval with DRIFT search for global queries.

    Wraps graph_retrieval() with DRIFT support:
    - Local queries: Pure graph traversal ranked by combined_degree
    - Global queries: DRIFT search (top-K communities via HNSW + primer + reduce)

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        top_k: Number of results to return.
        collection_name: Weaviate collection name.
        use_drift: Whether to use DRIFT for global queries.

    Returns:
        Tuple of (results, metadata).
        For global queries, metadata includes "drift_result".
    """
    from .map_reduce import should_use_map_reduce
    from .drift import drift_search

    # Get graph context to check for global query
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Global query path: DRIFT search over top-K communities
    if use_drift and should_use_map_reduce(query):
        logger.info("Global query detected, using DRIFT search")

        drift_result = drift_search(query, model=GRAPHRAG_SUMMARY_MODEL)

        if drift_result.communities_used or drift_result.final_answer:
            metadata = {
                "extracted_entities": graph_meta.get("extracted_entities", []),
                "query_entities": graph_meta.get("query_entities", []),
                "graph_context": graph_meta.get("graph_context", []),
                "query_type": "global",
                "drift_result": {
                    "final_answer": drift_result.final_answer,
                    "intermediate_answers": drift_result.intermediate_answers,
                    "communities_used": drift_result.communities_used,
                    "community_summaries": drift_result.community_summaries,
                    "primer_time_ms": drift_result.primer_time_ms,
                    "reduce_time_ms": drift_result.reduce_time_ms,
                    "total_time_ms": drift_result.total_time_ms,
                    "total_llm_calls": drift_result.total_llm_calls,
                },
            }

            logger.info(
                f"DRIFT complete: {len(drift_result.communities_used)} communities, "
                f"{drift_result.total_llm_calls} LLM calls, "
                f"{drift_result.total_time_ms:.0f}ms total"
            )

            # Return empty results (answer in metadata["drift_result"])
            return [], metadata

        logger.warning(
            "Global query detected but DRIFT returned no results, "
            "falling back to local graph retrieval"
        )

    # Local query path: pure graph retrieval (pass pre-computed data to avoid
    # redundant entity extraction + graph traversal)
    results, metadata = graph_retrieval(
        query=query,
        driver=driver,
        top_k=top_k,
        collection_name=collection_name,
        _precomputed=(graph_chunk_ids, graph_meta),
    )
    metadata["query_type"] = "local"
    return results, metadata
