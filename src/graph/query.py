"""Graph retrieval strategies for GraphRAG.

## RAG Theory: Graph-Based Retrieval (Microsoft GraphRAG)

GraphRAG uses two retrieval methods:
1. **Local search**: Entity matching → Graph traversal → Related chunks
   - Ranks by combined_degree (relationship hub importance)
2. **Global search**: Query → Community summary matching → Theme context
   - Map-reduce over Leiden communities

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

## Data Flow (Global Queries - Map-Reduce)

1. Query → LLM classifies as local or global
2. Retrieve ALL L0 communities (coarsest level)
3. Map: Parallel LLM calls → partial answers per community
4. Reduce: Synthesize final answer
"""

from typing import Any, Optional

import numpy as np
from neo4j import Driver

from src.config import (
    GRAPHRAG_TOP_COMMUNITIES,
    GRAPHRAG_TRAVERSE_DEPTH,
    GRAPHRAG_MAX_HIERARCHY_LEVELS,
    get_community_collection_name,
    get_collection_name,
)
from src.shared.files import setup_logging
from src.rag_pipeline.embedding.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_client import (
    get_client as get_weaviate_client,
    query_communities_by_vector,
)
from src.rag_pipeline.indexing.weaviate_query import SearchResult
from .neo4j_client import find_entity_neighbors, find_entities_by_names, get_entity_community_ids
from .community import load_communities
from .schemas import Community
# Entity extraction logic moved to query_entities.py
from .query_entities import extract_query_entities

logger = setup_logging(__name__)


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
    query_entities = extract_query_entities(query, driver)

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
    Handles both old format (source_chunk_id) and new format (source_chunk_ids list).

    Args:
        graph_context: List from retrieve_graph_context().

    Returns:
        List of unique chunk IDs.
    """
    chunk_ids = set()
    for entity in graph_context:
        # Handle new format: source_chunk_ids as list
        if entity.get("source_chunk_ids"):
            for cid in entity["source_chunk_ids"]:
                if cid:
                    chunk_ids.add(cid)
        # Backward compatibility: source_chunk_id as string
        elif entity.get("source_chunk_id"):
            chunk_ids.add(entity["source_chunk_id"])
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1, 1].
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm_product)


def retrieve_community_context_by_membership(
    entity_names: list[str],
    driver: Driver,
    communities: Optional[list[Community]] = None,
) -> list[dict[str, Any]]:
    """Retrieve community summaries by entity membership (Microsoft approach).

    Instead of embedding similarity, gets communities that CONTAIN the
    matched query entities. This is how Microsoft GraphRAG does local
    search community context.

    Args:
        entity_names: Entity names matched from query (already validated).
        driver: Neo4j driver instance.
        communities: Pre-loaded communities (optional, loads from file if None).

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

    # Load communities if not provided
    if communities is None:
        try:
            communities = load_communities()
        except FileNotFoundError:
            logger.warning("No communities file found")
            return []

    # Build lookup by community_id (need to parse the key format)
    from .hierarchy import parse_community_key, build_community_key

    # Finest level = GRAPHRAG_MAX_HIERARCHY_LEVELS - 1 (e.g., L2 for 3 levels)
    # Entities in Neo4j store community_id from Leiden's finest level
    finest_level = GRAPHRAG_MAX_HIERARCHY_LEVELS - 1

    community_lookup = {}
    for c in communities:
        # community_id format: "community_L2_42" → extract (level, id)
        try:
            level, cid = parse_community_key(c.community_id)
            # Only include finest level (where entities have community_id)
            if level == finest_level:
                community_lookup[cid] = c
        except ValueError:
            continue

    # Get matching communities
    results = []
    for cid in community_ids:
        if cid in community_lookup:
            c = community_lookup[cid]
            results.append({
                "community_id": c.community_id,
                "summary": c.summary,
                "member_count": c.member_count,
                "score": 1.0,  # Full membership match
            })

    logger.info(f"Retrieved {len(results)} communities by entity membership")
    return results


def retrieve_community_context(
    query: str,
    communities: Optional[list[Community]] = None,
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> list[dict[str, Any]]:
    """Retrieve relevant community summaries using embedding similarity.

    NOTE: This is the legacy approach. For local queries, prefer
    retrieve_community_context_by_membership() which uses entity
    membership (Microsoft GraphRAG approach).

    Tries Weaviate first for efficient HNSW search, then falls back to
    in-memory file-based search if Weaviate collection doesn't exist.

    Args:
        query: User query string.
        communities: List of Community objects (only for legacy fallback).
        top_k: Number of top communities to return.

    Returns:
        List of community dicts with summary, member info, and score.

    Example:
        >>> context = retrieve_community_context("What are the main themes?")
        >>> for c in context:
        ...     print(c["summary"][:100], c["score"])
    """
    # Try Weaviate first (preferred - O(log n) HNSW search)
    try:
        collection_name = get_community_collection_name()
        client = get_weaviate_client()

        if client.collections.exists(collection_name):
            logger.debug(f"Using Weaviate community retrieval from {collection_name}")
            query_embedding = embed_texts([query])[0]

            results = query_communities_by_vector(
                client=client,
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
            )

            client.close()
            logger.info(f"Weaviate community retrieval found {len(results)} communities")
            return results
        else:
            client.close()
            logger.debug(f"Weaviate collection {collection_name} not found, using file fallback")
    except Exception as e:
        logger.warning(f"Weaviate community retrieval failed: {e}, using file fallback")

    # Fallback: file-based retrieval (legacy - O(n) loop)
    if communities is None:
        try:
            communities = load_communities()
        except FileNotFoundError:
            logger.warning("No communities file found, skipping community retrieval")
            return []

    if not communities:
        return []

    # Check if embeddings are available
    has_embeddings = any(c.embedding for c in communities)

    if has_embeddings:
        # Embedding-based retrieval (preferred)
        logger.debug("Using file-based embedding community retrieval")
        query_embedding = embed_texts([query])[0]

        scored = []
        for community in communities:
            if community.embedding:
                similarity = cosine_similarity(query_embedding, community.embedding)
                scored.append((similarity, community))

        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        # Fallback: keyword matching (legacy)
        logger.debug("No community embeddings, using keyword fallback")
        query_words = set(query.lower().split())

        scored = []
        for community in communities:
            # Count keyword matches in summary
            summary_words = set(community.summary.lower().split())
            overlap = len(query_words & summary_words)

            # Also check member names
            member_names = " ".join(m.entity_name for m in community.members).lower()
            member_overlap = sum(1 for w in query_words if w in member_names)

            score = overlap + member_overlap * 2  # Weight member matches higher

            if score > 0:
                scored.append((score, community))

        scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k
    results = []
    for score, community in scored[:top_k]:
        results.append({
            "community_id": community.community_id,
            "summary": community.summary,
            "member_count": community.member_count,
            "score": float(score),
        })

    logger.info(f"File-based community retrieval found {len(results)} communities")
    return results


def retrieve_communities_for_map_reduce(
    query: str,
    level: Optional[int] = None,
) -> list[Community]:
    """Retrieve full Community objects for map-reduce processing.

    Unlike retrieve_community_context() which returns dicts,
    this returns full Community objects with members and relationships
    for use in map-reduce global queries.

    Microsoft GraphRAG uses ALL communities at the selected level for
    global queries (map-reduce over all community reports).

    Args:
        query: User query string.
        level: Hierarchy level filter (0=coarsest for global queries).

    Returns:
        List of Community objects sorted by relevance (full data for map-reduce).
    """
    # Load communities from JSON (has full data)
    try:
        communities = load_communities()
    except FileNotFoundError:
        logger.warning("No communities file found for map-reduce")
        return []

    if not communities:
        return []

    # Filter by level if specified (L0 = coarsest for global queries)
    if level is not None:
        communities = [c for c in communities if c.level == level]
        logger.debug(f"Filtered to {len(communities)} communities at level {level}")

    if not communities:
        return []

    # Sort by relevance (embedding similarity or keyword matching)
    has_embeddings = any(c.embedding for c in communities)

    if has_embeddings:
        query_embedding = embed_texts([query])[0]

        scored = []
        for community in communities:
            if community.embedding:
                similarity = cosine_similarity(query_embedding, community.embedding)
                scored.append((similarity, community))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [community for _, community in scored]
    else:
        # Fallback: keyword matching
        query_words = set(query.lower().split())

        scored = []
        for community in communities:
            summary_words = set(community.summary.lower().split())
            overlap = len(query_words & summary_words)
            scored.append((overlap, community))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [community for _, community in scored]


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
        "extracted_entities": [],  # What LLM found in query
        "query_entities": [],      # What matched in Neo4j
        "graph_context": [],
    }

    # Step 1: Extract entities from query using embedding similarity
    # (Matches Microsoft GraphRAG: map_query_to_entities via vector search)
    extracted = extract_query_entities(query)
    metadata["extracted_entities"] = extracted
    logger.info(f"Embedding extraction from query: {extracted}")

    # Step 2: Validate against Neo4j
    if extracted and driver:
        db_entities = find_entities_by_names(driver, extracted)
        matched = [e["name"] for e in db_entities]
        metadata["query_entities"] = matched
        logger.info(f"Matched in Neo4j: {matched}")
    else:
        metadata["query_entities"] = []

    if not metadata["query_entities"]:
        logger.info("No entities matched in graph, skipping traversal")
        return [], metadata

    # Step 3: Traverse graph from matched entities
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

        # Handle new format: source_chunk_ids as list
        chunk_ids = entity.get("source_chunk_ids", [])
        if not chunk_ids:
            # Backward compatibility: source_chunk_id as string
            chunk_id = entity.get("source_chunk_id")
            chunk_ids = [chunk_id] if chunk_id else []

        for chunk_id in chunk_ids:
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
    # Get graph chunk IDs and metadata
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


def graph_retrieval_with_map_reduce(
    query: str,
    driver: Driver,
    top_k: int = 10,
    collection_name: Optional[str] = None,
    use_map_reduce: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Graph-only retrieval with optional map-reduce for global queries.

    Wraps graph_retrieval() with map-reduce support:
    - Local queries: Pure graph traversal ranked by combined_degree
    - Global queries: Map-reduce over community summaries

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        top_k: Number of results to return.
        collection_name: Weaviate collection name.
        use_map_reduce: Whether to use map-reduce for global queries.

    Returns:
        Tuple of (results, metadata).
        For global queries, metadata includes "map_reduce_result".
    """
    from .map_reduce import map_reduce_global_query, should_use_map_reduce

    # Get graph context to check for global query
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Global query path: map-reduce over communities
    if use_map_reduce and should_use_map_reduce(query):
        logger.info("Global query detected, using map-reduce")

        communities = retrieve_communities_for_map_reduce(query, level=0)

        if communities:
            mr_result = map_reduce_global_query(query, communities)

            metadata = {
                "extracted_entities": graph_meta.get("extracted_entities", []),
                "query_entities": graph_meta.get("query_entities", []),
                "graph_context": graph_meta.get("graph_context", []),
                "community_context": [
                    {"community_id": c.community_id, "summary": c.summary, "member_count": c.member_count}
                    for c in communities
                ],
                "query_type": "global",
                "map_reduce_result": {
                    "final_answer": mr_result.final_answer,
                    "communities_used": mr_result.communities_used,
                    "map_time_ms": mr_result.map_time_ms,
                    "reduce_time_ms": mr_result.reduce_time_ms,
                    "total_time_ms": mr_result.total_time_ms,
                },
            }

            logger.info(
                f"Map-reduce complete: {len(mr_result.communities_used)} communities, "
                f"{mr_result.total_time_ms:.0f}ms total"
            )

            # Return empty results (answer in metadata["map_reduce_result"])
            return [], metadata
        else:
            logger.warning(
                "Global query detected but no communities available, "
                "falling back to local graph retrieval"
            )

    # Local query path: pure graph retrieval
    results, metadata = graph_retrieval(
        query=query,
        driver=driver,
        top_k=top_k,
        collection_name=collection_name,
    )
    metadata["query_type"] = "local"
    return results, metadata
