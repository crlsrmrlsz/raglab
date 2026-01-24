"""Graph retrieval strategy for hybrid GraphRAG + vector search.

## RAG Theory: Hybrid Retrieval in GraphRAG

GraphRAG combines two retrieval methods:
1. **Local search**: Entity matching → Graph traversal → Related chunks
2. **Global search**: Query → Community summary matching → Theme context

The hybrid approach uses RRF (Reciprocal Rank Fusion) to merge:
- Vector search results (semantic similarity)
- Graph traversal results (relationship-based via chunk lookup)
- Community summaries (thematic context)

## Library Usage

Uses existing infrastructure:
- Neo4j for graph queries
- Weaviate for vector search and chunk fetching
- RRF from src/rag_pipeline/retrieval/rrf.py

## Data Flow

1. Query → Extract entity mentions (embedding similarity + LLM fallback)
2. Match entities in Neo4j → Traverse graph → Get related chunk IDs
3. Vector search in Weaviate → Get similar chunks
4. Fetch graph-only chunks from Weaviate (not in vector results)
5. RRF merge vector + graph result sets → Return top-k chunks
"""

from typing import Any, Optional

import numpy as np
from neo4j import Driver

from src.config import (
    GRAPHRAG_TOP_COMMUNITIES,
    GRAPHRAG_TRAVERSE_DEPTH,
    GRAPHRAG_RRF_K,
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
from src.rag_pipeline.retrieval.rrf import reciprocal_rank_fusion
from .neo4j_client import find_entity_neighbors, find_entities_by_names
from .community import load_communities
from .schemas import Community
# Entity extraction logic moved to query_entities.py
from .query_entities import extract_query_entities, extract_query_entities_llm

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


def retrieve_community_context(
    query: str,
    communities: Optional[list[Community]] = None,
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> list[dict[str, Any]]:
    """Retrieve relevant community summaries using embedding similarity.

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
    top_k: Optional[int] = None,
) -> list[Community]:
    """Retrieve full Community objects for map-reduce processing.

    Unlike retrieve_community_context() which returns dicts,
    this returns full Community objects with members and relationships
    for use in map-reduce global queries.

    Microsoft GraphRAG uses ALL communities at the selected level for
    global queries (map-reduce over all community reports). The top_k
    parameter is optional for performance tuning on large corpora.

    Args:
        query: User query string.
        level: Hierarchy level filter (0=coarsest for global queries).
        top_k: Optional limit on communities (None = use all, Microsoft-aligned).

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
        sorted_communities = [community for _, community in scored]
    else:
        # Fallback: keyword matching
        query_words = set(query.lower().split())

        scored = []
        for community in communities:
            summary_words = set(community.summary.lower().split())
            overlap = len(query_words & summary_words)
            scored.append((overlap, community))

        scored.sort(key=lambda x: x[0], reverse=True)
        sorted_communities = [community for _, community in scored]

    # Apply top_k limit only if specified (Microsoft uses all for global queries)
    if top_k is not None:
        return sorted_communities[:top_k]
    return sorted_communities


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

    # Step 1: Extract entities from query using LLM
    extracted = extract_query_entities_llm(query)
    metadata["extracted_entities"] = extracted
    logger.info(f"LLM extracted from query: {extracted}")

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
    """Build a ranked list of graph results ordered by path_length.

    Shorter path_length = closer to query entity = higher rank.
    This list is used for RRF merging with vector results.

    Args:
        graph_context: List of entities with path_length and source_chunk_ids (list).
        fetched_chunks: SearchResult objects fetched from Weaviate.

    Returns:
        List of SearchResult ordered by graph relevance (path_length ascending).
    """
    # Build chunk_id -> minimum path_length mapping
    # A chunk might be reached via multiple entities; use shortest path
    chunk_path_lengths: dict[str, int] = {}
    for entity in graph_context:
        path_len = entity.get("path_length", 999)

        # Handle new format: source_chunk_ids as list
        chunk_ids = entity.get("source_chunk_ids", [])
        if not chunk_ids:
            # Backward compatibility: source_chunk_id as string
            chunk_id = entity.get("source_chunk_id")
            chunk_ids = [chunk_id] if chunk_id else []

        for chunk_id in chunk_ids:
            if chunk_id:
                if chunk_id not in chunk_path_lengths:
                    chunk_path_lengths[chunk_id] = path_len
                else:
                    chunk_path_lengths[chunk_id] = min(chunk_path_lengths[chunk_id], path_len)

    # Create lookup for fetched chunks
    fetched_by_id = {c.chunk_id: c for c in fetched_chunks}

    # Build ranked list: sort by path_length (ascending)
    ranked_chunk_ids = sorted(chunk_path_lengths.keys(), key=lambda x: chunk_path_lengths[x])

    # Create SearchResult list in ranked order
    ranked_results = []
    for chunk_id in ranked_chunk_ids:
        if chunk_id in fetched_by_id:
            ranked_results.append(fetched_by_id[chunk_id])

    return ranked_results


def hybrid_graph_retrieval(
    query: str,
    driver: Driver,
    vector_results: list[dict[str, Any]],
    top_k: int = 10,
    collection_name: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge vector search results with graph traversal using RRF.

    Enhances vector search with knowledge graph context:
    1. Traverses graph from query entities to find related chunk IDs
    2. Fetches ALL graph-discovered chunks from Weaviate
    3. Ranks graph chunks by path_length (closer to query entity = higher rank)
    4. Uses RRF to merge vector list + graph list (chunks in both get boosted)
    5. Adds community summaries for thematic context

    The key insight: RRF boosts chunks that appear in BOTH lists. A chunk
    with vector_rank=5 and graph_rank=3 gets score = 1/(k+5) + 1/(k+3),
    higher than a chunk appearing in only one list.

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        vector_results: Results from Weaviate vector search (list of dicts).
        top_k: Number of results to return.
        collection_name: Weaviate collection (default: from config).

    Returns:
        Tuple of:
        - RRF-merged results (vector + graph sources combined)
        - Metadata dict with entities, graph context, and communities

    Example:
        >>> driver = get_driver()
        >>> results, meta = hybrid_graph_retrieval("What is dopamine?", driver, vector_results)
        >>> print(meta["query_entities"])
    """
    # Get graph chunk IDs and metadata
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Get community context for thematic enrichment
    community_context = retrieve_community_context(query)

    # Build metadata
    metadata = {
        "extracted_entities": graph_meta.get("extracted_entities", []),
        "query_entities": graph_meta.get("query_entities", []),
        "graph_context": graph_meta.get("graph_context", []),
        "community_context": community_context,
        "graph_chunk_count": len(graph_chunk_ids),
    }

    if not graph_chunk_ids:
        # No graph results, return vector results as-is
        logger.info("No graph chunks found, returning vector results only")
        return vector_results[:top_k], metadata

    # Convert vector_results (dicts) to SearchResult objects for RRF
    vector_chunk_ids = {r.get("chunk_id") for r in vector_results if r.get("chunk_id")}
    vector_search_results = []
    for r in vector_results:
        vector_search_results.append(
            SearchResult(
                chunk_id=r.get("chunk_id", ""),
                book_id=r.get("book_id", ""),
                section=r.get("section", ""),
                context=r.get("context", ""),
                text=r.get("text", ""),
                token_count=r.get("token_count", 0),
                score=r.get("similarity", 0.0),
                is_summary=r.get("is_summary", False),
                tree_level=r.get("tree_level", 0),
            )
        )

    # Fetch ALL graph-discovered chunks from Weaviate (not just graph-only)
    # This enables proper RRF boosting for chunks in both lists
    all_graph_chunks = fetch_chunks_by_ids(graph_chunk_ids, collection_name)

    if all_graph_chunks:
        # Build graph-ranked list ordered by path_length (shorter = higher rank)
        graph_context = graph_meta.get("graph_context", [])
        graph_ranked_results = _build_graph_ranked_list(graph_context, all_graph_chunks)

        # RRF merge: chunks in BOTH lists get boosted
        # e.g., chunk at vector_rank=5, graph_rank=3 gets score = 1/(k+5) + 1/(k+3)
        rrf_result = reciprocal_rank_fusion(
            result_lists=[vector_search_results, graph_ranked_results],
            query_types=["vector", "graph"],
            k=GRAPHRAG_RRF_K,
            top_k=top_k,
        )
        merged_results = rrf_result.results

        # Count overlaps for logging
        graph_chunk_set = set(graph_chunk_ids)
        overlap_count = sum(1 for r in vector_search_results if r.chunk_id in graph_chunk_set)
        graph_only_count = len(graph_chunk_set - vector_chunk_ids)

        logger.info(
            f"RRF merge: {len(vector_search_results)} vector + "
            f"{len(graph_ranked_results)} graph (path-ranked) -> "
            f"{len(merged_results)} results ({overlap_count} overlapping, {graph_only_count} graph-only)"
        )
    else:
        # No graph chunks fetched (Weaviate empty or error), use vector results
        merged_results = vector_search_results[:top_k]
        graph_chunk_set = set(graph_chunk_ids)
        overlap_count = 0
        graph_only_count = len(graph_chunk_ids)
        logger.info(f"No graph chunks fetched, using {len(merged_results)} vector results")

    # Mark which results came from graph traversal (for visibility/debugging)
    merged_dicts = []
    for r in merged_results:
        result_dict = {
            "chunk_id": r.chunk_id,
            "book_id": r.book_id,
            "section": r.section,
            "context": r.context,
            "text": r.text,
            "token_count": r.token_count,
            "similarity": r.score,  # Now contains RRF score
            "is_summary": r.is_summary,
            "tree_level": r.tree_level,
        }
        # Mark if this chunk was found via graph traversal
        if r.chunk_id in graph_chunk_set:
            result_dict["graph_found"] = True
        # Mark if this was ONLY in graph results (not in original vector results)
        if r.chunk_id not in vector_chunk_ids:
            result_dict["graph_only"] = True
        merged_dicts.append(result_dict)

    # Update metadata
    metadata["overlap_count"] = overlap_count
    metadata["graph_only_count"] = graph_only_count
    metadata["graph_fetched"] = len(all_graph_chunks) if all_graph_chunks else 0
    metadata["rrf_merged"] = bool(all_graph_chunks)

    logger.info(
        f"Hybrid retrieval: {overlap_count} in both lists (RRF boosted), "
        f"{graph_only_count} graph-only, {metadata['graph_fetched']} total fetched"
    )

    return merged_dicts, metadata


def hybrid_graph_retrieval_with_map_reduce(
    query: str,
    driver: Driver,
    vector_results: list[dict[str, Any]],
    top_k: int = 10,
    collection_name: Optional[str] = None,
    use_map_reduce: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enhanced hybrid retrieval with optional map-reduce for global queries.

    Extends hybrid_graph_retrieval with map-reduce support:
    - For local queries: Standard RRF merge (entity traversal + vector)
    - For global queries: Map-reduce over community summaries

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        vector_results: Results from Weaviate vector search.
        top_k: Number of results to return.
        collection_name: Weaviate collection name.
        use_map_reduce: Whether to use map-reduce for global queries.

    Returns:
        Tuple of (results, metadata).
        For global queries with map-reduce, metadata includes "map_reduce_result".
    """
    from .map_reduce import classify_query, map_reduce_global_query, should_use_map_reduce

    # First, get graph context regardless of query type
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)
    extracted_entities = graph_meta.get("extracted_entities", [])

    # Determine if this is a global query that should use map-reduce
    if use_map_reduce and should_use_map_reduce(query, extracted_entities):
        logger.info("Global query detected, using map-reduce")

        # Retrieve ALL L0 (coarsest) communities for global query map-reduce
        # Microsoft GraphRAG: map-reduce over all community reports at selected level
        communities = retrieve_communities_for_map_reduce(
            query,
            level=0,  # L0 = coarsest level for global/abstract queries
            # top_k=None uses all communities (Microsoft-aligned)
        )

        if communities:
            # Run map-reduce
            mr_result = map_reduce_global_query(query, communities)

            # Build metadata with map-reduce info
            metadata = {
                "extracted_entities": extracted_entities,
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

            # Still return vector results for reference
            return vector_results[:top_k], metadata

    # Fall back to standard hybrid retrieval for local queries
    results, metadata = hybrid_graph_retrieval(
        query=query,
        driver=driver,
        vector_results=vector_results,
        top_k=top_k,
        collection_name=collection_name,
    )
    metadata["query_type"] = "local"
    return results, metadata


def format_graph_context_for_generation(
    metadata: dict[str, Any],
    max_chars: int = 2000,
) -> str:
    """Format graph metadata as additional context for answer generation.

    Includes entity relationships and community summaries
    to augment the retrieved chunks.

    Args:
        metadata: Dict from hybrid_graph_retrieval().
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
