"""Weaviate search service for RAG UI.

Provides semantic search with optional cross-encoder reranking.
This module wraps the vector_db query functions for use in the Streamlit UI.

## Architecture: Strategy Pattern for Retrieval

The search service uses the RetrievalStrategy pattern to handle different
preprocessing strategies (none, hyde, decomposition, graphrag). Each strategy
encapsulates its own retrieval logic:

- **StandardRetrieval**: Direct Weaviate search (no preprocessing)
- **HyDERetrieval**: Generate hypotheticals → average embeddings → single search
- **DecompositionRetrieval**: Break into sub-questions → parallel searches → union → rerank
- **GraphRAGRetrieval**: Extract entities → graph traversal → combined_degree ranking

This eliminates the confusing conditional logic that previously checked
`if strategy == "hyde" and multi_queries and len(multi_queries) > 1:`.

## Search Types

1. **Vector (Semantic)**: Uses embedding similarity to find related content.
   Best for conceptual queries like "What is consciousness?"

2. **Hybrid (Vector + Keyword)**: Combines embedding similarity with BM25 keyword
   matching. Best for queries with specific terms like "thalamocortical hub".

## Reranking

Optional cross-encoder reranking improves result quality by processing query
and document together. The cross-encoder sees both texts simultaneously,
enabling deeper semantic understanding than bi-encoders (embeddings).

Two-Stage Retrieval:
1. Fast bi-encoder retrieves top-50 candidates
2. Slow cross-encoder reranks to top-k with higher accuracy
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import streamlit as st

from src.config import (
    get_collection_name,
    DEFAULT_TOP_K,
    RERANK_INITIAL_K,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)
from src.rag_pipeline.indexing import get_client
from src.rag_pipeline.retrieval.reranking import apply_reranking_with_metadata

# Import strategy pattern components
from src.rag_pipeline.retrieval.strategy_registry import RetrievalContext, RetrievalResult, get_strategy


@dataclass
class SearchOutput:
    """Result of search operation including optional rerank and RRF data.

    Attributes:
        results: List of chunk dictionaries.
        rerank_data: If reranking was used, contains RerankResult for logging.
        rrf_data: If RRF merging was used, contains RRFResult for logging.
        graph_metadata: If GraphRAG was used, contains entity/community info.
    """
    results: list[dict[str, Any]] = field(default_factory=list)
    rerank_data: Optional[Any] = None  # RerankResult when reranking is used
    rrf_data: Optional[Any] = None  # RRFResult when multi-query is used
    graph_metadata: Optional[dict[str, Any]] = None  # GraphRAG metadata


def search_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = "vector",
    alpha: float = 0.5,
    collection_name: Optional[str] = None,
    use_reranking: bool = False,
    strategy: Optional[str] = None,
) -> SearchOutput:
    """
    Search Weaviate for relevant chunks using strategy-based retrieval.

    This is the main search function for the UI. It uses the RetrievalStrategy
    pattern to handle different preprocessing strategies, eliminating the
    confusing conditional logic that previously checked multi_queries.

    Args:
        query: User's search query.
        top_k: Number of results to return.
        search_type: Either "vector" (semantic) or "hybrid" (vector + keyword).
        alpha: For hybrid search, balance between vector (1.0) and keyword (0.0).
        collection_name: Override collection (for future multi-collection).
        use_reranking: If True, apply cross-encoder reranking for better accuracy.
        strategy: Preprocessing strategy name ("none", "hyde", "decomposition", "graphrag").

    Returns:
        SearchOutput with results list and optional rerank_data/rrf_data/graph_metadata.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is not running.

    Example:
        >>> # Basic hybrid search
        >>> output = search_chunks("What is consciousness?", search_type="hybrid")
        >>> results = output.results
        >>>
        >>> # With HyDE strategy (embedding averaging)
        >>> output = search_chunks("What is consciousness?", strategy="hyde")
        >>>
        >>> # With GraphRAG strategy
        >>> output = search_chunks("What is dopamine?", strategy="graphrag")
        >>> print(output.graph_metadata["query_entities"])
    """
    collection_name = collection_name or get_collection_name()
    strategy_id = strategy or "none"

    logger.info(f"[search_chunks] strategy={strategy_id}, collection={collection_name}")

    # Get Neo4j driver if needed for GraphRAG
    neo4j_driver = None
    if strategy_id == "graphrag":
        try:
            from src.graph.neo4j_client import get_driver
            neo4j_driver = get_driver()
        except Exception as e:
            logger.warning(f"[search_chunks] Neo4j driver unavailable for GraphRAG: {e}")

    # Build retrieval context
    client = get_client()
    initial_k = RERANK_INITIAL_K if use_reranking else top_k

    context = RetrievalContext(
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        use_reranking=use_reranking,
        initial_k=initial_k,
        alpha=alpha,
        search_type=search_type,
        neo4j_driver=neo4j_driver,
    )

    try:
        # Get strategy instance and execute (polymorphic dispatch)
        retrieval_strategy = get_strategy(strategy_id)
        result = retrieval_strategy.execute(query, context)

        # Convert SearchResult objects to dicts for Streamlit
        result_dicts = [
            {
                "chunk_id": r.chunk_id,
                "book_id": r.book_id,
                "section": r.section,
                "context": r.context,
                "text": r.text,
                "token_count": r.token_count,
                "similarity": r.score,
                "is_summary": r.is_summary,
                "tree_level": r.tree_level,
            }
            for r in result.results
        ]

        # Extract metadata from strategy result
        rerank_data = result.metadata.get("rerank_data")
        rrf_data = result.metadata.get("rrf_data")
        graph_metadata = result.metadata.get("graph_metadata")

        return SearchOutput(
            results=result_dicts,
            rerank_data=rerank_data,
            rrf_data=rrf_data,
            graph_metadata=graph_metadata,
        )

    finally:
        client.close()
        if neo4j_driver:
            neo4j_driver.close()


def list_collections() -> list[str]:
    """
    List all available RAG collections in Weaviate.

    Returns:
        List of collection names starting with 'RAG_'.
    """
    client = get_client()

    try:
        all_collections = client.collections.list_all()
        return sorted([name for name in all_collections.keys() if name.startswith("RAG_")])
    finally:
        client.close()


# ============================================================================
# COLLECTION METADATA ENRICHMENT
# ============================================================================

from src.config import get_strategy_metadata, StrategyMetadata


@dataclass
class CollectionInfo:
    """Enriched collection metadata for UI display.

    Attributes:
        collection_name: Full Weaviate collection name (e.g., "RAG_section_embed3large_v1").
        strategy: Strategy key extracted from collection name (e.g., "section").
        display_name: Human-readable name for UI (e.g., "Section-Based Chunking").
        description: Short description of the strategy.
        is_available: Whether the collection exists in Weaviate.
    """
    collection_name: str
    strategy: str
    display_name: str
    description: str
    is_available: bool


def extract_strategy_from_collection(collection_name: str) -> str:
    """
    Extract strategy key from collection name.

    Args:
        collection_name: Collection name like "RAG_section_embed3large_v1"
                        or "RAG_semantic_0.5_embed3large_v1".

    Returns:
        Strategy key like "section" or "semantic_0.5".

    Example:
        >>> extract_strategy_from_collection("RAG_section_embed3large_v1")
        'section'
        >>> extract_strategy_from_collection("RAG_semantic_0.5_embed3large_v1")
        'semantic_0.5'
        >>> extract_strategy_from_collection("RAG_contextual_embed3large_v1")
        'contextual'
    """
    # Format: RAG_{strategy}_{model}_v{version}
    # Strategy can be: "section", "contextual", "semantic_0.5", "semantic_0.75"
    if not collection_name.startswith("RAG_"):
        return "unknown"

    # Remove RAG_ prefix
    rest = collection_name[4:]

    # Find the model suffix pattern (embed3large or similar)
    # Strategy ends before the model suffix
    parts = rest.split("_")

    if len(parts) < 3:
        return "unknown"

    # Check for semantic variants
    if parts[0] == "semantic" and len(parts) >= 2:
        # Handle semantic_stdX pattern (e.g., "semantic_std2_embed3large_v1")
        if parts[1].startswith("std") and parts[1][3:].isdigit():
            return f"{parts[0]}_{parts[1]}"  # "semantic_std2"

        # Handle semantic_X_Y pattern (threshold X.Y was sanitized to X_Y)
        # Collection names replace "." with "_", so "semantic_0.5" becomes "semantic_0_5"
        if len(parts) >= 3:
            try:
                threshold = f"{parts[1]}.{parts[2]}"
                float(threshold)  # Validate it's a valid number
                return f"{parts[0]}_{threshold}"  # "semantic_0.5"
            except ValueError:
                pass  # Not a valid threshold pattern

    # Single-part strategy
    return parts[0]


@st.cache_data(ttl=60)
def get_available_collections() -> list[CollectionInfo]:
    """
    List all RAG collections with enriched metadata.

    Queries Weaviate for available collections, then enriches each with
    metadata from the strategy registry for UI display.

    Returns:
        List of CollectionInfo objects with display names and descriptions.

    Example:
        >>> collections = get_available_collections()
        >>> for c in collections:
        ...     print(f"{c.display_name}: {c.description}")
        Section-Based Chunking: Preserves document structure with sentence overlap
        Contextual Chunking: LLM-generated context prepended (+35% improvement)
    """
    existing_collections = list_collections()

    collection_infos = []
    for coll_name in existing_collections:
        strategy = extract_strategy_from_collection(coll_name)
        metadata = get_strategy_metadata(strategy)

        collection_infos.append(CollectionInfo(
            collection_name=coll_name,
            strategy=strategy,
            display_name=metadata.display_name,
            description=metadata.description,
            is_available=True,
        ))

    return collection_infos
