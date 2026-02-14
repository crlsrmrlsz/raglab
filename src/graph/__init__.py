"""GraphRAG module for knowledge graph construction and retrieval.

## RAG Theory: GraphRAG (arXiv:2404.16130)

GraphRAG augments RAG with knowledge graph structures for:
- Cross-document relationship discovery (entity linking)
- Hierarchical community detection (Leiden algorithm)
- Global query answering via community summaries

While vector search excels at local queries ("What does X say about Y?"),
GraphRAG enables global queries ("What are the main themes across all documents?").

## Module Structure

- schemas.py: Pydantic models for entities, relationships, communities
- extraction_utils.py: LLM-based entity/relationship extraction and consolidation
- neo4j_client.py: Neo4j connection and Cypher operations
- community.py: Leiden detection + community summarization
- hierarchy.py: Multi-level community parsing from Leiden
- centrality.py: PageRank computation for entity importance
- map_reduce.py: Query classification (classify_query, should_use_map_reduce)
- drift.py: DRIFT search for global queries (replaces map-reduce)
- query_entities.py: Query entity extraction (embedding similarity)
- query.py: Graph retrieval (pure graph traversal + DRIFT)

## Data Flow

1. Section chunks → Entity extraction → Neo4j upload
2. Neo4j graph → Leiden communities → Community summaries (stored in Weaviate)
3. Query → Embedding entity extraction → Graph traversal → Chunk ID discovery
4. Fetch graph-discovered chunks from Weaviate (batch filter) → Rank by combined_degree
5. For global queries: DRIFT search over top-K community summaries
"""

from .schemas import (
    GraphEntity,
    GraphRelationship,
    ExtractionResult,
    Community,
    CommunityMember,
    CommunityRelationship,
)
from .neo4j_client import (
    get_driver,
    get_gds_client,
    verify_connection,
    upload_extraction_results,
    get_graph_stats,
)
from .community import (
    detect_and_summarize_communities,
)
from .hierarchy import (
    CommunityLevel,
    parse_leiden_hierarchy,
    build_community_key,
    filter_communities_by_size,
)
from .centrality import (
    compute_pagerank,
    write_pagerank_to_neo4j,
)
from .map_reduce import (
    classify_query,
    should_use_map_reduce,
)
from .drift import (
    DriftResult,
    drift_search,
)
from .query_entities import (
    extract_query_entities,
)
from .query import (
    get_graph_chunk_ids,
    retrieve_graph_context,
    format_graph_context_for_generation,
    fetch_chunks_by_ids,
)

__all__ = [
    # Schemas
    "GraphEntity",
    "GraphRelationship",
    "ExtractionResult",
    "Community",
    "CommunityMember",
    "CommunityRelationship",
    # Neo4j
    "get_driver",
    "get_gds_client",
    "verify_connection",
    "upload_extraction_results",
    "get_graph_stats",
    # Community
    "detect_and_summarize_communities",
    # Hierarchy
    "CommunityLevel",
    "parse_leiden_hierarchy",
    "build_community_key",
    "filter_communities_by_size",
    # Centrality
    "compute_pagerank",
    "write_pagerank_to_neo4j",
    # Query classification (used by DRIFT)
    "classify_query",
    "should_use_map_reduce",
    # DRIFT Search
    "DriftResult",
    "drift_search",
    # Query entities
    "extract_query_entities",
    # Query
    "get_graph_chunk_ids",
    "retrieve_graph_context",
    "format_graph_context_for_generation",
    "fetch_chunks_by_ids",
]
