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
- extractor.py: LLM-based entity/relationship extraction
- neo4j_client.py: Neo4j connection and Cypher operations
- community.py: Leiden detection + community summarization
- hierarchy.py: Multi-level community parsing from Leiden
- centrality.py: PageRank computation for entity importance
- map_reduce.py: Async map-reduce for global queries
- query_entities.py: Query entity extraction (embedding + LLM)
- query.py: Graph retrieval strategy for hybrid search

## Data Flow

1. Section chunks → Entity extraction → Neo4j upload
2. Neo4j graph → Leiden communities → Community summaries (stored in Weaviate)
3. Query → Embedding/LLM entity extraction → Graph traversal → Chunk ID discovery
4. Vector search (Weaviate) + Fetch graph-only chunks → RRF merge → Answer
5. For global queries: Map-reduce over community summaries
"""

from .schemas import (
    GraphEntity,
    GraphRelationship,
    ExtractionResult,
    Community,
    CommunityMember,
    CommunityRelationship,
)
from .extractor import (
    load_chunks_for_extraction,
    save_extraction_results,
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
    save_communities,
    load_communities,
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
    MapReduceResult,
    classify_query,
    map_reduce_global_query,
    should_use_map_reduce,
)
from .query_entities import (
    extract_query_entities,
    extract_query_entities_embedding,
)
from .query import (
    get_graph_chunk_ids,
    retrieve_graph_context,
    retrieve_community_context,
    retrieve_communities_for_map_reduce,
    format_graph_context_for_generation,
    fetch_chunks_by_ids,
    hybrid_graph_retrieval,
    hybrid_graph_retrieval_with_map_reduce,
)

__all__ = [
    # Schemas
    "GraphEntity",
    "GraphRelationship",
    "ExtractionResult",
    "Community",
    "CommunityMember",
    "CommunityRelationship",
    # Extraction helpers
    "load_chunks_for_extraction",
    "save_extraction_results",
    # Neo4j
    "get_driver",
    "get_gds_client",
    "verify_connection",
    "upload_extraction_results",
    "get_graph_stats",
    # Community
    "detect_and_summarize_communities",
    "save_communities",
    "load_communities",
    # Hierarchy
    "CommunityLevel",
    "parse_leiden_hierarchy",
    "build_community_key",
    "filter_communities_by_size",
    # Centrality
    "compute_pagerank",
    "write_pagerank_to_neo4j",
    # Map-Reduce
    "MapReduceResult",
    "classify_query",
    "map_reduce_global_query",
    "should_use_map_reduce",
    # Query entities
    "extract_query_entities",
    "extract_query_entities_embedding",
    # Query
    "get_graph_chunk_ids",
    "retrieve_graph_context",
    "retrieve_community_context",
    "retrieve_communities_for_map_reduce",
    "format_graph_context_for_generation",
    "fetch_chunks_by_ids",
    "hybrid_graph_retrieval",
    "hybrid_graph_retrieval_with_map_reduce",
]
