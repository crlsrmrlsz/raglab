"""GraphRAG retrieval strategy with hybrid graph + vector search.

GraphRAG combines vector search with knowledge graph traversal,
using RRF to merge results from both sources.

Research: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
          arXiv:2404.16130 (Microsoft)
          Shows +72-83% win rate vs baseline on global queries.

Key algorithm:
    1. Extract entities from query (embedding similarity + LLM fallback)
    2. Execute vector search in Weaviate
    3. Traverse Neo4j graph from entities → find related chunk IDs
    4. Fetch graph-discovered chunks from Weaviate
    5. RRF merge vector results + graph results (overlapping chunks get boosted)
    6. Optionally: Map-reduce over community summaries for global queries
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_protocol import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.preprocessing import preprocess_query
from src.rag_pipeline.retrieval.preprocessing.retrieval_helpers import execute_search
from src.rag_pipeline.indexing.weaviate_query import SearchResult
from src.rag_pipeline.retrieval.reranking_utils import apply_reranking_if_enabled
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class GraphRAGRetrieval:
    """GraphRAG strategy: Vector search → Graph traversal → RRF merge.

    Research: GraphRAG shows +72-83% win rate vs baseline (arXiv:2404.16130).

    This encapsulates the graph retrieval logic from graph/query.py,
    integrating it into the unified strategy pattern.

    Flow:
        1. Extract entities from query (preprocessing)
        2. Execute vector search
        3. Traverse graph from entities → find related chunks
        4. RRF merge vector results + graph results
        5. Optionally: Map-reduce over community summaries for global queries

    Attributes:
        strategy_id: "graphrag" - identifies this as the GraphRAG strategy.
    """

    strategy_id = "graphrag"

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        preprocessing_model: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute GraphRAG hybrid retrieval.

        Args:
            query: User's original query text.
            context: Retrieval context with Weaviate client and Neo4j driver.
            preprocessing_model: Model for entity extraction fallback.

        Returns:
            RetrievalResult with RRF-merged results and graph metadata.

        Note:
            If context.neo4j_driver is None, falls back to vector-only search.
        """
        # Preprocessing: Extract query entities
        preprocessed = preprocess_query(
            query=query,
            strategy="graphrag",
            model=preprocessing_model,
        )

        # Execute vector search first (shared helper handles hybrid vs vector)
        vector_results = execute_search(context, query, context.initial_k)

        # Convert to dicts for hybrid_graph_retrieval
        vector_dicts = [
            {
                "chunk_id": r.chunk_id,
                "book_id": r.book_id,
                "section": r.section,
                "context": r.context,
                "text": r.text,
                "token_count": r.token_count,
                "similarity": r.score,
                "is_summary": getattr(r, "is_summary", False),
                "tree_level": getattr(r, "tree_level", 0),
            }
            for r in vector_results
        ]

        graph_metadata = {}

        # Graph traversal + RRF merge (if Neo4j driver available)
        if context.neo4j_driver is None:
            logger.warning("[graphrag] No Neo4j driver provided, returning vector results only")
            merged_dicts = vector_dicts[: context.top_k]
            graph_metadata = {"error": "No Neo4j driver provided", "query_type": "local"}
        else:
            try:
                from src.graph.query import hybrid_graph_retrieval_with_map_reduce

                merged_dicts, graph_metadata = hybrid_graph_retrieval_with_map_reduce(
                    query=query,
                    driver=context.neo4j_driver,
                    vector_results=vector_dicts,
                    top_k=context.top_k,
                    collection_name=context.collection_name,
                    use_map_reduce=True,
                )
            except Exception as e:
                logger.error(f"[graphrag] Graph retrieval failed: {e}, using vector only")
                merged_dicts = vector_dicts[: context.top_k]
                graph_metadata = {"error": str(e), "query_type": "local"}

        # Convert back to SearchResult objects
        results = [
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
            for r in merged_dicts
        ]

        # Apply reranking if enabled (after graph merge)
        results = apply_reranking_if_enabled(
            results,
            query,
            context.top_k,
            context.use_reranking,
        )

        logger.info(
            f"[graphrag] Retrieved {len(results)} results "
            f"(query_type={graph_metadata.get('query_type', 'local')})"
        )

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata={
                "strategy": "graphrag",
                "graph_metadata": graph_metadata,
                "query_entities": preprocessed.query_entities,
                "query_type": graph_metadata.get("query_type", "local"),
            },
        )
