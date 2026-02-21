"""GraphRAG retrieval strategy with pure graph traversal.

GraphRAG uses knowledge graph traversal to find relevant chunks,
ranking by combined_degree (relationship hub importance).

Research: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
          arXiv:2404.16130 (Microsoft)
          Shows +72-83% win rate vs baseline on global queries.

Key algorithm (local queries):
    1. Extract entities from query (embedding similarity)
    2. Traverse Neo4j graph from entities -> find related chunk IDs
    3. Fetch graph-discovered chunks from Weaviate (batch filter, not vector search)
    4. Rank by combined_degree (Microsoft approach: hub entities = more informative)
    5. Return top-k chunks (no reranking — StrategyConfig forbids it)

Key algorithm (global queries):
    1. LLM classifies query as local or global
    2. DRIFT search: embed query -> top-K communities via HNSW -> primer folds -> reduce
    3. Synthesize final answer from community perspectives (~5 LLM calls vs ~1000)
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_registry import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query
from src.rag_pipeline.indexing.weaviate_query import SearchResult
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class GraphRAGRetrieval:
    """GraphRAG strategy: Pure graph traversal ranked by combined_degree.

    Research: GraphRAG shows +72-83% win rate vs baseline (arXiv:2404.16130).

    This implements Microsoft's GraphRAG approach:
    - Local queries: Entity extraction -> Graph traversal -> Combined_degree ranking
    - Global queries: DRIFT search over Leiden community summaries

    Flow (local):
        1. Extract entities from query
        2. Traverse graph from entities -> related chunk IDs
        3. Rank by combined_degree (start_degree + neighbor_degree)
        4. Return top-k chunks (reranking forbidden by StrategyConfig)

    Flow (global):
        1. LLM classifies query as global
        2. DRIFT: embed query -> HNSW top-K -> primer folds -> reduce
        3. Synthesize answer (~5 LLM calls)

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
        """Execute GraphRAG pure graph retrieval.

        Args:
            query: User's original query text.
            context: Retrieval context with Neo4j driver.
            preprocessing_model: Model for entity extraction fallback.

        Returns:
            RetrievalResult with graph-ranked results and metadata.

        Note:
            If context.neo4j_driver is None, returns empty results with error.
        """
        # Preprocessing: Extract query entities
        preprocessed = preprocess_query(
            query=query,
            strategy="graphrag",
            model=preprocessing_model,
        )

        graph_metadata = {}

        # Graph retrieval (requires Neo4j)
        if context.neo4j_driver is None:
            logger.warning("[graphrag] No Neo4j driver provided, cannot execute graph retrieval")
            results_dicts = []
            graph_metadata = {"error": "No Neo4j driver provided", "query_type": "local"}
        else:
            try:
                from src.graph.query import graph_retrieval_with_drift

                results_dicts, graph_metadata = graph_retrieval_with_drift(
                    query=query,
                    driver=context.neo4j_driver,
                    top_k=context.top_k,
                    collection_name=context.collection_name,
                    use_drift=True,
                )
            except Exception as e:
                logger.error(f"[graphrag] Graph retrieval failed: {e}")
                results_dicts = []
                graph_metadata = {"error": str(e), "query_type": "local"}

        # Convert to SearchResult objects
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
            for r in results_dicts
        ]

        # DRIFT global queries: create SearchResult objects from community summaries
        # so RAGAS has real contexts to evaluate (instead of empty list)
        result_metadata = {
            "strategy": "graphrag",
            "graph_metadata": graph_metadata,
            "query_entities": preprocessed.query_entities,
            "query_type": graph_metadata.get("query_type", "local"),
        }

        if not results and graph_metadata.get("query_type") == "global":
            drift_data = graph_metadata.get("drift_result", {})
            summaries = drift_data.get("community_summaries", [])
            for i, summary in enumerate(summaries):
                results.append(SearchResult(
                    chunk_id=f"community_summary_{i}",
                    book_id="graphrag_community",
                    section="community_summary",
                    context="DRIFT community summary",
                    text=summary,
                    token_count=len(summary.split()),
                    score=1.0 - (i * 0.01),
                ))
            result_metadata["drift_final_answer"] = drift_data.get("final_answer", "")
            logger.info(
                f"[graphrag] DRIFT global: {len(results)} community summaries as contexts"
            )

        logger.info(
            f"[graphrag] Retrieved {len(results)} results "
            f"(query_type={graph_metadata.get('query_type', 'local')})"
        )

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata=result_metadata,
        )
