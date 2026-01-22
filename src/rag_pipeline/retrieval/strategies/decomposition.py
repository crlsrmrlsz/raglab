"""Decomposition retrieval strategy with RRF merging.

Query decomposition breaks complex questions into sub-questions,
executes each independently, and merges results with RRF.

Research: "Question Decomposition for Retrieval-Augmented Generation"
          arXiv:2507.00355 (Ammann et al.)
          Shows +36.7% MRR@10 improvement for complex multi-hop queries.

Key algorithm:
    1. Decompose query into 2-4 sub-questions
    2. Execute each sub-question + original as separate searches
    3. Merge results with Reciprocal Rank Fusion (RRF)
       - Documents appearing in multiple result lists get boosted
       - RRF formula: score(d) = sum(1 / (k + rank(d, q))) for each query q
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_protocol import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.preprocessing import preprocess_query
from src.rag_pipeline.retrieval.preprocessing.retrieval_helpers import execute_search
from src.rag_pipeline.retrieval.rrf import reciprocal_rank_fusion
from src.rag_pipeline.retrieval.reranking_utils import apply_reranking_if_enabled
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class DecompositionRetrieval:
    """Decomposition strategy: Break into sub-questions → RRF merge.

    Research: Query Decomposition shows +36.7% MRR@10 improvement
    for complex multi-hop queries (arXiv:2507.00355).

    This encapsulates all the RRF merging logic that was previously
    inline in search.py and ragas_evaluator.py.

    Flow:
        1. Decompose query into 2-4 sub-questions
        2. Execute each sub-question independently
        3. Merge results with RRF (chunks in multiple results get boosted)

    Attributes:
        strategy_id: "decomposition" - identifies this as the decomposition strategy.
    """

    strategy_id = "decomposition"

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        preprocessing_model: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute decomposition retrieval with RRF merging.

        Args:
            query: User's original query text.
            context: Retrieval context with Weaviate client and settings.
            preprocessing_model: Model for decomposing query into sub-questions.

        Returns:
            RetrievalResult with RRF-merged results and decomposition metadata.
        """
        # Preprocessing: Decompose into sub-questions
        preprocessed = preprocess_query(
            query=query,
            strategy="decomposition",
            model=preprocessing_model,
        )

        generated = preprocessed.generated_queries or []
        rrf_data = None

        if len(generated) <= 1:
            # Fallback: if decomposition failed, use standard search
            logger.warning("[decomposition] No sub-queries generated, falling back to standard search")
            results = execute_search(context, query, context.initial_k)
        else:
            # Multi-query RRF path
            logger.info(f"[decomposition] Executing {len(generated)} sub-queries with RRF merge")

            result_lists = []
            query_types = []

            # Retrieve more per query to give RRF enough candidates
            per_query_k = max(context.initial_k * 2, 20)

            for q in generated:
                query_text = q.get("query", "")
                query_type = q.get("type", "unknown")

                if not query_text:
                    continue

                # Execute search (shared helper handles hybrid vs vector)
                sub_results = execute_search(context, query_text, per_query_k)

                result_lists.append(sub_results)
                query_types.append(query_type)

            # RRF merge
            rrf_result = reciprocal_rank_fusion(
                result_lists=result_lists,
                query_types=query_types,
                top_k=context.initial_k,
            )
            results = rrf_result.results
            rrf_data = {
                "query_contributions": rrf_result.query_contributions,
                "merge_time_ms": rrf_result.merge_time_ms,
                "queries_merged": len(result_lists),
            }

        # Apply reranking if enabled
        results = apply_reranking_if_enabled(
            results,
            query,
            context.top_k,
            context.use_reranking,
        )

        logger.info(f"[decomposition] Retrieved {len(results)} results")

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata={
                "strategy": "decomposition",
                "sub_queries": len(generated),
                "rrf_data": rrf_data,
            },
        )
