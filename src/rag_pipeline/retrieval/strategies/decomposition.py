"""Decomposition retrieval strategy with union merge + reranking.

Query decomposition breaks complex questions into sub-questions,
executes each independently, pools results, and reranks against the
original query.

Research: "Question Decomposition for Retrieval-Augmented Generation"
          arXiv:2507.00355 (Ammann et al.)
          Shows +36.7% MRR@10 improvement for complex multi-hop queries.

Key algorithm (per paper):
    1. Decompose query into sub-questions (up to 5)
    2. Execute each sub-question + original as separate searches
    3. Pool all results (simple union)
    4. Rerank entire pool against original query using cross-encoder
    5. Return top-k by reranker score
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_registry import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query
from src.rag_pipeline.retrieval.query_helpers import execute_search
from src.rag_pipeline.retrieval.reranking import rerank
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class DecompositionRetrieval:
    """Decomposition strategy: Break into sub-questions → union → rerank.

    Research: Query Decomposition shows +36.7% MRR@10 improvement
    for complex multi-hop queries (arXiv:2507.00355).

    Flow (per paper):
        1. Decompose query into sub-questions
        2. Execute each sub-question independently
        3. Pool all results (simple union, deduplicate by chunk_id)
        4. Rerank entire pool against original query (mandatory)

    Note: Reranking is mandatory for this strategy per the paper.
    The StrategyConfig declares requires_reranking=True.

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
        """Execute decomposition retrieval with union merge + reranking.

        Args:
            query: User's original query text.
            context: Retrieval context with Weaviate client and settings.
            preprocessing_model: Model for decomposing query into sub-questions.

        Returns:
            RetrievalResult with reranked results and decomposition metadata.
        """
        # Preprocessing: Decompose into sub-questions
        preprocessed = preprocess_query(
            query=query,
            strategy="decomposition",
            model=preprocessing_model,
        )

        generated = preprocessed.generated_queries or []

        if len(generated) <= 1:
            # Fallback: if decomposition failed, use standard search + rerank
            logger.warning("[decomposition] No sub-queries generated, falling back to standard search")
            results = execute_search(context, query, context.initial_k)
        else:
            # Multi-query union path (per paper)
            logger.info(f"[decomposition] Executing {len(generated)} sub-queries with union merge")

            # Retrieve candidates for each query
            # Paper: pool all results then rerank, so retrieve generously
            per_query_k = max(context.initial_k, 20)

            all_results = []
            for q in generated:
                query_text = q.get("query", "")
                if not query_text:
                    continue

                sub_results = execute_search(context, query_text, per_query_k)
                all_results.extend(sub_results)

            # Simple union: deduplicate by chunk_id, keep first occurrence
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result.chunk_id not in seen_ids:
                    seen_ids.add(result.chunk_id)
                    unique_results.append(result)

            results = unique_results
            logger.info(f"[decomposition] Pooled {len(all_results)} -> {len(results)} unique chunks")

        # Rerank against original query (mandatory per paper)
        # Paper uses bge-reranker-large; RAGLab uses configurable reranker
        if results:
            rerank_result = rerank(query, results, top_k=context.top_k)
            results = rerank_result.results
            logger.info(f"[decomposition] Reranked to {len(results)} results")

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata={
                "strategy": "decomposition",
                "sub_queries": len(generated),
            },
        )
