"""Standard retrieval strategy (no preprocessing).

This is the baseline strategy - direct Weaviate search with optional reranking.
No LLM calls, no query transformation.
"""

from typing import Optional

from src.rag_pipeline.retrieval.strategy_registry import (
    RetrievalContext,
    RetrievalResult,
)
from src.rag_pipeline.retrieval.query_preprocessing import preprocess_query
from src.rag_pipeline.retrieval.query_helpers import execute_search
from src.rag_pipeline.retrieval.reranking import apply_reranking_if_enabled
from src.shared.files import setup_logging

logger = setup_logging(__name__)


class StandardRetrieval:
    """Standard retrieval: no preprocessing, direct Weaviate search.

    This is the baseline strategy for comparison. Returns the query exactly
    as entered, executes search, and optionally applies reranking.

    Attributes:
        strategy_id: "none" - identifies this as the no-preprocessing strategy.
    """

    strategy_id = "none"

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        preprocessing_model: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute standard retrieval without preprocessing.

        Args:
            query: User's original query text.
            context: Retrieval context with Weaviate client and settings.
            preprocessing_model: Ignored (no LLM call made).

        Returns:
            RetrievalResult with search results and minimal metadata.
        """
        # Preprocessing: none strategy just wraps query for consistency
        preprocessed = preprocess_query(
            query=query,
            strategy="none",
            model=preprocessing_model,
        )

        # Execute search (shared helper handles hybrid vs vector)
        results = execute_search(context, query, context.initial_k)

        # Apply reranking if enabled
        results = apply_reranking_if_enabled(
            results,
            query,
            context.top_k,
            context.use_reranking,
        )

        logger.info(f"[standard] Retrieved {len(results)} results")

        return RetrievalResult(
            results=results,
            preprocessing=preprocessed,
            metadata={
                "strategy": "none",
                "search_type": context.search_type,
                "alpha": context.alpha,
            },
        )
