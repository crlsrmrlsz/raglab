# Stage 7: Retrieval
# Query preprocessing, search, reranking, and diversity

from src.rag_pipeline.retrieval.query_preprocessing import (
    PreprocessedQuery,
    hyde_prompt,
    preprocess_query,
)
from src.rag_pipeline.retrieval.strategy_registry import (
    get_functional_strategy as get_strategy,
    list_strategies,
    STRATEGIES,
)
from src.rag_pipeline.retrieval.query_helpers import (
    execute_hyde_retrieval,
    execute_search,
)

__all__ = [
    # Core types
    "PreprocessedQuery",
    # Main entry point
    "preprocess_query",
    # Low-level functions (for direct use)
    "hyde_prompt",
    # Strategy registry
    "get_strategy",
    "list_strategies",
    "STRATEGIES",
    # Retrieval helpers
    "execute_hyde_retrieval",
    "execute_search",
]
