"""Query preprocessing module for RAGLab.

Provides query transformation strategies for improved retrieval:
- HyDE (Hypothetical Document Embeddings) for semantic matching (arXiv:2212.10496)
- Query decomposition for complex questions (+36.7% MRR@10, arXiv:2507.00355)
- Strategy-based preprocessing with registry pattern
"""

from src.rag_pipeline.retrieval.preprocessing.query_preprocessing import (
    PreprocessedQuery,
    hyde_prompt,
    preprocess_query,
)
from src.rag_pipeline.retrieval.preprocessing.strategies import (
    get_strategy,
    list_strategies,
    STRATEGIES,
)
from src.rag_pipeline.retrieval.preprocessing.retrieval_helpers import (
    execute_hyde_retrieval,
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
]
