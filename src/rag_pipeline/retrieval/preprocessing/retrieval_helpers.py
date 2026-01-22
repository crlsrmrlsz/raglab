"""Shared retrieval helpers for preprocessing strategies.

This module consolidates strategy-specific retrieval logic that was duplicated
across the UI (search.py) and CLI (ragas_evaluator.py) code paths.

## Problem Solved

Both UI and CLI need to execute HyDE retrieval with:
- Embedding averaging (original query + K hypotheticals)
- Pure semantic search (alpha=1.0)
- Single hybrid search with precomputed embedding

Previously this logic was duplicated (~35 lines each). Now it's shared.

## Data Flow

PreprocessedQuery → execute_hyde_retrieval() → (results, optional_metadata)

UI callers get (SearchResult list, RerankResult | None) for logging.
CLI callers extract texts with [r.text for r in results].
"""

from typing import TYPE_CHECKING, Any, Optional

import weaviate

from src.rag_pipeline.embedding.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_query import query_hybrid, SearchResult
from src.rag_pipeline.retrieval.preprocessing.strategy_config import get_strategy_config
from src.rag_pipeline.retrieval.reranking_utils import (
    apply_reranking_if_enabled,
    apply_reranking_with_metadata,
)
from src.shared.files import setup_logging

if TYPE_CHECKING:
    from src.rag_pipeline.retrieval.reranking import RerankResult

logger = setup_logging(__name__)


def execute_hyde_retrieval(
    client: weaviate.WeaviateClient,
    original_query: str,
    generated_queries: list[dict[str, str]],
    top_k: int,
    collection_name: str,
    use_reranking: bool,
    initial_k: int,
    return_metadata: bool = False,
) -> tuple[list[SearchResult], Optional["RerankResult"]]:
    """Execute HyDE retrieval with paper-aligned embedding averaging.

    HyDE (arXiv:2212.10496) averages embeddings of original query + K hypotheticals,
    then performs a single semantic search with the averaged embedding.

    This function consolidates the HyDE retrieval logic previously duplicated
    in src/ui/services/search.py and src/evaluation/ragas_evaluator.py.

    Args:
        client: Connected Weaviate client (caller manages lifecycle).
        original_query: The user's original query text.
        generated_queries: List of {type, query} dicts from hyde_strategy.
            Expected format: [{"type": "original", "query": "..."}, {"type": "hyde", "query": "..."}, ...]
        top_k: Final number of results to return.
        collection_name: Weaviate collection to search.
        use_reranking: Whether to apply cross-encoder reranking.
        initial_k: Number of candidates to retrieve before reranking.
        return_metadata: If True, returns RerankResult metadata for logging.
            If False, returns None for metadata (CLI path).

    Returns:
        Tuple of (results, rerank_metadata):
        - results: List of SearchResult objects
        - rerank_metadata: RerankResult if return_metadata=True and reranking applied, else None

    Paper Requirements (arXiv:2212.10496):
        - Embed original query + K hypotheticals
        - Average embeddings element-wise
        - Single search with averaged embedding
        - Pure semantic search (alpha=1.0)

    Example:
        >>> client = get_client()
        >>> try:
        ...     results, metadata = execute_hyde_retrieval(
        ...         client=client,
        ...         original_query="What is consciousness?",
        ...         generated_queries=[
        ...             {"type": "original", "query": "What is consciousness?"},
        ...             {"type": "hyde", "query": "Consciousness is..."},
        ...         ],
        ...         top_k=10,
        ...         collection_name="chunks",
        ...         use_reranking=True,
        ...         initial_k=50,
        ...         return_metadata=True,
        ...     )
        ... finally:
        ...     client.close()
    """
    config = get_strategy_config("hyde")
    hyde_alpha = config.alpha_constraint.fixed_value  # 1.0

    # Extract all queries for embedding (original + hypotheticals)
    all_queries = [q.get("query", "") for q in generated_queries if q.get("query")]

    logger.info(
        f"[hyde] Paper-aligned: averaging {len(all_queries)} embeddings, alpha={hyde_alpha}"
    )

    # Embed and average all queries
    embeddings = embed_texts(all_queries)
    avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]

    # Single search with averaged embedding (pure semantic, no BM25)
    results = query_hybrid(
        client=client,
        query_text=original_query,  # For BM25 component (ignored at alpha=1.0)
        top_k=initial_k,
        alpha=hyde_alpha,  # 1.0 - pure semantic
        collection_name=collection_name,
        precomputed_embedding=avg_embedding,
    )

    # Apply reranking
    if return_metadata:
        results, rerank_data = apply_reranking_with_metadata(
            results, original_query, top_k, use_reranking
        )
        return results, rerank_data
    else:
        results = apply_reranking_if_enabled(
            results, original_query, top_k, use_reranking
        )
        return results, None
