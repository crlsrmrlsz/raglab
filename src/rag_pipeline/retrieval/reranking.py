"""Cross-encoder reranking for improved retrieval quality.

## What is a Cross-Encoder?

A cross-encoder is a transformer model that takes TWO texts as input and
outputs a single relevance score. Unlike bi-encoders (embeddings) that
process texts separately, cross-encoders see both texts together.

## Why Cross-Encoders are More Accurate

**Bi-Encoder (Your Embedding Model):**
```
Query: "What metaphor does Marcus Aurelius use?"
        ↓
    embed(query) → [0.23, 0.87, ...]  (1536-dim vector)

Document: "He likens humans to puppets moved by wires"
        ↓
    embed(doc) → [0.45, 0.21, ...]    (1536-dim vector)

Similarity = cosine(query_vec, doc_vec) → 0.67
```

The model never sees "metaphor" and "puppets" together, so it may miss
the connection that "puppets moved by wires" IS the metaphor.

**Cross-Encoder (Reranker):**
```
Input: "[CLS] What metaphor does Marcus Aurelius use? [SEP] He likens humans to puppets... [SEP]"
        ↓
    Full Transformer Attention (query attends to document, and vice versa)
        ↓
    Relevance Score: 0.95
```

The model sees both texts simultaneously, enabling it to understand that
"puppets" directly answers the question about "metaphor".

## Model Selection

Configurable via `RERANK_MODEL` in `src/config.py`. Options range from:
- **CPU-friendly:** MiniLM (15-22M params, ~1-4k docs/sec)
- **GPU-optimized:** mxbai-rerank-large (560M params, SOTA quality)

See config.py for full model list with speed/quality tradeoffs.
"""

import time
from typing import Optional, Any
from dataclasses import dataclass, replace, field

from sentence_transformers import CrossEncoder

from src.config import RERANK_INITIAL_K, RERANK_MODEL
from src.rag_pipeline.indexing.weaviate_query import SearchResult
from src.shared.files import setup_logging

logger = setup_logging(__name__)


@dataclass
class RerankResult:
    """Result of cross-encoder reranking with order tracking for UI logging.

    Attributes:
        results: Reranked SearchResult objects with updated scores.
        order_changes: List of dicts tracking how rankings changed.
            Each dict: {"chunk_id", "before_rank", "after_rank", "before_score", "after_score", "text_preview"}
        rerank_time_ms: Time taken for reranking in milliseconds.
        model: The reranking model used.
    """

    results: list[SearchResult]
    order_changes: list[dict[str, Any]] = field(default_factory=list)
    rerank_time_ms: float = 0.0
    model: str = ""

# Singleton to hold loaded model (avoids reloading 1.2GB model on each call)
_reranker: Optional[CrossEncoder] = None


# =============================================================================
# RERANKER INITIALIZATION
# =============================================================================


def get_reranker() -> CrossEncoder:
    """
    Get or initialize the cross-encoder reranker model.

    Uses singleton pattern to avoid reloading the model on each call.
    The model is ~1.2GB and takes a few seconds to load.

    Returns:
        CrossEncoder: The loaded reranking model

    Example:
        >>> reranker = get_reranker()
        >>> scores = reranker.predict([["query", "doc1"], ["query", "doc2"]])
        >>> print(scores)  # [0.95, 0.23]

    Technical Notes:
        - First call downloads model from HuggingFace Hub (~1.2GB)
        - Model is cached locally after first download
        - Uses GPU automatically if available (CUDA/MPS)
        - Falls back to CPU if no GPU
    """
    global _reranker

    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANK_MODEL}")
        logger.info("(First load downloads ~1.2GB model, subsequent loads are instant)")

        # CrossEncoder loads to GPU automatically if available
        # trust_remote_code=True needed for some models
        _reranker = CrossEncoder(RERANK_MODEL)

        logger.info(f"Reranker loaded successfully")

    return _reranker


# =============================================================================
# MAIN RERANKING FUNCTION
# =============================================================================


def rerank(
    query: str,
    documents: list[SearchResult],
    top_k: int = 10,
    return_details: bool = False,
) -> RerankResult:
    """
    Rerank documents using cross-encoder for improved relevance.

    This function takes the initial retrieval results (from bi-encoder/hybrid search)
    and re-scores them using a cross-encoder that processes query and document together.

    Args:
        query: The user's search query
        documents: Initial retrieval results (typically top-50 from hybrid search)
        top_k: Number of documents to return after reranking
        return_details: If True, include detailed order_changes in result

    Returns:
        RerankResult: Contains reranked documents, order changes, and timing info

    Example:
        >>> # Get initial candidates from hybrid search
        >>> candidates = query_hybrid(client, "What is consciousness?", top_k=50)
        >>>
        >>> # Rerank to top-10 with cross-encoder
        >>> result = rerank("What is consciousness?", candidates, top_k=10)
        >>>
        >>> # result.results[0] is now the most relevant document
        >>> print(result.results[0].score)  # 0.95

    How It Works:
        1. Extract text from each SearchResult
        2. Create [query, text] pairs for the cross-encoder
        3. Score all pairs in a batch (efficient GPU usage)
        4. Create new SearchResult objects with updated scores
        5. Sort by score descending
        6. Return top_k results

    Performance Notes:
        - 50 documents: ~1 second on CPU, ~0.1s on GPU
        - 100 documents: ~2 seconds on CPU, ~0.2s on GPU
        - First call is slower due to model loading (~3-5 seconds)
    """
    if not documents:
        return RerankResult(results=[], model=RERANK_MODEL)

    if len(documents) <= top_k:
        # No need to rerank if we already have fewer documents than requested
        logger.debug(f"Skipping rerank: only {len(documents)} documents (top_k={top_k})")
        return RerankResult(results=documents, model=RERANK_MODEL)

    logger.info(f"Reranking {len(documents)} documents to top-{top_k}")
    start_time = time.time()

    # Step 1: Get the reranker model (lazy loading)
    reranker = get_reranker()

    # Step 2: Create query-document pairs for scoring
    # Format: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc.text] for doc in documents]

    # Step 3: Score all pairs in a batch
    # The model returns relevance scores (higher = more relevant)
    # These are typically logits or probabilities depending on the model
    scores = reranker.predict(pairs)

    logger.debug(f"Score range: [{min(scores):.3f}, {max(scores):.3f}]")

    # Step 4: Create new SearchResult objects with updated scores
    # We use dataclasses.replace() to create immutable copies
    # Also track original ranks and scores for order_changes
    reranked_results = []
    original_data = []  # [(original_rank, original_score, new_score, doc)]
    for i, (doc, new_score) in enumerate(zip(documents, scores)):
        # Convert numpy float to Python float for JSON serialization
        reranked_doc = replace(doc, score=float(new_score))
        reranked_results.append(reranked_doc)
        original_data.append({
            "chunk_id": doc.chunk_id,
            "before_rank": i + 1,  # 1-indexed
            "before_score": float(doc.score),
            "after_score": float(new_score),
            "text_preview": doc.text[:80] + "..." if len(doc.text) > 80 else doc.text,
        })

    # Step 5: Sort by score descending (highest relevance first)
    reranked_results.sort(key=lambda x: x.score, reverse=True)

    # Step 6: Return top_k results
    final_results = reranked_results[:top_k]

    # Build order_changes list for top_k results
    order_changes = []
    final_chunk_ids = {doc.chunk_id for doc in final_results}
    for after_rank, doc in enumerate(final_results, 1):
        for orig in original_data:
            if orig["chunk_id"] == doc.chunk_id:
                order_changes.append({
                    "chunk_id": doc.chunk_id,
                    "before_rank": orig["before_rank"],
                    "after_rank": after_rank,
                    "before_score": orig["before_score"],
                    "after_score": orig["after_score"],
                    "text_preview": orig["text_preview"],
                })
                break

    elapsed_ms = (time.time() - start_time) * 1000

    # Log improvement
    if documents and final_results:
        original_top = documents[0]
        new_top = final_results[0]
        if original_top.chunk_id != new_top.chunk_id:
            logger.info(
                f"Reranking changed top result: "
                f"'{original_top.text[:50]}...' → '{new_top.text[:50]}...'"
            )

    return RerankResult(
        results=final_results,
        order_changes=order_changes,
        rerank_time_ms=elapsed_ms,
        model=RERANK_MODEL,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_model_info() -> dict:
    """
    Get information about the current reranking model.

    Returns:
        dict: Model information including name, parameters, and status

    Example:
        >>> info = get_model_info()
        >>> print(info["model_name"])  # "mixedbread-ai/mxbai-rerank-large-v1"
        >>> print(info["loaded"])  # True/False
    """
    return {
        "model_name": RERANK_MODEL,
        "loaded": _reranker is not None,
        "default_initial_k": RERANK_INITIAL_K,
        "description": "Cross-encoder reranker for two-stage retrieval",
    }


# =============================================================================
# RERANKING UTILITIES
# =============================================================================


def apply_reranking_if_enabled(
    results: list[SearchResult],
    question: str,
    top_k: int,
    use_reranking: bool,
) -> list[SearchResult]:
    """Apply cross-encoder reranking if enabled and results exist.

    This is the simple interface for evaluation code that only needs
    the reranked results list.

    Args:
        results: Search results to potentially rerank.
        question: Original question for reranking context.
        top_k: Number of results to return after reranking.
        use_reranking: Whether reranking is enabled.

    Returns:
        Reranked list if use_reranking is True and results exist,
        otherwise returns original results.
    """
    if use_reranking and results:
        return rerank(question, results, top_k=top_k).results
    return results


def apply_reranking_with_metadata(
    results: list[SearchResult],
    question: str,
    top_k: int,
    use_reranking: bool,
) -> tuple[list[SearchResult], Optional[RerankResult]]:
    """Apply cross-encoder reranking and return both results and metadata.

    This is the interface for UI code that needs reranking metadata
    (timing info, order changes) for logging/display.

    Args:
        results: Search results to potentially rerank.
        question: Original question for reranking context.
        top_k: Number of results to return after reranking.
        use_reranking: Whether reranking is enabled.

    Returns:
        Tuple of (reranked_results, rerank_metadata):
        - reranked_results: List of SearchResult (reranked or original)
        - rerank_metadata: RerankResult with timing/order info, or None if not reranked
    """
    if use_reranking and results:
        rerank_result = rerank(question, results, top_k=top_k)
        return rerank_result.results, rerank_result
    return results, None
