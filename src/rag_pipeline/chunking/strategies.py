"""Chunking strategy implementations.

## RAG Theory: Strategy Pattern for Chunking

Different chunking strategies optimize for different retrieval scenarios:
- section: Sequential reading order with overlap (baseline, fast)
- semantic: Embedding similarity-based boundaries (improved coherence)
- contextual: LLM-generated chunk context (Anthropic-style, +35% improvement)
- raptor: Hierarchical summarization tree (multi-level retrieval)

The strategy pattern allows A/B testing chunking approaches without modifying
the pipeline. Each strategy outputs to its own subdirectory for isolated
Weaviate collections.

## Library Usage

Wraps existing chunking functions (section_chunker, semantic_chunker, etc.)
in a common interface for the stage runner to invoke.

## Data Flow

1. User selects strategy (CLI arg: --strategy semantic)
2. Stage 4 runner calls get_strategy() to get the function
3. Strategy function processes all files from DIR_NLP_CHUNKS
4. Outputs to DIR_FINAL_CHUNKS/{strategy_name}/
5. Returns stats dict {book_name: chunk_count}
"""

from functools import partial
from typing import Any, Callable, Optional

from src.config import (
    MAX_CHUNK_TOKENS,
    OVERLAP_SENTENCES,
    SEMANTIC_STD_COEFFICIENT,
    EMBEDDING_MAX_INPUT_TOKENS,
    CONTEXTUAL_MODEL,
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_CLUSTER_SIZE,
    RAPTOR_SUMMARY_MODEL,
)
from src.shared.files import setup_logging, OverwriteContext

logger = setup_logging(__name__)


# Type alias for strategy functions
# Input: None (reads from DIR_NLP_CHUNKS)
# Output: dict[book_name, chunk_count]
ChunkingStrategyFunction = Callable[[], dict[str, int]]


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================


def section_strategy(
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, int]:
    """Sequential chunking with sentence overlap (baseline).

    Algorithm:
    - Read paragraphs in document order
    - Build chunks up to MAX_CHUNK_TOKENS
    - Overlap OVERLAP_SENTENCES between chunks (same section only)
    - Respects markdown section boundaries (# Chapter, ## Section)

    Use case: Preserves reading order, best for linear narratives.
    Fast execution, no API calls during chunking.

    Args:
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts.
    """
    from src.rag_pipeline.chunking.section_chunker import run_section_chunking

    logger.info(f"[section] Using sequential chunking with overlap")
    logger.info(f"[section] Max tokens: {MAX_CHUNK_TOKENS}, overlap: {OVERLAP_SENTENCES}")
    return run_section_chunking(overwrite_context=overwrite_context)


def semantic_strategy(
    std_coefficient: float = SEMANTIC_STD_COEFFICIENT,
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, int]:
    """Semantic similarity-based chunking.

    Algorithm:
    - Embed sentences using text-embedding-3-large API
    - Compute cosine similarity between adjacent sentences
    - Split where similarity < mean - (coefficient * std) (statistical outliers)
    - Still respects section boundaries and token limits
    - Uses same overlap mechanism as section strategy

    Use case: Better topical coherence, improved retrieval precision.
    Research shows 8-12% improvement on Q&A tasks.

    Note: Requires API calls during chunking (costs apply).

    Args:
        std_coefficient: Standard deviation coefficient for breakpoint detection.
            Higher = fewer splits (only extreme drops). Default is 3.0.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts.
    """
    from src.rag_pipeline.chunking.semantic_chunker import run_semantic_chunking

    logger.info(f"[semantic] Using embedding similarity chunking")
    logger.info(f"[semantic] Std coefficient: {std_coefficient}, safeguard: {EMBEDDING_MAX_INPUT_TOKENS} tokens")
    return run_semantic_chunking(
        std_coefficient=std_coefficient,
        overwrite_context=overwrite_context,
    )


def contextual_strategy(
    model: str = CONTEXTUAL_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, int]:
    """Contextual chunking (Anthropic-style).

    Algorithm:
    - Load existing semantic_std2 chunks (topic-aligned boundaries)
    - For each chunk, use book title and section title as context
    - Call LLM to generate 50-100 token contextual snippet
    - Prepend snippet to chunk text before embedding

    Use case: Better disambiguation, improved retrieval for complex queries.
    Anthropic reports 35% failure reduction (recall@20).

    Note: Requires semantic_std2 chunks to exist first. Run semantic strategy
    with --std-coefficient 2.0 if contextual/ folder is empty.

    Note: Requires LLM API calls for each chunk (costs ~$0.50-1.00 for corpus).

    Args:
        model: OpenRouter model ID for context generation.
            Default: deepseek-v3.2 (fast, cheap).
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts.
    """
    from src.rag_pipeline.chunking.contextual_chunker import run_contextual_chunking

    logger.info(f"[contextual] Using LLM-generated context (Anthropic-style)")
    logger.info(f"[contextual] Context model: {model}")
    return run_contextual_chunking(
        model=model,
        overwrite_context=overwrite_context,
    )


def raptor_strategy(
    max_levels: int = RAPTOR_MAX_LEVELS,
    min_cluster_size: int = RAPTOR_MIN_CLUSTER_SIZE,
    summary_model: str = RAPTOR_SUMMARY_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, int]:
    """RAPTOR hierarchical summarization tree.

    Algorithm:
    - Load existing semantic chunks (std=2) as leaves (level 0)
    - Embed -> UMAP reduction -> GMM clustering
    - LLM summarizes each cluster -> new level
    - Repeat until max_levels or too few nodes
    - All nodes (leaves + summaries) stored for collapsed tree retrieval

    Use case: Queries requiring both thematic overview AND specific details.
    Paper reports +20% comprehension on multi-step reasoning tasks.

    Note: Requires semantic_std2 chunks to exist first.
    Note: Significant LLM costs (~$0.40 for full corpus) and time (~2-3 min/book).

    Args:
        max_levels: Maximum tree depth (default: 4).
        min_cluster_size: Minimum nodes for clustering (default: 3).
        summary_model: OpenRouter model ID for summarization.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to total node counts (leaves + summaries).
    """
    from src.rag_pipeline.chunking.raptor.raptor_chunker import run_raptor_chunking

    logger.info(f"[raptor] Using hierarchical summarization (RAPTOR)")
    logger.info(f"[raptor] Max levels: {max_levels}, min cluster size: {min_cluster_size}")
    logger.info(f"[raptor] Summary model: {summary_model}")
    return run_raptor_chunking(
        max_levels=max_levels,
        min_cluster_size=min_cluster_size,
        summary_model=summary_model,
        overwrite_context=overwrite_context,
    )


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================


STRATEGIES: dict[str, ChunkingStrategyFunction] = {
    "section": section_strategy,
    "semantic": semantic_strategy,
    "contextual": contextual_strategy,
    "raptor": raptor_strategy,
}


def get_strategy(strategy_id: str, **kwargs: Any) -> ChunkingStrategyFunction:
    """Get chunking strategy function by ID.

    Args:
        strategy_id: One of "section", "semantic", "contextual", "raptor".
        **kwargs: Optional parameters to pass to the strategy function.
            Common: overwrite_context (OverwriteContext).
            For semantic strategy: std_coefficient (float).
            For contextual strategy: model (str, OpenRouter model ID).
            For raptor strategy: max_levels (int), min_cluster_size (int),
                summary_model (str).

    Returns:
        Strategy function that takes no args and returns Dict[str, int].

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy_fn = get_strategy("semantic", std_coefficient=2.0)
        >>> stats = strategy_fn()
        >>> print(stats)  # {"book1": 45, "book2": 67}
        >>> strategy_fn = get_strategy("contextual", model="deepseek/deepseek-v3.2")
        >>> stats = strategy_fn()
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown chunking strategy '{strategy_id}'. Available: {available}")

    strategy_fn = STRATEGIES[strategy_id]
    if kwargs:
        return partial(strategy_fn, **kwargs)
    return strategy_fn


def list_strategies() -> list[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["section", "semantic"]).
    """
    return list(STRATEGIES.keys())
