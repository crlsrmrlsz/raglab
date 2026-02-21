"""Reciprocal Rank Fusion for multi-query retrieval.

## RAG Theory: RRF Merging

When using multiple search queries, each returns a ranked list of results.
RRF (Robertson et al., 1993) provides a simple, effective way to merge these
lists without requiring score normalization.

Formula: RRF_score(d) = sum(1 / (k + rank(d, q))) for each query q

Key properties:
- k parameter (default 60) controls how quickly lower ranks lose influence
- Higher k = more influence to lower-ranked items
- Results appearing in multiple query results get boosted

## Library Usage

Uses only standard library (collections.defaultdict) for the merge algorithm.
Works with SearchResult dataclass from vector_db.

## Data Flow

1. Execute each query independently via weaviate_query functions
2. Track chunk_id -> rank for each result list
3. Compute RRF scores across all lists
4. Sort by RRF score, return top-k
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.rag_pipeline.indexing.weaviate_query import SearchResult
from src.shared.files import setup_logging


logger = setup_logging(__name__)


# Standard k value from RRF literature (Cormack et al., 2009)
RRF_K = 60


@dataclass
class RRFResult:
    """Result of RRF merging operation.

    Attributes:
        results: Merged and sorted SearchResult list with RRF scores.
        query_contributions: Maps chunk_id to list of query types that found it.
        merge_time_ms: Time taken for merge operation.
    """

    results: list[SearchResult]
    query_contributions: dict[str, list[str]] = field(default_factory=dict)
    merge_time_ms: float = 0.0


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    query_types: list[str],
    k: int = RRF_K,
    top_k: int = 10,
) -> RRFResult:
    """Merge multiple result lists using Reciprocal Rank Fusion.

    RRF is a rank-based fusion method that doesn't require score normalization.
    Documents appearing in multiple result lists get boosted, with higher-ranked
    appearances contributing more to the final score.

    Args:
        result_lists: List of SearchResult lists, one per query.
        query_types: Labels for each query (e.g., ["original", "neuroscience"]).
        k: RRF k parameter (default: 60). Higher values give more weight to
           lower-ranked items.
        top_k: Number of results to return after merging.

    Returns:
        RRFResult with merged results sorted by RRF score.

    Example:
        >>> results_q1 = [SearchResult(chunk_id="a", score=0.9, ...), ...]
        >>> results_q2 = [SearchResult(chunk_id="b", score=0.8, ...), ...]
        >>> rrf = reciprocal_rank_fusion(
        ...     [results_q1, results_q2],
        ...     ["neuroscience", "philosophy"],
        ...     top_k=10
        ... )
        >>> len(rrf.results) <= 10
        True
    """
    start_time = time.time()

    # Track scores and contributions
    scores: dict[str, float] = defaultdict(float)
    results_by_id: dict[str, SearchResult] = {}
    contributions: dict[str, list[str]] = defaultdict(list)

    for query_idx, results in enumerate(result_lists):
        query_type = (
            query_types[query_idx] if query_idx < len(query_types) else f"query_{query_idx}"
        )

        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # RRF formula: 1 / (k + rank + 1)
            # Note: rank is 0-indexed, so we add 1 to match 1-indexed ranking
            rrf_score = 1.0 / (k + rank + 1)
            scores[chunk_id] += rrf_score

            # Keep the result object (first occurrence or highest original score)
            if chunk_id not in results_by_id or result.score > results_by_id[chunk_id].score:
                results_by_id[chunk_id] = result

            contributions[chunk_id].append(query_type)

    # Sort by RRF score (descending)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build merged results with RRF scores
    merged_results = []
    for chunk_id in sorted_ids[:top_k]:
        original_result = results_by_id[chunk_id]
        # Create new SearchResult with RRF score replacing original score
        merged_result = SearchResult(
            chunk_id=original_result.chunk_id,
            book_id=original_result.book_id,
            section=original_result.section,
            context=original_result.context,
            text=original_result.text,
            token_count=original_result.token_count,
            score=scores[chunk_id],  # RRF score
        )
        merged_results.append(merged_result)

    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(
        f"[RRF] Merged {len(result_lists)} queries, "
        f"{sum(len(r) for r in result_lists)} total results -> {len(merged_results)} unique"
    )

    return RRFResult(
        results=merged_results,
        query_contributions=dict(contributions),
        merge_time_ms=elapsed_ms,
    )
