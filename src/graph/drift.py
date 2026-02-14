"""DRIFT search for GraphRAG global queries.

## RAG Theory: Simplified DRIFT (Dynamic Reasoning and Inference with Flexible Traversal)

DRIFT replaces brute-force map-reduce with selective community search.
Instead of processing ALL L0 communities (~1000 LLM calls), DRIFT:
1. Embeds the query and finds the top-K most relevant communities via Weaviate HNSW
2. Splits top-K into folds, runs parallel LLM "primer" calls per fold
3. Reduces all intermediate answers into a single final answer

This achieves ~99.5% reduction in LLM calls (1000 -> ~5) with comparable quality.

## Libraries

- Weaviate HNSW: O(log n) approximate nearest neighbor for community selection
- asyncio: Parallel primer LLM calls across folds

## Data Flow

Query -> embed_texts() -> Weaviate HNSW (top-20 communities)
     -> Split into 4 folds (5 communities each) -> 4 parallel primer LLM calls
     -> 4 intermediate answers (scored) -> 1 reduce LLM call -> final answer
Total: ~5 LLM calls (4 primer + 1 reduce) + 1 embed call
"""

import asyncio
import re
import time
from dataclasses import dataclass, field

from src.config import (
    DRIFT_TOP_K_COMMUNITIES,
    DRIFT_PRIMER_FOLDS,
    DRIFT_PRIMER_MAX_TOKENS,
    DRIFT_REDUCE_MAX_TOKENS,
    GRAPHRAG_SUMMARY_MODEL,
    get_community_collection_name,
)
from src.prompts import DRIFT_PRIMER_PROMPT, DRIFT_REDUCE_PROMPT
from src.rag_pipeline.embedder import embed_texts
from src.rag_pipeline.indexing.weaviate_client import (
    get_client as get_weaviate_client,
    query_communities_by_vector,
)
from src.shared.openrouter_client import call_chat_completion
from src.shared.files import setup_logging

logger = setup_logging(__name__)


@dataclass
class DriftResult:
    """Result from DRIFT search (replaces MapReduceResult for global queries).

    Attributes:
        final_answer: Synthesized answer from reduce phase.
        intermediate_answers: Scored answers from primer phase.
        communities_used: Community IDs that contributed.
        community_summaries: Full summary texts from retrieved communities (for evaluation).
        query_type: Always "global" for DRIFT.
        primer_time_ms: Time for primer phase (parallel LLM calls).
        reduce_time_ms: Time for reduce phase (single LLM call).
        total_time_ms: Total processing time.
        total_llm_calls: Number of LLM calls made (primer folds + reduce).
    """

    final_answer: str
    intermediate_answers: list[str] = field(default_factory=list)
    communities_used: list[str] = field(default_factory=list)
    community_summaries: list[str] = field(default_factory=list)
    query_type: str = "global"
    primer_time_ms: float = 0.0
    reduce_time_ms: float = 0.0
    total_time_ms: float = 0.0
    total_llm_calls: int = 0


def _parse_primer_response(response: str) -> tuple[str, float]:
    """Extract answer text and relevance score from primer LLM response.

    Args:
        response: Raw LLM response with [Score: X/10] prefix.

    Returns:
        Tuple of (answer_text, score). Falls back to score=5.0 if unparseable.
    """
    score = 5.0
    match = re.search(r"\[Score:\s*(\d+(?:\.\d+)?)\s*/\s*10\]", response)
    if match:
        score = float(match.group(1))

    # Strip the score line and [Answer] marker from the text
    answer = re.sub(r"\[Score:.*?/10\]\s*", "", response).strip()
    answer = re.sub(r"\[Answer\]\s*", "", answer).strip()

    return answer, score


async def _primer_fold_async(
    query: str,
    community_summaries: list[dict],
    model: str,
    max_tokens: int,
) -> tuple[str, float, list[str]]:
    """Process one fold of community reports in the primer phase.

    Args:
        query: User query.
        community_summaries: List of community dicts with 'community_id' and 'summary'.
        model: LLM model for primer call.
        max_tokens: Maximum response tokens.

    Returns:
        Tuple of (intermediate_answer, score, community_ids).
    """
    # Format community reports for the prompt
    reports = "\n\n".join(
        f"--- Community {c['community_id']} ---\n{c['summary']}"
        for c in community_summaries
    )

    prompt = DRIFT_PRIMER_PROMPT.format(query=query, community_reports=reports)

    loop = asyncio.get_running_loop()

    def _call():
        return call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
        )

    response = await loop.run_in_executor(None, _call)
    answer, score = _parse_primer_response(response)
    community_ids = [c["community_id"] for c in community_summaries]

    return answer, score, community_ids


async def drift_search_async(
    query: str,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    top_k: int = DRIFT_TOP_K_COMMUNITIES,
    primer_folds: int = DRIFT_PRIMER_FOLDS,
) -> DriftResult:
    """Execute DRIFT search over community summaries (async).

    Phase 1 (Primer): Embed query, find top-K communities via HNSW,
    split into folds, run parallel LLM calls.
    Phase 2 (Reduce): Synthesize all intermediate answers into final response.

    Args:
        query: User query string.
        model: LLM model for primer and reduce phases.
        top_k: Number of communities to retrieve via HNSW.
        primer_folds: Number of parallel primer LLM calls.

    Returns:
        DriftResult with final answer, timing, and LLM call counts.
    """
    start_time = time.time()

    # Phase 1: PRIMER
    # 1a. Embed query
    query_embedding = embed_texts([query])[0]

    # 1b. Find top-K communities via Weaviate HNSW
    collection_name = get_community_collection_name()
    client = get_weaviate_client()
    try:
        communities = query_communities_by_vector(
            client, collection_name, query_embedding, top_k=top_k,
        )
    finally:
        client.close()

    if not communities:
        logger.warning("DRIFT: No communities found via HNSW search")
        return DriftResult(
            final_answer="Unable to answer - no relevant communities found.",
            total_time_ms=(time.time() - start_time) * 1000,
        )

    # Collect community summary texts for evaluation (RAGAS contexts)
    community_summary_texts = [c["summary"] for c in communities if c.get("summary")]

    logger.info(f"DRIFT: Retrieved {len(communities)} communities via HNSW")

    # 1c. Split communities into folds
    fold_size = max(1, len(communities) // primer_folds)
    folds = [
        communities[i : i + fold_size]
        for i in range(0, len(communities), fold_size)
    ]

    # 1d. Parallel primer calls
    primer_start = time.time()
    tasks = [
        _primer_fold_async(query, fold, model, DRIFT_PRIMER_MAX_TOKENS)
        for fold in folds
    ]
    primer_results = await asyncio.gather(*tasks, return_exceptions=True)
    primer_time_ms = (time.time() - primer_start) * 1000

    # Collect results, filter errors and "not relevant" folds
    scored_answers = []
    all_community_ids = []
    primer_llm_calls = 0

    for result in primer_results:
        if isinstance(result, Exception):
            logger.warning(f"DRIFT primer fold error: {result}")
            continue
        primer_llm_calls += 1
        answer, score, community_ids = result
        if "not relevant" not in answer.lower() and score > 0:
            scored_answers.append((answer, score, community_ids))
            all_community_ids.extend(community_ids)

    logger.info(
        f"DRIFT primer: {len(scored_answers)}/{len(folds)} relevant folds "
        f"in {primer_time_ms:.0f}ms"
    )

    if not scored_answers:
        return DriftResult(
            final_answer="The retrieved communities do not contain information relevant to this query.",
            communities_used=all_community_ids,
            community_summaries=community_summary_texts,
            primer_time_ms=primer_time_ms,
            total_time_ms=(time.time() - start_time) * 1000,
            total_llm_calls=primer_llm_calls,
        )

    # Sort by score descending for the reduce prompt
    scored_answers.sort(key=lambda x: x[1], reverse=True)

    # Phase 2: REDUCE
    reduce_start = time.time()

    formatted_answers = "\n\n".join(
        f"[Relevance: {score:.0f}/10]\n{answer}"
        for answer, score, _ in scored_answers
    )

    reduce_prompt = DRIFT_REDUCE_PROMPT.format(
        query=query, intermediate_answers=formatted_answers,
    )

    final_answer = call_chat_completion(
        messages=[{"role": "user", "content": reduce_prompt}],
        model=model,
        temperature=0.3,
        max_tokens=DRIFT_REDUCE_MAX_TOKENS,
    )

    reduce_time_ms = (time.time() - reduce_start) * 1000
    total_time_ms = (time.time() - start_time) * 1000
    total_llm_calls = primer_llm_calls + 1  # +1 for reduce

    logger.info(
        f"DRIFT complete: {total_llm_calls} LLM calls, "
        f"{len(all_community_ids)} communities, {total_time_ms:.0f}ms total"
    )

    return DriftResult(
        final_answer=final_answer.strip(),
        intermediate_answers=[a for a, _, _ in scored_answers],
        communities_used=all_community_ids,
        community_summaries=community_summary_texts,
        primer_time_ms=primer_time_ms,
        reduce_time_ms=reduce_time_ms,
        total_time_ms=total_time_ms,
        total_llm_calls=total_llm_calls,
    )


def drift_search(
    query: str,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    top_k: int = DRIFT_TOP_K_COMMUNITIES,
    primer_folds: int = DRIFT_PRIMER_FOLDS,
) -> DriftResult:
    """Execute DRIFT search (sync wrapper).

    Convenience function for non-async contexts.

    Args:
        query: User query string.
        model: LLM model for primer and reduce.
        top_k: Number of communities to retrieve.
        primer_folds: Number of parallel primer LLM calls.

    Returns:
        DriftResult with final answer.
    """
    return asyncio.run(
        drift_search_async(query, model=model, top_k=top_k, primer_folds=primer_folds)
    )
