"""Map-reduce query handling for GraphRAG global queries.

## RAG Theory: Map-Reduce for Global Queries

Microsoft GraphRAG (arXiv:2404.16130) uses map-reduce for global queries:

**Local queries**: "What is dopamine?" → Entity traversal + vector search
**Global queries**: "What are the main themes?" → Map-reduce over L0 communities

For abstract queries like "What are the main themes?", we:
1. Retrieve L0 communities (coarsest level = corpus-wide themes)
2. Map: Generate partial answer per community (parallel LLM calls)
3. Reduce: Synthesize partial answers into coherent response

Benefits:
- L0 communities capture corpus-wide themes (fewer, larger communities)
- Parallelizes LLM calls (50% latency reduction with async)
- Captures diverse perspectives from different communities
- Scales to large corpora without exceeding context limits

## Data Flow

1. Query classification: local vs global
2. For global: Retrieve ALL L0 (coarsest) communities (matches Microsoft)
3. Map: Parallel LLM calls → partial answers
4. Reduce: Single LLM call → final synthesis
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import (
    GRAPHRAG_MAP_MAX_TOKENS,
    GRAPHRAG_REDUCE_MAX_TOKENS,
    GRAPHRAG_SUMMARY_MODEL,
    GRAPHRAG_EXTRACTION_MODEL,
)
from src.prompts import (
    GRAPHRAG_CLASSIFICATION_PROMPT,
    GRAPHRAG_MAP_PROMPT,
    GRAPHRAG_REDUCE_PROMPT,
)
from src.shared.openrouter_client import call_chat_completion
from src.shared.files import setup_logging
from .schemas import Community

logger = setup_logging(__name__)


@dataclass
class MapReduceResult:
    """Result from map-reduce query processing.

    Attributes:
        final_answer: Synthesized answer from reduce phase.
        partial_answers: List of community-specific answers (map phase).
        communities_used: List of community IDs that contributed.
        query_type: Classification result ("local" or "global").
        map_time_ms: Time for map phase (parallel LLM calls).
        reduce_time_ms: Time for reduce phase (single LLM call).
        total_time_ms: Total processing time.

    Example:
        >>> result = await map_reduce_global_query(query, communities)
        >>> print(f"Answer: {result.final_answer}")
        >>> print(f"Map: {result.map_time_ms}ms, Reduce: {result.reduce_time_ms}ms")
    """

    final_answer: str
    partial_answers: list[str] = field(default_factory=list)
    communities_used: list[str] = field(default_factory=list)
    query_type: str = "global"
    map_time_ms: float = 0.0
    reduce_time_ms: float = 0.0
    total_time_ms: float = 0.0


def classify_query(
    query: str,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> str:
    """Classify query as 'local' or 'global'.

    Local queries ask about specific entities, concepts, or facts.
    Global queries ask about themes, patterns, or overviews.

    Args:
        query: User query string.
        model: LLM model for classification (fast model recommended).

    Returns:
        "local" or "global"

    Example:
        >>> classify_query("What is dopamine?")
        'local'
        >>> classify_query("What are the main themes in this corpus?")
        'global'
    """
    prompt = GRAPHRAG_CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=10,
        )

        # Parse response (expect "local" or "global")
        result = response.strip().lower()

        if result not in ("local", "global"):
            # Default to local if unclear
            logger.warning(
                f"Unclear classification '{result}' for query, defaulting to local"
            )
            return "local"

        logger.info(f"Query classified as: {result}")
        return result

    except Exception as e:
        logger.warning(f"Query classification failed: {e}, defaulting to local")
        return "local"


def _format_community_context(community: Community, max_members: int = 5) -> dict[str, str]:
    """Format community data for map prompt.

    Args:
        community: Community object with summary, members, relationships.
        max_members: Maximum members to include.

    Returns:
        Dict with formatted strings for prompt template.
    """
    # Top members by PageRank
    sorted_members = sorted(
        community.members,
        key=lambda m: m.pagerank,
        reverse=True,
    )[:max_members]

    top_entities = "\n".join(
        f"- {m.entity_name} ({m.entity_type}): {m.description[:100]}"
        for m in sorted_members
        if m.description
    ) or "No entity details available."

    # Top relationships
    relationships = "\n".join(
        f"- {r.source} --[{r.relationship_type}]--> {r.target}"
        for r in community.relationships[:5]
    ) or "No relationship details available."

    return {
        "community_summary": community.summary,
        "top_entities": top_entities,
        "relationships": relationships,
    }


async def _map_community_async(
    query: str,
    community: Community,
    model: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Async map phase: Generate partial answer from one community.

    Args:
        query: User query.
        community: Community object.
        model: LLM model.
        max_tokens: Maximum response tokens.

    Returns:
        Tuple of (community_id, partial_answer).
    """
    context = _format_community_context(community)

    prompt = GRAPHRAG_MAP_PROMPT.format(
        query=query,
        community_summary=context["community_summary"],
        top_entities=context["top_entities"],
        relationships=context["relationships"],
    )

    # Run LLM call in executor (OpenRouter client is sync)
    loop = asyncio.get_event_loop()

    def _call():
        return call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
        )

    partial_answer = await loop.run_in_executor(None, _call)

    return community.community_id, partial_answer.strip()


async def map_reduce_global_query_async(
    query: str,
    communities: list[Community],
    model: str = GRAPHRAG_SUMMARY_MODEL,
    map_max_tokens: int = GRAPHRAG_MAP_MAX_TOKENS,
    reduce_max_tokens: int = GRAPHRAG_REDUCE_MAX_TOKENS,
) -> MapReduceResult:
    """Execute map-reduce query across communities (async).

    Map phase runs in parallel for better latency.
    Reduce phase synthesizes all partial answers.

    Args:
        query: User query.
        communities: List of Community objects to process.
        model: LLM model for both map and reduce.
        map_max_tokens: Max tokens per map response.
        reduce_max_tokens: Max tokens for reduce response.

    Returns:
        MapReduceResult with final answer and timing info.

    Example:
        >>> communities = retrieve_community_context(query, top_k=5)
        >>> result = await map_reduce_global_query_async(query, communities)
        >>> print(result.final_answer)
    """
    start_time = time.time()

    if not communities:
        logger.warning("No communities provided for map-reduce")
        return MapReduceResult(
            final_answer="Unable to answer - no relevant communities found.",
            query_type="global",
        )

    # MAP PHASE: Parallel partial answer generation
    map_start = time.time()

    tasks = [
        _map_community_async(query, community, model, map_max_tokens)
        for community in communities
    ]

    map_results = await asyncio.gather(*tasks, return_exceptions=True)

    map_time_ms = (time.time() - map_start) * 1000

    # Process results, handling any errors
    partial_answers = []
    communities_used = []

    for result in map_results:
        if isinstance(result, Exception):
            logger.warning(f"Map phase error: {result}")
            continue

        community_id, partial_answer = result

        # Filter out "Not relevant" responses
        if "not relevant" not in partial_answer.lower():
            partial_answers.append(partial_answer)
            communities_used.append(community_id)

    logger.info(
        f"Map phase: {len(partial_answers)}/{len(communities)} relevant responses "
        f"in {map_time_ms:.0f}ms"
    )

    if not partial_answers:
        return MapReduceResult(
            final_answer="The retrieved communities do not contain information relevant to this query.",
            communities_used=communities_used,
            query_type="global",
            map_time_ms=map_time_ms,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    # REDUCE PHASE: Synthesize partial answers
    reduce_start = time.time()

    formatted_partials = "\n\n".join(
        f"Community {i+1}:\n{answer}"
        for i, answer in enumerate(partial_answers)
    )

    reduce_prompt = GRAPHRAG_REDUCE_PROMPT.format(
        query=query,
        partial_answers=formatted_partials,
    )

    final_answer = call_chat_completion(
        messages=[{"role": "user", "content": reduce_prompt}],
        model=model,
        temperature=0.3,
        max_tokens=reduce_max_tokens,
    )

    reduce_time_ms = (time.time() - reduce_start) * 1000
    total_time_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Reduce phase: synthesized {len(partial_answers)} answers "
        f"in {reduce_time_ms:.0f}ms (total: {total_time_ms:.0f}ms)"
    )

    return MapReduceResult(
        final_answer=final_answer.strip(),
        partial_answers=partial_answers,
        communities_used=communities_used,
        query_type="global",
        map_time_ms=map_time_ms,
        reduce_time_ms=reduce_time_ms,
        total_time_ms=total_time_ms,
    )


def map_reduce_global_query(
    query: str,
    communities: list[Community],
    model: str = GRAPHRAG_SUMMARY_MODEL,
    map_max_tokens: int = GRAPHRAG_MAP_MAX_TOKENS,
    reduce_max_tokens: int = GRAPHRAG_REDUCE_MAX_TOKENS,
) -> MapReduceResult:
    """Execute map-reduce query (sync wrapper for async version).

    Convenience function for non-async contexts.

    Args:
        query: User query.
        communities: List of Community objects.
        model: LLM model.
        map_max_tokens: Max tokens per map response.
        reduce_max_tokens: Max tokens for reduce response.

    Returns:
        MapReduceResult with final answer.

    Example:
        >>> communities = retrieve_community_context(query, top_k=5)
        >>> result = map_reduce_global_query(query, communities)
        >>> print(result.final_answer)
    """
    return asyncio.run(
        map_reduce_global_query_async(
            query=query,
            communities=communities,
            model=model,
            map_max_tokens=map_max_tokens,
            reduce_max_tokens=reduce_max_tokens,
        )
    )


def should_use_map_reduce(query: str) -> bool:
    """Determine if map-reduce should be used for a query.

    Uses LLM classification to decide if the query is global
    (themes, patterns, overviews) or local (specific entities, facts).

    Args:
        query: User query.

    Returns:
        True if map-reduce should be used, False for local retrieval.

    Example:
        >>> should_use_map_reduce("What is dopamine?")
        False
        >>> should_use_map_reduce("What are the main themes?")
        True
    """
    query_type = classify_query(query)
    return query_type == "global"
