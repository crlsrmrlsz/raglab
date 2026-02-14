"""Query classification for GraphRAG local vs global routing.

## RAG Theory: Query Classification

Microsoft GraphRAG (arXiv:2404.16130) distinguishes between:
- **Local queries**: "What is dopamine?" -> Entity traversal + vector search
- **Global queries**: "What are the main themes?" -> DRIFT search over communities

This module provides the classification logic used by DRIFT (src/graph/drift.py)
to determine when to activate community-based search.

## Data Flow

1. User query -> classify_query() -> "local" or "global"
2. should_use_map_reduce() wraps classification for boolean dispatch
"""

from src.config import GRAPHRAG_EXTRACTION_MODEL
from src.prompts import GRAPHRAG_CLASSIFICATION_PROMPT
from src.shared.openrouter_client import call_chat_completion
from src.shared.files import setup_logging

logger = setup_logging(__name__)


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


def should_use_map_reduce(query: str) -> bool:
    """Determine if a global (DRIFT) search should be used for a query.

    Uses LLM classification to decide if the query is global
    (themes, patterns, overviews) or local (specific entities, facts).

    Args:
        query: User query.

    Returns:
        True if global search should be used, False for local retrieval.

    Example:
        >>> should_use_map_reduce("What is dopamine?")
        False
        >>> should_use_map_reduce("What are the main themes?")
        True
    """
    query_type = classify_query(query)
    return query_type == "global"
