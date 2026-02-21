"""LLM-based cluster summarization for RAPTOR.

## RAG Theory: Cluster Summarization

RAPTOR builds tree levels by summarizing clusters of similar chunks:

1. **Cluster Contents**: After GMM clustering, we have groups of semantically
   related chunks. Each cluster captures a theme or topic from the document.

2. **LLM Summarization**: An LLM generates a comprehensive summary that:
   - Captures the main ideas from all cluster members
   - Preserves key details, names, and concepts
   - Creates a higher-level abstraction useful for thematic queries

3. **Context Inheritance**: Summary nodes get context strings derived from
   their children, enabling hierarchical navigation.

## Library Usage

- src.shared.openrouter_client: LLM API calls (same as contextual chunking)
- Uses RAPTOR_SUMMARY_PROMPT from config

## Data Flow

1. Input: List of RaptorNodes (cluster members)
2. Concatenate member texts (truncated if needed)
3. LLM call with summary prompt
4. Output: Summary text (100-150 tokens typical)
"""


from src.config import (
    RAPTOR_SUMMARY_MODEL,
    RAPTOR_SUMMARY_PROMPT,
    RAPTOR_MAX_SUMMARY_TOKENS,
    RAPTOR_MAX_CONTEXT_TOKENS,
)
from src.shared.openrouter_client import call_chat_completion, OpenRouterError
from src.shared.tokens import count_tokens
from src.shared.files import setup_logging
from src.rag_pipeline.chunking.raptor.schemas import RaptorNode

logger = setup_logging(__name__)


def generate_cluster_summary(
    nodes: list[RaptorNode],
    model: str = RAPTOR_SUMMARY_MODEL,
    max_context_tokens: int = RAPTOR_MAX_CONTEXT_TOKENS,
    max_summary_tokens: int = RAPTOR_MAX_SUMMARY_TOKENS,
) -> str:
    """Generate LLM summary for a cluster of nodes.

    Concatenates the text from all nodes in the cluster and calls the LLM
    to generate a comprehensive summary.

    Args:
        nodes: List of RaptorNodes to summarize.
        model: OpenRouter model ID for summarization.
        max_context_tokens: Maximum tokens for input context.
        max_summary_tokens: Maximum tokens for output summary.

    Returns:
        Summary text (typically 100-150 tokens).

    Raises:
        OpenRouterError: If LLM call fails after retries.
        ValueError: If nodes list is empty.
    """
    if not nodes:
        raise ValueError("Cannot summarize empty cluster")

    # Concatenate node texts with separators
    context_parts = []
    total_tokens = 0

    for node in nodes:
        node_text = node.text.strip()
        node_tokens = count_tokens(node_text)

        # Check if adding this node would exceed limit
        if total_tokens + node_tokens > max_context_tokens:
            # Truncate this node to fit
            remaining_tokens = max_context_tokens - total_tokens
            if remaining_tokens > 50:  # Only include if meaningful amount left
                # Rough truncation (4 chars per token)
                truncate_chars = remaining_tokens * 4
                node_text = node_text[:truncate_chars] + "..."
                context_parts.append(node_text)
            break

        context_parts.append(node_text)
        total_tokens += node_tokens

    # Join with double newlines for clear separation
    context = "\n\n---\n\n".join(context_parts)

    # Build prompt from template
    prompt = RAPTOR_SUMMARY_PROMPT.format(context=context)

    logger.info(
        f"Summarizing cluster: {len(nodes)} nodes, "
        f"~{total_tokens} input tokens"
    )

    try:
        summary = call_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3,  # Some creativity but mostly factual
            max_tokens=max_summary_tokens,
            timeout=30,
            max_retries=2,
        )
        return summary.strip()

    except OpenRouterError as e:
        logger.error(f"Cluster summarization failed: {e}")
        raise


def create_summary_context(
    nodes: list[RaptorNode],
    level: int,
    cluster_id: int,
) -> str:
    """Create hierarchical context string for a summary node.

    Derives context from child nodes to describe what this summary covers.
    For a cluster of chapter 1-3 content, might produce:
    "Book > Chapters 1-3 Summary"

    Args:
        nodes: Child nodes being summarized.
        level: Tree level for this summary (1, 2, 3, ...).
        cluster_id: Cluster identifier.

    Returns:
        Context string for the summary node.
    """
    if not nodes:
        return f"Level {level} Summary"

    # Get unique book_ids (should all be same, but be safe)
    book_ids = set(node.book_id for node in nodes)
    book_id = list(book_ids)[0] if len(book_ids) == 1 else "Multiple Books"

    # Get unique sections from children
    sections = set()
    for node in nodes:
        if node.section:
            sections.add(node.section)

    # Build context based on what we find
    if len(sections) <= 3:
        sections_str = ", ".join(sorted(sections)[:3])
        return f"{book_id} > Level {level} Summary ({sections_str})"
    else:
        return f"{book_id} > Level {level} Summary ({len(sections)} sections)"


def create_summary_section(level: int, cluster_id: int) -> str:
    """Create section name for a summary node.

    Args:
        level: Tree level for this summary.
        cluster_id: Cluster identifier.

    Returns:
        Section name string.
    """
    return f"Level {level} Cluster {cluster_id} Summary"
