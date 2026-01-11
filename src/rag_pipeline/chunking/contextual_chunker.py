"""Contextual chunking: Add LLM-generated context to chunks.

## RAG Theory: Contextual Retrieval (Anthropic)

Traditional chunking loses document-level context when encoding chunks. A chunk
saying "The company's revenue grew by 3%" loses the context of which company
and time period. Contextual retrieval prepends a short LLM-generated snippet:

"[This chunk discusses ACME Corp's Q2 2023 financial performance...] The company's
revenue grew by 3%"

This improves embedding quality by:
1. Adding disambiguation terms (entities, topics)
2. Situating the chunk within broader arguments
3. Connecting isolated facts to their context

Anthropic reports 35% failure reduction (recall@20) with contextual embeddings,
up to 67% with BM25 hybrid + reranking.

## Library Usage

- src.shared.openrouter_client: LLM calls for context generation
- Uses section chunking as baseline (loads from section/ folder)
- Reuses chunk structure, only modifies text field

## Data Flow

1. Load existing chunks from DIR_FINAL_CHUNKS/section/{book}.json
2. For each chunk:
   a. Gather neighboring chunks as document context
   b. Call LLM with contextual prompt
   c. Prepend snippet to chunk text
3. Save to DIR_FINAL_CHUNKS/contextual/{book}.json
"""

import json
from pathlib import Path
from typing import Optional

from src.config import (
    DIR_FINAL_CHUNKS,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_NEIGHBOR_CHUNKS,
    CONTEXTUAL_MAX_SNIPPET_TOKENS,
    CONTEXTUAL_PROMPT,
)
from src.shared.openrouter_client import call_chat_completion, OpenRouterError
from src.shared.tokens import count_tokens
from src.shared.files import get_file_list, setup_logging, OverwriteContext, OverwriteMode

logger = setup_logging(__name__)

# Output folder name
CONTEXTUAL_FOLDER = "contextual"


# ============================================================================
# CONTEXT GENERATION
# ============================================================================


def gather_document_context(
    chunks: list[dict],
    current_index: int,
    neighbor_count: int = CONTEXTUAL_NEIGHBOR_CHUNKS,
    max_context_tokens: int = 2000,
) -> str:
    """Gather text from neighboring chunks as document context.

    Collects chunks before and after the current chunk to provide
    the LLM with surrounding document context.

    Args:
        chunks: List of all chunks in the document.
        current_index: Index of the chunk being contextualized.
        neighbor_count: Number of chunks to include before and after.
        max_context_tokens: Maximum tokens for the context window.

    Returns:
        Combined text from neighboring chunks, truncated if needed.
    """
    # Determine range of neighbors
    start_idx = max(0, current_index - neighbor_count)
    end_idx = min(len(chunks), current_index + neighbor_count + 1)

    # Gather neighbor texts (excluding current chunk)
    context_parts = []
    for i in range(start_idx, end_idx):
        if i != current_index:
            chunk_text = chunks[i].get("text", "")
            section = chunks[i].get("section", "")
            if chunk_text:
                # Add section marker for context
                if section:
                    context_parts.append(f"[{section}] {chunk_text}")
                else:
                    context_parts.append(chunk_text)

    context = "\n\n".join(context_parts)

    # Truncate if too long (simple character-based truncation)
    # Rough estimate: 4 chars per token
    max_chars = max_context_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "..."

    return context


def generate_contextual_snippet(
    chunk: dict,
    document_context: str,
    model: str = CONTEXTUAL_MODEL,
    max_tokens: int = CONTEXTUAL_MAX_SNIPPET_TOKENS,
) -> str:
    """Generate a contextual snippet for a chunk using LLM.

    Calls the LLM with the contextual prompt to generate a short
    snippet (2-3 sentences) situating the chunk within the document.

    Args:
        chunk: The chunk dict with 'text', 'context', 'book_id' fields.
        document_context: Text from neighboring chunks.
        model: OpenRouter model ID for generation.
        max_tokens: Maximum tokens for the generated snippet.

    Returns:
        Generated contextual snippet (50-100 tokens typically).
        Returns empty string if generation fails.
    """
    chunk_text = chunk.get("text", "")
    book_name = chunk.get("book_id", "Unknown")
    context_path = chunk.get("context", "")

    # Build prompt from template
    prompt = CONTEXTUAL_PROMPT.format(
        document_context=document_context,
        chunk_text=chunk_text,
        book_name=book_name,
        context_path=context_path,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        snippet = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,  # Some creativity but mostly factual
            max_tokens=max_tokens,
            timeout=30,
            max_retries=2,
        )
        return snippet.strip()
    except OpenRouterError as e:
        logger.warning(f"Context generation failed for chunk {chunk.get('chunk_id')}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Unexpected error generating context: {e}")
        return ""


# ============================================================================
# CHUNK PROCESSING
# ============================================================================


def contextualize_chunk(
    chunk: dict,
    contextual_snippet: str,
) -> dict:
    """Create a contextualized version of a chunk.

    Prepends the contextual snippet to the chunk text and updates
    metadata to reflect the contextual strategy.

    Args:
        chunk: Original chunk dict.
        contextual_snippet: LLM-generated context (or empty string).

    Returns:
        New chunk dict with contextualized text.
    """
    original_text = chunk.get("text", "")

    # Prepend snippet if available
    if contextual_snippet:
        contextualized_text = f"[{contextual_snippet}] {original_text}"
    else:
        contextualized_text = original_text

    # Create new chunk with updated fields
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "book_id": chunk.get("book_id", ""),
        "context": chunk.get("context", ""),
        "section": chunk.get("section", ""),
        "text": contextualized_text,
        "token_count": count_tokens(contextualized_text),
        "chunking_strategy": "contextual",
        "original_text": original_text,  # Preserve original for debugging
        "contextual_snippet": contextual_snippet,  # Store snippet separately
    }


def process_single_file(
    file_path: Path,
    model: str = CONTEXTUAL_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> tuple[str, int, int, bool]:
    """Process a single book's chunks with contextual enrichment.

    Loads existing section chunks, generates contextual snippets,
    and saves contextualized chunks to output directory.

    Args:
        file_path: Path to input JSON file (section chunks).
        model: OpenRouter model ID for context generation.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Tuple of (book_name, chunk_count, snippets_generated, was_processed).
        was_processed is False if file was skipped.
    """
    book_name = file_path.stem

    # Output path
    output_dir = Path(DIR_FINAL_CHUNKS) / CONTEXTUAL_FOLDER
    output_path = output_dir / f"{book_name}.json"

    # Check if we should process
    if overwrite_context is None:
        overwrite_context = OverwriteContext(OverwriteMode.ALL)
    if not overwrite_context.should_overwrite(output_path, logger):
        return book_name, 0, 0, False

    # Load existing chunks
    with file_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Processing {book_name} ({len(chunks)} chunks)")

    # Process each chunk
    contextualized_chunks = []
    snippets_generated = 0

    for i, chunk in enumerate(chunks):
        # Gather context from neighbors
        doc_context = gather_document_context(chunks, i)

        # Generate contextual snippet
        snippet = generate_contextual_snippet(chunk, doc_context, model=model)
        if snippet:
            snippets_generated += 1

        # Create contextualized chunk
        ctx_chunk = contextualize_chunk(chunk, snippet)
        contextualized_chunks.append(ctx_chunk)

        # Progress logging every 50 chunks
        if (i + 1) % 50 == 0:
            logger.info(f"  {book_name}: {i + 1}/{len(chunks)} chunks processed")

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(contextualized_chunks, f, ensure_ascii=False, indent=2)

    return book_name, len(contextualized_chunks), snippets_generated, True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_contextual_chunking(
    model: str = CONTEXTUAL_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> dict[str, int]:
    """Process all section chunks with contextual enrichment.

    Main entry point for contextual chunking strategy. Reads section chunks
    and adds LLM-generated contextual snippets to each chunk.

    Note: This is a POST-PROCESSING step on section chunks, not a new
    chunking algorithm. Run section chunking first if needed.

    Args:
        model: OpenRouter model ID for context generation.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts (only processed files).

    Raises:
        FileNotFoundError: If section chunks don't exist.
        Exception: Re-raises any exception from processing (fail-fast).
    """
    # Input: section chunks
    section_dir = Path(DIR_FINAL_CHUNKS) / "section"

    if not section_dir.exists():
        raise FileNotFoundError(
            f"Section chunks not found at {section_dir}. "
            "Run section chunking first: python -m src.stages.run_stage_4_chunking --strategy section"
        )

    input_files = list(section_dir.glob("*.json"))

    if not input_files:
        raise FileNotFoundError(
            f"No chunk files found in {section_dir}. "
            "Run section chunking first."
        )

    results = {}
    skipped_count = 0
    total_snippets = 0

    logger.info("Starting contextual chunking (Anthropic-style)...")
    logger.info(f"Processing {len(input_files)} files from section/")
    logger.info(f"Output folder: {CONTEXTUAL_FOLDER}/")
    logger.info(f"Context model: {model}")
    logger.info(f"Neighbor chunks: {CONTEXTUAL_NEIGHBOR_CHUNKS} before + after")

    for file_path in sorted(input_files):
        book_name, chunk_count, snippets, was_processed = process_single_file(
            file_path, model=model, overwrite_context=overwrite_context
        )
        if was_processed:
            results[book_name] = chunk_count
            total_snippets += snippets
            logger.info(f"  {book_name}: {chunk_count} chunks, {snippets} snippets generated")
        else:
            skipped_count += 1

    logger.info(f"Contextual chunking complete: {sum(results.values())} total chunks")
    logger.info(f"Contextual snippets generated: {total_snippets}")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files (already exist)")

    return results


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================


if __name__ == "__main__":
    logger.info("Starting contextual chunking (standalone)...")
    stats = run_contextual_chunking()
    logger.info(f"Completed! Processed {len(stats)} books")
    logger.info(f"Total chunks: {sum(stats.values())}")
