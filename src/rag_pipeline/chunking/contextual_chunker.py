"""Contextual chunking: Add LLM-generated context to chunks.

## RAG Theory: Contextual Retrieval (Anthropic)

Traditional chunking loses document-level context when encoding chunks. A chunk
saying "In 1954, James Olds and Peter Milner implanted electrodes in a rat..."
loses the context that this is about the ventral striatum and pleasure/reward.
Contextual retrieval prepends a short LLM-generated snippet:

"[This chunk discusses the ventral striatum's role in pleasure and reward...]
In 1954, James Olds and Peter Milner implanted electrodes in a rat..."

Anthropic's original approach passes the FULL DOCUMENT to the LLM for each chunk.
This is impractical for books (300-800 pages), so this implementation uses:

- Book title (LLM may have knowledge of well-known books/authors)
- Section title (contains disambiguation terms like "Ventral Striatum: Pleasure and Reward")
- The chunk text

The LLM's job is to connect the chunk's content to the section title's concepts.

Anthropic reports 35% failure reduction (recall@20) with contextual embeddings,
up to 67% with BM25 hybrid + reranking.

## Library Usage

- src.shared.openrouter_client: LLM calls for context generation
- Uses semantic chunking (std=2) as baseline (loads from semantic_std2/ folder)
- Reuses chunk structure, only modifies text field

## Data Flow

1. Load existing chunks from DIR_FINAL_CHUNKS/semantic_std2/{book}.json
2. For each chunk:
   a. Get book title and section title
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
    CONTEXTUAL_MAX_SNIPPET_TOKENS,
    CONTEXTUAL_PROMPT,
)
from src.shared.openrouter_client import call_chat_completion, OpenRouterError
from src.shared.tokens import count_tokens
from src.shared.files import setup_logging, OverwriteContext, OverwriteMode

logger = setup_logging(__name__)

CONTEXTUAL_FOLDER = "contextual"


# ============================================================================
# CONTEXT GENERATION
# ============================================================================


def generate_contextual_snippet(
    chunk: dict,
    model: str = CONTEXTUAL_MODEL,
    max_tokens: int = CONTEXTUAL_MAX_SNIPPET_TOKENS,
) -> str:
    """Generate a contextual snippet for a chunk using LLM.

    Calls the LLM with the book title, section title, and chunk text
    to generate a short snippet situating the chunk for improved retrieval.

    Args:
        chunk: The chunk dict with 'text', 'book_id', 'section' fields.
        model: OpenRouter model ID for generation.
        max_tokens: Maximum tokens for the generated snippet.

    Returns:
        Generated contextual snippet (50-100 tokens typically).
        Returns empty string if generation fails.
    """
    chunk_text = chunk.get("text", "")
    book_title = chunk.get("book_id", "Unknown")
    section_title = chunk.get("section", "Unknown")

    prompt = CONTEXTUAL_PROMPT.format(
        book_title=book_title,
        section_title=section_title,
        chunk_text=chunk_text,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        snippet = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.3,
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


def contextualize_chunk(chunk: dict, contextual_snippet: str) -> dict:
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

    if contextual_snippet:
        contextualized_text = f"[{contextual_snippet}] {original_text}"
    else:
        contextualized_text = original_text

    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "book_id": chunk.get("book_id", ""),
        "context": chunk.get("context", ""),
        "section": chunk.get("section", ""),
        "text": contextualized_text,
        "token_count": count_tokens(contextualized_text),
        "chunking_strategy": "contextual",
        "original_text": original_text,
        "contextual_snippet": contextual_snippet,
    }


def process_single_file(
    file_path: Path,
    model: str = CONTEXTUAL_MODEL,
    overwrite_context: Optional[OverwriteContext] = None,
) -> tuple[str, int, int, bool]:
    """Process a single book's chunks with contextual enrichment.

    Args:
        file_path: Path to input JSON file (semantic chunks).
        model: OpenRouter model ID for context generation.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Tuple of (book_name, chunk_count, snippets_generated, was_processed).
    """
    book_name = file_path.stem

    output_dir = Path(DIR_FINAL_CHUNKS) / CONTEXTUAL_FOLDER
    output_path = output_dir / f"{book_name}.json"

    if overwrite_context is None:
        overwrite_context = OverwriteContext(OverwriteMode.ALL)
    if not overwrite_context.should_overwrite(output_path, logger):
        return book_name, 0, 0, False

    with file_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Processing {book_name} ({len(chunks)} chunks)")

    contextualized_chunks = []
    snippets_generated = 0

    for i, chunk in enumerate(chunks):
        snippet = generate_contextual_snippet(chunk, model=model)
        if snippet:
            snippets_generated += 1

        ctx_chunk = contextualize_chunk(chunk, snippet)
        contextualized_chunks.append(ctx_chunk)

        if (i + 1) % 50 == 0:
            logger.info(f"  {book_name}: {i + 1}/{len(chunks)} chunks processed")

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
    """Process all semantic chunks with contextual enrichment.

    Reads semantic chunks (std=2) and adds LLM-generated contextual
    snippets using book title and section title.

    Args:
        model: OpenRouter model ID for context generation.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts (only processed files).

    Raises:
        FileNotFoundError: If semantic chunks (std=2) don't exist.
    """
    input_dir = Path(DIR_FINAL_CHUNKS) / "semantic_std2"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Semantic chunks not found at {input_dir}. "
            "Run semantic chunking first: python -m src.stages.run_stage_4_chunking --strategy semantic --std-coefficient 2.0"
        )

    input_files = list(input_dir.glob("*.json"))

    if not input_files:
        raise FileNotFoundError(
            f"No chunk files found in {input_dir}. "
            "Run semantic chunking first."
        )

    results = {}
    skipped_count = 0
    total_snippets = 0

    logger.info("Starting contextual chunking...")
    logger.info(f"Processing {len(input_files)} files from semantic_std2/")
    logger.info(f"Output folder: {CONTEXTUAL_FOLDER}/")
    logger.info(f"Model: {model}")

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


if __name__ == "__main__":
    logger.info("Starting contextual chunking (standalone)...")
    stats = run_contextual_chunking()
    logger.info(f"Completed! Processed {len(stats)} books")
    logger.info(f"Total chunks: {sum(stats.values())}")
