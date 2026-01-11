"""Semantic chunking: Split text by embedding similarity.

## RAG Theory: Semantic Coherence

Traditional chunking (fixed tokens, sentence count) often splits mid-topic,
fragmenting semantically related content across chunks. Semantic chunking
uses embedding similarity to detect topic boundaries:

1. Embed each sentence using the same model as Stage 5
2. Compute cosine similarity between adjacent sentences
3. Split when similarity drops below threshold (topic shift)
4. Still respect section boundaries and token limits

Research shows 8-12% improvement in retrieval precision for Q&A tasks when
using semantic boundaries vs fixed-size chunks.

## Library Usage

- numpy: Cosine similarity computation
- src.rag_pipeline.embedding.embedder: API embeddings (same as Stage 5)
- Reuses section_chunker helpers (split_oversized_sentence, parse_section_name)

## Data Flow

1. Load paragraphs from DIR_NLP_CHUNKS/{book}.json
2. For each paragraph:
   a. Embed all sentences (batch API call)
   b. Compute pairwise cosine similarity
   c. Find breakpoints where similarity < threshold
3. Build chunks respecting breakpoints and token limits
4. Add 2-sentence overlap between chunks (same section only)
5. Save to DIR_FINAL_CHUNKS/semantic/{book}.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Deque
from collections import deque

from src.config import (
    DIR_FINAL_CHUNKS,
    DIR_NLP_CHUNKS,
    EMBEDDING_MAX_INPUT_TOKENS,
    OVERLAP_SENTENCES,
    CHUNK_ID_PREFIX,
    CHUNK_ID_SEPARATOR,
    SEMANTIC_SIMILARITY_THRESHOLD,
    get_semantic_folder_name,
)
from src.rag_pipeline.embedding.embedder import embed_texts
from src.rag_pipeline.chunking.section_chunker import (
    split_oversized_sentence,
    parse_section_name,
)
from src.shared.tokens import count_tokens
from src.shared.files import get_file_list, setup_logging, OverwriteContext, OverwriteMode

logger = setup_logging(__name__)


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================


def compute_similarity_breakpoints(
    sentences: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[int]:
    """Find semantic breakpoints by embedding similarity.

    Computes cosine similarity between adjacent sentences and identifies
    indices where similarity drops below threshold, indicating topic shifts.

    Args:
        sentences: List of sentence strings.
        threshold: Cosine similarity threshold (0-1). Lower values mean
            more splits. Default 0.75 works well for most content.

    Returns:
        List of indices where new chunks should start (0-indexed).
        Always includes 0 (first chunk starts at index 0).

    Example:
        >>> sentences = ["Intro.", "More intro.", "New topic!", "Continues."]
        >>> # If similarity between "More intro." and "New topic!" < threshold
        >>> breakpoints = compute_similarity_breakpoints(sentences)
        >>> breakpoints  # [0, 2] - split before "New topic!"
    """
    if len(sentences) <= 1:
        return [0]

    # Embed all sentences (batch API call)
    try:
        embeddings = embed_texts(sentences)
    except Exception as e:
        logger.warning(f"Embedding failed, using paragraph boundaries: {e}")
        return [0]  # Fall back to no splitting

    if not embeddings or len(embeddings) != len(sentences):
        logger.warning("Embedding count mismatch, using paragraph boundaries")
        return [0]

    embeddings_array = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings_array / norms

    # Compute pairwise cosine similarity (adjacent sentences only)
    similarities = []
    for i in range(len(normalized) - 1):
        sim = np.dot(normalized[i], normalized[i + 1])
        similarities.append(float(sim))

    # Find breakpoints where similarity drops below threshold
    breakpoints = [0]  # Always start at 0
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)  # Start new chunk at next sentence
            logger.debug(f"Semantic split at index {i+1}, similarity={sim:.3f}")

    return breakpoints


# ============================================================================
# CHUNK CREATION
# ============================================================================


def _create_chunk_dict(
    text: str,
    context: str,
    book_name: str,
    chunk_id: int,
    similarity_threshold: float,
) -> dict:
    """Create standardized chunk dictionary with semantic strategy marker.

    Args:
        text: Chunk text content.
        context: Hierarchical context string (Book > Chapter > Section).
        book_name: Book identifier.
        chunk_id: Sequential chunk number.
        similarity_threshold: Actual threshold used for this chunking run.

    Returns:
        Chunk dictionary with all required metadata fields.
    """
    return {
        "chunk_id": f"{book_name}{CHUNK_ID_SEPARATOR}{CHUNK_ID_PREFIX}_{chunk_id}",
        "book_id": book_name,
        "context": context,
        "section": parse_section_name(context),
        "text": text,
        "token_count": count_tokens(text),
        "chunking_strategy": f"semantic_{similarity_threshold}",
    }


def create_semantic_chunks(
    paragraphs: list[dict],
    book_name: str,
    max_tokens: int = EMBEDDING_MAX_INPUT_TOKENS,
    overlap_sentences: int = OVERLAP_SENTENCES,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Create chunks using semantic similarity breakpoints.

    Algorithm:
    1. Process paragraphs in order, respecting section boundaries
    2. For each paragraph, find semantic breakpoints via embedding similarity
    3. Build chunks from breakpoint segments (semantic coherence drives size)
    4. Add sentence overlap between chunks (same section only)

    The max_tokens parameter is a safeguard against embedding model limits,
    not an optimization target. Semantic boundaries (similarity < threshold)
    are the primary split mechanism.

    Args:
        paragraphs: List of paragraph dicts with 'context', 'sentences' keys.
        book_name: Book identifier.
        max_tokens: Safeguard limit for embedding model (default: 8191).
        overlap_sentences: Number of sentences to overlap between chunks.
        similarity_threshold: Cosine similarity threshold for splitting.

    Returns:
        List of chunk dicts ready for embedding.
    """
    chunks = []
    chunk_id = 0
    current_context = None
    current_chunk_sentences: list[str] = []
    num_overlap_sentences = 0

    # Overlap buffer for continuity between chunks
    overlap_buffer: Deque[str] = deque(maxlen=overlap_sentences if overlap_sentences > 0 else None)

    def _save_current_chunk():
        """Save current chunk and update overlap buffer."""
        nonlocal chunk_id, num_overlap_sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(_create_chunk_dict(
                text=chunk_text,
                context=current_context,
                book_name=book_name,
                chunk_id=chunk_id,
                similarity_threshold=similarity_threshold,
            ))
            chunk_id += 1

            # Update overlap buffer
            if overlap_sentences > 0:
                overlap_buffer.clear()
                overlap_buffer.extend(current_chunk_sentences[-overlap_sentences:])

            num_overlap_sentences = 0

    def _start_new_chunk_with_overlap() -> list[str]:
        """Initialize new chunk with overlap from previous."""
        nonlocal num_overlap_sentences
        if overlap_sentences > 0 and len(overlap_buffer) > 0:
            num_overlap_sentences = len(overlap_buffer)
            return list(overlap_buffer)
        num_overlap_sentences = 0
        return []

    for paragraph in paragraphs:
        context = paragraph.get("context", "")
        sentences = paragraph.get("sentences", [])

        if not sentences or not context:
            continue

        # Section boundary: save current chunk, start fresh, clear overlap
        if context != current_context:
            _save_current_chunk()
            current_chunk_sentences = []
            current_context = context
            num_overlap_sentences = 0
            overlap_buffer.clear()

        # Find semantic breakpoints within this paragraph
        breakpoints = compute_similarity_breakpoints(sentences, similarity_threshold)

        # Process sentences using breakpoints as hints
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if this is a semantic breakpoint
            is_breakpoint = i in breakpoints and i > 0

            # If breakpoint, save current chunk and start new one
            if is_breakpoint and current_chunk_sentences:
                has_new_content = len(current_chunk_sentences) > num_overlap_sentences
                if has_new_content:
                    _save_current_chunk()
                    current_chunk_sentences = _start_new_chunk_with_overlap()

            # Handle oversized sentences
            if count_tokens(sentence) > max_tokens:
                # Save current chunk first
                if current_chunk_sentences:
                    _save_current_chunk()
                    current_chunk_sentences = _start_new_chunk_with_overlap()

                # Split oversized sentence
                # Note: No overlap between split parts (they're from the same sentence)
                parts = split_oversized_sentence(sentence, max_tokens)
                for part in parts[:-1]:
                    current_chunk_sentences.append(part)
                    _save_current_chunk()
                    # Start fresh, no overlap for split parts
                    current_chunk_sentences = []
                    num_overlap_sentences = 0
                # Keep last part for next iteration
                sentence = parts[-1]

            # Try adding sentence to current chunk
            test_text = " ".join(current_chunk_sentences + [sentence])

            if count_tokens(test_text) <= max_tokens:
                current_chunk_sentences.append(sentence)
            else:
                # Doesn't fit - save and start new chunk
                has_new_content = len(current_chunk_sentences) > num_overlap_sentences

                if has_new_content:
                    _save_current_chunk()
                    current_chunk_sentences = _start_new_chunk_with_overlap()
                    current_chunk_sentences.append(sentence)
                else:
                    # Only overlap, drop it and add sentence
                    current_chunk_sentences = [sentence]
                    num_overlap_sentences = 0

    # Save final chunk
    _save_current_chunk()

    return chunks


# ============================================================================
# FILE I/O
# ============================================================================


def process_single_file(
    file_path: Path,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    overwrite_context: OverwriteContext = None,
) -> tuple[str, int, bool]:
    """Process a single JSON file with semantic chunking.

    Args:
        file_path: Path to input JSON file from Stage 3.
        similarity_threshold: Cosine similarity threshold for detecting topic shifts.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Tuple of (book_name, chunk_count, was_processed).
        was_processed is False if file was skipped due to overwrite decision.
    """
    book_name = file_path.stem

    # Determine output path using threshold-based folder name
    folder_name = get_semantic_folder_name(similarity_threshold)
    output_dir = Path(DIR_FINAL_CHUNKS) / folder_name
    output_path = output_dir / f"{book_name}.json"

    # Check if we should process (overwrite check)
    if overwrite_context is None:
        overwrite_context = OverwriteContext(OverwriteMode.ALL)
    if not overwrite_context.should_overwrite(output_path, logger):
        return book_name, 0, False

    # Read input and process
    with file_path.open("r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    logger.info(f"Processing {book_name} ({len(paragraphs)} paragraphs)")

    chunks = create_semantic_chunks(
        paragraphs, book_name, similarity_threshold=similarity_threshold
    )

    # Save to threshold-based subdirectory (e.g., semantic_0.5/)
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return book_name, len(chunks), True


def run_semantic_chunking(
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    overwrite_context: OverwriteContext = None,
) -> dict[str, int]:
    """Process all files with semantic chunking.

    Main entry point for semantic chunking strategy. Reads paragraph files
    from Stage 3 output and creates semantically-coherent chunks.

    Args:
        similarity_threshold: Cosine similarity threshold (0.0-1.0) for detecting
            topic shifts. Lower = fewer splits (larger chunks).
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts (only includes processed files).

    Raises:
        Exception: Re-raises any exception from file processing (fail-fast).
    """
    input_files = get_file_list(DIR_NLP_CHUNKS, "json")
    results = {}
    skipped_count = 0

    folder_name = get_semantic_folder_name(similarity_threshold)
    logger.info(f"Starting semantic chunking...")
    logger.info(f"Processing {len(input_files)} files")
    logger.info(f"Output folder: {folder_name}/")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    logger.info(f"Overlap sentences: {OVERLAP_SENTENCES}")
    logger.info(f"Token safeguard: {EMBEDDING_MAX_INPUT_TOKENS} (embedding model limit)")

    for file_path in input_files:
        book_name, chunk_count, was_processed = process_single_file(
            file_path, similarity_threshold, overwrite_context
        )
        if was_processed:
            results[book_name] = chunk_count
            logger.info(f"  {book_name}: {chunk_count} chunks")
        else:
            skipped_count += 1

    logger.info(f"Semantic chunking complete: {sum(results.values())} total chunks")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files (already exist)")
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    logger.info("Starting semantic chunking (standalone)...")
    stats = run_semantic_chunking()
    logger.info(f"Completed! Processed {len(stats)} books")
    logger.info(f"Total chunks created: {sum(stats.values())}")
