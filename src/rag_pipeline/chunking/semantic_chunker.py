"""Semantic chunking: Split text by embedding similarity.

## RAG Theory: Semantic Coherence

Traditional chunking (fixed tokens, sentence count) often splits mid-topic,
fragmenting semantically related content across chunks. Semantic chunking
uses embedding similarity to detect topic boundaries:

1. Embed each sentence using the same model as Stage 5
2. Compute cosine similarity between adjacent sentences
3. Split when similarity drops below mean - (coefficient * std) (statistical outlier)
4. Still respect section boundaries and token limits

## Section-Scope vs Document-Scope

This implementation uses SECTION-SCOPE breakpoint detection:
- LangChain/LlamaIndex use document-scope (embed entire document at once)
- Document-scope suits short documents (articles, pages)
- For books with chapters/sections, section-scope is more appropriate because:
  1. Authors create sections to group coherent topics (strong structural prior)
  2. Cross-section comparison introduces noise (unrelated topics may have similar embeddings)
  3. The std deviation threshold needs a homogeneous distribution to work well
  4. Qu et al. 2024 found semantic chunking helps most on "high topic diversity" content,
     which exists WITHIN sections (sub-topics), not across them

## Library Usage

- numpy: Cosine similarity computation, mean/std calculation
- src.rag_pipeline.embedding.embedder: API embeddings (same as Stage 5)
- Reuses section_chunker helpers (split_oversized_sentence, parse_section_name)

## Data Flow

1. Load paragraphs from DIR_NLP_CHUNKS/{book}.json
2. Group paragraphs by section (context)
3. For each section:
   a. Aggregate all sentences
   b. Embed all sentences (batch API call)
   c. Compute pairwise cosine similarity
   d. Find breakpoints where similarity < mean - (coefficient * std)
4. Build chunks respecting breakpoints and token limits
5. Add 2-sentence overlap between chunks (same section only)
6. Save to DIR_FINAL_CHUNKS/semantic_std{coefficient}/{book}.json
"""

import json
import numpy as np
from pathlib import Path
from collections import deque

from src.config import (
    DIR_FINAL_CHUNKS,
    DIR_NLP_CHUNKS,
    EMBEDDING_MAX_INPUT_TOKENS,
    OVERLAP_SENTENCES,
    CHUNK_ID_PREFIX,
    CHUNK_ID_SEPARATOR,
    SEMANTIC_STD_COEFFICIENT,
    get_semantic_folder_name,
)
from src.rag_pipeline.embedder import embed_texts
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
    std_coefficient: float = SEMANTIC_STD_COEFFICIENT,
) -> list[int]:
    """Find semantic breakpoints by embedding similarity using standard deviation.

    Computes cosine similarity between adjacent sentences and identifies
    indices where similarity is statistically low (below mean - k*std),
    indicating topic shifts.

    Args:
        sentences: List of sentence strings.
        std_coefficient: Number of standard deviations below mean for breakpoint.
            Higher values = fewer splits (only extreme drops trigger breakpoints).
            Default is 3.0 (statistically significant drops only).

    Returns:
        List of indices where new chunks should start (0-indexed).
        Always includes 0 (first chunk starts at index 0).

    Example:
        >>> sentences = ["Intro.", "More intro.", "New topic!", "Continues."]
        >>> # If similarity between "More intro." and "New topic!" < mean - 3*std
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

    # Compute cutoff using standard deviation
    similarities_array = np.array(similarities)
    mean_sim = np.mean(similarities_array)
    std_sim = np.std(similarities_array)
    cutoff = mean_sim - (std_coefficient * std_sim)

    # Find breakpoints where similarity drops below cutoff
    breakpoints = [0]  # Always start at 0
    for i, sim in enumerate(similarities):
        if sim < cutoff:
            breakpoints.append(i + 1)  # Start new chunk at next sentence
            logger.debug(f"Semantic split at index {i+1}, similarity={sim:.3f} < cutoff={cutoff:.3f}")

    return breakpoints


# ============================================================================
# CHUNK CREATION
# ============================================================================


def _create_chunk_dict(
    text: str,
    context: str,
    book_name: str,
    chunk_id: int,
    std_coefficient: float,
) -> dict:
    """Create standardized chunk dictionary with semantic strategy marker.

    Args:
        text: Chunk text content.
        context: Hierarchical context string (Book > Chapter > Section).
        book_name: Book identifier.
        chunk_id: Sequential chunk number.
        std_coefficient: Std coefficient used for this chunking run.

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
        "chunking_strategy": get_semantic_folder_name(std_coefficient),
    }


def create_semantic_chunks(
    paragraphs: list[dict],
    book_name: str,
    max_tokens: int = EMBEDDING_MAX_INPUT_TOKENS,
    overlap_sentences: int = OVERLAP_SENTENCES,
    std_coefficient: float = SEMANTIC_STD_COEFFICIENT,
) -> list[dict]:
    """Create chunks using semantic similarity breakpoints at SECTION scope.

    Algorithm:
    1. Group paragraphs by section (context)
    2. For each section, aggregate all sentences and find semantic breakpoints
    3. Build chunks from breakpoint segments (semantic coherence drives size)
    4. Add sentence overlap between chunks (same section only)

    Section-scope (vs paragraph-scope) produces more meaningful similarity
    distributions for the std deviation threshold. With 5 sentences per paragraph,
    variance is minimal and breakpoints rarely trigger. With 50+ sentences per
    section, outliers become statistically meaningful.

    Args:
        paragraphs: List of paragraph dicts with 'context', 'sentences' keys.
        book_name: Book identifier.
        max_tokens: Safeguard limit for embedding model (default: 8191).
        overlap_sentences: Number of sentences to overlap between chunks.
        std_coefficient: Standard deviation coefficient for breakpoint detection.

    Returns:
        List of chunk dicts ready for embedding.
    """
    chunks = []
    chunk_id = 0

    # --- PHASE 1: Group paragraphs by section ---
    sections: dict[str, list[str]] = {}  # context -> list of all sentences
    section_order: list[str] = []  # preserve section order

    for paragraph in paragraphs:
        context = paragraph.get("context", "")
        sentences = paragraph.get("sentences", [])

        if not sentences or not context:
            continue

        if context not in sections:
            sections[context] = []
            section_order.append(context)

        # Aggregate all sentences from this paragraph into the section
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sections[context].append(sentence)

    # --- PHASE 2: Process each section with section-scope breakpoints ---
    for context in section_order:
        section_sentences = sections[context]

        if not section_sentences:
            continue

        # Find semantic breakpoints across ALL sentences in this section
        breakpoints = compute_similarity_breakpoints(section_sentences, std_coefficient)
        breakpoint_set = set(breakpoints)

        logger.debug(
            f"Section '{context[-50:]}': {len(section_sentences)} sentences, "
            f"{len(breakpoints)} breakpoints"
        )

        # Build chunks from this section
        current_chunk_sentences: list[str] = []
        overlap_buffer: deque[str] = deque(
            maxlen=overlap_sentences if overlap_sentences > 0 else None
        )
        num_overlap_sentences = 0

        def _save_chunk():
            nonlocal chunk_id, num_overlap_sentences
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(_create_chunk_dict(
                    text=chunk_text,
                    context=context,
                    book_name=book_name,
                    chunk_id=chunk_id,
                    std_coefficient=std_coefficient,
                ))
                chunk_id += 1

                if overlap_sentences > 0:
                    overlap_buffer.clear()
                    overlap_buffer.extend(current_chunk_sentences[-overlap_sentences:])

                num_overlap_sentences = 0

        def _start_with_overlap() -> list[str]:
            nonlocal num_overlap_sentences
            if overlap_sentences > 0 and len(overlap_buffer) > 0:
                num_overlap_sentences = len(overlap_buffer)
                return list(overlap_buffer)
            num_overlap_sentences = 0
            return []

        for i, sentence in enumerate(section_sentences):
            # Check if this is a semantic breakpoint (skip index 0)
            is_breakpoint = i in breakpoint_set and i > 0

            # If breakpoint, save current chunk and start new one
            if is_breakpoint and current_chunk_sentences:
                has_new_content = len(current_chunk_sentences) > num_overlap_sentences
                if has_new_content:
                    _save_chunk()
                    current_chunk_sentences = _start_with_overlap()

            # Handle oversized sentences
            if count_tokens(sentence) > max_tokens:
                if current_chunk_sentences:
                    _save_chunk()
                    current_chunk_sentences = _start_with_overlap()

                parts = split_oversized_sentence(sentence, max_tokens)
                for part in parts[:-1]:
                    current_chunk_sentences.append(part)
                    _save_chunk()
                    current_chunk_sentences = []
                    num_overlap_sentences = 0
                sentence = parts[-1]

            # Try adding sentence to current chunk
            test_text = " ".join(current_chunk_sentences + [sentence])

            if count_tokens(test_text) <= max_tokens:
                current_chunk_sentences.append(sentence)
            else:
                has_new_content = len(current_chunk_sentences) > num_overlap_sentences
                if has_new_content:
                    _save_chunk()
                    current_chunk_sentences = _start_with_overlap()
                    current_chunk_sentences.append(sentence)
                else:
                    current_chunk_sentences = [sentence]
                    num_overlap_sentences = 0

        # Save final chunk for this section
        _save_chunk()

    return chunks


# ============================================================================
# FILE I/O
# ============================================================================


def process_single_file(
    file_path: Path,
    std_coefficient: float = SEMANTIC_STD_COEFFICIENT,
    overwrite_context: OverwriteContext = None,
) -> tuple[str, int, bool]:
    """Process a single JSON file with semantic chunking.

    Args:
        file_path: Path to input JSON file from Stage 3.
        std_coefficient: Standard deviation coefficient for breakpoint detection.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Tuple of (book_name, chunk_count, was_processed).
        was_processed is False if file was skipped due to overwrite decision.
    """
    book_name = file_path.stem

    # Determine output path using coefficient-based folder name
    folder_name = get_semantic_folder_name(std_coefficient)
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
        paragraphs, book_name, std_coefficient=std_coefficient
    )

    # Save to coefficient-based subdirectory (e.g., semantic_std3/)
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return book_name, len(chunks), True


def run_semantic_chunking(
    std_coefficient: float = SEMANTIC_STD_COEFFICIENT,
    overwrite_context: OverwriteContext = None,
) -> dict[str, int]:
    """Process all files with semantic chunking.

    Main entry point for semantic chunking strategy. Reads paragraph files
    from Stage 3 output and creates semantically-coherent chunks.

    Args:
        std_coefficient: Standard deviation coefficient for breakpoint detection.
            Higher = fewer splits (only extreme drops). Default is 3.0.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dict mapping book names to chunk counts (only includes processed files).

    Raises:
        Exception: Re-raises any exception from file processing (fail-fast).
    """
    input_files = get_file_list(DIR_NLP_CHUNKS, "json")
    results = {}
    skipped_count = 0

    folder_name = get_semantic_folder_name(std_coefficient)
    logger.info(f"Starting semantic chunking...")
    logger.info(f"Processing {len(input_files)} files")
    logger.info(f"Output folder: {folder_name}/")
    logger.info(f"Std coefficient: {std_coefficient}")
    logger.info(f"Overlap sentences: {OVERLAP_SENTENCES}")
    logger.info(f"Token safeguard: {EMBEDDING_MAX_INPUT_TOKENS} (embedding model limit)")

    for file_path in input_files:
        book_name, chunk_count, was_processed = process_single_file(
            file_path, std_coefficient, overwrite_context
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
