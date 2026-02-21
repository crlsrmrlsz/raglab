"""
Sequential text chunker for RAG system with sentence overlap.
Processes paragraphs in reading order, respecting section boundaries and token limits.
Includes configurable overlap between consecutive chunks within same section.
"""

import json
from pathlib import Path
from collections import deque

from src.config import (
    DIR_FINAL_CHUNKS,
    MAX_CHUNK_TOKENS,
    DIR_NLP_CHUNKS,
    OVERLAP_SENTENCES,
    MAX_LOOP_ITERATIONS,
    CHUNK_ID_PREFIX,
    CHUNK_ID_SEPARATOR,
)
from src.shared.tokens import count_tokens
from src.shared.files import get_file_list, setup_logging, OverwriteContext, OverwriteMode

logger = setup_logging(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def split_oversized_sentence(sentence: str, max_tokens: int) -> list[str]:
    """
    Split a sentence that exceeds token limit.
    
    Strategy:
    1. Try splitting by punctuation ("; ", ": ", ", ")
    2. Fallback to word boundary splitting
    
    Args:
        sentence: Text to split
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of sentence parts, each ≤ max_tokens (or single oversized part if splitting fails)
    """
    # If sentence fits, no splitting needed
    if count_tokens(sentence) <= max_tokens:
        return [sentence]
    
    # Try splitting by punctuation marks (in priority order)
    for separator in ["; ", ": ", ", "]:
        if separator not in sentence:
            continue
            
        parts = sentence.split(separator)
        result = []
        current = ""
        
        for i, part in enumerate(parts):
            # Reconstruct with separator (add separator prefix except for first part)
            if i > 0:
                part = separator + part
            
            # Try adding part to current chunk
            test_text = current + part if current else part
            
            if count_tokens(test_text) <= max_tokens:
                current = test_text
            else:
                # Part doesn't fit
                if current:
                    result.append(current.strip())
                    current = part.lstrip(separator)  # Start fresh without separator
                else:
                    # Single part too large, abandon this separator
                    break
        
        # Add remaining text
        if current:
            result.append(current.strip())
        
        # If we successfully split, return results
        if len(result) > 1:
            return result
    
    # Fallback: split by word boundaries
    return split_by_words(sentence, max_tokens)


def split_by_words(text: str, max_tokens: int) -> list[str]:
    """
    Split text at word boundaries to respect token limit.
    Last resort when punctuation splitting fails.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks (guaranteed to make progress, even if parts exceed max_tokens)
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    
    words = text.split()
    
    # Edge case: single word or no spaces - can't split further
    if len(words) <= 1:
        logger.warning(f"Unsplittable text ({count_tokens(text)} tokens), including anyway")
        return [text]  # Return as-is, accept token overflow
    
    chunks = []
    current_words = []
    
    for word in words:
        test_text = " ".join(current_words + [word])
        
        if count_tokens(test_text) <= max_tokens:
            current_words.append(word)
        else:
            # Save current chunk and start new one
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                # Single word exceeds limit (extremely rare) - include anyway to make progress
                logger.warning(f"Single word exceeds limit ({count_tokens(word)} tokens): {word[:50]}...")
                chunks.append(word)
    
    # Add remaining words
    if current_words:
        chunks.append(" ".join(current_words))
    
    return chunks if chunks else [text]  # Ensure we return something


def parse_section_name(context: str) -> str:
    """Extract section name from hierarchical context.

    Context format: "BookTitle > Chapter > Section > Subsection"
    Returns the last component (most specific section).

    Args:
        context: Hierarchical context string.

    Returns:
        Section name or empty string.
    """
    if not context:
        return ""
    return context.split(">")[-1].strip()


# ============================================================================
# CORE CHUNKING LOGIC WITH OVERLAP
# ============================================================================

def create_chunks_from_paragraphs(
    paragraphs: list[dict],
    book_name: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_sentences: int = OVERLAP_SENTENCES
) -> list[dict]:
    """
    Process paragraphs sequentially to create chunks with overlap.
    
    Algorithm:
    1. Read paragraphs in order (preserves reading sequence)
    2. When context changes → save current chunk, start new chunk, clear overlap
    3. For each sentence:
       - Add to chunk if it fits
       - If doesn't fit but chunk has content → save chunk, start new chunk with overlap
       - If sentence too large → split it
    4. Overlap: Last N sentences from previous chunk initialize next chunk (same section only)
    
    Args:
        paragraphs: List of paragraph dicts with 'context', 'sentences', etc.
        book_name: Book identifier
        max_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks (0 = no overlap)
        
    Returns:
        List of chunk dicts ready for embedding
    """
    chunks = []
    current_chunk_sentences = []  # Track sentences in current chunk
    current_context = None
    chunk_id = 0
    sentence_counter = 0  # For debugging
    num_overlap_sentences = 0  # Track how many sentences are from overlap (not new content)
    
    # Overlap buffer: stores last N sentences from previous chunk (within same section)
    overlap_buffer: deque[str] = deque(maxlen=overlap_sentences if overlap_sentences > 0 else None)
    
    def _save_current_chunk():
        """Helper to save current chunk and update overlap buffer"""
        nonlocal chunk_id, num_overlap_sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(_create_chunk_dict(
                text=chunk_text,
                context=current_context,
                book_name=book_name,
                chunk_id=chunk_id,
                overlap_sentences=overlap_sentences,
            ))
            chunk_id += 1

            # Update overlap buffer with last N sentences (for next chunk in same section)
            if overlap_sentences > 0:
                overlap_buffer.clear()
                overlap_buffer.extend(current_chunk_sentences[-overlap_sentences:])

            # Reset overlap counter after saving
            num_overlap_sentences = 0
    
    def _start_new_chunk_with_overlap():
        """Helper to initialize new chunk with overlap from previous chunk"""
        nonlocal num_overlap_sentences
        if overlap_sentences > 0 and len(overlap_buffer) > 0:
            # Start new chunk with overlapping sentences from previous chunk
            num_overlap_sentences = len(overlap_buffer)
            return list(overlap_buffer)
        num_overlap_sentences = 0
        return []
    
    # Process each paragraph sequentially
    for paragraph_idx, paragraph in enumerate(paragraphs):
        context = paragraph.get("context", "")
        sentences = paragraph.get("sentences", [])
        
        # Skip paragraphs without content
        if not sentences or not context:
            continue
        
        # CONTEXT CHANGE: Save current chunk, start fresh, clear overlap
        if context != current_context:
            _save_current_chunk()
            current_chunk_sentences = []
            current_context = context
            num_overlap_sentences = 0
            overlap_buffer.clear()  # Don't overlap across section boundaries
        
        # Process each sentence in the paragraph
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_counter += 1
            iteration_count = 0  # Safety counter to detect infinite loops
            
            # Try to add sentence to current chunk
            while sentence:  # Loop until sentence is fully processed
                iteration_count += 1
                
                # SAFETY CHECK: Detect infinite loops
                if iteration_count > MAX_LOOP_ITERATIONS:
                    logger.warning(
                        f"Max iterations reached. Paragraph {paragraph_idx}, Sentence {sent_idx}, "
                        f"Context: {context}, Sentence ({count_tokens(sentence)} tokens): {sentence[:200]}..., "
                        f"Current chunk size: {len(current_chunk_sentences)} sentences, "
                        f"Overlap sentences: {num_overlap_sentences}"
                    )
                    # Force progress by including sentence anyway
                    current_chunk_sentences.append(sentence)
                    num_overlap_sentences = 0
                    break
                
                # Build test chunk
                test_sentences = current_chunk_sentences + [sentence]
                test_chunk_text = " ".join(test_sentences)
                
                # Check if sentence fits
                if count_tokens(test_chunk_text) <= max_tokens:
                    # SUCCESS: Sentence fits in current chunk
                    current_chunk_sentences.append(sentence)
                    num_overlap_sentences = 0  # We've added new content, no longer just overlap
                    break  # Move to next sentence
                
                else:
                    # DOESN'T FIT: Handle based on current chunk state
                    
                    # Check if we have NEW content (not just overlap)
                    # IMPORTANT: This prevents infinite loops when overlap + sentence > max_tokens
                    # In that case, we drop the overlap and try just the sentence
                    has_new_content = len(current_chunk_sentences) > num_overlap_sentences
                    
                    if has_new_content:
                        # Chunk has new content → save it and retry sentence in new chunk WITH OVERLAP
                        _save_current_chunk()
                        current_chunk_sentences = _start_new_chunk_with_overlap()
                        # Loop continues with same sentence (will be added to new chunk)
                    
                    else:
                        # Chunk is empty OR only has overlap (no new content yet)
                        # This means: overlap + sentence > max_tokens
                        # Solution: drop overlap and try just the sentence
                        
                        if current_chunk_sentences:
                            # We have overlap but it doesn't help - clear it
                            current_chunk_sentences = []
                            num_overlap_sentences = 0
                            # Loop continues - will retry sentence without overlap
                        else:
                            # Truly empty, sentence itself is too large → split sentence
                            original_sentence = sentence
                            original_token_count = count_tokens(sentence)
                            sentence_parts = split_oversized_sentence(sentence, max_tokens)
                            
                            # CRITICAL: Check if splitting actually worked
                            if len(sentence_parts) == 1 and sentence_parts[0] == original_sentence:
                                # Splitting failed - sentence unchanged
                                # Force progress by including it anyway (accept token overflow)
                                logger.warning(
                                    f"Could not split sentence ({original_token_count} tokens), "
                                    f"preview: {sentence[:100]}..."
                                )
                                current_chunk_sentences.append(sentence)
                                num_overlap_sentences = 0
                                break  # Exit while loop, move to next sentence
                            
                            # Splitting succeeded - process parts
                            for part in sentence_parts[:-1]:
                                current_chunk_sentences.append(part)
                                num_overlap_sentences = 0
                                _save_current_chunk()
                                current_chunk_sentences = _start_new_chunk_with_overlap()
                            
                            # Keep last part for next iteration
                            sentence = sentence_parts[-1]
                            # Loop continues with last part (which should be smaller)
    
    # Save final chunk if it has content
    _save_current_chunk()
    
    return chunks


def _create_chunk_dict(
    text: str,
    context: str,
    book_name: str,
    chunk_id: int,
    overlap_sentences: int,
) -> dict:
    """
    Create standardized chunk dictionary.

    Args:
        text: Chunk text content
        context: Hierarchical context string
        book_name: Book identifier
        chunk_id: Sequential chunk number
        overlap_sentences: Actual overlap count used for this chunking run

    Returns:
        Chunk dictionary with metadata
    """
    return {
        "chunk_id": f"{book_name}{CHUNK_ID_SEPARATOR}{CHUNK_ID_PREFIX}_{chunk_id}",
        "book_id": book_name,
        "context": context,
        "section": parse_section_name(context),
        "text": text,
        "token_count": count_tokens(text),
        "chunking_strategy": f"sequential_overlap_{overlap_sentences}"
    }


# ============================================================================
# FILE I/O
# ============================================================================

def process_single_file(
    file_path: Path,
    overwrite_context: OverwriteContext = None,
) -> tuple[str, int, bool]:
    """Process a single JSON file and create chunks.

    Args:
        file_path: Path to input JSON file.
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Tuple of (book_name, chunk_count, was_processed).
        was_processed is False if file was skipped due to overwrite decision.
    """
    book_name = file_path.stem

    # Determine output path
    output_dir = Path(DIR_FINAL_CHUNKS) / "section"
    output_path = output_dir / f"{book_name}.json"

    # Check if we should process (overwrite check)
    if overwrite_context is None:
        overwrite_context = OverwriteContext(OverwriteMode.ALL)
    if not overwrite_context.should_overwrite(output_path, logger):
        return book_name, 0, False

    # Read paragraphs from file
    with file_path.open("r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    # Create chunks with overlap
    chunks = create_chunks_from_paragraphs(paragraphs, book_name)

    # Save chunks to output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return book_name, len(chunks), True


def run_section_chunking(
    overwrite_context: OverwriteContext = None,
) -> dict[str, int]:
    """Process all JSON files in DIR_NLP_CHUNKS directory.

    Args:
        overwrite_context: Context for handling existing file overwrites.

    Returns:
        Dictionary mapping book names to chunk counts (only includes processed files).

    Raises:
        Exception: Re-raises any exception from file processing (fail-fast).
    """
    input_files = get_file_list(DIR_NLP_CHUNKS, "json")
    results = {}
    skipped_count = 0

    logger.info(f"Processing {len(input_files)} files...")
    logger.info(f"Max tokens per chunk: {MAX_CHUNK_TOKENS}")
    logger.info(f"Sentence overlap: {OVERLAP_SENTENCES}")

    for file_path in input_files:
        book_name, chunk_count, was_processed = process_single_file(
            file_path, overwrite_context
        )
        if was_processed:
            results[book_name] = chunk_count
            logger.info(f"{book_name}: {chunk_count} chunks")
        else:
            skipped_count += 1

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files (already exist)")
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting sequential chunking with overlap...")

    stats = run_section_chunking()

    logger.info(f"Completed! Processed {len(stats)} books")
    logger.info(f"Total chunks created: {sum(stats.values())}")